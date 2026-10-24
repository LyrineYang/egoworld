"""R02/R08/R10: real concurrent tasks, real commits, injected model work only."""
import json
import threading
import time
from pathlib import Path
import pytest
from egoworld.pipeline import driver, stages
from egoworld.pipeline.artifacts import WriterActor
from egoworld.pipeline.state_store import get_clip_state
from egoworld.utils.errors import TransientIOError


@pytest.fixture
def pipeline_case(tmp_path, monkeypatch, thread_backend):
    config = {"num_gpus": 1, "run_id": "A", "paths": {"output_root": str(tmp_path / "output"), "state_db_path": str(tmp_path / "state.db")},
              "operators": {"sam2": {"enabled": True, "params": {}}},
              "backpressure": {"max_in_flight_cpu": 2, "max_in_flight_gpu": 2, "max_in_flight_write": 4},
              "retry": {"max_retries": 1, "base_delay_s": 0}, "execution": {"max_active_clips": 4}}
    cfg, videos, clips = (tmp_path / name for name in ("config.json", "videos.jsonl", "clips.jsonl"))
    videos.write_text(json.dumps({"video_id": "v", "path": "unused.mp4", "fps": 10}) + "\n")
    rows = [{"clip_id": f"c{i}", "video_id": "v", "start_s": 0., "end_s": 1., "frame_start": 0, "frame_end": 10} for i in range(3)]
    monkeypatch.setattr(stages.PrepareActor, "prepare", lambda self, job: {"video_path": "prepared.mp4", "frames": 10})
    def infer(self, path, start, end):
        return {"frames": [], "instances": [], "processed_frames": 10, "empty_mask_rate": 1.}
    monkeypatch.setattr(stages.Sam2Operator, "run", infer)
    created = []
    def factory(config):
        backend = thread_backend(config); created.append(backend); return backend
    def run():
        cfg.write_text(json.dumps(config))
        clips.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return driver.run_pipeline(str(cfg), str(videos), str(clips), backend_factory=factory)
    return {"config": config, "root": tmp_path, "run": run, "rows": rows, "backends": created}


def test_resume_skips_valid_stages_and_detects_corrupt_parquet(pipeline_case):
    case = pipeline_case; case["run"]()
    root = case["root"] / "output/run_id=A/video_id=v/clip_id=c0"
    output = root / "instances.parquet"; original = output.stat().st_mtime_ns
    case["run"](); assert output.stat().st_mtime_ns == original
    output.write_bytes(b"broken")
    case["run"](); assert output.stat().st_mtime_ns != original
    assert json.loads((root / "stage_sam2.json").read_text())["files"]["instances.parquet"]["sha256"]


def test_missing_final_qc_is_rebuilt_without_model_inference(pipeline_case, monkeypatch):
    case = pipeline_case; case["run"]()
    root = case["root"] / "output/run_id=A/video_id=v/clip_id=c0"
    (root / "qc.json").unlink()
    monkeypatch.setattr(stages.ModelActor, "process", lambda *args: pytest.fail("Committed model work must be reused"))
    case["run"]()
    assert json.loads((root / "qc.json").read_text())["status"] == "review"


def test_independent_clips_finish_after_model_failure(pipeline_case, monkeypatch):
    original = stages.ModelActor.process
    def fail(self, job, prepared, inputs):
        if job["clip"]["clip_id"] == "c0": raise ValueError("model failure")
        return original(self, job, prepared, inputs)
    monkeypatch.setattr(stages.ModelActor, "process", fail)
    case = pipeline_case
    with pytest.raises(RuntimeError, match="failed for 1 clip"): case["run"]()
    assert get_clip_state(case["config"]["paths"]["state_db_path"], "c1", run_id="A").status == "Done"
    assert case["backends"][0].closed
    assert json.loads((case["root"] / "output/run_id=A/run_summary.json").read_text())["failed_clips"] == 1


@pytest.mark.parametrize("where", ["model", "writer", "prepare"])
def test_transient_failures_retry_without_restarting_successful_stages(pipeline_case, monkeypatch, where):
    cls, method = {"model": (stages.ModelActor, "process"), "writer": (WriterActor, "write_stage"), "prepare": (stages.PrepareActor, "prepare")}[where]
    original = getattr(cls, method); calls = []
    def fail_once(self, job, *args):
        calls.append(job["clip"]["clip_id"])
        if len(calls) == 1: raise TransientIOError("temporary")
        return original(self, job, *args)
    monkeypatch.setattr(cls, method, fail_once)
    pipeline_case["run"](); assert len(calls) == 4


def test_preparation_overlaps_inference_and_writer(pipeline_case, monkeypatch):
    # Barrier-based proof of overlap; independent of machine speed/timing assertions.
    infer_running, second_prepared = threading.Event(), threading.Event()
    original = stages.ModelActor.process
    def prepare(self, job):
        if job["clip"]["clip_id"] == "c1":
            assert infer_running.wait(3); second_prepared.set()
        return {"video_path": "prepared.mp4", "frames": 10}
    def infer(self, job, prepared, inputs):
        if job["clip"]["clip_id"] == "c0":
            infer_running.set(); assert second_prepared.wait(3)
        return original(self, job, prepared, inputs)
    monkeypatch.setattr(stages.PrepareActor, "prepare", prepare)
    monkeypatch.setattr(stages.ModelActor, "process", infer)
    pipeline_case["run"](); assert second_prepared.is_set()


def test_queue_bounds_and_identity_change(pipeline_case):
    case = pipeline_case; case["run"]()
    summary = json.loads((case["root"] / "output/run_id=A/run_summary.json").read_text())
    assert summary["queue_peaks"]["active_clips"] <= 4
    assert summary["queue_peaks"]["uncommitted_results"] <= 4
    case["config"]["operators"]["sam2"]["params"] = {"precision": "fp16"}
    with pytest.raises(ValueError, match="new run_id"): case["run"]()


def test_delayed_retry_does_not_block_other_clips(pipeline_case, monkeypatch):
    case = pipeline_case; case["config"]["retry"]["base_delay_s"] = .2
    original = stages.ModelActor.process; log = []
    def fail(self, job, prepared, inputs):
        cid = job["clip"]["clip_id"]; log.append(cid)
        if cid == "c0" and log.count(cid) == 1: raise TransientIOError("temporary")
        return original(self, job, prepared, inputs)
    monkeypatch.setattr(stages.ModelActor, "process", fail)
    case["run"]()
    assert log.index("c1") < max(i for i, cid in enumerate(log) if cid == "c0")


def full_packet(stage, clip):
    import numpy as np
    base = {"frame_index": clip["frame_start"], "timestamp_s": clip["frame_start"] / clip["fps"]}
    if stage == "sam2":
        instance = dict(base, object_id=1, label="right hand", detection_score=.9,
                        mask_rle='{"size":[1,1],"counts":[0,1]}', area_px=1, bbox_xyxy=[0, 0, 1, 1])
        return {"tables": {"masks": [dict(base, mask_rle=instance["mask_rle"])], "instances": [instance],
                           "tracks": [dict(base, object_id=1, label="right hand", area_px=1, center_xy=[.5, .5])]},
                "stats": {"processed_frames": 10, "empty_mask_rate": 0.}}
    if stage == "hamer":
        return {"tables": {"hand_pose": [dict(base, object_id=1, handedness="right", joints_3d=[0.] * 63,
            global_orient=np.eye(3).ravel().tolist(), mano_pose=np.tile(np.eye(3), (15, 1, 1)).ravel().tolist(),
            betas=[0.] * 10, camera_translation=[0., 0., 1.], coord_frame="hand_root", length_unit="meters",
            scale_status="model_prior", joint_order="openpose21")]}}
    if stage == "foundationpose":
        return {"tables": {"object_pose": [dict(base, object_id=1, label="cup", transform=np.eye(4).ravel().tolist(),
                    coord_frame="object_to_camera", length_unit="meters", mesh_path="fixture.obj")]}}
    if stage == "dex_retarget":
        return {"tables": {"mapping": [dict(base, object_id=1, handedness="right", robot_id="test-hand",
            joint_names=["a"], joint_positions=[.1], joint_units=["radians"], source_scale_status="model_prior")]}}
    return {"tables": {"reconstruction": [dict(base, view_index=0, camera_to_world=np.eye(4).ravel().tolist(),
        intrinsics=np.eye(3).ravel().tolist(), image_width=1, image_height=1,
        pointcloud_path="reconstruction-pointcloud.npz", coord_frame="first_view_world", length_unit="relative", scale_status="unscaled")]},
        "arrays": {"reconstruction-pointcloud.npz": {"points": np.zeros((1, 3)), "confidence": np.ones(1), "view_index": np.zeros(1, dtype=int)}}}


def enable_all(case):
    ops = case["config"]["operators"]
    ops.update({"hamer": {"enabled": True, "params": {"checkpoint": "not-loaded.ckpt"}},
                "foundationpose": {"enabled": True, "params": {"assets_manifest": "not-read.json", "objects": {}}},
                "dex_retarget": {"enabled": True, "params": {"config": "not-read.yaml", "robot_id": "fixture",
                    "human_to_robot_rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "handedness": "right", "joint_units": "radians"}},
                "fast3r": {"enabled": True, "params": {"model_name_or_path": "not-loaded"}}})


def test_all_branches_commit_typed_outputs_and_resume_per_stage(pipeline_case, monkeypatch):
    import pyarrow.parquet as pq
    from egoworld.contracts import STAGE_ARTIFACTS
    case = pipeline_case; enable_all(case); calls = []
    def process(self, job, prepared, inputs):
        stage = job["stage"]; calls.append((job["clip"]["clip_id"], stage))
        dependencies = {"hamer": ("sam2",), "foundationpose": ("sam2",), "dex_retarget": ("hamer",)}
        for dependency in dependencies.get(stage, ()):
            assert Path(inputs[dependency]["directory"], f"stage_{dependency}.json").is_file()
        return full_packet(stage, job["clip"])
    monkeypatch.setattr(stages.ModelActor, "process", process)
    case["run"](); assert len(calls) == 15
    root = case["root"] / "output/run_id=A/video_id=v/clip_id=c0"
    for artifacts in STAGE_ARTIFACTS.values():
        for artifact in artifacts: assert pq.ParquetFile(root / f"{artifact}.parquet").metadata.num_rows > 0
    calls.clear(); case["run"](); assert not calls
    # Corrupt only the hand branch: segmentation and independent reconstruction stay cached.
    (root / "hand_pose.parquet").write_bytes(b"corrupt")
    case["run"](); assert set(calls) == {("c0", "hamer"), ("c0", "dex_retarget")}


def test_failed_branch_does_not_cancel_independent_reconstruction(pipeline_case, monkeypatch):
    case = pipeline_case; enable_all(case)
    def process(self, job, prepared, inputs):
        if job["stage"] == "sam2": raise ValueError("segmentation failed")
        return full_packet(job["stage"], job["clip"])
    monkeypatch.setattr(stages.ModelActor, "process", process)
    with pytest.raises(RuntimeError, match="failed for 3 clip"): case["run"]()
    root = case["root"] / "output/run_id=A/video_id=v/clip_id=c0"
    meta = json.loads((root / "meta.json").read_text())
    assert meta["stages"] == ["fast3r"]
    assert set(meta["errors"]) == {"sam2", "hamer", "foundationpose", "dex_retarget"}


def test_second_interruption_does_not_restore_stale_descendants(pipeline_case, monkeypatch):
    from egoworld.config import load_config
    from egoworld.pipeline.engine import Engine
    case = pipeline_case; enable_all(case)
    monkeypatch.setattr(stages.ModelActor, "process", lambda self, job, *_: full_packet(job["stage"], job["clip"]))
    case["run"]()
    root = case["root"] / "output/run_id=A/video_id=v/clip_id=c0"
    (root / "instances.parquet").write_bytes(b"corrupt")
    fingerprint = json.loads((root / "stage_hamer.json").read_text())["fingerprint"]
    config = load_config(str(case["root"] / "config.json")).resolved()
    clip = dict(case["rows"][0], fps=10.)
    interrupted = Engine(config, None, fingerprint, lambda **_: None)
    interrupted.admit(clip)
    ctx = interrupted.active["c0"]
    assert ctx["states"]["fast3r"] == "Done"
    assert not (root / "stage_hamer.json").exists()
    packet = full_packet("sam2", clip)
    for artifact in ("instances", "tracks"): packet["tables"][artifact][0]["object_id"] = 2
    from dataclasses import asdict
    WriterActor(asdict(config)).write_stage(interrupted.job(ctx, "sam2"), packet)
    # Stop here: new SAM2 is committed; HaMeR/FP/Dex have not been rerun.
    restarted = Engine(config, None, fingerprint, lambda **_: None)
    restarted.admit(clip)
    assert restarted.skipped == 0
    assert restarted.active["c0"]["states"] == {
        "sam2": "Done", "hamer": "Pending", "foundationpose": "Pending", "fast3r": "Done", "dex_retarget": "Pending"}


def test_resume_repairs_database_after_final_files_committed(pipeline_case, monkeypatch):
    from egoworld.pipeline.state_store import upsert_clip_status, set_stage
    case = pipeline_case; case["run"]()
    db = case["config"]["paths"]["state_db_path"]
    upsert_clip_status(db, "c0", "v", "Running", run_id="A")
    set_stage(db, "A", "c0", "sam2", "Running")
    monkeypatch.setattr(stages.ModelActor, "process", lambda *args: pytest.fail("Do not rerun committed model work"))
    case["run"]()
    assert get_clip_state(db, "c0", run_id="A").status == "Done"
    import sqlite3
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM stage_status WHERE run_id='A' AND clip_id='c0' AND stage='sam2'").fetchone() == ("Done",)
