import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_product_recipes import recipe_case
from test_product_geometry import world_case
from egoworld.product.store import DerivedCache, inspect_cache
from egoworld.utils.hashing import sha256_file, hash_session, register_assets


def test_source_identity_tracks_import_changes_after_reusing_parser(tmp_path, monkeypatch):
    from egoworld.product import identity

    (tmp_path / "operators").mkdir()
    source = tmp_path / "operators" / "sam2_op.py"
    (tmp_path / "first.py").write_text("value = 1\n")
    (tmp_path / "other.py").write_text("value = 2\n")
    monkeypatch.setattr(identity, "__file__", str(tmp_path / "product" / "identity.py"))
    source.write_text("import egoworld.first\n")
    first = identity.source_identity("sam2")
    assert first == identity.source_identity("sam2")
    source.write_text("import egoworld.other\n")
    changed = identity.source_identity("sam2")
    assert set(first) == {"operators/sam2_op.py", "first.py"}
    assert set(changed) == {"operators/sam2_op.py", "other.py"}
    assert first["operators/sam2_op.py"] != changed["operators/sam2_op.py"]


def test_rejected_inputs_never_reach_models(recipe_case):
    data, run, calls, root = recipe_case
    duplicate = root / "duplicate.mp4"; shutil.copyfile(data["inputs"][0], duplicate)
    bad = root / "corrupt.mp4"; bad.write_bytes(b"not video")
    data["inputs"] += [str(duplicate), str(bad)]
    result = run("screened")
    assert result["cleaning"]["accepted"] == 1 and result["cleaning"]["rejected"] == 2
    assert calls == ["sam2", "hamer"] and len(result["exports"]) == 1


def test_all_rejected_needs_no_runtime_or_assets(recipe_case, monkeypatch):
    from egoworld.product import recipes
    data, run, calls, root = recipe_case
    data["inputs"] = [str(root / "missing.mp4")]
    monkeypatch.setattr(recipes, "runtime_identity", lambda *a: pytest.fail("No environment probing for rejected data"))
    result = run("allbad")
    assert result["status"] == "rejected" and calls == []
    assert Path(result["run_dir"], "cleaning.json").exists()


def test_partial_hit_runs_only_cpu_without_prepare(recipe_case, monkeypatch):
    from egoworld.pipeline import stages
    data, run, calls, root = recipe_case
    run("hand-only")
    data["outputs"] = ["robot_joints"]
    data["operators"] = {"dex_retarget": {"config": str(root / "robot.yml"), "robot_id": "fixture",
        "human_to_robot_rotation": np.eye(3).ravel().tolist(), "handedness": "right", "joint_units": "radians"}}
    monkeypatch.setattr(stages.PrepareActor, "prepare", lambda *a: pytest.fail("Numeric retargeting must not re-encode video"))
    result = run("new-robot")
    assert calls == ["sam2", "hamer", "dex_retarget"]
    assert result["required_compute_stages"] == ["dex_retarget"]


def test_content_rename_reuses_model_results(recipe_case, monkeypatch):
    from egoworld.pipeline import driver
    data, run, calls, root = recipe_case
    run("original")
    renamed = root / "renamed.mp4"; shutil.copyfile(data["inputs"][0], renamed)
    data["inputs"] = [str(renamed)]
    monkeypatch.setattr(driver, "run_pipeline", lambda *a, **kw: pytest.fail("Content identity must survive filename changes"))
    result = run("renamed")
    assert len(result["cache_hits"]) == 2 and len(result["exports"]) == 1
    clip = next(iter(result["exports"]))
    assert clip.startswith("renamed-")


def test_projection_cache_and_unrelated_versions_reuse(recipe_case, monkeypatch):
    from egoworld.pipeline import driver
    data, run, calls, root = recipe_case
    first = run("first")
    cfg = root / "config.json"
    config = json.loads(cfg.read_text()); config["model_versions"] = {"dex_retarget": "different-unused"}; cfg.write_text(json.dumps(config))
    monkeypatch.setattr(driver, "run_pipeline", lambda *a, **k: pytest.fail("Unrelated model version must not rerun SAM/HaMeR"))
    second = run("second")
    assert len(second["cache_hits"]) == 2 and second["derived_cache_hits"][0]["kernel"] == "projection"
    assert all(str(Path(second["run_dir"]) / "exports") in x["file"] for e in second["exports"].values() for x in e.values())
    assert second["hashing"]["memo_hits"] > 0 and second["timings"]["entry_wall_seconds"] > 0


def test_hash_session_and_registered_mutation(tmp_path):
    file = tmp_path / "weights"; file.write_bytes(b"first")
    registry = tmp_path / "assets.json"; register_assets([file], registry)
    with hash_session(registry) as stats:
        old = sha256_file(file); assert sha256_file(file) == old
        assert stats["registered_hits"] == 1 and stats["files_hashed"] == 0
        stamp = file.stat()
        file.write_bytes(b"other")
        times = stamp.st_atime_ns, stamp.st_mtime_ns
        os.utime(file, ns=times)
        # The registry observes stat changes. Let coarse filesystem clocks
        # advance before asserting invalidation with a restored mtime.
        deadline = time.monotonic() + 2
        while file.stat().st_ctime_ns == stamp.st_ctime_ns and time.monotonic() < deadline:
            time.sleep(.01)
            os.utime(file, ns=times)
        assert file.stat().st_ctime_ns != stamp.st_ctime_ns
        assert sha256_file(file) != old and stats["files_hashed"] == 1
    with hash_session() as stats:
        sha256_file(file); sha256_file(file)
        assert stats["files_hashed"] == 1 and stats["memo_hits"] == 1


def test_derived_corruption_and_gc_preserve_unknown_files(tmp_path):
    cache = DerivedCache(tmp_path / "cache")
    calls = []
    def build(dest):
        dest.mkdir(parents=True, exist_ok=True); (dest / "result.json").write_text('{"value":1}')
        calls.append(1); return {"value": 1}
    identity = {"kind": "test", "version": 1}
    assert not cache.compute(identity, tmp_path / "one", build)[1]
    assert cache.compute(identity, tmp_path / "two", build)[1] and len(calls) == 1
    next((cache.root / "derived").glob("*/result.json")).write_text("corrupt")
    assert not cache.compute(identity, tmp_path / "three", build)[1] and len(calls) == 2
    unrelated = cache.root / "keep-me"; unrelated.mkdir(); (unrelated / "data").write_text("keep")
    external = tmp_path / "external"; external.mkdir(); (external / "data").write_text("keep")
    (cache.root / "derived" / ("f" * 64)).symlink_to(external, target_is_directory=True)
    preview = inspect_cache(cache.root, max_bytes=0)
    assert len(preview["selected_for_removal"]) == 1 and preview["dry_run"]
    inspect_cache(cache.root, max_bytes=0, delete=True)
    assert (unrelated / "data").read_text() == "keep" and (external / "data").read_text() == "keep"


def test_branch_filter_preserves_committed_sam_and_bounds_private_projection(tmp_path):
    from egoworld.contracts import schemas
    from egoworld.product.gates import branch_admission
    from test_pipeline_logic import full_packet
    clip = {"clip_id": "c", "frame_start": 0, "frame_end": 10, "fps": 10}
    row = full_packet("sam2", clip)["tables"]["instances"][0]
    rows = [dict(row, object_id=i+1, area_px=area) for i, area in enumerate((1, 100))]
    pq.write_table(pa.Table.from_pylist(rows, schema=schemas()["instances"]), tmp_path / "instances.parquet")
    original = sha256_file(tmp_path / "instances.parquet")
    config = {"extra": {"branch_cleaning": {"min_hand_area_px": 10}}, "operators": {"hamer": {"params": {}}}}
    job = {"stage": "hamer", "clip": clip, "work_dir": str(tmp_path / "work")}
    result, inputs = branch_admission(config, job, {}, {"sam2": {"directory": str(tmp_path)}})
    assert result is None and sha256_file(tmp_path / "instances.parquet") == original
    file = Path(inputs["sam2"]["directory"]) / "instances.parquet"
    assert pq.ParquetFile(file).read()["object_id"].to_pylist() == [2]
    # Native runner re-admission must not truncate its parent's private view.
    assert branch_admission(config, job, {}, inputs)[0] is None
    assert pq.ParquetFile(file).metadata.num_rows == 1


def test_matched_benchmark_detects_changed_workload(recipe_case):
    from egoworld.product.benchmark import compare_runs
    data, run, _, _ = recipe_case
    first, second = run("first"), run("second")
    compared = compare_runs([first["run_dir"], second["run_dir"]])
    assert compared["status"] == "matched"
    assert compared["runs"][1]["observed_model_stages"] == 0
    assert compared["runs"][1]["peaks"]["process_peak_rss_bytes"] == 0
    data["outputs"] = ["hand_mano"]
    third = run("different-output")
    assert compare_runs([first["run_dir"], third["run_dir"]])["status"] == "different_workloads"


def test_world_cache_checks_bindings_before_reuse(world_case):
    from egoworld.product.derived import world_identity, build_world
    source, calibration, destination = world_case
    clip = json.loads((source / "meta.json").read_text())["clip"]
    cfg = {"path": str(calibration)}; outputs = ["world_camera", "world_scene"]
    key = world_identity(source, cfg, outputs, clip, {})
    cache = DerivedCache(destination.parent / "cache")
    result, hit = cache.compute(key, destination, lambda d: build_world(source, cfg, outputs, d))
    assert result["status"] == "passed" and not hit
    assert cache.compute(key, destination.parent / "other", lambda d: pytest.fail("World CPU work must be reused"))[1]
    (source / "reconstruction-pointcloud.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        world_identity(source, cfg, outputs, clip, {})


def test_optional_datajuicer_op_uses_native_contract(recipe_case, monkeypatch):
    import importlib
    import sys
    import types
    data, run, calls, root = recipe_case
    baseline = run("native")
    result_root = Path(baseline["run_dir"])
    folder = next((result_root / "model-output").glob("run_id=*/video_id=*/clip_id=*"))
    meta = json.loads((folder / "meta.json").read_text())
    module = types.ModuleType("data_juicer.ops.base_op")
    class Mapper:
        def __init__(self, *args, **kwargs): pass
    class Registry:
        def register_module(self, name): return lambda cls: cls
    module.Mapper, module.OPERATORS = Mapper, Registry()
    monkeypatch.setitem(sys.modules, "data_juicer.ops.base_op", module)
    sys.modules.pop("egoworld.product.datajuicer_adapter", None)
    adapter = importlib.import_module("egoworld.product.datajuicer_adapter").EgoWorldStageMapper(
        str(result_root / "config.json"), "hamer", str(root / "dj-output"))
    try:
        mapped = adapter.process_single({"egoworld_clip": meta["clip"], "egoworld_prepared": {"video_path": data["inputs"][0]},
            "egoworld_fingerprint": meta["fingerprint"], "egoworld_inputs": {}})
        receipt = mapped["egoworld_inputs"]["hamer"]
        assert receipt["files"]["hand_pose.parquet"]["rows"] == 1
        assert receipt["files"]["hand_pose.parquet"]["sha256"] == json.loads((folder / "stage_hamer.json").read_text())["files"]["hand_pose.parquet"]["sha256"]
        assert adapter.__getstate__()["_actor"] is None
    finally:
        adapter.close(); sys.modules.pop("egoworld.product.datajuicer_adapter", None)


@pytest.mark.parametrize("operation", [None, "input", "infer", "post"])
def test_empty_branch_skips_model_for_all_execution_paths(tmp_path, monkeypatch, operation):
    from egoworld.contracts import schemas
    from egoworld.pipeline.stages import ModelActor
    pq.write_table(pa.Table.from_pylist([], schema=schemas()["instances"]), tmp_path / "instances.parquet")
    actor = object.__new__(ModelActor)
    actor.config = {"extra": {"branch_cleaning": {}}, "operators": {"hamer": {"params": {"working_dir": "missing-workdir"}}}}
    monkeypatch.setattr(actor, "_process", lambda *a: pytest.fail("Filtered branch must not load or run a model"))
    job = {"stage": "hamer", "clip": {"clip_id": "test", "frame_start": 0, "frame_end": 10}, "work_dir": str(tmp_path)}
    if operation:
        job["_operation"] = operation
    packet = actor.process(job, {}, {"sam2": {"directory": str(tmp_path)}})
    assert packet["stats"]["branch_filtered"] == 1 and packet["stats"]["model_load_seconds"] == 0
    assert packet.get("_handoff", False) == (operation in ("input", "infer"))
