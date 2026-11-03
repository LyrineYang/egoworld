import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from egoworld.product.registry import resolve_outputs
from egoworld.product.recipes import compile_recipe, run_recipe


def test_output_selection_deduplicates_shared_kernels_and_keeps_dependencies():
    selections, graph = resolve_outputs(["hand_joints", "hand_mano", "robot_joints", "boxes"])
    assert list(graph) == ["sam2", "hamer", "dex_retarget"]
    assert len(selections) == 4
    with pytest.raises(ValueError, match="Unknown data type"):
        resolve_outputs(["guessed_metric_hand"])
    with pytest.raises(ValueError, match="Invalid fields"):
        resolve_outputs([{"type": "hand_joints", "fields": ["not_a_column"]}])


@pytest.fixture
def recipe_case(tmp_path, monkeypatch, thread_backend):
    from egoworld.pipeline import stages
    from egoworld.product import recipes
    from test_pipeline_logic import full_packet
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10., (32, 32))
    for i in range(3):
        image = np.full((32, 32, 3), i * 20, np.uint8)
        image[8:24, 8:24] = i * 20 + 40
        writer.write(image)
    writer.release()
    config = tmp_path / "config.json"
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"first model identity")
    config.write_text(json.dumps({"num_gpus": 1, "scenedetect": {"method": "none"},
        "operators": {"sam2": {"enabled": True}, "hamer": {"enabled": True, "params": {"checkpoint": str(checkpoint)}}}}))
    recipe = tmp_path / "recipe.json"
    data = {"version": 1, "config": str(config), "inputs": [str(video)],
            "outputs": ["hand_joints"], "output_dir": str(tmp_path / "results"),
            "cache": {"root": str(tmp_path / "cache")}}
    calls = []
    monkeypatch.setattr(recipes, "runtime_identity", lambda command: {"test_runtime": True})
    monkeypatch.setattr(stages.PrepareActor, "prepare", lambda self, job: {"video_path": str(video), "frames": 3})
    def process(self, job, prepared, inputs):
        calls.append(job["stage"])
        packet = full_packet(job["stage"], job["clip"])
        if job["stage"] == "sam2":
            packet["stats"]["processed_frames"] = 3
        return packet
    monkeypatch.setattr(stages.ModelActor, "process", process)
    def run(identifier):
        recipe.write_text(json.dumps(data))
        return run_recipe(recipe, run_id=identifier, backend_factory=thread_backend)
    return data, run, calls, tmp_path


def test_recipe_cross_run_cache_reuse_and_per_kernel_invalidation(recipe_case, monkeypatch):
    from egoworld.pipeline import driver
    data, run, calls, root = recipe_case
    first = run("first")
    assert calls == ["sam2", "hamer"]
    data["outputs"] = ["hand_mano", {"type": "hand_joints", "fields": ["joints_3d"]}]
    original = driver.run_pipeline
    monkeypatch.setattr(driver, "run_pipeline", lambda *a, **k: pytest.fail("A full cache hit must not start Ray or models"))
    second = run("second")
    monkeypatch.setattr(driver, "run_pipeline", original)
    assert calls == ["sam2", "hamer"] and len(second["cache_hits"]) == 2
    export = next(iter(second["exports"].values()))["hand_joints"]["file"]
    exported_fields = pq.ParquetFile(export).schema_arrow.names
    assert "joints_3d" in exported_fields and "mano_pose" not in exported_fields
    assert {"coord_frame", "length_unit", "scale_status", "handedness"} <= set(exported_fields)
    data["operators"] = {"hamer": {"frame_stride": 2}}
    third = run("third")
    assert calls == ["sam2", "hamer", "hamer"]
    assert [h["kernel"] for h in third["cache_hits"]] == ["sam2"]
    # Cache files are copies, not writable aliases to another run.
    source = next((Path(second["run_dir"]) / "model-output").glob("run_id=*/video_id=*/clip_id=*/instances.parquet"))
    source.write_bytes(b"damage only this restored run")
    fourth = run("fourth")
    assert fourth["status"] == "passed" and len(fourth["cache_hits"]) == 2
    assert calls == ["sam2", "hamer", "hamer"]


def test_corrupt_cache_is_recomputed_and_read_disabled_is_honored(recipe_case):
    data, run, calls, root = recipe_case
    run("first")
    next((root / "cache/sam2").glob("*/instances.parquet")).write_bytes(b"broken")
    run("after-corruption")
    assert calls.count("sam2") == 2 and calls.count("hamer") == 2
    data["cache"]["read"] = False
    run("no-read")
    assert calls.count("sam2") == 3 and calls.count("hamer") == 3


def test_changed_recipe_cannot_overwrite_an_existing_run(recipe_case):
    data, run, calls, root = recipe_case
    run("fixed")
    data["outputs"] = ["boxes"]
    with pytest.raises(ValueError, match="Recipe changed"):
        run("fixed")


def test_changed_model_bytes_invalidate_only_its_kernel_and_dependents(recipe_case):
    data, run, calls, root = recipe_case
    run("first")
    (root / "weights.ckpt").write_bytes(b"different weights at same path")
    result = run("changed-model")
    assert calls == ["sam2", "hamer", "hamer"]
    assert [h["kernel"] for h in result["cache_hits"]] == ["sam2"]


def test_empty_requested_data_is_partial_even_when_model_stage_commits(recipe_case, monkeypatch):
    from egoworld.pipeline import stages
    data, run, calls, root = recipe_case
    original = stages.ModelActor.process
    def no_hand(self, job, prepared, inputs):
        if job["stage"] == "hamer":
            return {"tables": {"hand_pose": []}}
        return original(self, job, prepared, inputs)
    monkeypatch.setattr(stages.ModelActor, "process", no_hand)
    result = run("no-hands")
    assert result["status"] == "partial"
    assert result["empty_outputs"][0]["type"] == "hand_joints"


def test_explicit_pipeline_executor_keeps_exports_and_verified_cache(recipe_case, thread_backend):
    from egoworld.pipeline.driver import run_pipeline
    data, _, calls, root = recipe_case
    recipe = root / "recipe.json"
    recipe.write_text(json.dumps(data))
    executions = []
    def executor(*args, **kwargs):
        executions.append(kwargs["run_id"])
        return run_pipeline(*args, **kwargs)
    first = run_recipe(recipe, run_id="custom-first", backend_factory=thread_backend, pipeline_executor=executor)
    assert first["status"] == "passed" and calls == ["sam2", "hamer"]
    assert next(iter(first["exports"].values()))["hand_joints"]["rows"] > 0
    second = run_recipe(recipe, run_id="custom-cache", backend_factory=thread_backend, pipeline_executor=executor)
    assert second["status"] == "passed" and len(second["cache_hits"]) == 2
    assert executions == ["custom-first"]
