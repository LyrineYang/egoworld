"""Two real Data-Juicer executors sharing the frozen EgoWorld operator chain."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from product_comparison import package_snapshot, write

repo = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text())


def shard_pipeline(config_path, videos, clips, *, run_id, backend_factory=None, device_ids, python):
    from egoworld.config import load_config
    from egoworld.contracts import OUTPUT_SCHEMA_VERSION, enabled_stages
    from egoworld.pipeline.driver import load_manifest, _build_clip_tasks, _clip_to_dict
    from egoworld.pipeline.input_cache import cache_root, clear_cache
    from egoworld.utils.provenance import provenance

    config = load_config(config_path).resolved()
    if backend_factory is not None or config.execution.split_execution or config.num_gpus != len(device_ids):
        raise ValueError("Expected unsplit two-device matched recipe")
    stages = list(enabled_stages(config))
    if any(not getattr(config.operators, stage).params.get("runner_command") for stage in stages):
        raise ValueError("Expected frozen external runners")
    index = {row["video_id"]: row for row in load_manifest(videos)}
    tasks = [_clip_to_dict(clip) for clip in _build_clip_tasks(load_manifest(clips), index)]
    for task in tasks:
        task["fps"] = float(index[task["video_id"]]["fps"])
    if len(tasks) < len(device_ids):
        raise ValueError("Need at least one unique clip per GPU")
    identity = provenance(config, videos, clips)
    model_root = Path(config.paths.output_root) / f"run_id={run_id}"
    work = model_root / "matched-executors"
    work.mkdir()
    write(model_root / "run_manifest.json", {**config.to_run_manifest(), "config_fingerprint": identity["fingerprint"],
          "provenance": identity, "config": asdict(config), "stages": stages,
          "output_schema_version": OUTPUT_SCHEMA_VERSION, "executor": "two_sharded_datajuicer_default_executors"})
    spec = {"config": asdict(config), "stages": stages, "model_root": str(model_root), "map_mode": "inprocess",
            "model_config": config_path, "work_root": str(model_root / ".work")}
    write(work / "spec.json", spec)
    common = os.environ.copy()
    before = package_snapshot(python, common)
    write(work / "environment-before.json", before)

    def execute_shard(number, device):
        shard = work / f"shard-{number}"
        shard.mkdir()
        assigned = tasks[number::len(device_ids)]
        samples = shard / "samples.jsonl"
        rows = [{"text": "shared-model video recipe", "egoworld_state": json.dumps({
            "egoworld_clip": clip, "egoworld_fingerprint": identity["fingerprint"], "egoworld_inputs": {}})}
            for clip in assigned]
        samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
        argv = [python, str(repo / "egoworld/scripts/matched_product_worker.py"), "--config", str(work / "spec.json"),
                "--samples", str(samples), "--output", str(shard / "execution")]
        with (shard / "stdout.log").open("w") as stdout, (shard / "stderr.log").open("w") as stderr:
            subprocess.run(argv, env={**common, "CUDA_VISIBLE_DEVICES": device}, stdout=stdout, stderr=stderr, check=True)
        report = load(shard / "execution/execution.json")
        if report["status"] != "passed" or report["clips"] != len(assigned):
            raise ValueError(f"Incomplete shard {shard}")
        return {"device": int(device), "clips": [clip["clip_id"] for clip in assigned]}

    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures = [pool.submit(execute_shard, index, device) for index, device in enumerate(device_ids)]
        shards = [future.result() for future in futures]
    after = package_snapshot(python, common)
    write(work / "environment-after.json", after)
    if before != after:
        raise RuntimeError("Matched Data-Juicer environment changed")
    clear_cache(cache_root(asdict(config), model_root / ".work"))
    write(work / "execution.json", {"status": "passed", "shards": shards})
