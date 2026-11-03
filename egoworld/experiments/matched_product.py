"""Full raw-video recipe execution with real DJ DefaultExecutor and shared kernels.

The parent uses the normal GPU coordinator environment for both recipes. Only
DJ orchestration uses its isolated interpreter. All model calls, preparation,
QC and exports are production APIs; this is not the native WiLoR comparison.
"""
from dataclasses import asdict
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(config_path, samples_path, output):
    import matched_product_ops  # registers the explicit comparison OPs
    from data_juicer.config import init_configs
    from data_juicer.core import DefaultExecutor
    spec = json.loads(Path(config_path).read_text())
    config, stages = spec["config"], spec["stages"]
    root = Path(output)
    root.mkdir(exist_ok=False)
    common = {"num_proc": 1, "auto_op_parallelism": False}
    process = [{"egoworld_matched_prepare_mapper": {"config": config,
        "work_root": spec["work_root"], **common}}]
    process += [{"egoworld_matched_stage_mapper": {"config_path": spec["model_config"],
        "stage": stage, "output_root": spec["model_root"], **common}} for stage in stages]
    process += [{"egoworld_matched_finalize_mapper": {"config": config,
        "output_root": spec["model_root"], **common}}]
    dj = {"project_name": "matched-full-recipe", "dataset_path": samples_path,
        "export_path": str(root / "clips.jsonl"), "work_dir": str(root / "dj-work"),
        "ds_cache_dir": str(root / "hf-cache"), "np": 1, "text_keys": "text",
        "use_cache": False, "use_checkpoint": False, "skip_op_error": False,
        "op_fusion": False, "export_in_parallel": False, "process": process}
    write(root / "datajuicer-config.json", dj)
    dist = metadata.distribution("py-data-juicer")
    write(root / "installation.json", {"version": dist.version,
        "direct_url": json.loads(dist.read_text("direct_url.json") or "{}"),
        "resource_policy": "Explicit matched-model baseline: one serial OP process, one visible external GPU; no native resource claim"})
    DefaultExecutor(init_configs(["--config", str(root / "datajuicer-config.json")])).run()
    rows = [json.loads(json.loads(line)["egoworld_state"]) for line in (root / "clips.jsonl").read_text().splitlines() if line.strip()]
    expected = [json.loads(json.loads(line)["egoworld_state"])["egoworld_clip"]["clip_id"] for line in Path(samples_path).read_text().splitlines() if line.strip()]
    actual = [r["egoworld_clip"]["clip_id"] for r in rows]
    if sorted(expected) != sorted(actual) or len(set(actual)) != len(actual):
        raise ValueError("Full Data-Juicer recipe lost/duplicated clips")
    if any(r["egoworld_final_status"] != "Done" or set(r["egoworld_inputs"]) != set(stages) for r in rows):
        raise ValueError("Incomplete shared-model chain")
    write(root / "execution.json", {"status": "passed", "clips": len(rows), "stages": stages})


def run_pipeline(config_path, video_manifest_path, clip_manifest_path, *, run_id, python, backend_factory=None):
    from egoworld.config import load_config
    from egoworld.contracts import OUTPUT_SCHEMA_VERSION, enabled_stages
    from egoworld.pipeline.driver import load_manifest, _build_clip_tasks, _clip_to_dict
    from egoworld.utils.provenance import provenance
    if backend_factory is not None:
        raise ValueError("The matched DefaultExecutor does not use an EgoWorld backend")
    config = load_config(config_path).resolved()
    if config.execution.split_execution or config.num_gpus != 1:
        raise ValueError("Initial shared-model baseline requires full stages and one GPU")
    graph = enabled_stages(config)
    if any(not getattr(config.operators, s).params.get("runner_command") for s in graph):
        raise ValueError("Use the same explicit external model environments on both sides")
    index = {r["video_id"]: r for r in load_manifest(video_manifest_path)}
    clips = [_clip_to_dict(task) for task in _build_clip_tasks(load_manifest(clip_manifest_path), index)]
    for clip in clips:
        clip.update(fps=float(index[clip["video_id"]]["fps"]))
    identity = provenance(config, video_manifest_path, clip_manifest_path)
    model_root = Path(config.paths.output_root) / f"run_id={run_id}"
    work = model_root / "matched-executor"
    work.mkdir(parents=True, exist_ok=False)
    spec = {"config": asdict(config), "stages": list(graph), "model_root": str(model_root),
        "model_config": str(Path(config_path).resolve()), "work_root": str(model_root / ".work")}
    write(work / "spec.json", spec)
    samples = work / "samples.jsonl"
    samples.write_text("\n".join(json.dumps({"text": "shared-model video recipe", "egoworld_state": json.dumps({
        "egoworld_clip": clip, "egoworld_fingerprint": identity["fingerprint"], "egoworld_inputs": {}})}) for clip in clips) + "\n")
    write(model_root / "run_manifest.json", {**config.to_run_manifest(), "config_fingerprint": identity["fingerprint"],
        "provenance": identity, "config": asdict(config), "stages": list(graph),
        "output_schema_version": OUTPUT_SCHEMA_VERSION, "executor": "datajuicer_shared_models"})
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    from product_comparison import package_snapshot
    before = package_snapshot(python, env)
    write(work / "environment-before.json", before)
    argv = [python, str(ROOT / "scripts/matched_product_worker.py"), "--config", str(work / "spec.json"),
        "--samples", str(samples), "--output", str(work / "execution")]
    # Child shares this trial's process group so the product observer's timeout
    # can terminate the whole job. Native runner guards own their subprocesses.
    with (work / "stdout.log").open("w") as stdout, (work / "stderr.log").open("w") as stderr:
        completed = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr)
    after = package_snapshot(python, env)
    write(work / "environment-after.json", after)
    if before != after:
        raise RuntimeError("Matched Data-Juicer environment changed during execution")
    if completed.returncode:
        raise RuntimeError(f"Matched DefaultExecutor exited {completed.returncode}; see {work}")
    result = json.loads((work / "execution/execution.json").read_text())
    if result["status"] != "passed" or result["clips"] != len(clips):
        raise ValueError("Matched DefaultExecutor produced no complete recipe")
    from egoworld.pipeline.input_cache import clear_cache, cache_root
    clear_cache(cache_root(asdict(config), model_root / ".work"))
    return result
