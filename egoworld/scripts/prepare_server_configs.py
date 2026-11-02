#!/usr/bin/env python3
"""Prepare portable single-server 2–16 GPU profiles without running Ray/models."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoworld.config import load_config, PipelineConfig
from egoworld.contracts import GPU_STAGES


def relocate(value, mappings):
    """Require every absolute deployment dependency to have an explicit map."""
    if isinstance(value, dict):
        return {k: relocate(v, mappings) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, mappings) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        for source, target in sorted(mappings, key=lambda pair: len(str(pair[0])), reverse=True):
            if path == source or source in path.parents:
                return str(target / path.relative_to(source))
        raise ValueError(f"Unmapped absolute deployment path: {value}")
    return value


def resource_profile(config, gpus, cpus, ram_gib, runner_mode="auto"):
    import copy
    result = copy.deepcopy(config)
    if not 2 <= gpus <= 16 or cpus < 1 or ram_gib <= 0:
        raise ValueError("Require 2–16 GPUs and positive CPU/RAM budgets")
    result["num_gpus"] = gpus
    execution = result["execution"]
    writers = min(4, max(1, cpus // 8))
    dex = execution["cpu_operator_workers"] if result["operators"]["dex_retarget"]["enabled"] else 0
    phases = execution["phase_workers"] if execution["split_execution"] else 0
    gpu_pool = gpus if any(result["operators"][s]["enabled"] for s in GPU_STAGES) else 0
    prepare = min(8, math.floor(cpus - gpu_pool - dex - phases - .5 * writers))
    if prepare < 1:
        raise ValueError(f"{gpus} GPUs: {cpus} CPUs cannot admit GPU/phase/Dex/Writer/prepare pools")
    execution.update(cpu_workers=prepare, write_workers=writers, gpu_stage_groups=[], external_runner_mode=runner_mode,
                     max_active_clips=max(2, min(32, 2 * gpus + prepare)))
    # Admission estimates only: third-party heaps and OS page cache are outside.
    ram = int(ram_gib * 1024**3)
    bounds = result["backpressure"]
    bounds.update(max_in_flight_cpu=None, max_in_flight_gpu=None, max_in_flight_write=None,
                  max_in_flight_bytes=min(bounds.get("max_in_flight_bytes") or 2 * 1024**3, ram // 8),
                  max_prepared_bytes=min(bounds.get("max_prepared_bytes") or 4 * 1024**3, ram // 4))
    resources = {"configured_gpus": gpus, "cpu_budget": cpus, "ram_budget_gib": ram_gib,
                 "actor_cpu_reservation": prepare + gpu_pool + dex + phases + .5 * writers,
                 "gpu_actors": gpu_pool, "prepare_actors": prepare, "writer_actors": writers,
                 "dex_actors": dex, "phase_actors": phases}
    return result, resources


def prepare(config_path, output, mappings, deployment_root, env_root, gpus, cpus, ram_gib, runner_mode="auto"):
    source = load_config(str(config_path)).resolved()
    original = asdict(source)
    # Run paths belong to the target server; never reuse a WSL state DB or cache.
    original["paths"] = {"data_root": "data", "output_root": "output", "manifest_path": "manifests",
                         "state_db_path": "state/pipeline.db", "runlog_path": "runlog.md"}
    for stage, operator in original["operators"].items():
        if operator["enabled"]:
            operator["params"].pop("runner_command", None)
    portable = relocate(original, mappings)
    target = Path(deployment_root)
    portable["paths"] = {key: str(target / value) for key, value in original["paths"].items()}
    for stage, operator in portable["operators"].items():
        if operator["enabled"]:
            operator["params"]["runner_command"] = [str(Path(env_root) / stage / "bin/python"), "-m", "egoworld.worker"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "source_config": str(Path(config_path).resolve()),
                "source_config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                "single_server": True, "hardware_validated": False,
                "path_mappings": [[str(a), str(b)] for a, b in mappings], "profiles": [],
                "required_actions": ["Transfer source/data/model assets, preserving versions and license access.",
                    "Recreate per-model environments; rebuild CUDA/native extensions on the target host.",
                    "Rebase paths inside RGB-D assets JSON and preserve input video identity.",
                    "Run check-only, doctor, native smoke, output audit, then a many-clip concurrency test.",
                    "Measure NVML/process-tree peaks and correct admission estimates before a full server run."],
                "limits": ["Configuration validation is not physical multi-GPU validation.",
                    "RAM budgets bound admission estimates, not total machine RAM.",
                    "This prepares one Linux server, not a multi-node Ray deployment.",
                    "Model precision, sampling and output semantics are preserved, not automatically reduced."]}
    for count in gpus:
        profile, resources = resource_profile(portable, count, cpus, ram_gib, runner_mode)
        name = f"server-{count}gpu.json"
        file = output / name
        file.write_text(json.dumps(profile, indent=2) + "\n")
        # Structural validation does not require those remote assets to exist.
        load_config(str(file)).validate()
        manifest["profiles"].append({"file": name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), **resources})
    (output / "deployment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map", action="append", default=[], metavar="LOCAL=SERVER")
    parser.add_argument("--deployment-root", required=True)
    parser.add_argument("--environment-root", required=True)
    parser.add_argument("--gpus", default="2,4,8,16")
    parser.add_argument("--cpus", required=True, type=int, help="Explicit target Ray CPU budget; not this laptop's CPU count")
    parser.add_argument("--ram-gib", required=True, type=float, help="Explicit target RAM budget; not a whole-machine cap")
    parser.add_argument("--runner-mode", choices=("auto", "persistent", "per_task"), default="auto", help="Server default auto retains built-in model workers; per_task is for isolation diagnostics")
    args = parser.parse_args()
    mappings = []
    for item in args.map:
        left, sep, right = item.partition("=")
        if not sep or not Path(left).is_absolute() or not Path(right).is_absolute():
            parser.error("Each --map must use absolute LOCAL=SERVER paths")
        mappings.append((Path(left), Path(right)))
    counts = [int(value) for value in args.gpus.split(",")]
    print(json.dumps(prepare(args.config, args.output, mappings, args.deployment_root,
                            args.environment_root, counts, args.cpus, args.ram_gib, args.runner_mode), indent=2))


if __name__ == "__main__":
    main()
