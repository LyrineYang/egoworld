#!/usr/bin/env python3
"""Run the single-host two-V100 matched-model study on a frozen corpus."""
import argparse
from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
from matched_audit import audit_recipe_pair
from output_quality import egoworld_predictions, evaluate
from product_comparison import audit_egoworld, package_snapshot, sha, write
from two_gpu_shards import shard_pipeline


def read(path):
    return json.loads(Path(path).read_text())


def freeze(args):
    target = args.study
    target.mkdir(parents=True)
    corpus = read(args.corpus)
    data = [{"path": row["video_path"], "sha256": row["video_sha256"], "expected": "accepted"}
            for row in corpus["samples"].values()]
    if len(data) < 2:
        raise ValueError("Two GPUs require at least two unique inputs")
    for index, row in enumerate(data[:2]):
        duplicate = target / f"byte-duplicate-{index}.mp4"
        duplicate.write_bytes(Path(row["path"]).read_bytes())
        blank = target / f"blank-{index}.mp4"
        color = ("black", "white")[index]
        subprocess.run([args.ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                        f"color={color}:s=224x224:r=1:d=3", "-c:v", "libx264", "-threads", "1", str(blank)], check=True)
        corrupt = target / f"corrupt-{index}.mp4"
        corrupt.write_bytes(f"synthetic broken video header {index}\n".encode())
        data.extend({"path": str(p), "sha256": sha(p), "expected": "rejected"} for p in (duplicate, blank, corrupt))
    recipe = read(args.recipe)
    recipe["config"] = str((args.recipe.parent / recipe["config"]).resolve())
    recipe["inputs"] = [row["path"] for row in data]
    cache = recipe.setdefault("cache", {})
    cache.update(read=False, write=False)
    paths = [args.corpus, Path(recipe["config"])]
    if recipe["cache"].get("asset_registry"):
        registry = (args.recipe.parent / recipe["cache"]["asset_registry"]).resolve()
        recipe["cache"]["asset_registry"] = str(registry)
        paths.append(registry)
    spec = target / "recipe.json"
    write(spec, recipe)
    paths += [spec, Path(__file__), EXPERIMENTS / "two_gpu_stream.py", EXPERIMENTS / "two_gpu_shards.py"]
    write(target / "plan.json", {"inputs": data, "corpus": str(args.corpus), "devices": args.devices,
          "hashes": {str(p): sha(p) for p in paths},
          "pose_policy": {"wrist_match_diagonal_fraction": .25, "mpjpe_thresholds_mm": [20, 50]}})


def worker(args):
    from egoworld.product.recipes import run_recipe
    from two_gpu_stream import pipeline

    recipe = read(args.study / "recipe.json")
    recipe["output_dir"] = str(args.output / "runs")
    recipe["cache"]["root"] = str(args.output / "empty-result-cache")
    spec = args.output / "recipe.json"
    write(spec, recipe)
    native = args.engine != "datajuicer"
    if native:
        execute = partial(pipeline, layout=args.engine)
    else:
        execute = partial(shard_pipeline, device_ids=tuple(read(args.study / "plan.json")["devices"]),
                          python=str(args.dj_python))
    result = run_recipe(spec, run_id="c1", pipeline_executor=execute, backend_factory=True if native else None)
    if result["status"] not in ("passed", "partial"):
        raise ValueError(f"Product did not produce verified output: {result['status']}")


@contextmanager
def monitor(output, devices):
    stop, records, errors = threading.Event(), [], []

    def sample():
        try:
            import psutil
            import pynvml as nvml
            nvml.nvmlInit()
            handles = {int(index): nvml.nvmlDeviceGetHandleByIndex(int(index)) for index in devices}
            parent = psutil.Process()
            with (output / "gpu-observations.jsonl").open("w") as log:
                while not stop.is_set():
                    own = {child.pid for child in parent.children(recursive=True)}
                    group = []
                    for index, handle in handles.items():
                        pids = {proc.pid for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle) if proc.pid > 0}
                        memory = nvml.nvmlDeviceGetMemoryInfo(handle, nvml.nvmlMemory_v2)
                        group.append({"seconds": time.monotonic(), "device": index, "used_bytes": int(memory.used),
                                      "own_pids": sorted(pids & own), "outside_pids": sorted(pids - own)})
                    log.write(json.dumps(group) + "\n")
                    records.extend(group)
                    stop.wait(.5)
            nvml.nvmlShutdown()
        except Exception as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        own = {pid for row in records for pid in row["own_pids"]}
        outside = {pid for row in records for pid in row["outside_pids"]} - own
        observed = {row["device"] for row in records if row["own_pids"]}
        write(output / "gpu-monitor.json", {"errors": errors, "devices": sorted(observed),
              "outside_only_pids": sorted(outside), "samples": len(records)})


def measure(args, engine, deadline):
    plan = read(args.study / "plan.json")
    output = args.study / engine
    output.mkdir()
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(plan["devices"]),
           "EGOWORLD_STUDY_DEVICES": ",".join(plan["devices"]), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
           "UV_OFFLINE": "1", "PIP_NO_INDEX": "1"}
    pythonpath = [str(EXPERIMENTS), str(EXPERIMENTS.parent / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    before = package_snapshot(str(args.coordinator), env)
    argv = [str(args.coordinator), "-B", str(Path(__file__).resolve()), "--study", str(args.study),
            "worker", "--engine", engine, "--output", str(output), "--dj-python", str(args.dj_python)]
    if time.monotonic() >= deadline - 15:
        raise TimeoutError("No remaining time for another arm")
    with monitor(output, plan["devices"]), (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        started = time.monotonic()
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        status, expired = -1, False
        try:
            status = process.wait(timeout=deadline - started - 15)
        except subprocess.TimeoutExpired:
            expired = True
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        wall = time.monotonic() - started
    stable = before == package_snapshot(str(args.coordinator), env)
    trial = {"wall_seconds": wall, "returncode": status, "environment_stable": stable, "timeout": expired}
    write(output / "trial.json", trial)
    if status or not stable:
        raise RuntimeError(f"Failed or timed out: {output}")
    observed = read(output / "gpu-monitor.json")
    if observed["errors"] or observed["devices"] != sorted(map(int, plan["devices"])):
        raise RuntimeError("Unable to verify both GPUs")
    return wall


def audit(args):
    plan = read(args.study / "plan.json")
    corpus = read(plan["corpus"])
    engines = (args.layout, "datajuicer")
    outputs = [args.study / engine for engine in engines]
    manifests = [path / "runs/run_id=c1/data_manifest.json" for path in outputs]
    parity = audit_recipe_pair(*manifests, allow_review=True)
    metrics, qualities = {}, []
    for engine, path, manifest in zip(engines, outputs, manifests):
        artifacts = audit_egoworld(manifest)
        pred, _ = egoworld_predictions(manifest, corpus["samples"], 5000 / 256)
        quality = evaluate(pred, corpus["ground_truth"], corpus["samples"], plan["pose_policy"])
        qualities.append(quality)
        write(path / "quality.json", quality)
        decisions = {row["path"]: row["status"] for row in read(manifest.parent / "cleaning.json")["records"]}
        if set(decisions) != {row["path"] for row in plan["inputs"]}:
            raise ValueError("Cleaning coverage differs from frozen inputs")
        good = [row for row in plan["inputs"] if row["expected"] == "accepted"]
        bad = [row for row in plan["inputs"] if row["expected"] == "rejected"]
        wall = read(path / "trial.json")["wall_seconds"]
        metrics[engine] = {"wall_seconds": wall, "qualified_20mm": quality["quality_hands"]["20"],
                           "qualified_per_second": quality["quality_hands"]["20"] / wall,
                           "mpjpe_root_mm": quality["mpjpe_root_mm_detected_only"],
                           "false_rejects": sum(decisions[row["path"]] != "accepted" for row in good),
                           "missed_rejects": sum(decisions[row["path"]] != "rejected" for row in bad),
                           "good": len(good), "bad": len(bad), "output_counts": artifacts["output_counts"]}
    if qualities[0] != qualities[1]:
        raise ValueError("Independent quality differs")
    shared = any(read(path / "gpu-monitor.json")["outside_only_pids"] for path in outputs)
    report = {"parity": parity, "metrics": metrics, "observed_other_compute": shared,
              "datajuicer_over_egoworld_wall_ratio": metrics["datajuicer"]["wall_seconds"] / metrics[args.layout]["wall_seconds"]}
    write(args.study / "report.json", report)
    print(json.dumps(report, indent=2))


def run(args):
    plan = read(args.study / "plan.json")
    devices(",".join(plan["devices"]))
    files = {**plan["hashes"], **{row["path"]: row["sha256"] for row in plan["inputs"]}}
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError("Frozen file changed")
    if not 60 <= args.budget_seconds <= 1800:
        raise ValueError("Budget must be between 60 and 1800 seconds")
    deadline = time.monotonic() + args.budget_seconds
    engines = [args.layout, "datajuicer"]
    if args.datajuicer_first:
        engines.reverse()
    for engine in engines:
        measure(args, engine, deadline)
    audit(args)


def absolute(value):
    return Path(value).resolve()


def executable(value):
    # Resolving the final symlink would escape a virtual environment.
    return Path(os.path.abspath(value))


def devices(value):
    selected = value.split(",")
    if len(selected) != 2 or len(set(selected)) != 2 or not all(part.isdecimal() for part in selected):
        raise argparse.ArgumentTypeError("Provide two distinct physical GPU indices")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=absolute)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("freeze")
    init.add_argument("--recipe", required=True, type=absolute)
    init.add_argument("--corpus", required=True, type=absolute)
    init.add_argument("--devices", type=devices, required=True)
    init.add_argument("--ffmpeg", default="ffmpeg")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--coordinator", required=True, type=executable)
    run_parser.add_argument("--dj-python", required=True, type=executable)
    run_parser.add_argument("--budget-seconds", type=int, default=1765)
    run_parser.add_argument("--datajuicer-first", action="store_true")
    check = sub.add_parser("audit")
    for command in (run_parser, check):
        command.add_argument("--layout", choices=("stream-checked-dual", "stream-checked-pinned"),
                             default="stream-checked-dual")
    child = sub.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--engine", required=True)
    child.add_argument("--output", required=True, type=absolute)
    child.add_argument("--dj-python", required=True, type=executable)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "audit": audit, "worker": worker}[args.action](args)


if __name__ == "__main__":
    main()
