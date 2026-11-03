"""Fresh-process product trials using production EgoWorld or unmodified DJ OPs.

Native model/policy differences are retained. This module never turns a native
capability comparison into a matched-model speedup or treats an exit code as QC.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def package_snapshot(python, env):
    code = "import sys,importlib.metadata as m,json;print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"
    return json.loads(subprocess.check_output([python, "-c", code], text=True, env=env, timeout=60))


def worker(kind, config_path, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    config = json.loads(Path(config_path).read_text())
    if kind in ("egoworld_recipe", "datajuicer_matched_recipe"):
        from egoworld.product.recipes import run_recipe
        executor = None
        if kind == "datajuicer_matched_recipe":
            from functools import partial
            from matched_product import run_pipeline
            native_python = config["datajuicer_python"]
            map_mode = config.get("map_mode", "process")
            config = json.loads(Path(config["recipe"]).read_text())
            executor = partial(run_pipeline, python=native_python, map_mode=map_mode)
        config["output_dir"] = str(output / "runs")
        config["cache"]["root"] = str(output / "empty-result-cache")
        if config["cache"].get("read", True):
            raise ValueError("Fresh-model trial requires explicit cache.read=false")
        write(output / "recipe.json", config)
        result = run_recipe(output / "recipe.json", run_id="c1", pipeline_executor=executor)
        write(output / "worker.json", {"kind": kind, "status": result["status"],
              "data_manifest": str(Path(result["run_dir"], "data_manifest.json"))})
        if result["status"] != "passed":
            raise RuntimeError(f"Product recipe ended {result['status']}")
    elif kind == "egoworld_cleaning":
        from egoworld.product.cleaning import screen_inputs
        from egoworld.utils.hashing import hash_session
        with hash_session():
            result = screen_inputs(config["inputs"], config.get("cleaning"))
        write(output / "screening.json", result)
        write(output / "worker.json", {"kind": kind, "status": "passed", "screening": str(output / "screening.json")})
    elif kind == "datajuicer_native":
        # No egoworld_stage_mapper, custom_operator_paths, monkeypatch or device override.
        if config.get("custom_operator_paths") or any(next(iter(op)).startswith("egoworld_") for op in config["process"]):
            raise ValueError("Native comparison forbids EgoWorld custom operators")
        from data_juicer.config import init_configs
        from data_juicer.core import DefaultExecutor
        config.update(export_path=str(output / "datajuicer.jsonl"), work_dir=str(output / "dj-work"),
                      ds_cache_dir=str(output / "hf-cache"))
        write(output / "config.json", config)
        cfg = init_configs(["--config", str(output / "config.json")])
        settings = {key: getattr(cfg, key, None) for key in
                    ("executor_type", "np", "use_cache", "skip_op_error", "op_fusion", "auto_op_parallelism")}
        import torch
        dist = metadata.distribution("py-data-juicer")
        settings.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                        datajuicer_version=dist.version, direct_url=json.loads(dist.read_text("direct_url.json") or "{}"))
        write(output / "native-settings.json", settings)
        DefaultExecutor(cfg).run()
        write(output / "worker.json", {"kind": kind, "status": "passed", "export": str(output / "datajuicer.jsonl")})
    else:
        raise ValueError(f"Unsupported worker {kind}")


def audit_egoworld(manifest_path):
    import pyarrow.parquet as pq
    from egoworld.pipeline.artifacts import load_stage
    result = json.loads(Path(manifest_path).read_text())
    if result["status"] != "passed":
        raise ValueError("Incomplete product result")
    rows, artifacts, qcs = {}, [], []
    for clip, exports in result["exports"].items():
        for name, info in exports.items():
            if name == "world":
                import numpy as np
                directory = Path(result["run_dir"]) / "exports" / clip / "world"
                if info["status"] != "passed":
                    raise ValueError("Incomplete world export")
                for filename, digest in info["files"].items():
                    if Path(filename).name != filename or sha(directory / filename) != digest:
                        raise ValueError("World export changed")
                for output_name, count_info in info["outputs"].items():
                    file = directory / (output_name + (".npz" if output_name == "world_scene" else ".parquet"))
                    if file.suffix == ".parquet":
                        count = pq.ParquetFile(file).metadata.num_rows
                    else:
                        with np.load(file, allow_pickle=False) as data:
                            count = len(data["points"])
                    if count != count_info.get("rows", count_info.get("points")):
                        raise ValueError("World export coverage mismatch")
                    rows[output_name] = rows.get(output_name, 0) + count
                    artifacts.append({"clip_id": clip, "output": output_name, "file": str(file),
                                      "rows_or_points": count, "sha256": info["files"][file.name]})
                continue
            file = Path(info["file"])
            if sha(file) != info["sha256"]:
                raise ValueError(f"Export changed: {file}")
            count = pq.ParquetFile(file).metadata.num_rows if file.suffix == ".parquet" else info["points"]
            if count != info.get("rows", info.get("points")):
                raise ValueError("Export row count mismatch")
            rows[name] = rows.get(name, 0) + count
            artifacts.append({"clip_id": clip, "output": name, "file": str(file), "rows_or_points": count, "sha256": info["sha256"]})
    root = Path(result["run_dir"]) / "model-output"
    stages = []
    for file in root.glob("run_id=*/video_id=*/clip_id=*/stage_*.json"):
        receipt = json.loads(file.read_text())
        if load_stage(file.parent, receipt["stage"], receipt["fingerprint"]) is None:
            raise ValueError("Invalid production stage commit")
        stages.append(receipt["stage"])
    for file in root.glob("run_id=*/video_id=*/clip_id=*/qc.json"):
        qcs.append(json.loads(file.read_text()))
    return {"validated": bool(artifacts), "output_counts": rows, "artifacts": artifacts,
            "cleaning": result["cleaning"], "computed_stages": result["computed"],
            "stage_commits": stages, "qc": qcs,
            "validity": "Production contracts and checked exports; accuracy and downstream usefulness unlabelled"}


def audit_datajuicer(export):
    import numpy as np
    ids, hands, invalid_hands, tag_keys = [], 0, 0, set()
    main_meta = []
    with Path(export).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(row.get("sample_id"))
            main_meta.append(row.get("__dj__meta__", {}))
    # Native Exporter removes metadata from the main JSONL by default and
    # writes it to <stem>_stats.jsonl. Preserve that layout, audit both files.
    sidecar = Path(export).with_name(Path(export).stem + "_stats.jsonl")
    metadata_rows = main_meta
    if sidecar.exists():
        records = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
        if len(records) != len(ids):
            raise ValueError("Native stats sidecar coverage differs from main export")
        metadata_rows = [r.get("__dj__meta__", {}) for r in records]
    frame_counts, known = [], set()
    for meta in metadata_rows:
        meta = json.loads(meta) if isinstance(meta, str) else meta
        tag_keys.update(meta)
        tags = meta.get("hand_reconstruction_tags", {})
        if tags:
            key = meta.get("input_id")
            if key is not None and (key not in ids or key in known):
                raise ValueError("Native metadata identity missing from main export or duplicated")
            if key is not None:
                known.add(key)
            frame_counts.append(len(tags.get("joints", [])))
        for frame in tags.get("joints", []):
            for hand in frame:
                array = np.asarray(hand, dtype=float)
                if array.shape == (21, 3) and np.isfinite(array).all():
                    hands += 1
                else:
                    invalid_hands += 1
    return {"exported_samples": len(ids), "sample_ids": ids, "finite_21x3_hand_predictions": hands,
            "invalid_hand_predictions": invalid_hands, "metadata_keys": sorted(tag_keys),
            "native_hand_frame_counts": frame_counts,
            "export_sha256": sha(export), "stats_sidecar": str(sidecar) if sidecar.exists() else None,
            "stats_sidecar_sha256": sha(sidecar) if sidecar.exists() else None,
            "validity": "Finite 21x3 predictions only; native model differs, no pose-accuracy or coordinate-equivalence guarantee"}


def run_trial(job, output, timeout):
    import psutil
    env = os.environ.copy()
    env.update(job.get("environment", {}))
    env.update(PYTHONDONTWRITEBYTECODE="1", UV_OFFLINE="1", PIP_NO_INDEX="1", HF_HUB_OFFLINE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    output.mkdir(exist_ok=False)
    before = package_snapshot(job["python"], env)
    write(output / "environment-before.json", before)
    argv = [job["python"], str(ROOT / "scripts/benchmark_products.py"), "worker", "--kind", job["kind"],
            "--config", job["config"], "--output", str(output / "worker")]
    trial = {"name": job["name"], "kind": job["kind"], "argv": argv, "started_utc": datetime.now(timezone.utc).isoformat(),
             "status": "running", "load_before": list(os.getloadavg()), "environment": job.get("environment", {}),
             "memory_method": "20ms sampled process-tree RSS; shared pages can be counted repeatedly",
             "gpu_method": "NVML whole-device samples; includes unrelated processes, not attributable model allocation"}
    nvml, handles = None, []
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml
        handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]
    except Exception as exc:
        trial["gpu_observation_error"] = str(exc)
    peaks = [None] * len(handles)
    baseline = []
    for handle in handles:
        try:
            baseline.append(int(nvml.nvmlDeviceGetMemoryInfo(handle).used))
        except Exception:
            baseline.append(None)
    rss, samples, proc = 0, 0, None
    started = time.perf_counter()
    try:
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr, (output / "observations.jsonl").open("w") as observations:
            proc = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            while proc.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > timeout:
                    raise TimeoutError(f"Trial exceeded {timeout}s")
                current = 0
                try:
                    parent = psutil.Process(proc.pid)
                    for child in [parent, *parent.children(recursive=True)]:
                        try:
                            current += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                rss = max(rss, current)
                devices = []
                for i, handle in enumerate(handles):
                    memory, utilization = None, None
                    try:
                        memory = int(nvml.nvmlDeviceGetMemoryInfo(handle).used)
                        peaks[i] = max(peaks[i] or 0, memory)
                    except Exception:
                        pass
                    try:
                        utilization = int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except Exception:
                        pass
                    devices.append({"index": i, "used_bytes": memory, "utilization_percent": utilization})
                observations.write(json.dumps({"seconds": elapsed, "tree_rss_bytes": current, "devices": devices}) + "\n")
                samples += 1
                time.sleep(.02)
        trial.update(returncode=proc.returncode, wall_seconds=time.perf_counter() - started)
        after = package_snapshot(job["python"], env)
        write(output / "environment-after.json", after)
        trial["environment_stable"] = before == after
        if proc.returncode or before != after:
            raise RuntimeError(f"Worker exited {proc.returncode} or environment changed; inspect raw logs")
        result = json.loads((output / "worker/worker.json").read_text())
        if job["kind"] in ("egoworld_recipe", "datajuicer_matched_recipe"):
            trial["audit"] = audit_egoworld(result["data_manifest"])
        elif job["kind"] == "datajuicer_native":
            trial["audit"] = audit_datajuicer(result["export"])
            for key, minimum in job.get("minimum_outputs", {}).items():
                if trial["audit"].get(key, 0) < minimum:
                    raise ValueError(f"Native pipeline exited successfully but {key} < {minimum}")
        else:
            trial["audit"] = json.loads(Path(result["screening"]).read_text())
        trial["status"] = "passed"
    except BaseException as exc:
        trial.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        trial.setdefault("wall_seconds", time.perf_counter() - started)
        trial.update(process_tree_peak_rss_bytes=rss, gpu_baseline_used_bytes=baseline,
                     gpu_whole_device_peak_used_bytes=peaks, observation_samples=samples)
        write(output / "trial.json", trial)
        if nvml:
            nvml.nvmlShutdown()
    return trial


def run(spec_path, output):
    spec = json.loads(Path(spec_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {str(Path(p).resolve()): sha(p) for p in spec["inputs"]}
    configs = {job["config"]: sha(job["config"]) for job in spec["jobs"]}
    dependencies = {str(Path(p).resolve()): sha(p) for p in spec.get("dependencies", [])}
    sources = {str(p): sha(p) for p in [*(ROOT / "src/egoworld").rglob("*.py"), Path(__file__), ROOT / "scripts/benchmark_products.py"]}
    write(output / "spec.json", spec)
    report = {"version": 1, "scope": spec["scope"], "input_hashes": inputs, "config_hashes": configs, "dependency_hashes": dependencies,
              "source_hashes": sources, "trials": [], "same_model_speed_ratio": None,
              "timing": "Fresh subprocess creation to exit, including imports, complete pipeline and export; audit and asset preparation excluded",
              "limits": ["Single paired pilot, not a repeat/CI or server scaling result.",
                         "Native models, policies or outputs may differ; no matched-model ratio.",
                         "Warm OS/model-file caches uncontrolled; fresh result directories, no prior predictions loaded."]}
    write(output / "report.json", report)
    for job in spec["jobs"]:
        if not job["name"] or Path(job["name"]).name != job["name"]:
            raise ValueError("Trial name must be a plain directory name")
        report["trials"].append(run_trial(job, output / job["name"], spec.get("timeout_seconds", 1200)))
        report["inputs_unchanged"] = all(sha(p) == h for p, h in inputs.items())
        report["configs_unchanged"] = all(sha(p) == h for p, h in configs.items())
        report["sources_unchanged"] = all(sha(p) == h for p, h in sources.items())
        report["dependencies_unchanged"] = all(sha(p) == h for p, h in dependencies.items())
        write(output / "report.json", report)
    report["status"] = "completed" if all(t["status"] == "passed" for t in report["trials"]) and all(report[k] for k in ("inputs_unchanged", "configs_unchanged", "sources_unchanged", "dependencies_unchanged")) else "completed_with_failures"
    write(output / "report.json", report)
    return report
