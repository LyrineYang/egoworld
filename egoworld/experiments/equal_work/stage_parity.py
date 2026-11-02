"""Engineering parity check for a full native stage, not a throughput benchmark.

Both subprocesses use the same interpreter, request, external model runner and
Writer. Inputs are read in place and verified before and after; this is a small
fixture check, not an immutable dataset release or production recipe executor.
"""
from dataclasses import asdict
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess

from equal_work.contract import DJ_REVISION, ROOT, canonical, sha256, write_json


def installation():
    dist = metadata.distribution("py-data-juicer")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    vcs = direct.get("vcs_info", {})
    url = direct.get("url", "")
    if not (url == f"https://github.com/datajuicer/data-juicer/archive/{DJ_REVISION}.zip" or
            (url.removesuffix(".git") == "https://github.com/datajuicer/data-juicer" and
             vcs.get("commit_id") == DJ_REVISION)):
        raise ValueError("Data-Juicer source revision does not match the pilot pin")
    return {"version": dist.version, "direct_url": direct}


def environment(python, env):
    code = "import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"
    return json.loads(subprocess.check_output([python, "-c", code], env=env, text=True, timeout=60))


def input_identity(request):
    if request["job"].get("_operation") or any(k in request["prepared"] for k in ("input_cache", "_phase_input")):
        raise ValueError("Use a full-stage request without ephemeral phase/input caches")
    files = set()
    video = Path(request["prepared"]["video_path"]).resolve()
    files.add(video)
    for receipt in request["inputs"].values():
        for name, info in receipt["files"].items():
            if Path(name).name != name:
                raise ValueError("Upstream artifact must be a plain filename")
            file = Path(receipt["directory"], name).resolve()
            if sha256(file) != info["sha256"]:
                raise ValueError(f"Upstream artifact changed: {file}")
            files.add(file)
    # Explicit local stage assets and asset directories, including robot URDFs.
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            path = Path(value)
            if path.is_file():
                files.add(path.resolve())
            elif path.is_dir():
                files.update(p.resolve() for p in path.rglob("*") if p.is_file())
    # Recursing a model's working directory would include its entire checkout.
    params = request["config"]["operators"][request["job"]["stage"]]["params"]
    visit({k: v for k, v in params.items() if k != "working_dir"})
    files.update((ROOT / "src/egoworld").rglob("*.py"))
    files.update([Path(__file__), ROOT / "scripts/verify_stage_datajuicer.py"])
    return {str(file): sha256(file) for file in sorted(files)}


def worker(backend, request_path, output):
    from egoworld.config import load_config
    from egoworld.pipeline.stages import ModelActor
    from egoworld.pipeline.artifacts import WriterActor
    request, folder = json.loads(Path(request_path).read_text()), Path(output)
    folder.mkdir(exist_ok=False)
    config_path = folder / "config.json"
    write_json(config_path, request["config"])
    config = asdict(load_config(str(config_path)).resolved())
    clip, stage = request["job"]["clip"], request["job"]["stage"]
    if backend == "native_stage":
        directory = folder / "outputs" / f"video_id={clip['video_id']}" / f"clip_id={clip['clip_id']}"
        work = directory / ".work" / stage
        work.mkdir(parents=True)
        job = {**request["job"], "directory": str(directory), "work_dir": str(work)}
        actor = ModelActor(config)
        try:
            receipt = WriterActor(config).write_stage(job, actor.process(job, request["prepared"], request["inputs"]))
        finally:
            actor.close()
    elif backend == "datajuicer":
        install = installation()
        from egoworld.product import datajuicer_adapter  # register before configuration parsing
        from data_juicer.config import init_configs
        from data_juicer.core import DefaultExecutor
        sample = {"text": "stage parity", "egoworld_clip": clip, "egoworld_prepared": request["prepared"],
                  "egoworld_inputs": request["inputs"], "egoworld_fingerprint": request["job"]["fingerprint"]}
        samples = folder / "samples.jsonl"
        samples.write_text(canonical(sample) + "\n")
        cfg = {"project_name": "egoworld-stage-parity", "dataset_path": str(samples),
               "export_path": str(folder / "datajuicer.jsonl"), "work_dir": str(folder / "dj-work"),
               "ds_cache_dir": str(folder / "hf-cache"), "np": 1, "text_keys": "text",
               "use_cache": False, "use_checkpoint": False, "skip_op_error": False,
               "op_fusion": False, "export_in_parallel": False,
               "process": [{"egoworld_stage_mapper": {"config_path": str(config_path),
                   "stage": stage, "output_root": str(folder / "outputs")}}]}
        write_json(folder / "datajuicer-config.json", cfg)
        DefaultExecutor(init_configs(["--config", str(folder / "datajuicer-config.json")])).run()
        rows = [json.loads(line) for line in (folder / "datajuicer.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("Stage check must return exactly one clip")
        receipt = rows[0]["egoworld_inputs"][stage]
        write_json(folder / "installation.json", install)
    else:
        raise ValueError(backend)
    write_json(folder / "receipt.json", receipt)


def audit_receipts(receipts):
    import numpy as np
    import pyarrow.parquet as pq
    from egoworld.contracts import STAGE_ARTIFACTS, schemas
    from egoworld.pipeline.artifacts import load_stage
    if (len(receipts) != 2 or receipts[0]["stage"] != receipts[1]["stage"] or
            receipts[0]["fingerprint"] != receipts[1]["fingerprint"]):
        raise ValueError("Expected two receipts for the same stage")
    stage = receipts[0]["stage"]
    for receipt in receipts:
        if load_stage(receipt["directory"], stage, receipt["fingerprint"]) is None:
            raise ValueError("Stage commit failed integrity validation")
    counts = {}
    for name in STAGE_ARTIFACTS[stage]:
        tables = [pq.ParquetFile(Path(r["directory"], name + ".parquet")).read() for r in receipts]
        if any(not table.schema.equals(schemas()[name]) for table in tables):
            raise ValueError(f"Schema mismatch: {name}")
        # Full logical fields and multiplicity, ignoring serialization and row order.
        rows = [sorted(canonical(row) for row in table.to_pylist()) for table in tables]
        if rows[0] != rows[1]:
            raise ValueError(f"Output mismatch: {name}")
        counts[name] = len(rows[0])
    if not any(counts.values()):
        raise ValueError("Empty stage outputs do not establish model parity")
    arrays = [{name for name in r["files"] if name.endswith(".npz")} for r in receipts]
    if arrays[0] != arrays[1]:
        raise ValueError("Array file coverage mismatch")
    for name in arrays[0]:
        with np.load(Path(receipts[0]["directory"], name), allow_pickle=False) as a, np.load(Path(receipts[1]["directory"], name), allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError(f"Array mismatch: {name}")
    return {"passed": True, "rows": counts, "comparison": "all logical fields exact, row order ignored; arrays exact"}


def run(request_path, python, output, timeout=300):
    request = json.loads(Path(request_path).read_text())
    identity = input_identity(request)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", request)
    write_json(output / "inputs.json", identity)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0",
               UV_OFFLINE="1", PIP_NO_INDEX="1", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    before = environment(python, env)
    write_json(output / "environment-before.json", before)
    report = {"version": 1, "scope": "single-stage engineering parity, not a throughput or quality benchmark",
              "stage": request["job"]["stage"], "status": "running", "trials": []}
    write_json(output / "report.json", report)
    try:
        for backend in ("native_stage", "datajuicer"):
            argv = [str(Path(python).absolute()), str(ROOT / "scripts/verify_stage_datajuicer.py"),
                    "--worker", backend, "--request", str(output / "request.json"), "--output", str(output / backend)]
            with (output / (backend + ".log")).open("w") as log:
                proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=timeout)
                except BaseException:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    raise
            report["trials"].append({"backend": backend, "argv": argv, "returncode": code})
            write_json(output / "report.json", report)
            if code:
                raise RuntimeError(f"{backend} exited {code}; see {backend}.log")
        after = environment(python, env)
        write_json(output / "environment-after.json", after)
        if before != after or input_identity(request) != identity:
            raise ValueError("Environment, source or inputs changed during parity check")
        report["audit"] = audit_receipts([json.loads((output / b / "receipt.json").read_text()) for b in ("native_stage", "datajuicer")])
        report.update(status="passed", environment_stable=True, inputs_unchanged=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "report.json", report)
    return report
