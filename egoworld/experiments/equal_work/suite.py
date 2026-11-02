"""Sequential subprocess trials, alternating order, with persistent failure records."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from equal_work.audit import audit_trial, summarize
from equal_work.contract import ROOT, sha256, verify_corpus, write_json


def run_suite(corpus, output, reference_python, datajuicer_python, *, repeats=5, timeout=300):
    import psutil

    if repeats < 1 or timeout <= 0:
        raise ValueError("repeats and timeout must be positive")
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    contract = verify_corpus(corpus, verify_sources=True)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    controlled = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                  "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                  "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
                  "UV_OFFLINE": "1", "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1"}
    env.update(controlled)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    interpreters = {"stream_reference": str(Path(reference_python).absolute()),
                    "datajuicer": str(Path(datajuicer_python).absolute())}
    manifest = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "contract": contract, "contract_sha256": sha256(corpus / "contract.json"),
                "repeats": repeats, "controlled_environment": controlled,
                "platform": platform.platform(), "cpu_count": os.cpu_count(),
                "cpu_affinity": psutil.Process().cpu_affinity(), "interpreters": interpreters,
                "timing": "Fresh process start to successful exit, including imports, input verification, executor and final writes; excludes corpus freezing, source admission and audit",
                "cache_state": "Unique empty work/cache directories per trial, OS caches uncontrolled, no model execution",
                "memory_method": "20ms sampled sum RSS of child process tree; shared pages may be double counted",
                "trials": []}
    # Runtime package snapshots outside timed intervals, including installation provenance.
    environments = {}
    def environment(python):
        command = [python, "-c", "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"]
        return subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
    for backend, python in interpreters.items():
        result = environment(python)
        environments[backend] = result.stdout if result.returncode == 0 else None
        (output / f"environment-{backend}.json").write_text(result.stdout)
        if result.returncode:
            (output / f"environment-{backend}.error.txt").write_text(result.stderr)
    write_json(output / "report.json", manifest)
    for repeat in range(repeats):
        order = list(interpreters)
        if repeat % 2:
            order.reverse()
        for backend in order:
            folder = output / f"{repeat:03d}-{backend}"
            folder.mkdir()
            argv = [interpreters[backend], str(ROOT / "scripts/benchmark_equal_work.py"), "worker",
                    "--backend", backend, "--corpus", str(corpus), "--output", str(folder)]
            trial = {"repeat": repeat, "backend": backend, "argv": argv, "directory": folder.name,
                     "started_utc": datetime.now(timezone.utc).isoformat(), "load_average_before": list(os.getloadavg())}
            peak = 0
            proc = None
            started = time.perf_counter()
            try:
                with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                    proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
                    while proc.poll() is None:
                        if time.perf_counter() - started > timeout:
                            raise TimeoutError(f"Invocation exceeded {timeout}s")
                        try:
                            root = psutil.Process(proc.pid)
                            processes = [root, *root.children(recursive=True)]
                            rss = 0
                            for p in processes:
                                try:
                                    rss += p.memory_info().rss
                                except psutil.NoSuchProcess:
                                    pass
                            peak = max(peak, rss)
                        except psutil.NoSuchProcess:
                            pass
                        time.sleep(.02)
                trial["returncode"] = proc.returncode
                trial["wall_seconds"] = time.perf_counter() - started
                if proc.returncode:
                    raise RuntimeError(f"Worker exited with code {proc.returncode}; see stderr.log")
                trial["audit"] = audit_trial(corpus, folder)
                trial["status"] = "ok"
            except BaseException as exc:
                trial.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if proc and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                trial.setdefault("wall_seconds", time.perf_counter() - started)
                trial["process_tree_peak_rss_bytes"] = peak
                write_json(folder / "trial.json", trial)
                manifest["trials"].append(trial)
                # No eligible ratio until the post-run environment check completes.
                manifest["summary"] = summarize(manifest["trials"], repeats, environment_stable=False)
                write_json(output / "report.json", manifest)
    stable = True
    for backend, python in interpreters.items():
        result = environment(python)
        (output / f"environment-after-{backend}.json").write_text(result.stdout)
        stable = stable and result.returncode == 0 and environments[backend] == result.stdout
    manifest["summary"] = summarize(manifest["trials"], repeats, environment_stable=stable)
    write_json(output / "report.json", manifest)
    return manifest
