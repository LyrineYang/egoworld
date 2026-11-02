"""Fresh-process executor adapters. No configuration auto-tuning or fallback."""
import importlib.metadata
import json
from pathlib import Path
import resource
import time

from equal_work.contract import (DJ_REVISION, accepted_record, canonical, evaluate,
                                 records, sha256, verify_corpus, write_json)


def run_worker(backend, corpus, output):
    folder, corpus = Path(output), Path(corpus)
    if any((folder / name).exists() for name in ("identity.json", "worker.json", "decisions.jsonl", "accepted.jsonl")):
        raise FileExistsError("Worker output already exists; retain it and use a new directory")
    started = time.perf_counter()
    contract = verify_corpus(corpus)
    identity = {"backend": backend, "contract_sha256": sha256(corpus / "contract.json"),
                "corpus_sha256": contract["corpus_sha256"]}
    write_json(folder / "identity.json", identity)
    phases = {}
    if backend == "datajuicer":
        # Refuse an unpinned or different installation, including PyPI versions
        # with the same version number but different source code.
        dist = importlib.metadata.distribution("py-data-juicer")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        url = direct.get("url", "")
        pinned_url = "https://github.com/datajuicer/data-juicer/archive/" + DJ_REVISION + ".zip"
        vcs = direct.get("vcs_info", {})
        pinned_git = url.removesuffix(".git") == "https://github.com/datajuicer/data-juicer" and vcs.get("commit_id") == DJ_REVISION
        if url != pinned_url and not pinned_git:
            raise ValueError("Data-Juicer installation does not match the pinned revision")
        identity.update(datajuicer_version=dist.version, datajuicer_install=direct)
        write_json(folder / "identity.json", identity)
        from equal_work import dj_operator  # Register before init_configs parses OP arguments.
        from data_juicer.config import init_configs
        from data_juicer.core import DefaultExecutor

        cfg = {"project_name": "egoworld-equal-work", "dataset_path": str(corpus / "samples.jsonl"),
               "export_path": str(folder / "datajuicer.jsonl"), "work_dir": str(folder / "dj-work"),
               "ds_cache_dir": str(folder / "hf-cache"), "text_keys": "payload", "np": 1,
               "use_cache": False, "use_checkpoint": False, "skip_op_error": False,
               "op_fusion": False, "export_in_parallel": False,
               "process": [{"egoworld_hand_contract_mapper": {}}]}
        write_json(folder / "datajuicer-config.json", cfg)
        executor = DefaultExecutor(init_configs(["--config", str(folder / "datajuicer-config.json")]))
        phases["setup_seconds"] = time.perf_counter() - started
        before = time.perf_counter()
        executor.run()
        phases["executor_seconds"] = time.perf_counter() - before
        samples = records(folder / "datajuicer.jsonl")
    elif backend == "stream_reference":
        # Experimental streaming reference, not EgoWorld's production recipe executor.
        from egoworld import contracts  # Include rule imports in setup on both paths.
        phases["setup_seconds"] = time.perf_counter() - started
        samples = records(corpus / "samples.jsonl")
    else:
        raise ValueError("Unknown backend")
    before = time.perf_counter()
    total = kept = 0
    with (folder / "decisions.jsonl").open("w") as decisions, (folder / "accepted.jsonl").open("w") as accepted:
        for sample in samples:
            decision = json.loads(sample["decision"]) if backend == "datajuicer" else evaluate(sample)
            decisions.write(canonical(decision) + "\n")
            total += 1
            if decision["accepted"]:
                accepted.write(canonical(accepted_record(sample)) + "\n")
                kept += 1
    phases["normalize_and_write_seconds"] = time.perf_counter() - before
    result = {**identity, "status": "ok", "rows": total, "accepted": kept, "phases": phases,
              "worker_seconds": time.perf_counter() - started,
              "self_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "output_sha256": {name: sha256(folder / name) for name in ("decisions.jsonl", "accepted.jsonl")}}
    write_json(folder / "worker.json", result)
    return result
