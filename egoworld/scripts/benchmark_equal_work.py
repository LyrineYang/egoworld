#!/usr/bin/env python3
"""Artifact-cleaning pilot for the equal-input/output research protocol."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("freeze")
    for name in ("hand-pose", "meta", "provenance", "output"):
        corpus.add_argument("--" + name, required=True)
    corpus.add_argument("--fault-fixtures", action="store_true", help="Append four labeled synthetic rejection fixtures; QA only")
    run = sub.add_parser("run")
    for name in ("corpus", "output", "reference-python", "datajuicer-python"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--timeout", type=float, default=300)
    worker = sub.add_parser("worker")
    worker.add_argument("--backend", choices=("stream_reference", "datajuicer"), required=True)
    for name in ("corpus", "output"):
        worker.add_argument("--" + name, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--corpus", required=True)
    audit.add_argument("--trial", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "freeze":
        from equal_work.contract import freeze_corpus
        result = freeze_corpus(**args)
    elif command == "worker":
        from equal_work.worker import run_worker
        result = run_worker(**args)
    elif command == "audit":
        from equal_work.audit import audit_trial
        result = audit_trial(**args)
    else:
        from equal_work.suite import run_suite
        result = run_suite(**args)
    print(json.dumps(result.get("summary", result), indent=2))
    if command == "run" and not result["summary"]["all_trials_valid"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
