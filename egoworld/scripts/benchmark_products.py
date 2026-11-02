#!/usr/bin/env python3
"""Measure complete product jobs; retain native Data-Juicer resource defaults."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--output", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--kind", required=True, choices=("egoworld_recipe", "egoworld_cleaning", "datajuicer_native"))
    worker.add_argument("--config", required=True)
    worker.add_argument("--output", required=True)
    args = parser.parse_args()
    import product_comparison as experiment
    if args.command == "worker":
        experiment.worker(args.kind, args.config, args.output)
    else:
        result = experiment.run(args.spec, args.output)
        print(json.dumps({"status": result["status"], "trials": [{k: t.get(k) for k in
              ("name", "status", "wall_seconds", "process_tree_peak_rss_bytes", "error")} for t in result["trials"]]}, indent=2))
        if result["status"] != "completed":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
