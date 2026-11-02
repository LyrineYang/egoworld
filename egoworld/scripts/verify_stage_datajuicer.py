#!/usr/bin/env python3
"""Verify a full-stage native request through real Data-Juicer; no speed ratio."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Existing full-stage egoworld.worker request.json")
    parser.add_argument("--output", required=True, help="New directory; never overwrite prior attempts")
    parser.add_argument("--python", help="Same pinned Data-Juicer interpreter for both subprocesses")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--worker", choices=("native_stage", "datajuicer"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    from equal_work.stage_parity import run, worker
    if args.worker:
        worker(args.worker, args.request, args.output)
    else:
        if not args.python or args.timeout <= 0:
            parser.error("--python and a positive --timeout are required")
        print(json.dumps(run(args.request, args.python, args.output, args.timeout), indent=2))


if __name__ == "__main__":
    main()
