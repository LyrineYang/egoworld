#!/usr/bin/env python3
"""Isolated DefaultExecutor entrypoint for the explicit shared-model baseline."""
import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "experiments")]
from matched_product import execute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    execute(args.config, args.samples, args.output)
