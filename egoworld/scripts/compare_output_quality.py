#!/usr/bin/env python3
"""Score native pipeline predictions against a frozen common-output corpus."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from output_quality import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.spec, args.output)
    print(json.dumps({k: {n: v for n, v in values.items() if n not in ("details", "source_outputs")}
                      for k, values in result["pipelines"].items()}, indent=2))
