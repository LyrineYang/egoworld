"""Run one native model stage inside a separate Python/Conda environment."""
import argparse
import json
from pathlib import Path
import numpy as np
import traceback
from egoworld.pipeline.stages import ModelActor
from egoworld.io.writers import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--response")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--stage")
    args = parser.parse_args()
    if args.doctor:
        if not args.config or not args.stage:
            parser.error("--doctor requires --config and --stage")
        from egoworld.preflight import runtime_report
        from contextlib import redirect_stdout
        import sys
        with redirect_stdout(sys.stderr):
            report = runtime_report(args.config, gpu=True, stages=[args.stage], external=False, coordinator=False)
        print(json.dumps(report))
        raise SystemExit(0 if report["ok"] else 1)
    if not args.request or not args.response:
        parser.error("--request and --response are required for inference")
    request = json.loads(Path(args.request).read_text())
    request["config"]["operators"][request["job"]["stage"]]["params"].pop("runner_command", None)
    try:
        packet = ModelActor(request["config"]).process(request["job"], request["prepared"], request["inputs"])
    except Exception as exc:
        from egoworld.utils.errors import classify_error
        write_json(args.response, {"error": {"type": type(exc).__name__, "message": str(exc),
                    "retryable": classify_error(exc).retryable or isinstance(exc, TimeoutError)}})
        traceback.print_exc()
        raise SystemExit(1)
    files = {}
    for name, arrays in packet.pop("arrays", {}).items():
        file = Path(args.response).parent / name
        np.savez_compressed(file, **arrays)
        files[name] = str(file.resolve())
    packet["array_files"] = files
    write_json(args.response, packet)


if __name__ == "__main__":
    main()
