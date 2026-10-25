"""Run native stages in a separate environment, once or in a serial session."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from egoworld.pipeline.stages import ModelActor
from egoworld.io.writers import write_json


def execute(request_path, response_path, actor=None):
    request = json.loads(Path(request_path).read_text())
    request["config"]["operators"][request["job"]["stage"]]["params"].pop("runner_command", None)
    actor = actor or ModelActor(request["config"])
    try:
        packet = actor.process(request["job"], request["prepared"], request["inputs"])
    except Exception as exc:
        from egoworld.utils.errors import classify_error
        write_json(response_path, {"error": {"type": type(exc).__name__, "message": str(exc),
                    "retryable": classify_error(exc).retryable or isinstance(exc, TimeoutError)}})
        traceback.print_exc()
        return actor, False
    files = {}
    for name, arrays in packet.pop("arrays", {}).items():
        file = Path(response_path).parent / name
        np.savez_compressed(file, **arrays)
        files[name] = str(file.resolve())
    packet["array_files"] = files
    write_json(response_path, packet)
    return actor, True


def serve(folder, stage):
    actor = None
    signature = None
    task_path = Path(folder) / "task.json"
    try:
        while True:
            if not task_path.is_file():
                time.sleep(.02)
                continue
            task = json.loads(task_path.read_text())
            task_path.unlink()  # one request in flight; response is the acknowledgement
            request = json.loads(Path(task["request"]).read_text())
            current = json.dumps(request["config"], sort_keys=True)
            if request["job"]["stage"] != stage or (signature is not None and signature != current):
                write_json(task["response"], {"error": {"type": "ValueError", "retryable": False,
                    "message": "Runner session stage/config changed; start a new session"}})
                return 1
            signature = current
            actor, ok = execute(task["request"], task["response"], actor)
            if not ok:
                return 1  # discard potentially corrupted model/CUDA state
    finally:
        if actor:
            actor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--response")
    parser.add_argument("--serve", help="Serial file-backed session mailbox; used by ModelActor")
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
    if args.serve:
        if not args.stage:
            parser.error("--serve requires --stage")
        raise SystemExit(serve(args.serve, args.stage))
    if not args.request or not args.response:
        parser.error("--request and --response are required for inference")
    actor = None
    try:
        actor, ok = execute(args.request, args.response)
        raise SystemExit(0 if ok else 1)
    finally:
        if actor:
            actor.close()


if __name__ == "__main__":
    main()
