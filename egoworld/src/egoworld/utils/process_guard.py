"""Run an external model until completion or the owning actor's pipe closes."""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("Expected an external runner command")
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
    def cancelled(*_):
        raise SystemExit(125)
    # Ray may SIGTERM descendants directly on Linux. Python's default handler
    # would bypass finally and leave the runner's separate process group alive.
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        while child.poll() is None:
            if select.select([sys.stdin.fileno()], [], [], .1)[0]:
                if not os.read(sys.stdin.fileno(), 1):
                    return 125  # actor exited or explicitly cancelled this task
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
