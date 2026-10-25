"""Serial, file-backed runner sessions. The ownership pipe is reserved for cleanup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

from egoworld.io.writers import write_json


def runner_environment(threads):
    return dict(os.environ, **{name: str(threads) for name in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


class RunnerSession:
    def __init__(self, command, stage, log_dir, threads):
        self.key = (tuple(command), stage)
        self.folder = tempfile.TemporaryDirectory(prefix="ew-runner-")
        self.mailbox = Path(self.folder.name)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{stage}-{uuid4().hex}.log"
        env = runner_environment(threads)
        with self.log_path.open("wb") as log:
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "egoworld.utils.process_guard", "--"] + list(command) +
                    ["--serve", str(self.mailbox), "--stage", stage], stdin=subprocess.PIPE,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            except BaseException:
                self.folder.cleanup()
                raise

    def run(self, request, response, timeout):
        write_json(str(self.mailbox / "task.json"), {"request": str(request), "response": str(response)})
        deadline = time.monotonic() + timeout
        while not response.is_file():
            if self.process.poll() is not None:
                raise RuntimeError(f"Persistent runner exited {self.process.returncode}; see {self.log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Persistent runner timed out; see {self.log_path}")
            time.sleep(.02)
        return json.loads(response.read_text())

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=5)
        finally:
            self.folder.cleanup()
