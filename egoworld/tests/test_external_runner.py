"""Real subprocess cleanup, without Ray, CUDA, or model dependencies."""
from dataclasses import asdict
import os
import signal
import sys
import time

import pytest

from egoworld.config import PipelineConfig
from egoworld.pipeline.stages import ModelActor


@pytest.mark.parametrize("hang", [False, True])
def test_external_runner_cleans_up_descendants_on_exit_and_timeout(tmp_path, hang):
    heartbeat, pid_file = tmp_path / "heartbeat", tmp_path / "pid"
    child = ("import os,time;from pathlib import Path;"
             f"Path({str(pid_file)!r}).write_text(str(os.getpid()));p=Path({str(heartbeat)!r});\n"
             "while True: p.write_text(str(time.time()));time.sleep(.05)")
    runner = tmp_path / "runner.py"
    runner.write_text("import json,subprocess,sys,time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        f"while not Path({str(heartbeat)!r}).is_file(): time.sleep(.01)\n" +
        ("time.sleep(60)\n" if hang else
         "Path(sys.argv[sys.argv.index('--response')+1]).write_text(json.dumps({'tables':{'masks':[],'instances':[],'tracks':[]}}))\n"))
    config = asdict(PipelineConfig().resolved())
    config["execution"]["stage_timeout_s"] = 3
    actor = ModelActor(config)
    job = {"stage": "sam2", "clip": {}, "work_dir": str(tmp_path / "work")}
    try:
        if hang:
            with pytest.raises(TimeoutError, match="timed out"):
                actor._external([sys.executable, str(runner)], job, {}, {})
        else:
            assert actor._external([sys.executable, str(runner)], job, {}, {})["tables"]
        assert heartbeat.is_file()
        before = heartbeat.read_text(); time.sleep(.2)
        assert heartbeat.read_text() == before
    finally:
        if pid_file.is_file():
            try: os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError: pass


def test_native_weights_are_released_before_external_runner(monkeypatch):
    import weakref
    from types import SimpleNamespace
    config = asdict(PipelineConfig().resolved())
    config["operators"]["sam2"]["params"]["runner_command"] = ["fixture-python"]
    actor = ModelActor(config)
    class Weights: pass
    weights = Weights(); reference = weakref.ref(weights)
    actor.cache["hamer"] = weights
    del weights
    emptied = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: emptied.append(True))))
    def external(*args):
        assert reference() is None and not actor.cache and emptied
        return {"tables": {"masks": [], "instances": [], "tracks": []}}
    monkeypatch.setattr(actor, "_external", external)
    assert actor.process({"stage": "sam2", "clip": {}}, {}, {})["tables"]
