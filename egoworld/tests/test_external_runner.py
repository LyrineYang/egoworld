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


def session_case(tmp_path, mode):
    runner = tmp_path / 'native_fixture.py'
    counts = tmp_path / 'initializations'
    runner.write_text('''import os,sys
from pathlib import Path
from egoworld.pipeline import stages
from egoworld import worker
class Model:
    def __init__(self, **params):
        with Path(COUNTS).open('a') as f: f.write(str(os.getpid())+'\\n')
    def run(self, *args):
        if args[0] == 'FAIL': raise ValueError('fixture failure')
        if args[0] == 'WAIT': __import__('time').sleep(10)
        return {'frames': [], 'instances': [], 'processed_frames': 1}
stages.OPERATORS['sam2'] = Model
worker.main()
'''.replace('COUNTS', repr(str(counts))))
    config = asdict(PipelineConfig().resolved())
    config['execution']['external_runner_mode'] = mode
    config['operators']['sam2']['params']['runner_command'] = [sys.executable, str(runner)]
    actor = ModelActor(config)
    def run(cid, video='unused'):
        return actor.process({'stage':'sam2', 'clip':{'clip_id':cid}, 'work_dir':str(tmp_path / cid)}, {'video_path':video}, {})
    return actor, run, counts


def test_persistent_runner_reuses_loaded_model_after_original_clip_is_removed(tmp_path):
    import shutil
    actor, run, counts = session_case(tmp_path, 'persistent')
    try:
        first = run('A')
        shutil.rmtree(tmp_path / 'A')
        second = run('B')
        assert len(counts.read_text().splitlines()) == 1
        assert first['stats']['model_cache_hit'] == 0
        assert second['stats']['model_cache_hit'] == 1
        assert second['stats']['result_bytes'] > 0
        process = actor.session.process
    finally:
        actor.close()
    assert process.poll() is not None


def test_per_task_mode_keeps_isolation(tmp_path):
    actor, run, counts = session_case(tmp_path, 'per_task')
    try:
        run('A'); run('B')
        assert len(counts.read_text().splitlines()) == 2
        assert actor.session is None
    finally:
        actor.close()


def test_persistent_model_error_discards_session_and_next_task_restarts(tmp_path):
    actor, run, counts = session_case(tmp_path, 'persistent')
    try:
        run('A')
        process = actor.session.process
        with pytest.raises(ValueError, match='fixture failure'):
            run('B', 'FAIL')
        assert actor.session is None and process.poll() is not None
        run('C')
        assert len(counts.read_text().splitlines()) == 2
    finally:
        actor.close()


def test_persistent_timeout_cleans_up_session(tmp_path):
    actor, run, _ = session_case(tmp_path, 'persistent')
    actor.config['execution']['stage_timeout_s'] = 2
    try:
        run('A')
        process = actor.session.process
        with pytest.raises(TimeoutError, match='timed out'):
            run('B', 'WAIT')
        assert actor.session is None and process.poll() is not None
    finally:
        actor.close()


def test_guard_sigterm_kills_runner_process_group(tmp_path, monkeypatch):
    import concurrent.futures
    import subprocess
    pid_file = tmp_path / 'runner.pid'
    command = "import os,time;from pathlib import Path;Path(PID).write_text(str(os.getpid()));time.sleep(30)".replace('PID', repr(str(pid_file)))
    config = asdict(PipelineConfig().resolved())
    config['execution']['stage_timeout_s'] = 5
    actor = ModelActor(config)
    launched = []
    original = subprocess.Popen
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs); launched.append(child); return child
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    job = {'stage':'sam2', 'clip':{}, 'work_dir':str(tmp_path / 'work')}
    try:
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            task = pool.submit(actor._external, [sys.executable, '-c', command], job, {}, {})
            deadline = time.monotonic() + 4
            while not pid_file.is_file() and time.monotonic() < deadline: time.sleep(.01)
            assert pid_file.is_file()
            os.kill(launched[0].pid, signal.SIGTERM)
            with pytest.raises(RuntimeError, match='exited 125'): task.result(timeout=6)
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)  # guard must have killed AND reaped its child
    finally:
        if pid_file.is_file():
            try: os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError: pass
        actor.close()
