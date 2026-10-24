"""R02/R08: real Ray processes + real FFmpeg/Parquet; model outputs are explicit fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import pytest


@pytest.mark.ray_integration
def test_real_ray_all_branches_and_resume(tmp_path):
    if os.getenv("EGOWORLD_RAY_TEST") != "1": pytest.skip("Set EGOWORLD_RAY_TEST=1 to run real Ray scheduling")
    ray = pytest.importorskip("ray")
    from egoworld.pipeline.backend import RayBackend
    from egoworld.pipeline.driver import run_pipeline
    from egoworld.manifests.build_manifest import build_manifests, write_manifest_json
    from egoworld.config import SceneDetectConfig
    import shutil
    if not shutil.which("ffmpeg"): pytest.fail("FFmpeg required for Ray integration")
    video = tmp_path / "input.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=1:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(video)], check=True)
    videos, clips = build_manifests([str(video)], scenedetect=SceneDetectConfig(method="none", overlap_s=0))
    clips = [{**clips[0], "clip_id": f"c{i}"} for i in range(3)]
    vm, cm = tmp_path / "videos.jsonl", tmp_path / "clips.jsonl"
    write_manifest_json(str(vm), videos); write_manifest_json(str(cm), clips)
    runner = tmp_path / "fixture_runner.py"
    runner.write_text('''import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, TEST_DIRECTORY)
from test_pipeline_logic import full_packet
parser=argparse.ArgumentParser();parser.add_argument('--request');parser.add_argument('--response');args=parser.parse_args()
request=json.loads(Path(args.request).read_text());stage=request['job']['stage']
for dependency in {'hamer':['sam2'],'foundationpose':['sam2'],'dex_retarget':['hamer']}.get(stage,[]):
    assert Path(request['inputs'][dependency]['directory'],f'stage_{dependency}.json').is_file()
packet=full_packet(stage,request['job']['clip']);packet['array_files']={}
for name, arrays in packet.pop('arrays',{}).items():
    file=Path(args.response).parent/name;np.savez_compressed(file,**arrays);packet['array_files'][name]=str(file)
Path(args.response).write_text(json.dumps(packet))
'''.replace("TEST_DIRECTORY", repr(str(Path(__file__).parent))))
    config = json.loads((Path(__file__).parents[1] / "configs/full.json").read_text())
    config.update(num_gpus=2, run_id="ray_test")
    config["paths"].update(output_root=str(tmp_path / "output"), state_db_path=str(tmp_path / "state.db"))
    for name, operator in config["operators"].items():
        if operator["enabled"]:
            operator["params"].pop("working_dir", None)  # fixtures do not load upstream assets
            operator["params"]["runner_command"] = [sys.executable, str(runner)]
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps(config))
    # Ray's two declared accelerator resources test scheduling only. No CUDA inference occurs.
    # Keep AF_UNIX paths short on macOS as well as Linux.
    with tempfile.TemporaryDirectory(prefix="ew-ray-", dir="/tmp") as ray_temp:
        ray.init(num_cpus=8, num_gpus=2, include_dashboard=False, object_store_memory=100 * 1024**2, _temp_dir=ray_temp)
        try:
            run_pipeline(str(cfg), str(vm), str(cm), backend_factory=RayBackend)
            root = tmp_path / "output/run_id=ray_test"
            summary = json.loads((root / "run_summary.json").read_text())
            assert summary["completed_clips"] == 3 and summary["failed_clips"] == 0
            assert set(summary["stage_metrics"]) == {"sam2", "hamer", "foundationpose", "dex_retarget", "fast3r"}
            run_pipeline(str(cfg), str(vm), str(cm), backend_factory=RayBackend)
            assert json.loads((root / "run_summary.json").read_text())["skipped_clips"] == 3
        finally:
            ray.shutdown()


@pytest.mark.ray_integration
def test_real_ray_replaces_crashed_and_timed_out_model_workers(tmp_path):
    if os.getenv("EGOWORLD_RAY_TEST") != "1": pytest.skip("Set EGOWORLD_RAY_TEST=1")
    ray = pytest.importorskip("ray")
    from dataclasses import asdict
    from egoworld.config import PipelineConfig
    from egoworld.pipeline.backend import RayBackend
    runner = tmp_path / "runner.py"
    runner.write_text("import json,sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--response')+1]).write_text(json.dumps({'tables':{'masks':[],'instances':[],'tracks':[]},'stats':{}}))\n")
    config = asdict(PipelineConfig().resolved())
    config["operators"]["sam2"]["params"]["runner_command"] = [sys.executable, str(runner)]
    job = {"stage": "sam2", "fingerprint": "fixture", "work_dir": str(tmp_path / "work"), "directory": str(tmp_path / "out"),
           "clip": {"clip_id": "c", "video_id": "v", "frame_start": 0, "frame_end": 10, "fps": 10.}}
    with tempfile.TemporaryDirectory(prefix="ew-ray-", dir="/tmp") as folder:
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=100 * 1024**2, _temp_dir=folder)
        backend = RayBackend(config)
        def event():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                events = backend.poll(.1)
                if events: return events[0]
            pytest.fail("Ray backend did not produce an event")
        try:
            index = backend.available("compute", "sam2")[0]
            slot = backend.slots[index]
            ray.kill(slot["actor"], no_restart=True)
            @ray.remote(num_cpus=1, num_gpus=1)
            class Crash:
                def compute(self, *args): os._exit(9)
            slot["actor"] = Crash.remote()
            crashed_actor = slot["actor"]
            backend.submit("compute", job, prepared={"video_path": "unused"}, inputs={})
            kind, _, _, error, _ = event()
            assert kind == "compute" and error is not None
            assert slot["actor"] != crashed_actor
            # The replacement runs production ModelActor + a declared fixture runner.
            job.pop("_retry_compute", None)
            backend.submit("compute", job, prepared={"video_path": "unused"}, inputs={})
            _, _, reference, _, _ = event()
            backend.submit("write", job, result=reference)
            _, _, receipt, error, _ = event()
            assert error is None and receipt["stage"] == "sam2"
            previous = slot["actor"]
            ref = backend.submit("compute", job, prepared={"video_path": "unused"}, inputs={})
            # Advance the watchdog timestamp rather than spending 15 minutes asleep.
            backend.pending[ref]["started"] -= config["execution"]["stage_timeout_s"] + 1
            _, _, _, error, _ = event()
            assert isinstance(error, TimeoutError) and slot["actor"] != previous
            assert not backend.pending
        finally:
            backend.close(); ray.shutdown()


@pytest.mark.ray_integration
def test_model_error_is_checked_before_slot_reuse(tmp_path, monkeypatch):
    if os.getenv("EGOWORLD_RAY_TEST") != "1": pytest.skip("Set EGOWORLD_RAY_TEST=1")
    ray = pytest.importorskip("ray")
    from dataclasses import asdict
    from egoworld.config import PipelineConfig
    from egoworld.pipeline.backend import RayBackend
    runner = tmp_path / "runner.py"
    runner.write_text("""import json,sys
from pathlib import Path
request=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_text())
if request['job']['clip']['clip_id']=='A':
    reply={'error':{'type':'OutOfMemoryError','message':'CUDA out of memory','retryable':True}}
else:
    reply={'tables':{'masks':[],'instances':[],'tracks':[]},'stats':{}}
Path(sys.argv[sys.argv.index('--response')+1]).write_text(json.dumps(reply))
""")
    config = asdict(PipelineConfig().resolved())
    config["operators"]["sam2"]["params"]["runner_command"] = [sys.executable, str(runner)]
    job = {"stage": "sam2", "fingerprint": "fixture", "work_dir": str(tmp_path / "work"), "directory": str(tmp_path / "out"),
           "clip": {"clip_id": "A", "video_id": "v", "frame_start": 0, "frame_end": 10, "fps": 10.}}
    with tempfile.TemporaryDirectory(prefix="ew-ray-", dir="/tmp") as folder:
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=100 * 1024**2, _temp_dir=folder)
        backend = RayBackend(config)
        def event():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if events := backend.poll(.1): return events[0]
            pytest.fail("No Ray event")
        try:
            slot = backend.slots[backend.available("compute", "sam2")[0]]
            old_actor = slot["actor"]
            backend.submit("compute", job, prepared={"video_path": "unused"}, inputs={})
            assert not backend.available("compute", "sam2")
            kind, _, value, error, _ = event()
            assert kind == "compute" and error is not None and value is None
            assert slot["actor"] != old_actor
            healthy_actor = slot["actor"]
            other = {**job, "clip": {**job["clip"], "clip_id": "B"}}
            control = backend.submit("compute", other, prepared={"video_path": "unused"}, inputs={})
            payload = backend.pending[control]["result"]
            fetched = []
            original_get = ray.get
            def get(ref, *args, **kwargs):
                fetched.append(ref)
                return original_get(ref, *args, **kwargs)
            monkeypatch.setattr(ray, "get", get)
            kind, _, reference, error, _ = event()
            assert kind == "compute" and error is None and reference == payload
            assert payload not in fetched  # the driver only fetched the status return
            backend.submit("write", other, result=reference)
            _, _, receipt, error, _ = event()
            assert error is None and receipt["stage"] == "sam2" and slot["actor"] == healthy_actor
        finally:
            backend.close(); ray.shutdown()


@pytest.mark.ray_integration
@pytest.mark.parametrize("stop", ["close", "kill"])
def test_external_process_group_stops_when_actor_exits(tmp_path, stop):
    if os.getenv("EGOWORLD_RAY_TEST") != "1": pytest.skip("Set EGOWORLD_RAY_TEST=1")
    ray = pytest.importorskip("ray")
    import signal
    from dataclasses import asdict
    from egoworld.config import PipelineConfig
    from egoworld.pipeline.backend import RayBackend
    heartbeat = tmp_path / "heartbeat"
    pid_file = tmp_path / "pid"
    program = ("import os,time;from pathlib import Path;"
               f"Path({str(pid_file)!r}).write_text(str(os.getpid()));p=Path({str(heartbeat)!r});\n"
               "while True: p.write_text(str(time.time()));time.sleep(.05)")
    config = asdict(PipelineConfig().resolved())
    config["operators"]["sam2"]["params"]["runner_command"] = [sys.executable, "-c", program]
    with tempfile.TemporaryDirectory(prefix="ew-ray-", dir="/tmp") as folder:
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=100 * 1024**2, _temp_dir=folder)
        backend = RayBackend(config)
        try:
            job = {"stage": "sam2", "work_dir": str(tmp_path / "work"), "clip": {"clip_id": "c"}}
            slot = backend.slots[backend.available("compute", "sam2")[0]]
            backend.submit("compute", job, prepared={"video_path": "unused"}, inputs={})
            deadline = time.monotonic() + 20
            while not heartbeat.is_file() and time.monotonic() < deadline: time.sleep(.05)
            assert heartbeat.is_file(), "External runner never started"
            if stop == "close": backend.close()
            else: ray.kill(slot["actor"], no_restart=True)
            # A stable heartbeat alone could be a zombie; additionally require
            # disappearance or a zombie state, never a live inference process.
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                                       capture_output=True, text=True).stdout.strip()
                if not state or state.startswith("Z"):
                    break
                time.sleep(.05)
            else: pytest.fail("External runner survived actor exit")
            stamp = heartbeat.read_text(); time.sleep(.15)
            assert heartbeat.read_text() == stamp
        finally:
            if pid_file.is_file():
                try: os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError: pass
            backend.close(); ray.shutdown()
