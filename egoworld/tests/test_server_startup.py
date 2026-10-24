"""Server configuration/startup tests. Model files and inference are explicit fixtures."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import pytest

from egoworld.config import load_config
from egoworld.preflight import asset_report


@pytest.fixture
def server_case(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/full.json").read_text())
    config["paths"].update(output_root=str(tmp_path / "output"), manifest_path=str(tmp_path / "manifests"), state_db_path=str(tmp_path / "state.db"))
    config["scenedetect"].update(method="none", overlap_s=0)
    def file(relative, content="fixture"):
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content); return str(path)
    def checkpoint(name):
        path = Path(file(name))
        with zipfile.ZipFile(path, "w") as archive: archive.writestr("archive/data.pkl", "fixture metadata, not a real model")
        return str(path)
    sam = config["operators"]["sam2"]["params"]
    sam["checkpoint"] = checkpoint("sam2.pt"); sam["config"] = file("sam2.yaml")
    sam["prompting"].update(gd_checkpoint=checkpoint("gd.pth"), gd_config=file("gd.py"))
    hand = config["operators"]["hamer"]["params"]
    hand.update(checkpoint=file("hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"), working_dir=str(tmp_path / "hamer"))
    file("hamer/_DATA/hamer_ckpts/model_config.yaml"); file("hamer/_DATA/data/mano/MANO_RIGHT.pkl")
    pose = config["operators"]["foundationpose"]["params"]
    pose.update(repo_path=str(tmp_path / "fp"), assets_manifest=file("assets.json", json.dumps({"v": {
        "intrinsics": [100, 0, 32, 0, 100, 32, 0, 0, 1], "depth_pattern": "depth/{frame_index:09d}.npy", "depth_scale_to_meters": .001}})))
    file("fp/estimater.py")
    for version in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        file(f"fp/weights/{version}/model_best.pth"); file(f"fp/weights/{version}/config.yml")
    pose["objects"]["cup"]["mesh_path"] = file("cup.obj", "v 0 0 0\nv 0 1 0\nv 1 0 0\nf 1 2 3\n")
    (tmp_path / "depth").mkdir(); np.save(tmp_path / "depth/000000000.npy", np.ones((64, 64), dtype=np.uint16))
    robot = config["operators"]["dex_retarget"]["params"]
    robot.update(config=file("robot.yaml", "retargeting:\n  type: vector\n  urdf_path: hand.urdf\n"), urdf_dir=str(tmp_path / "robots"))
    file("robots/hand.urdf", '<robot name="fixture"><link name="base"/></robot>')
    config["operators"]["fast3r"]["params"]["model_name_or_path"] = str(tmp_path / "fast3r")
    file("fast3r/config.json", "{}"); file("fast3r/model.safetensors")
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps(config))
    return cfg, config


def test_static_assets_for_all_enabled_models_without_gpu(server_case):
    path, _ = server_case
    report = asset_report(load_config(str(path)).resolved(), videos=[{"video_id": "v", "width": 64, "height": 64}], clips=[{"video_id": "v", "frame_start": 0}])
    assert report["ok"], [c for c in report["checks"] if not c["ok"]]


@pytest.mark.parametrize("missing", ["sam2.pt", "hamer/_DATA/data/mano/MANO_RIGHT.pkl", "fp/weights/2024-01-11-20-02-45/model_best.pth", "robots/hand.urdf", "fast3r/model.safetensors"])
def test_missing_model_assets_fail_before_launch(server_case, missing):
    path, _ = server_case; (path.parent / missing).unlink()
    assert not asset_report(load_config(str(path)).resolved())["ok"]


def test_rgbd_manifest_must_cover_requested_video_and_match_resolution(server_case):
    path, _ = server_case; config = load_config(str(path)).resolved()
    assert not asset_report(config, videos=[{"video_id": "missing"}])["ok"]
    assert not asset_report(config, videos=[{"video_id": "v", "width": 128, "height": 64}])["ok"]


def test_model_working_directory_is_restored_and_paths_are_absolute(server_case, monkeypatch):
    from egoworld.pipeline import stages
    path, _ = server_case; config = load_config(str(path)).resolved()
    original = Path.cwd(); expected = Path(config.operators.hamer.params["working_dir"])
    class Model:
        def __init__(self, **params):
            assert Path(params["checkpoint"]).is_absolute()
        def run(self, *args):
            assert Path.cwd() == expected
            assert Path("_DATA/data/mano/MANO_RIGHT.pkl").is_file()
            return {"tables": {"hand_pose": []}}
    monkeypatch.setitem(stages.OPERATORS, "hamer", Model)
    actor = stages.ModelActor(asdict(config))
    actor.process({"stage": "hamer", "clip": {}}, {"video_path": "unused"}, {})
    assert Path.cwd() == original
    monkeypatch.setattr(Model, "run", lambda *args: (_ for _ in ()).throw(ValueError("fixture failure")))
    with pytest.raises(ValueError): actor.process({"stage": "hamer", "clip": {}}, {"video_path": "unused"}, {})
    assert Path.cwd() == original


def test_external_doctor_runs_inside_selected_environment(server_case, monkeypatch):
    from egoworld import preflight
    path, config = server_case
    runner = path.parent / "doctor_fixture.py"
    runner.write_text("import json,sys\nfrom pathlib import Path\nassert '--doctor' in sys.argv\nconfig=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())\nassert Path(config['operators']['hamer']['params']['checkpoint']).is_absolute()\nprint(json.dumps({'ok':False,'checks':[{'name':'fixture missing native library','ok':False}]}))\n")
    config["operators"]["hamer"]["params"]["runner_command"] = [sys.executable, str(runner)]
    path.write_text(json.dumps(config))
    original_import = preflight.importlib.import_module
    torch = SimpleNamespace(__version__="2.5.1", cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda _: "fixture"))
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: torch if name == "torch" else SimpleNamespace() if name == "ray" else original_import(name))
    report = preflight.runtime_report(str(path), gpu=True, stages=["hamer"])
    assert not report["ok"]
    assert any(c.get("report", {}).get("checks", [{}])[0].get("name") == "fixture missing native library" for c in report["checks"])


def test_worker_doctor_protocol_reports_missing_runtime_as_json(server_case):
    path, _ = server_case
    result = subprocess.run([sys.executable, "-m", "egoworld.worker", "--doctor", "--config", str(path), "--stage", "fast3r"], capture_output=True, text=True)
    report = json.loads(result.stdout)
    # This local environment has no Fast3R/CUDA. Failure must be a diagnostic report.
    assert result.returncode == (0 if report["ok"] else 1)
    assert not any(c["name"] == "ray" for c in report["checks"])


def test_documented_launch_cli_commits_all_stages_and_resumes(server_case, monkeypatch, thread_backend, capsys):
    from egoworld import cli
    import egoworld.launch as entry
    from egoworld.pipeline import stages
    from egoworld.manifests.build_manifest import build_manifests
    from egoworld.config import SceneDetectConfig
    from egoworld.pipeline.driver import run_pipeline
    from egoworld.preflight import runtime_report
    from test_pipeline_logic import full_packet
    path, config = server_case
    if not shutil.which("ffmpeg"): pytest.skip("FFmpeg required")
    folder = path.parent / "input"; folder.mkdir(); video = folder / "real-container.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(video)], check=True)
    videos, _ = build_manifests([str(video)], scenedetect=SceneDetectConfig(method="none", overlap_s=0))
    assets = path.parent / "assets.json"; data = json.loads(assets.read_text()); data[videos[0]["video_id"]] = data.pop("v"); assets.write_text(json.dumps(data))
    # Bypass only CUDA prerequisites in this test. Run real CPU checks/assets,
    # manifests, extraction, driver, async backend and Parquet commits.
    monkeypatch.setattr(entry, "runtime_report", lambda cfg, **kwargs: runtime_report(cfg, gpu=False, assets=True))
    monkeypatch.setattr(entry, "run_pipeline", lambda *args, **kwargs: run_pipeline(*args, **kwargs, backend_factory=thread_backend))
    monkeypatch.setattr(stages.ModelActor, "process", lambda self, job, prepared, inputs: full_packet(job["stage"], job["clip"]))
    argv = ["egoworld", "launch", "--config", str(path), "--input-dir", str(folder), "--run-id", "launch_test"]
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", argv); cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "passed" and set(result["stages"]) == {"sam2", "hamer", "foundationpose", "dex_retarget", "fast3r"}
    root = Path(result["run_dir"])
    assert json.loads((root / "run_summary.json").read_text())["skipped_clips"] == 1
    assert len(list(root.glob("video_id=*/clip_id=*/*.parquet"))) == 7
    video.touch()
    with pytest.raises(ValueError, match="new run_id"): cli.main()


def test_launch_check_only_does_not_create_runtime_files(server_case, monkeypatch):
    from egoworld.launch import launch
    path, _ = server_case; folder = path.parent / "input"; folder.mkdir(); (folder / "sample.mp4").write_bytes(b"fixture")
    before = {p for p in path.parent.rglob("*")}
    result = launch(str(path), str(folder), run_id="dry_run", check_only=True)
    assert result["status"] == "ready" and result["startup"]["mode"] == "cpu"
    assert {p for p in path.parent.rglob("*")} == before


def test_setup_uses_environment_directory_for_relative_constraints(tmp_path):
    root = Path(__file__).parents[2]
    command = tmp_path / "conda"
    command.write_text('#!/bin/bash\npwd >> "$TEST_CONDA_LOG"\nexit 0\n'); command.chmod(0o755)
    log = tmp_path / "cwd.log"
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "TEST_CONDA_LOG": str(log)}
    subprocess.run(["bash", str(root / "egoworld/scripts/setup_env.sh")], cwd=tmp_path, env=environment, check=True, capture_output=True)
    rows = log.read_text().splitlines()
    assert rows[:2] == [str(root / "egoworld/env")] * 2
    assert rows[-1] == str(root)
