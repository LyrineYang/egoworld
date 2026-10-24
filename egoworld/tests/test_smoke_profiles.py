"""Smoke profile configuration and complete local acceptance flow, no CUDA."""
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import pytest

from egoworld.acceptance import prepare_smoke_config, single_gpu_smoke
from egoworld.config import load_config, PipelineConfig


def test_16gb_profile_bounds_work_and_preserves_model_environments():
    config = load_config(str(Path(__file__).parents[1] / "configs/full.json"))
    command = ["/configured/hamer/python", "-m", "egoworld.worker"]
    config.operators.hamer.params["runner_command"] = command
    config = prepare_smoke_config(config, profile="16gb")
    assert config.num_gpus == config.execution.max_active_clips == config.execution.max_cached_models == 1
    assert config.execution.cpu_workers == config.execution.write_workers == config.execution.cpu_operator_workers == 1
    assert config.backpressure.max_in_flight_cpu == config.backpressure.max_in_flight_gpu == config.backpressure.max_in_flight_write == 1
    assert config.retry.max_retries == 0 and config.execution.keep_prepared
    assert config.operators.hamer.params["batch_size"] == 1
    assert config.operators.hamer.params["runner_command"] == command
    assert config.operators.fast3r.params["max_views"] == 2
    assert config.operators.fast3r.params["dtype"] == "bf16"
    assert config.operators.fast3r.params["attn_implementation"] == "pytorch_auto"
    assert config.operators.sam2.params["runner_command"] == [sys.executable, "-m", "egoworld.worker"]


def test_smoke_selection_precedes_validation_and_does_not_require_disabled_assets():
    config = PipelineConfig(num_gpus=4)
    config.execution.gpu_stage_groups = [["sam2"], ["hamer"], ["foundationpose"], ["fast3r"]]
    result = prepare_smoke_config(config, stages=["sam2"], profile="16gb")
    assert result.num_gpus == 1 and not result.execution.gpu_stage_groups
    assert result.operators.sam2.enabled and not result.operators.fast3r.enabled


@pytest.mark.parametrize("profile,seconds,error", [("16gb", 4, "<= 3"), ("invalid", 2, "Unknown smoke profile")])
def test_bad_smoke_profile_is_explicit(tmp_path, profile, seconds, error):
    video = tmp_path / "input.mp4"; video.write_bytes(b"fixture")
    config = tmp_path / "config.json"; config.write_text("{}")
    with pytest.raises(ValueError, match=error):
        single_gpu_smoke(str(config), str(video), str(tmp_path / "out"), seconds, "hand", profile=profile)
    assert not (tmp_path / "out").exists()


def test_16gb_smoke_cli_runs_real_io_and_resume_with_declared_model_fixtures(tmp_path, monkeypatch, thread_backend, capsys):
    from egoworld import acceptance, cli, preflight
    from egoworld.pipeline import driver, stages
    from test_pipeline_logic import full_packet
    config = json.loads((Path(__file__).parents[1] / "configs/rgb.json").read_text())
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    original_config = config_path.read_bytes()
    video = tmp_path / "input.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc=duration=1:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(video)], check=True)
    calls = []
    def process(self, job, *_):
        calls.append(job["stage"])
        return full_packet(job["stage"], job["clip"])
    monkeypatch.setattr(stages.ModelActor, "process", process)
    real_run = driver.run_pipeline
    monkeypatch.setattr(driver, "run_pipeline", lambda *args, **kwargs: real_run(*args, **kwargs, backend_factory=thread_backend))
    checked = []
    def runtime(path, **kwargs):
        checked.append(kwargs)
        return {"ok": True, "checks": [], "validation_scope": "CUDA and native assets explicitly bypassed in local fixture test"}
    monkeypatch.setattr(preflight, "runtime_report", runtime)
    monkeypatch.setattr(sys, "argv", ["egoworld", "smoke", "--config", str(config_path), "--input-video", str(video),
        "--output-dir", str(tmp_path / "out"), "--seconds", "2", "--profile", "16gb"])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed" and report["resume_verified"] and report["profile"] == "16gb"
    assert set(report["stage_rows"]) == {"sam2", "hamer", "dex_retarget", "fast3r"}
    assert len(calls) == 4  # second run reuses every committed stage
    assert checked == [{"gpu": True, "assets": True}]
    assert config_path.read_bytes() == original_config
    snapshot = json.loads(Path(report["config_path"]).read_text())
    assert snapshot["execution"]["max_active_clips"] == 1
    assert snapshot["operators"]["fast3r"]["params"]["max_views"] == 2
