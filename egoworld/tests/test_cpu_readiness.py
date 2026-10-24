import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from egoworld.config import PipelineConfig, SceneDetectConfig, load_config
from egoworld.manifests import build_manifest as bm
from egoworld.preflight import checkpoint_error


def test_fallback_clips_have_frame_aligned_duration_bound(monkeypatch):
    monkeypatch.setattr(bm, "parse_video_meta", lambda *args, **kwargs: bm.VideoMeta(
        "v", "/tmp/input.mp4", 101.0, 29.97, 640, 480, False, "abc", "train"))
    _, clips = bm.build_manifests(["unused"], scenedetect=SceneDetectConfig(
        method="none", overlap_s=1, max_clip_duration_s=15))
    assert len(clips) == 8
    assert clips[0]["frame_start"] == 0
    assert clips[-1]["frame_end"] == round(101 * 29.97)
    assert all(c["end_s"] - c["start_s"] <= 15 for c in clips)
    assert all(a["frame_end"] - b["frame_start"] == round(29.97) for a, b in zip(clips, clips[1:]))


def test_truncated_checkpoint_is_detected_without_torch(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", "test metadata only")
    assert checkpoint_error(str(checkpoint)) is None
    checkpoint.write_bytes(checkpoint.read_bytes()[:-24])
    assert "complete" in checkpoint_error(str(checkpoint))


@pytest.mark.parametrize("field,value", [("num_gpus", 0), ("num_gpus", -1), ("num_gpus", 1.5)])
def test_invalid_gpu_counts_rejected(field, value):
    config = PipelineConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError, match="num_gpus"):
        config.resolved()


def test_enabled_model_requires_its_runtime_parameters():
    config = PipelineConfig()
    config.operators.hamer.enabled = True
    with pytest.raises(ValueError, match="checkpoint is required"):
        config.resolved()


def test_manifest_and_input_check_in_process_without_torch_or_ray(tmp_path):
    import shutil
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required")
    video = tmp_path / "input.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=1:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(video)], check=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"scenedetect": {"method": "none"}}))
    # Importing or initializing a GPU dependency would fail in this subprocess.
    program = '''
import sys
sys.modules['torch'] = None
sys.modules['ray'] = None
from egoworld.cli import main
main()
'''
    command = [sys.executable, "-c", program]
    subprocess.run(command + ["make-manifest", "--config", str(config), "--input-dir", str(tmp_path),
                              "--output-dir", str(tmp_path / "manifests")], check=True, capture_output=True)
    result = subprocess.run(command + ["run", "--config", str(config), "--check-only",
                            "--video-manifest", str(tmp_path / "manifests/video_manifest.jsonl"),
                            "--clip-manifest", str(tmp_path / "manifests/clip_manifest.jsonl")],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["clips"] == 1
    assert not (tmp_path / "state").exists()


def test_example_defaults_to_one_gpu_and_bounded_work():
    path = Path(__file__).resolve().parents[1] / "configs/example.json"
    config = load_config(str(path)).resolved()
    assert config.num_gpus == 1
    assert config.backpressure.max_in_flight_gpu == 2
    assert config.scenedetect.max_clip_duration_s == 15


def test_longer_audio_does_not_extend_video_clip(tmp_path):
    from egoworld.pipeline.stages import PrepareActor
    video = tmp_path / "audio-tail.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc=duration=1:size=64x64:rate=10", "-f", "lavfi", "-i", "sine=duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    videos, clips = bm.build_manifests([str(video)], scenedetect=SceneDetectConfig(method="none", overlap_s=0))
    assert videos[0]["duration_s"] == 1. and clips[0]["frame_end"] == 10
    prepared = PrepareActor({}).prepare({"clip": dict(clips[0], video_path=str(video), fps=10.), "work_dir": str(tmp_path / "work")})
    assert prepared["frames"] == 10


def test_sam2_external_config_loads_in_hydra_without_model_weights(tmp_path, monkeypatch):
    hydra = pytest.importorskip("hydra")
    from types import ModuleType
    from egoworld.operators.sam2_op import Sam2Operator

    config_dir = tmp_path / "config with spaces"
    config_dir.mkdir()
    config = config_dir / "custom.yaml"
    config.write_text("defaults: [base, _self_]\nmodel:\n  value: 7\n")
    (config_dir / "base.yaml").write_text("model:\n  scale: 2\n")
    main = tmp_path / "main"
    main.mkdir()
    builder = ModuleType("sam2.build_sam")
    seen = []

    def build(name, checkpoint, **kwargs):
        with hydra.initialize_config_dir(config_dir=str(main), version_base="1.2"):
            cfg = hydra.compose(config_name=name, overrides=kwargs["hydra_overrides_extra"])
        seen.append((cfg.model.value, cfg.model.scale, kwargs["device"]))
        return object()

    builder.build_sam2_video_predictor = build
    monkeypatch.setitem(sys.modules, "sam2", ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", builder)
    operator = Sam2Operator(config=str(config), checkpoint="not-loaded.pt", device="cuda")
    operator._ensure_predictor()
    assert seen == [(7, 2, "cuda")]
