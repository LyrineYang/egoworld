import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import types
import sys

import pytest

from egoworld.operators.sam2_op import Sam2Operator, _resolve_model_cfg


SAM2_SMOKE_ENV = "EGOWORLD_SAM2_SMOKE"
SAM2_CHECKPOINT_ENV = "EGOWORLD_SAM2_CHECKPOINT"
SAM2_CONFIG_ENV = "EGOWORLD_SAM2_CONFIG"
GD_CONFIG_ENV = "EGOWORLD_GD_CONFIG"
GD_CHECKPOINT_ENV = "EGOWORLD_GD_CHECKPOINT"


def _resolve_path(env_key: str, default: str) -> str | None:
    path = os.getenv(env_key, default)
    if path and os.path.isfile(path):
        return path
    return None


def _ensure_video(path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=30",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_resolve_model_cfg_prefers_existing_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "sam2_test.yaml"
    cfg_path.write_text("test", encoding="utf-8")
    assert _resolve_model_cfg(str(cfg_path)) == str(cfg_path)


def test_resolve_model_cfg_from_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_name = "sam2.1_hiera_s.yaml"
    config_dir = tmp_path / "configs" / "sam2.1"
    config_dir.mkdir(parents=True)
    cfg_path = config_dir / config_name
    cfg_path.write_text("test", encoding="utf-8")

    fake_module = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    monkeypatch.setitem(sys.modules, "sam2", fake_module)

    resolved = _resolve_model_cfg(config_name)
    assert resolved == str(cfg_path)


def _sam2_prereqs_ok() -> tuple[bool, str, dict]:
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found", {}
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"torch import failed: {exc}", {}
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false", {}
    try:
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"sam2 import failed: {exc}", {}
    try:
        import groundingdino  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"groundingdino import failed: {exc}", {}

    checkpoint = _resolve_path(SAM2_CHECKPOINT_ENV, "./models/sam2/sam2.1_hiera_small.pt")
    config_raw = os.getenv(SAM2_CONFIG_ENV, "./models/sam2/sam2.1_hiera_s.yaml")
    config = _resolve_model_cfg(config_raw)

    if checkpoint is None:
        return False, f"SAM2 checkpoint missing; set {SAM2_CHECKPOINT_ENV}", {}
    if config is None or not os.path.isfile(config):
        return False, f"SAM2 config missing; set {SAM2_CONFIG_ENV}", {}

    gd_config = _resolve_path(GD_CONFIG_ENV, "./models/groundingdino/GroundingDINO_SwinT_OGC.py")
    gd_checkpoint = _resolve_path(GD_CHECKPOINT_ENV, "./models/groundingdino/groundingdino_swint_ogc.pth")
    if gd_config is None:
        return False, f"GroundingDINO config missing; set {GD_CONFIG_ENV}", {}
    if gd_checkpoint is None:
        return False, f"GroundingDINO checkpoint missing; set {GD_CHECKPOINT_ENV}", {}

    params = {
        "checkpoint": checkpoint,
        "config": config,
        "device": "cuda",
        "precision": "bf16",
        "vos_optimized": True,
        "prompting": {
            "source": "groundingdino",
            "gd_config": gd_config,
            "gd_checkpoint": gd_checkpoint,
            "gd_device": "cuda",
        },
    }
    return True, "", params


def _require_prereqs() -> dict:
    env_flag = os.getenv(SAM2_SMOKE_ENV)
    ok, reason, params = _sam2_prereqs_ok()
    if env_flag == "1":
        if not ok:
            pytest.fail(f"sam2 prerequisites missing: {reason}")
        return params
    if env_flag == "0":
        pytest.skip(f"{SAM2_SMOKE_ENV}=0 set; skipping sam2 smoke tests")
    if not ok:
        pytest.skip(f"sam2 smoke prerequisites not met: {reason}")
    return params


@pytest.mark.sam2_smoke
def test_sam2_operator_smoke() -> None:
    params = _require_prereqs()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "input.mp4"
        _ensure_video(video_path)

        op = Sam2Operator(**params)
        result = op.run(str(video_path), 0.0, 1.0, params=params)

        assert "frames" in result
        assert result.get("mask_encoding") == "rle"
        assert result.get("video_path") == str(video_path)
        assert 0.0 <= float(result.get("empty_mask_rate", 0.0)) <= 1.0
        assert result.get("start_s") == 0.0
        assert result.get("end_s") == 1.0
