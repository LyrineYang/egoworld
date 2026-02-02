import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from egoworld.operators.sam2_op import _resolve_model_cfg
from egoworld.pipeline.driver import run_pipeline
from egoworld.pipeline.state_store import get_clip_state


ENV_FLAG = "EGOWORLD_PIPELINE_SMOKE"
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


def _sam2_prereqs_ok() -> tuple[bool, str, dict]:
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

    try:
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"sam2 import failed: {exc}", {}
    try:
        import groundingdino  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"groundingdino import failed: {exc}", {}

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


def _prereqs_ok() -> tuple[bool, str, dict]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found", {}
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found", {}
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"torch import failed: {exc}", {}
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false", {}
    try:
        import ray  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"ray import failed: {exc}", {}
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        return False, f"pyarrow import failed: {exc}", {}

    sam2_ok, sam2_reason, sam2_params = _sam2_prereqs_ok()
    if not sam2_ok:
        return False, sam2_reason, {}
    return True, "", sam2_params


def _require_prereqs() -> dict:
    env_flag = os.getenv(ENV_FLAG)
    ok, reason, params = _prereqs_ok()
    if env_flag == "1":
        if not ok:
            pytest.fail(f"pipeline prerequisites missing: {reason}")
        return params
    if env_flag == "0":
        pytest.skip(f"{ENV_FLAG}=0 set; skipping pipeline smoke tests")
    if not ok:
        pytest.skip(f"pipeline smoke prerequisites not met: {reason}")
    return params


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


@pytest.mark.pipeline_smoke
def test_pipeline_smoke_minimal() -> None:
    sam2_params = _require_prereqs()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_root = tmp_path / "output"
        state_db = tmp_path / "state" / "pipeline.db"
        run_id = "test_run"
        video_path = tmp_path / "input.mp4"
        _ensure_video(video_path)

        config = {
            "num_gpus": 1,
            "run_id": run_id,
            "paths": {
                "output_root": str(output_root),
                "state_db_path": str(state_db),
            },
            "operators": {
                "sam2": {"enabled": True, "params": sam2_params},
                "hamer": {"enabled": False, "params": {}},
                "foundationpose": {"enabled": False, "params": {}},
                "dex_retarget": {"enabled": False, "params": {}},
                "fast3r": {"enabled": False, "params": {}},
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        video_id = "video-abc"
        clip_id = "video-abc-000000000-000000030-deadbeef"
        video_manifest = [
            {
                "video_id": video_id,
                "path": str(video_path),
                "duration_s": 1.0,
                "fps": 30.0,
                "width": 320,
                "height": 240,
                "audio": False,
                "checksum": "deadbeef",
                "split": "train",
            }
        ]
        clip_manifest = [
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "start_s": 0.0,
                "end_s": 1.0,
                "frame_start": 0,
                "frame_end": 30,
                "overlap_s": 0.0,
                "scenedetect_failed": False,
                "status": "Pending",
                "last_error": "",
                "retry_count": 0,
            }
        ]

        video_manifest_path = tmp_path / "video_manifest.jsonl"
        clip_manifest_path = tmp_path / "clip_manifest.jsonl"
        _write_jsonl(video_manifest_path, video_manifest)
        _write_jsonl(clip_manifest_path, clip_manifest)

        run_pipeline(str(config_path), str(video_manifest_path), str(clip_manifest_path))

        run_root = output_root / f"run_id={run_id}"
        clip_dir = run_root / f"video_id={video_id}" / f"clip_id={clip_id}"

        assert (run_root / "run_manifest.json").exists()
        assert (clip_dir / "meta.json").exists()
        assert (clip_dir / "masks.parquet").exists()
        assert (clip_dir / "hand_pose.parquet").exists()
        assert (clip_dir / "object_pose.parquet").exists()
        assert (clip_dir / "mapping.parquet").exists()

        state = get_clip_state(str(state_db), clip_id)
        assert state is not None
        assert state.status == "Done"


@pytest.mark.pipeline_smoke
def test_pipeline_resume_skips_done() -> None:
    sam2_params = _require_prereqs()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_root = tmp_path / "output"
        state_db = tmp_path / "state" / "pipeline.db"
        run_id = "resume_run"
        video_path = tmp_path / "input.mp4"
        _ensure_video(video_path)

        config = {
            "num_gpus": 1,
            "run_id": run_id,
            "paths": {
                "output_root": str(output_root),
                "state_db_path": str(state_db),
            },
            "operators": {
                "sam2": {"enabled": True, "params": sam2_params},
                "hamer": {"enabled": False, "params": {}},
                "foundationpose": {"enabled": False, "params": {}},
                "dex_retarget": {"enabled": False, "params": {}},
                "fast3r": {"enabled": False, "params": {}},
            },
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        video_id = "video-abc"
        clip_id = "video-abc-000000000-000000030-deadbeef"
        video_manifest = [
            {
                "video_id": video_id,
                "path": str(video_path),
                "duration_s": 1.0,
                "fps": 30.0,
                "width": 320,
                "height": 240,
                "audio": False,
                "checksum": "deadbeef",
                "split": "train",
            }
        ]
        clip_manifest = [
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "start_s": 0.0,
                "end_s": 1.0,
                "frame_start": 0,
                "frame_end": 30,
                "overlap_s": 0.0,
                "scenedetect_failed": False,
                "status": "Pending",
                "last_error": "",
                "retry_count": 0,
            }
        ]

        video_manifest_path = tmp_path / "video_manifest.jsonl"
        clip_manifest_path = tmp_path / "clip_manifest.jsonl"
        _write_jsonl(video_manifest_path, video_manifest)
        _write_jsonl(clip_manifest_path, clip_manifest)

        run_pipeline(str(config_path), str(video_manifest_path), str(clip_manifest_path))
        first_state = get_clip_state(str(state_db), clip_id)
        assert first_state is not None
        assert first_state.status == "Done"
        first_updated = first_state.updated_at

        run_pipeline(str(config_path), str(video_manifest_path), str(clip_manifest_path))
        second_state = get_clip_state(str(state_db), clip_id)
        assert second_state is not None
        assert second_state.status == "Done"
        assert second_state.updated_at == first_updated
