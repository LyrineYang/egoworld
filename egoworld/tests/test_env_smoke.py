import os
import shutil
import subprocess
from pathlib import Path

import pytest


ENV_FLAG = "EGOWORLD_ENV_SMOKE"


def _should_run() -> None:
    env_flag = os.getenv(ENV_FLAG)
    if env_flag == "1":
        return
    if env_flag == "0":
        pytest.skip(f"{ENV_FLAG}=0 set; skipping environment smoke tests")
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not found; set EGOWORLD_ENV_SMOKE=1 to force")


def _run_ok(command: list[str]) -> bool:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0


@pytest.mark.env_smoke
def test_gpu_driver_and_torch_cuda_available() -> None:
    _should_run()

    assert _run_ok(["nvidia-smi", "-L"]), "nvidia-smi not available or no GPU visible"

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment check
        pytest.fail(f"torch import failed: {exc}")

    assert torch.cuda.is_available() is True


@pytest.mark.env_smoke
def test_ffmpeg_tools_available() -> None:
    _should_run()

    assert _run_ok(["ffmpeg", "-version"]), "ffmpeg not available on PATH"
    assert _run_ok(["ffprobe", "-version"]), "ffprobe not available on PATH"


@pytest.mark.env_smoke
def test_model_cache_paths_writable() -> None:
    _should_run()

    repo_root = Path.cwd()
    model_home = Path(os.getenv("EGOWORLD_MODEL_HOME", repo_root / "models"))
    cache_home = Path(os.getenv("EGOWORLD_CACHE", repo_root / "cache"))

    assert model_home.exists(), f"model path missing: {model_home}"
    assert cache_home.exists(), f"cache path missing: {cache_home}"

    assert os.access(model_home, os.W_OK), f"model path not writable: {model_home}"
    assert os.access(cache_home, os.W_OK), f"cache path not writable: {cache_home}"
