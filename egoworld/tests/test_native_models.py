"""R03–R07 native runtime entry point; never succeeds through a model double."""
import os
import pytest


@pytest.mark.native_models
@pytest.mark.parametrize("stage", ["sam2", "hamer", "foundationpose", "dex_retarget", "fast3r"])
def test_native_stage(stage, tmp_path):
    if os.getenv("EGOWORLD_NATIVE_TEST") != "1": pytest.skip("Set EGOWORLD_NATIVE_TEST=1 with model assets to run native inference")
    from egoworld.acceptance import single_gpu_smoke
    config = os.environ.get("EGOWORLD_NATIVE_CONFIG")
    video = os.environ.get("EGOWORLD_NATIVE_VIDEO")
    if not config or not video: pytest.fail("EGOWORLD_NATIVE_CONFIG and EGOWORLD_NATIVE_VIDEO are required")
    report = single_gpu_smoke(config, video, str(tmp_path), 3., "left hand . right hand . cup .", [stage],
                              profile=os.getenv("EGOWORLD_NATIVE_PROFILE", "default"))
    assert report["status"] == "passed" and report["stage_rows"][stage] > 0
