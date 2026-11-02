"""Portable server profiles must fit resource pools and preserve model semantics."""
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest
from egoworld.config import PipelineConfig

spec = importlib.util.spec_from_file_location("server_preparation", Path(__file__).parents[1] / "scripts/prepare_server_configs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("gpus", [2, 4, 8, 16])
def test_profiles_fit_cpu_and_keep_model_behavior(gpus):
    config = asdict(PipelineConfig())
    config["operators"]["sam2"]["enabled"] = True
    config["execution"].update(split_execution=True, phase_workers=2)
    result, resources = module.resource_profile(config, gpus, 48, 128)
    assert resources["actor_cpu_reservation"] <= 48
    assert resources["gpu_actors"] == gpus
    assert result["operators"] == config["operators"]
    assert result["execution"]["gpu_stage_groups"] == []
    assert result["execution"]["external_runner_mode"] == "auto"
    assert result["backpressure"]["max_in_flight_bytes"] <= 128 * 1024**3 // 8


def test_profiles_fail_underprovisioned_cpu():
    config = asdict(PipelineConfig())
    config["operators"]["sam2"]["enabled"] = True
    with pytest.raises(ValueError, match="cannot admit"):
        module.resource_profile(config, 16, 8, 128)


def test_path_mapping_longest_prefix_and_refuses_unmapped():
    maps = [(Path('/local'), Path('/srv')), (Path('/local/models'), Path('/weights'))]
    assert module.relocate({'path': '/local/models/m.ckpt'}, maps)['path'] == '/weights/m.ckpt'
    with pytest.raises(ValueError, match="Unmapped"):
        module.relocate('/mnt/c/Users/private/model.ckpt', maps)
