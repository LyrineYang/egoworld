"""R05/R07/R09: reject misleading structured data and check reproducible contracts."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from egoworld.config import PipelineConfig, load_config
from egoworld.contracts import schemas, validate_rows
from egoworld.pipeline.artifacts import WriterActor
from test_pipeline_logic import full_packet

CLIP = {"frame_start": 0, "frame_end": 10, "fps": 10., "video_id": "v", "clip_id": "c"}


@pytest.mark.parametrize("stage,artifact,field,value", [
    ("hamer", "hand_pose", "joints_3d", [0.] * 62),
    ("hamer", "hand_pose", "length_unit", "millimeters"),
    ("hamer", "hand_pose", "timestamp_s", .07),
    ("foundationpose", "object_pose", "transform", [0.] * 16),
    ("dex_retarget", "mapping", "joint_units", ["unknown"]),
    ("fast3r", "reconstruction", "scale_status", "calibrated"),
    ("fast3r", "reconstruction", "intrinsics", [float("nan")] * 9),
])
def test_invalid_model_output_rejected(stage, artifact, field, value):
    data = full_packet(stage, CLIP)["tables"][artifact]
    data[0][field] = value
    with pytest.raises(ValueError): validate_rows(artifact, data, CLIP)


@pytest.mark.parametrize("bad", ["missing", "nonfinite", "shape", "object"])
def test_pointcloud_must_be_present_finite_and_well_formed(tmp_path, bad):
    packet = full_packet("fast3r", CLIP)
    arrays = packet["arrays"]["reconstruction-pointcloud.npz"]
    if bad == "missing": packet.pop("arrays")
    elif bad == "nonfinite": arrays["points"][0, 0] = np.inf
    elif bad == "shape": arrays["confidence"] = np.ones(2)
    else: arrays["points"] = np.array([[object(), object(), object()]], dtype=object)
    writer = WriterActor(asdict(PipelineConfig().resolved()))
    with pytest.raises(ValueError):
        writer.write_stage({"directory": str(tmp_path), "stage": "fast3r", "clip": CLIP, "fingerprint": "test"}, packet)
    assert not (tmp_path / "stage_fast3r.json").exists()


def test_schema_cli_reports_actual_contracts():
    result = subprocess.run([sys.executable, "-m", "egoworld.cli", "schema"], check=True, capture_output=True, text=True)
    declared = json.loads(result.stdout)["artifacts"]
    assert declared == {name: {f.name: str(f.type) for f in schema} for name, schema in schemas().items()}


@pytest.mark.parametrize("data", [
    {"typo": 1}, {"operators": {"hmar": {}}},
    {"operators": {"hamer": {"enabld": True}}},
    {"operators": {"hamer": {"enabled": "false"}}},
    {"operators": {"hamer": {"params": []}}},
    {"coordinates": {"frame_index_base": 1}},
    {"execution": {"stage_timeout_s": float("inf")}},
])
def test_unsupported_or_mistyped_settings_fail(tmp_path, data):
    path = tmp_path / "config.json"; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_config(str(path)).resolved()


def test_provenance_changes_with_local_assets_and_manifests(tmp_path):
    from egoworld.utils.provenance import provenance
    config = PipelineConfig().resolved()
    model, videos, clips = (tmp_path / name for name in ("model.pt", "videos.jsonl", "clips.jsonl"))
    for file in (model, videos, clips): file.write_text("first")
    config.operators.sam2.params["checkpoint"] = str(model)
    first = provenance(config, str(videos), str(clips))["fingerprint"]
    config.execution.cpu_workers += 1
    assert provenance(config, str(videos), str(clips))["fingerprint"] == first
    model.write_text("different checkpoint")
    second = provenance(config, str(videos), str(clips))["fingerprint"]
    assert second != first
    clips.write_text("different source manifest")
    assert provenance(config, str(videos), str(clips))["fingerprint"] != second
    before = provenance(config, str(videos), str(clips))["fingerprint"]
    config.dataset_hash = "depth-and-robot-assets-v2"
    assert provenance(config, str(videos), str(clips))["fingerprint"] != before


@pytest.mark.parametrize("bf16_supported", [False, True])
def test_doctor_only_imports_selected_models(monkeypatch, tmp_path, bf16_supported):
    from egoworld import preflight
    config = json.loads((Path(__file__).parents[1] / "configs/full.json").read_text())
    config["operators"]["fast3r"]["params"]["model_name_or_path"] = str(tmp_path / "snapshot")
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps(config))
    imports = []
    torch = SimpleNamespace(__version__="2.5.1", cuda=SimpleNamespace(device_count=lambda: 1,
        get_device_name=lambda _: "fixture",
        is_bf16_supported=lambda including_emulation=True: True if including_emulation else bf16_supported))
    def import_module(name):
        imports.append(name)
        return torch if name == "torch" else SimpleNamespace()
    monkeypatch.setattr(preflight.importlib, "import_module", import_module)
    report = preflight.runtime_report(str(cfg), gpu=True, stages=["fast3r"])
    assert report["ok"] == bf16_supported
    assert not any(name.startswith(("sam2", "hamer", "estimater")) for name in imports)
    assert any(c["name"] == "Fast3R BF16 attention" and c["ok"] == bf16_supported for c in report["checks"])


def test_foundationpose_rejects_bad_calibration_before_gpu_load(monkeypatch, tmp_path):
    from egoworld.operators.foundationpose_op import FoundationPoseOperator
    import types
    monkeypatch.setitem(sys.modules, "trimesh", types.ModuleType("trimesh"))
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"v": {"intrinsics": [0] * 9, "depth_scale_to_meters": .001}}))
    operator = FoundationPoseOperator(assets_manifest=str(assets), objects={})
    monkeypatch.setattr(operator, "_load", lambda: pytest.fail("Invalid calibration must fail before model loading"))
    with pytest.raises(ValueError, match="intrinsics"): operator.run(CLIP, "unused", {})


def test_external_runner_preserves_retry_classification(tmp_path):
    from egoworld.pipeline.stages import ModelActor
    from egoworld.utils.errors import RetryableError
    runner = tmp_path / "runner.py"
    runner.write_text("import json,sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--response')+1]).write_text(json.dumps({'error':{'type':'OutOfMemoryError','message':'CUDA out of memory','retryable':True}}))\nsys.exit(1)\n")
    config = asdict(PipelineConfig().resolved())
    job = {"stage": "sam2", "clip": CLIP, "work_dir": str(tmp_path)}
    with pytest.raises(RetryableError, match="CUDA out of memory"):
        ModelActor(config)._external([sys.executable, str(runner)], job, {}, {})
