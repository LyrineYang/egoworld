"""Do not mistake exported native rows for valid hand annotations."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from product_comparison import audit_datajuicer, audit_egoworld
from test_product_geometry import world_case


def test_native_empty_export_is_zero_output(tmp_path):
    file = tmp_path / 'output.jsonl'
    file.write_text('')
    assert audit_datajuicer(file)['finite_21x3_hand_predictions'] == 0
    assert audit_datajuicer(file)['exported_samples'] == 0


def test_native_finite_shape_and_missing_identity_remain_explicit(tmp_path):
    file = tmp_path / 'output.jsonl'
    row = {'sample_id': 'v', '__dj__meta__': {'hand_reconstruction_tags':
            {'joints': [[[[0., 0., 0.]] * 21, [[0., 0., 0.]] * 20]]}}}
    file.write_text(json.dumps(row) + '\n')
    result = audit_datajuicer(file)
    assert result['finite_21x3_hand_predictions'] == 1
    assert result['invalid_hand_predictions'] == 1
    assert result['sample_ids'] == ['v']


def test_nested_world_exports_are_verified_and_corruption_rejected(world_case, tmp_path):
    from egoworld.product.world import fuse_world
    source, calibration, _ = world_case
    run = tmp_path / 'run'
    destination = run / 'exports/c/world'
    result = fuse_world(source, calibration, ['world_camera', 'world_scene'], destination)
    manifest = run / 'data_manifest.json'
    manifest.write_text(json.dumps({'status': 'passed', 'run_dir': str(run),
        'exports': {'c': {'world': result}}, 'cleaning': {'accepted': 1}, 'computed': []}))
    audit = audit_egoworld(manifest)
    assert audit['output_counts'] == {'world_camera': 1, 'world_scene': 1}
    (destination / 'world_scene.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='World export changed'):
        audit_egoworld(manifest)


def test_native_default_sidecar_layout_and_identity_are_audited(tmp_path):
    file = tmp_path / 'output.jsonl'
    file.write_text(json.dumps({'sample_id': 'v', 'text': 'input'}) + '\n')
    sidecar = tmp_path / 'output_stats.jsonl'
    meta = {'input_id': 'v', 'hand_reconstruction_tags': {'joints': [[[[0., 0., 0.]] * 21]]}}
    sidecar.write_text(json.dumps({'__dj__meta__': meta}) + '\n')
    assert audit_datajuicer(file)['finite_21x3_hand_predictions'] == 1
    meta['input_id'] = 'different'
    sidecar.write_text(json.dumps({'__dj__meta__': meta}) + '\n')
    with pytest.raises(ValueError, match='identity'):
        audit_datajuicer(file)
    sidecar.write_text('')
    with pytest.raises(ValueError, match='coverage'):
        audit_datajuicer(file)


def test_nvml_utilization_failure_keeps_memory_observation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from product_comparison import run_trial
    def unsupported(*args):
        raise RuntimeError('utilization unsupported')
    fake = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(used=123),
        nvmlDeviceGetUtilizationRates=unsupported)
    monkeypatch.setitem(sys.modules, 'pynvml', fake)
    cfg = tmp_path / 'screen.json'
    cfg.write_text(json.dumps({'inputs': [str(tmp_path / 'missing.mp4')]}))
    result = run_trial({'name': 'probe', 'kind': 'egoworld_cleaning',
        'python': sys.executable, 'config': str(cfg)}, tmp_path / 'trial', 30)
    assert result['status'] == 'passed'
    assert result['gpu_whole_device_peak_used_bytes'] == [123]
