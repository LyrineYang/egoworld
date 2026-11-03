"""A failed or source-changing group must not turn into a speed claim."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import convergence_study as study


@pytest.mark.parametrize('changed', [False, True])
def test_failed_group_stops_and_preserves_raw_attempt(tmp_path, monkeypatch, changed):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'jobs': [{'name': 'native', 'kind': 'datajuicer_native'}]}))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'groups': [{'name': name, 'track': 'native_product', 'spec': str(spec)}
        for name in ['first', 'must-not-run']]}))
    calls = []
    def benchmark(path, output):
        calls.append(output.name)
        output.mkdir()
        report = {'status': 'completed_with_failures', 'sources_unchanged': not changed,
            'inputs_unchanged': True, 'configs_unchanged': True, 'dependencies_unchanged': True,
            'trials': [{'name': 'native', 'status': 'failed', 'error': 'fixture failure', 'wall_seconds': 2.}]}
        (output / 'report.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(study, 'benchmark', benchmark)
    output = tmp_path / 'study'
    with pytest.raises((RuntimeError, ValueError)):
        study.run(plan, output)
    state = json.loads((output / 'study-state.json').read_text())
    assert calls == ['first'] and state['status'] == 'failed'
    assert state['same_model_speed_ratio'] is None
    assert (output / 'first/report.json').exists()
    with pytest.raises(FileExistsError):
        study.run(plan, output)


def test_common_native_product_preserves_duplicate_slots_and_all_frames(tmp_path):
    export = tmp_path / 'native.jsonl'
    export.write_text(json.dumps({'sample_id': 's'}) + '\n')
    sidecar = tmp_path / 'native_stats.jsonl'
    tags = {'joints': [[[[0., 0., 0.]] * 21]] * 3,
        'if_right_hand': [[1]] * 3, 'camera_translation': [[[0., 0., 1.]]] * 3}
    sidecar.write_text(json.dumps({'__dj__meta__': {'input_id': 's', 'hand_reconstruction_tags': tags}}) + '\n')
    rows = study.native_common_predictions(export, [0, 2, 2], {'s': {'image_size': [224, 224]}}, 5000 / 256)
    assert [r['frame_index'] for r in rows] == [0, 2, 2]
    assert len({r['prediction_id'] for r in rows}) == 3


def test_config_changed_before_another_group_is_rejected(tmp_path):
    dependency = tmp_path / 'policy.json'
    dependency.write_text('{}')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'jobs': [], 'dependencies': [str(dependency)]}))
    plan_path = tmp_path / 'plan.json'
    plan = {'groups': [{'spec': str(spec)}]}
    plan_path.write_text(json.dumps(plan))
    identities = study.frozen_files(plan_path, plan)
    dependency.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed across groups'):
        study.verify_frozen(identities)
