"""Each pipeline invocation keeps its own evidence, including resume and interruption."""
import json
from pathlib import Path
import pytest
from egoworld.pipeline import stages
from egoworld.profiling import profile_run, list_invocations
from test_pipeline_logic import pipeline_case


def test_resume_preserves_original_performance_and_event_identity(pipeline_case):
    case = pipeline_case
    case['run']()
    root = case['root'] / 'output/run_id=A'
    first = json.loads((root / 'run_summary.json').read_text())
    history = root / 'invocations' / first['invocation_id']
    original = (history / 'run_summary.json').read_bytes()
    case['run']()
    assert (history / 'run_summary.json').read_bytes() == original
    assert len(list_invocations(root)['invocations']) == 2
    latest = json.loads((root / 'run_summary.json').read_text())
    assert latest['skipped_clips'] == 3 and latest['stage_metrics'] == {}
    report = profile_run(root, first['invocation_id'])
    assert report['stages']['sam2']['completed_calls'] == 3
    assert all(json.loads(line)['invocation_id'] == first['invocation_id']
               for line in (history / 'events.jsonl').read_text().splitlines())
    manifest = json.loads((history / 'run_manifest.json').read_text())
    assert all(Path(path).is_file() and Path(path).parent == history for path in manifest['input_manifest_paths'].values())


def test_interruption_keeps_committed_metrics(pipeline_case, monkeypatch):
    from egoworld.pipeline.engine import Engine
    original = Engine.finish
    def interrupt(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise KeyboardInterrupt('fixture stop')
    monkeypatch.setattr(Engine, 'finish', interrupt)
    with pytest.raises(KeyboardInterrupt): pipeline_case['run']()
    root = pipeline_case['root'] / 'output/run_id=A'
    summary = json.loads((root / 'run_summary.json').read_text())
    assert summary['status'] == 'interrupted'
    assert summary['stage_metrics']['sam2']['calls'] >= 1
    assert profile_run(root, summary['invocation_id'])['status'] == 'interrupted'


def test_legacy_latest_summary_is_saved_before_overwrite(pipeline_case):
    from egoworld.pipeline.records import preserve_legacy_summary
    case = pipeline_case; case['run']()
    root = case['root'] / 'output/run_id=A'
    data = json.loads((root / 'run_summary.json').read_text()); data.pop('invocation_id')
    (root / 'run_summary.json').write_text(json.dumps(data))
    preserve_legacy_summary(root)
    records = list_invocations(root)['invocations']
    legacy = next(row for row in records if row['legacy'])
    assert profile_run(root, legacy['invocation_id'])['stages']['sam2']['completed_calls'] == 3


def test_profile_cannot_escape_invocation_directory(tmp_path):
    with pytest.raises(ValueError): profile_run(tmp_path, '../outside')
