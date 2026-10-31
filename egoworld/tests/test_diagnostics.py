"""Real guarded runner, video and stage files; model work is an explicit CPU fixture."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from egoworld.config import PipelineConfig
from egoworld.contracts import OUTPUT_SCHEMA_VERSION
from egoworld.diagnostics.replay import run_diagnostic, recommendations, _device_environment
from egoworld.io.writers import write_json, write_json_lines
from egoworld.pipeline.artifacts import WriterActor
from egoworld.utils.hashing import sha256_file
from egoworld.utils.provenance import provenance
from test_pipeline_logic import full_packet


@pytest.fixture
def diagnostic_case(tmp_path):
    video = tmp_path / 'sample.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=duration=1:size=32x32:rate=10',
                    '-pix_fmt', 'yuv420p', str(video)], check=True)
    config = PipelineConfig().resolved()
    config.execution.external_runner_mode = 'persistent'
    config.operators.hamer.enabled = True
    config.operators.hamer.params = {'checkpoint': str(tmp_path / 'model.ckpt')}
    Path(config.operators.hamer.params['checkpoint']).write_bytes(b'fixture weights')
    calls, instances = tmp_path / 'calls', tmp_path / 'instances'
    runner = tmp_path / 'runner.py'
    runner.write_text('''import json,os,time
from pathlib import Path
from egoworld.pipeline import stages
from egoworld.operators.common import rows
from egoworld import worker
class Model:
    def __init__(self, **params):
        self.params = params
        with Path(INSTANCES).open('a') as stream: stream.write(str(os.getpid())+'\\n')
    def run(self, *args):
        with Path(CALLS).open('a') as stream: stream.write(str(os.getpid())+'\\n')
        if self.params.get('cache_fixture'):
            from egoworld.operators.common import frame_reader, read_frame
            cap = frame_reader(args[1])
            try: read_frame(cap, 0)
            finally: cap.release()
        if self.params.get('fail'): raise RuntimeError('CUDA out of memory fixture')
        if self.params.get('hang'): time.sleep(30)
        if isinstance(args[0], dict):
            assert rows(args[2], 'sam2', 'instances')[0]['label'] == 'right hand'
            return {'tables': {'hand_pose': []}}
        return {'frames': [], 'instances': [], 'processed_frames': 10}
stages.OPERATORS['sam2'] = Model
stages.OPERATORS['hamer'] = Model
worker.main()
'''.replace('CALLS', repr(str(calls))).replace('INSTANCES', repr(str(instances))))
    for stage in ('sam2', 'hamer'):
        getattr(config.operators, stage).params['runner_command'] = [sys.executable, str(runner)]
    cfg = tmp_path / 'config.json'; write_json(cfg, asdict(config))
    clip = {'clip_id': 'c', 'video_id': 'v', 'video_path': str(video), 'start_s': 0., 'end_s': 1.,
            'frame_start': 0, 'frame_end': 10, 'fps': 10., 'scenedetect_failed': False, 'retry_count': 0}
    videos, clips = tmp_path / 'videos.jsonl', tmp_path / 'clips.jsonl'
    write_json_lines(videos, [{'video_id': 'v', 'path': str(video), 'fps': 10., 'width': 32, 'height': 32,
                              'checksum': sha256_file(str(video))}])
    write_json_lines(clips, [clip])
    identity = provenance(config, str(videos), str(clips))
    source = tmp_path / 'source'; source.mkdir()
    write_json(source / 'run_manifest.json', {'run_id': 'original', 'output_schema_version': OUTPUT_SCHEMA_VERSION,
        'config_fingerprint': identity['fingerprint'], 'provenance': identity, 'config': asdict(config),
        'input_manifest_paths': {'videos': str(videos), 'clips': str(clips)}})
    folder = source / 'video_id=v/clip_id=c'
    writer = WriterActor(asdict(config))
    for stage in ('sam2', 'hamer'):
        writer.write_stage({'directory': str(folder), 'clip': clip, 'stage': stage,
                            'fingerprint': identity['fingerprint']}, full_packet(stage, clip))
    def run(stage='hamer', **kwargs):
        return run_diagnostic(str(cfg), str(source), 'c', stage, str(tmp_path / 'diagnostics'), **kwargs)
    return {'run': run, 'cfg': cfg, 'config': config, 'source': source, 'folder': folder,
            'video': video, 'videos': videos, 'clips': clips, 'calls': calls, 'instances': instances}


def snapshot(root):
    return {str(path): (sha256_file(str(path)), path.stat().st_mtime_ns) for path in root.rglob('*') if path.is_file()}


def test_cold_warm_probe_reuses_runner_and_keeps_every_trial(diagnostic_case):
    case = diagnostic_case
    before = snapshot(case['source']); original_config = case['cfg'].read_bytes()
    report = case['run'](mode='probe', warmup=1, repeats=3)
    assert report['status'] == 'passed', report.get('error')
    assert [trial['phase'] for trial in report['trials']] == ['cold', 'warmup', 'hot', 'hot', 'hot']
    assert [trial['stats']['model_cache_hit'] for trial in report['trials']] == [0, 1, 1, 1, 1]
    assert len(case['instances'].read_text().splitlines()) == 1
    assert len(case['calls'].read_text().splitlines()) == 5
    assert snapshot(case['source']) == before and case['cfg'].read_bytes() == original_config
    assert report['recommendations']['warm_model_reuse_verified']
    assert report['recommendations']['device_headroom_status'] == 'unknown'
    assert not report['automatic_resource_plan'] and not report['configuration_modified']
    assert report['lineage']['upstream']['sam2']['files']['instances.parquet']['sha256']
    for trial in report['trials']:
        assert Path(trial['directory'], 'record.json').is_file()
        assert Path(trial['directory'], 'outputs/stage_hamer.json').is_file()
        assert trial['runtime']['source_tree_hash']
        assert trial['diagnostics']['process_tree_peak_rss_bytes'] > 0
    with pytest.raises(ProcessLookupError):
        os.kill(int(case['instances'].read_text().strip()), 0)


def test_repeated_replay_runs_instead_of_accepting_target_commit(diagnostic_case):
    case = diagnostic_case
    one = case['run'](); first = Path(one['report_path']).read_bytes()
    two = case['run']()
    assert one['status'] == two['status'] == 'passed'
    assert one['invocation_id'] != two['invocation_id']
    assert len(case['calls'].read_text().splitlines()) == 2
    assert Path(one['report_path']).read_bytes() == first
    assert one['trials'][0]['stats']['model_cache_hit'] == two['trials'][0]['stats']['model_cache_hit'] == 0
    assert one['trials'][0]['runtime']['process_id'] != two['trials'][0]['runtime']['process_id']
    assert one['observed_identity_hash'] == two['observed_identity_hash']


@pytest.mark.parametrize('fail', [False, True])
def test_runner_cache_metrics_and_cleanup_survive_success_or_error(diagnostic_case, fail):
    case = diagnostic_case
    case['config'].extra['input_cache'] = {'enabled': True}
    case['config'].operators.hamer.params.update(cache_fixture=True, fail=fail)
    write_json(case['cfg'], asdict(case['config']))
    report = case['run'](mode='probe', warmup=0, repeats=1)
    assert report['status'] == ('failed' if fail else 'passed')
    if not fail:
        assert report['trials'][0]['stats']['input_cache_frame_misses'] == 1
        assert report['trials'][1]['stats']['input_cache_frame_hits'] == 1
    assert report['released_input_cache_bytes'] > 0
    assert not list(Path(report['report_path']).parent.rglob('*.dat'))


@pytest.mark.parametrize('damage', ['upstream', 'video', 'manifest', 'missing_commit', 'wrong_fingerprint'])
def test_changed_or_uncommitted_inputs_rejected_before_runner(diagnostic_case, damage):
    case = diagnostic_case
    if damage == 'upstream': (case['folder'] / 'instances.parquet').write_bytes(b'changed')
    elif damage == 'video': case['video'].write_bytes(b'changed')
    elif damage == 'manifest': case['clips'].write_text('{}')
    elif damage == 'missing_commit': (case['folder'] / 'stage_sam2.json').unlink()
    else:
        path = case['folder'] / 'stage_sam2.json'; data = json.loads(path.read_text())
        data['fingerprint'] = 'different'; write_json(path, data)
    report = case['run']()
    assert report['status'] == 'failed'
    assert not case['calls'].exists()
    assert json.loads(Path(report['report_path']).read_text())['error']


@pytest.mark.parametrize('failure', ['fail', 'hang'])
def test_failed_probe_retains_error_and_cleans_runner(diagnostic_case, failure):
    case = diagnostic_case
    case['config'].operators.hamer.params[failure] = True
    write_json(case['cfg'], asdict(case['config']))
    report = case['run'](mode='probe', timeout_s=4)
    assert report['status'] == 'failed'
    assert len(report['trials']) == 1 and report['trials'][0]['status'] == 'failed'
    assert Path(report['trials'][0]['directory'], 'record.json').is_file()
    assert 'out of memory' in str(report['error']) if failure == 'fail' else 'timed out' in str(report['error'])
    with pytest.raises(ProcessLookupError):
        os.kill(int(case['instances'].read_text().strip()), 0)


def test_invalid_output_cannot_touch_source(diagnostic_case):
    case = diagnostic_case; before = snapshot(case['source'])
    with pytest.raises(ValueError, match='outside'):
        run_diagnostic(str(case['cfg']), str(case['source']), 'c', 'hamer', str(case['source'] / 'probe'))
    assert snapshot(case['source']) == before


def test_stage_inputs_are_read_from_verified_copy_not_receipt_directory(diagnostic_case):
    case = diagnostic_case
    path = case['folder'] / 'stage_sam2.json'; data = json.loads(path.read_text())
    data['directory'] = '/does/not/exist'; write_json(path, data)
    report = case['run']()
    assert report['status'] == 'passed', report.get('error')


def test_changed_model_assets_produce_different_identity(diagnostic_case):
    case = diagnostic_case
    first = case['run']()
    Path(case['config'].operators.hamer.params['checkpoint']).write_bytes(b'new fixture weights')
    second = case['run']()
    assert first['status'] == second['status'] == 'passed'
    assert first['model_identity']['identity_hash'] != second['model_identity']['identity_hash']


def test_gpu_mapping_respects_existing_visibility(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b')
    assert _device_environment(1, True)['CUDA_VISIBLE_DEVICES'] == 'GPU-b'
    assert _device_environment(0, False)['CUDA_VISIBLE_DEVICES'] == ''
    with pytest.raises(ValueError): _device_environment(2, True)


def test_missing_gpu_observation_is_unknown_and_failed_probe_has_no_fit_claim():
    assert recommendations([], 'sam2')['device_headroom_status'] == 'unknown'
    passed = {'status': 'passed', 'phase': 'cold', 'diagnostics': {'device_observation': 'nvml',
        'device_free_min_bytes': 4 * 1024**3, 'device_total_bytes': 8 * 1024**3}}
    assert recommendations([passed], 'sam2')['device_headroom_status'] == 'observed_for_this_sample'
    assert recommendations([passed, {'status': 'failed'}], 'sam2')['device_headroom_status'] == 'unknown'


def test_cli_reports_failed_diagnostic_with_nonzero_exit(diagnostic_case):
    case = diagnostic_case; case['video'].write_bytes(b'changed')
    result = subprocess.run([sys.executable, '-m', 'egoworld.cli', 'gpu-probe', '--config', str(case['cfg']),
        '--source-run', str(case['source']), '--clip-id', 'c', '--stage', 'hamer', '--output-dir', str(case['source'].parent / 'cli')],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['status'] == 'failed'


def test_runner_start_failure_is_recorded_per_trial(diagnostic_case, monkeypatch):
    from egoworld.diagnostics import replay
    def fail(*args, **kwargs): raise OSError('fixture spawn failure')
    monkeypatch.setattr(replay, 'RunnerSession', fail)
    report = diagnostic_case['run']()
    assert report['status'] == 'failed'
    assert report['trials'][0]['status'] == 'failed'
    assert Path(report['trials'][0]['directory'], 'record.json').is_file()


def test_hot_measurement_requires_reuse_receipt(diagnostic_case, monkeypatch):
    from egoworld.pipeline.external import RunnerSession
    original = RunnerSession.run
    calls = []
    def run(self, *args, **kwargs):
        packet = original(self, *args, **kwargs)
        calls.append(1)
        if len(calls) > 1:
            packet['stats']['model_cache_hit'] = 0
        return packet
    monkeypatch.setattr(RunnerSession, 'run', run)
    report = diagnostic_case['run'](mode='probe', warmup=0, repeats=1)
    assert report['status'] == 'failed'
    assert 'Hot trial' in report['error']['message']
    assert not report['recommendations']['warm_model_reuse_verified']


def test_transitive_dependencies_are_verified(diagnostic_case, tmp_path):
    from egoworld.diagnostics.inputs import freeze_inputs
    case = diagnostic_case
    (case['folder'] / 'stage_sam2.json').unlink()
    destination = tmp_path / 'frozen'; destination.mkdir()
    with pytest.raises(ValueError, match='sam2'):
        freeze_inputs(case['source'], 'c', 'dex_retarget', destination)


def test_interrupt_keeps_trial_record(diagnostic_case, monkeypatch):
    from egoworld.pipeline.external import RunnerSession
    def interrupt(*args, **kwargs): raise KeyboardInterrupt('fixture interrupt')
    monkeypatch.setattr(RunnerSession, 'run', interrupt)
    report = diagnostic_case['run']()
    assert report['status'] == report['trials'][0]['status'] == 'interrupted'
    assert Path(report['trials'][0]['directory'], 'record.json').is_file()


@pytest.mark.parametrize('cuda_uuid,nvml_uuid', [('GPU-actual', 'GPU-actual'),
    ('d69560c2-3ab1-d989-ff13-013d1db9d0dd', 'GPU-d69560c2-3ab1-d989-ff13-013d1db9d0dd')])
def test_nvml_mapping_uses_actual_cuda_uuid_not_numeric_index(monkeypatch, cuda_uuid, nvml_uuid):
    import types
    from egoworld.diagnostics.observations import TrialObserver
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    calls = []
    nvml = types.SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlSystemGetDriverVersion=lambda: 'fixture', nvmlDeviceGetHandleByUUID=lambda value: calls.append(value) or 'handle',
        nvmlDeviceGetUUID=lambda handle: 'GPU-actual',
        nvmlDeviceGetMemoryInfo=lambda handle: types.SimpleNamespace(total=100, used=25, free=75))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True, synchronize=lambda: None,
        current_device=lambda: 0, get_device_properties=lambda _: types.SimpleNamespace(uuid=cuda_uuid)))
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    observer = TrialObserver(True).__enter__()
    data = observer.finish()
    assert calls == [nvml_uuid] and data['device_observation'] == 'nvml'
    assert data['device_free_min_bytes'] == 75


def test_runtime_identity_tracks_editable_model_source_changes(tmp_path, monkeypatch):
    import types
    from egoworld.diagnostics.observations import model_source_identities
    package = tmp_path / 'hamer'; package.mkdir()
    init = package / '__init__.py'; init.write_text('')
    implementation = package / 'model.py'; implementation.write_text('VERSION = 1')
    monkeypatch.setitem(sys.modules, 'hamer', types.SimpleNamespace(__file__=str(init)))
    first = model_source_identities()['hamer']['python_source_sha256']
    implementation.write_text('VERSION = 2')
    assert model_source_identities()['hamer']['python_source_sha256'] != first


def test_nvml_real_sample_semantics_and_partial_failure(monkeypatch):
    import types
    from egoworld.diagnostics.observations import TrialObserver
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fixture')
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    fail = [False]
    def memory(handle):
        if fail[0]: raise RuntimeError('device disappeared')
        return types.SimpleNamespace(total=1000, used=300, free=700)
    nvml = types.SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlSystemGetDriverVersion=lambda: 'fixture', nvmlDeviceGetHandleByUUID=lambda _: 'h',
        nvmlDeviceGetUUID=lambda _: 'GPU-fixture', nvmlDeviceGetMemoryInfo=memory,
        nvmlDeviceGetUtilizationRates=lambda _: types.SimpleNamespace(gpu=25),
        nvmlDeviceGetPowerUsage=lambda _: 10000,
        nvmlDeviceGetComputeRunningProcesses=lambda _: [types.SimpleNamespace(pid=123)])
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    observer = TrialObserver(True, interval=60).__enter__()
    data = observer.finish()
    assert data['device_observation'] == 'nvml'
    assert data['device_memory_coverage'] == 'from_trial_start'
    assert data['device_memory_samples'] == 2
    assert data['device_gpu_utilization_percent_mean'] == 25
    assert data['device_power_watts_peak'] == 10
    assert data['device_compute_pids_observed'] == [123]
    observer = TrialObserver(True, interval=60).__enter__()
    fail[0] = True
    data = observer.finish()
    assert data['device_observation'] == 'partial'
    assert data['device_used_peak_bytes'] == 300
    trial = {'status': 'passed', 'phase': 'cold', 'diagnostics': data}
    assert recommendations([trial], 'hamer')['device_headroom_status'] == 'unknown'
