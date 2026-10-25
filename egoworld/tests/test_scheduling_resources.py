"""Exercise bounded execution, retry draining, model reuse and wide GPU pools without CUDA."""
import json
import threading
import time

import pytest
from test_pipeline_logic import pipeline_case
from egoworld.pipeline import stages
from egoworld.pipeline.scheduling import ready_priority
from egoworld.utils.errors import TransientIOError


def test_writer_retry_drains_when_producer_budget_is_full(pipeline_case, monkeypatch):
    from egoworld.pipeline.artifacts import WriterActor
    case = pipeline_case
    case['config']['backpressure'].update(max_in_flight_write=1, max_in_flight_bytes=1024,
        estimated_result_bytes=1024)
    calls = []
    original = WriterActor.write_stage
    def write(self, job, packet):
        calls.append(job['clip']['clip_id'])
        if len(calls) == 1:
            raise TransientIOError('writer temporarily unavailable')
        return original(self, job, packet)
    monkeypatch.setattr(WriterActor, 'write_stage', write)
    case['run']()
    summary = json.loads((case['root'] / 'output/run_id=A/run_summary.json').read_text())
    assert summary['completed_clips'] == 3 and len(calls) == 4
    assert summary['memory_budget']['reserved_peak_bytes'] == 1024
    assert summary['memory_budget']['remaining_bytes'] == 0


def test_impossible_reservation_fails_instead_of_hanging(pipeline_case):
    pipeline_case['config']['backpressure'].update(max_in_flight_bytes=1)
    with pytest.raises(ValueError, match='reservation.*exceeds'):
        pipeline_case['run']()


@pytest.mark.parametrize('gpus', [2, 16])
def test_logical_gpu_pool_runs_ready_clips_concurrently(pipeline_case, monkeypatch, gpus):
    case = pipeline_case
    case['config'].update(num_gpus=gpus)
    case['config']['execution']['max_active_clips'] = gpus
    case['config']['backpressure'].update(max_in_flight_gpu=gpus, max_in_flight_write=gpus,
        max_in_flight_bytes=gpus * 1024, estimated_result_bytes=1024)
    case['rows'][:] = [dict(case['rows'][0], clip_id=f'c{i}') for i in range(gpus)]
    gate = threading.Barrier(gpus)
    original = stages.ModelActor.process
    def infer(self, *args):
        gate.wait(timeout=10)
        return original(self, *args)
    monkeypatch.setattr(stages.ModelActor, 'process', infer)
    case['run']()
    summary = json.loads((case['root'] / 'output/run_id=A/run_summary.json').read_text())
    assert summary['completed_clips'] == gpus
    assert summary['queue_peaks']['compute'] == gpus
    assert summary['memory_budget']['reserved_peak_bytes'] == gpus * 1024
    assert summary['memory_budget']['remaining_bytes'] == 0


def test_overdue_stage_beats_warm_model_preference():
    contexts = [({'states': {'sam2': 'Pending'}, 'ready_since': {'sam2': 90.}}, 'sam2'),
                ({'states': {'hamer': 'Pending'}, 'ready_since': {'hamer': 1.}}, 'hamer')]
    preference = lambda stage: 0 if stage == 'sam2' else 1
    picked = min(contexts, key=lambda pair: ready_priority(*pair, 100., 60., preference))
    assert picked[1] == 'hamer'


def test_prepared_byte_pressure_pauses_preparation_until_clip_finishes(pipeline_case, monkeypatch):
    case = pipeline_case
    case['config']['execution']['cpu_workers'] = 1
    case['config']['backpressure']['max_prepared_bytes'] = 10
    events = []
    def prepare(self, job):
        events.append(('prepare', job['clip']['clip_id']))
        return {'video_path': 'unused', 'prepared_bytes': 10}
    original = stages.ModelActor.process
    def infer(self, job, *args):
        events.append(('compute', job['clip']['clip_id']))
        time.sleep(.02)
        return original(self, job, *args)
    monkeypatch.setattr(stages.PrepareActor, 'prepare', prepare)
    monkeypatch.setattr(stages.ModelActor, 'process', infer)
    case['run']()
    assert events.index(('compute', 'c0')) < events.index(('prepare', 'c1'))


def test_profile_reports_observations_without_running_models(pipeline_case, monkeypatch):
    from egoworld.profiling import profile_run
    pipeline_case['run']()
    monkeypatch.setattr(stages.ModelActor, 'process', lambda *a: pytest.fail('profile must not infer'))
    report = profile_run(pipeline_case['root'] / 'output/run_id=A')
    assert report['stages']['sam2']['completed_calls'] == 3
    assert report['stages']['sam2']['observed_result_peak_bytes'] > 0
    assert not report['stages']['sam2']['gpu_memory_observed']
    assert not report['automatic_resource_plan']
    assert report['config_fingerprint']


@pytest.mark.parametrize('field,value', [('max_in_flight_bytes', -1), ('max_prepared_bytes', 1.5),
                                        ('stage_memory_bytes', {'unknown': 100})])
def test_bad_budget_settings_are_rejected(field, value):
    from egoworld.config import PipelineConfig
    config = PipelineConfig()
    setattr(config.backpressure, field, value)
    with pytest.raises(ValueError): config.resolved()


def test_observed_oversize_is_drained_before_budget_error():
    from egoworld.config import BackpressureConfig
    from egoworld.pipeline.scheduling import MemoryBudget
    budget = MemoryBudget(BackpressureConfig(max_in_flight_bytes=1024, estimated_result_bytes=512))
    job = {'clip': {'clip_id':'c'}, 'stage':'sam2'}
    budget.reserve(job)
    budget.completed(dict(job, _result_bytes=2048))
    assert not budget.fits(job)
    budget.release(job)
    with pytest.raises(ValueError, match='reservation.*exceeds'): budget.fits(job)


def test_optional_cuda_observations_do_not_initialize_runtime(monkeypatch):
    import sys
    from types import SimpleNamespace
    from egoworld.utils.performance import cuda_metrics
    resets = []
    cuda = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    assert cuda_metrics(reset=True) == {} and cuda_metrics() == {}
    cuda.is_initialized = lambda: True
    cuda.reset_peak_memory_stats = lambda: resets.append(True)
    cuda.current_device = lambda: 0
    cuda.get_device_properties = lambda _: SimpleNamespace(name='fixture GPU', total_memory=10000)
    cuda.max_memory_allocated = lambda: 100
    cuda.max_memory_reserved = lambda: 200
    cuda_metrics(reset=True)
    report = cuda_metrics()
    assert resets == [True] and report['torch_peak_reserved_bytes'] == 200
