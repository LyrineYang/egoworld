"""Phase contracts: immutable bounded handoffs, fair scheduling and safe replay."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from egoworld.config import PipelineConfig
from egoworld.pipeline.spool import Spool
from egoworld.utils.timing import Profile, CURRENT, phase, profile_session


def test_spool_bounds_corruption_and_mapping_lifetime(tmp_path):
    spool = Spool(tmp_path, 8000, 5000)
    source = np.arange(256, dtype=np.float32).reshape(16, 16)
    ref = spool.put({'x': source, 'scalar': np.array(7, dtype=np.int32)})
    restored = spool.get(ref)
    np.testing.assert_array_equal(restored['x'], source)
    assert type(restored['x']) is np.ndarray and restored['x'].flags.owndata
    spool.discard(ref)
    assert restored['scalar'].shape == () and restored['scalar'].item() == 7
    restored['x'][:] = 0  # valid after mappings/files are closed
    with pytest.raises(ValueError, match='budget'):
        spool.put(np.ones(5000))
    assert spool.used == 0
    a = spool.put(source); b = spool.put(source)
    (tmp_path / a['block'] / 'arrays.dat').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Corrupt'): spool.get(a)
    with pytest.raises(ValueError, match='Invalid phase block'): spool.get(dict(a, block='../escape'))
    spool.clear(); assert not tmp_path.exists()


def test_profile_records_failure_no_cuda_init_and_bounded_trace(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    with profile_session(True, {'clip_id': 'c'}) as profiler:
        profiler.max_events = 2
        for _ in range(4):
            with phase('forward', cuda=True): pass
        with pytest.raises(ValueError), phase('decode'):
            raise ValueError('fixture')
        report = profiler.finish()
    assert CURRENT.get() is None
    assert report['stats']['phase_forward_calls'] == 4
    assert report['stats']['phase_forward_cuda_unobserved_calls'] == 4
    assert report['stats']['phase_decode_errors'] == 1
    assert len(report['events']) == 2 and report['stats']['phase_trace_dropped_events'] == 3


def test_cuda_events_resolve_only_at_boundary(monkeypatch):
    log = []
    class Event:
        def __init__(self, **kwargs): pass
        def record(self): log.append('record')
        def synchronize(self): log.append('sync')
        def elapsed_time(self, other): return 2.5
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True, Event=Event)))
    with profile_session(True) as profiler:
        with phase('forward', cuda=True): pass
        with phase('transfer', cuda=True): pass
        assert log == ['record'] * 4
        report = profiler.finish()
    assert report['stats']['phase_forward_cuda_seconds'] == .0025
    assert log.count('sync') == 2


def test_sam_masks_one_thresholded_transfer_and_exact_cpu_derivatives():
    from egoworld.operators.sam2_op import _instance_masks, encode_masks
    from egoworld.operators.common import decode_rle
    transfers = []
    data = np.zeros((2, 1, 4, 5), np.float32); data[0, 0, 1:3, 1:4] = 2.; data[1, 0, 0, 0] = 1.
    class Tensor:
        def __init__(self, value): self.value = value
        def __gt__(self, threshold): return Tensor(self.value > threshold)
        def detach(self): return self
        def cpu(self):
            assert self.value.dtype == np.bool_; transfers.append(self.value.nbytes); return self
        def numpy(self): return self.value
    masks = _instance_masks(Tensor(data))
    meta = {'frame_index': 7, 'fps': 10., 'object_ids': [8, 3], 'objects': {'8': {'label': 'left hand', 'detection_score': .9}},
            'shape': list(masks.shape), 'packed': np.packbits(masks.reshape(-1))}
    direct = encode_masks(meta, masks); packed = encode_masks(meta)
    assert direct == packed and transfers == [40]
    np.testing.assert_array_equal(decode_rle(direct[0][0]['mask_rle']), np.any(masks, 0))
    assert [r['object_id'] for r in direct[1]] == [8, 3]
    assert direct[1][0]['bbox_xyxy'] == [1, 1, 4, 3] and direct[1][0]['area_px'] == 6


def test_hamer_cross_frame_batches_boundaries_and_tail(monkeypatch):
    from egoworld.operators import hamer_op as target
    from test_model_adapters import module
    rows = [{'frame_index': frame, 'object_id': oid, 'label': side + ' hand', 'area_px': 4, 'bbox_xyxy': [0, 0, 2, 2]}
            for frame in (10, 11, 12) for oid, side in ((1, 'left'), (2, 'right'))]
    monkeypatch.setattr(target, 'rows', lambda *args: rows)
    monkeypatch.setattr(target, 'frame_reader', lambda *a: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(target, 'read_frame', lambda cap, index: np.full((4, 4, 3), index, np.uint8))
    class Dataset:
        def __init__(self, cfg, image, boxes, rights, **kw): self.image, self.rights = image, rights
        def __getitem__(self, i): return {'img': self.image.copy(), 'right': self.rights[i]}
    module(monkeypatch, 'hamer.datasets.vitdet_dataset', ViTDetDataset=Dataset)
    op = target.HamerOperator(checkpoint='unused', frame_stride=1, batch_size=4, max_batch_crops=3)
    op.model_cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000), MODEL=SimpleNamespace(IMAGE_SIZE=256))
    batches = list(op.prepare_batches({'frame_start': 10}, 'unused', {}, cross_frame=True))
    assert list(map(lambda b: len(b['metadata']), batches)) == [3, 3]
    assert [(m['index'], m['detection']['object_id'], m['side']) for b in batches for m in b['metadata']] == [(r['frame_index'], r['object_id'], r['label'].split()[0]) for r in rows]
    op.params['max_batch_crops'] = 4
    assert [len(b['metadata']) for b in op.prepare_batches({'frame_start': 10}, 'unused', {}, True)] == [4, 2]
    op.params['max_crop_bytes'] = 1
    with pytest.raises(ValueError, match='crop exceeds'):
        list(op.prepare_batches({'frame_start': 10}, 'unused', {}, True))


def test_hamer_baseline_profile_measures_lazy_crop_work(monkeypatch):
    from egoworld.operators import hamer_op as target
    from egoworld.utils import timing
    from test_model_adapters import module
    clock = [0.]
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    def loader(*args, **kwargs):
        for count, seconds in ((2, 2.), (1, .5)):
            clock[0] += seconds  # official dataset work happens on next(), not construction
            yield {'right': np.ones(count)}
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(DataLoader=loader))))
    module(monkeypatch, 'hamer.datasets.vitdet_dataset', ViTDetDataset=lambda *a, **kw: object())
    detections = [{'frame_index': 0, 'object_id': i, 'label': 'right hand',
                   'area_px': 4, 'bbox_xyxy': [0, 0, 2, 2]} for i in range(3)]
    monkeypatch.setattr(target, 'rows', lambda *args: detections)
    monkeypatch.setattr(target, 'frame_reader', lambda *args: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(target, 'read_frame', lambda *args: np.zeros((4, 4, 3), np.uint8))
    op = target.HamerOperator(checkpoint='unused', frame_stride=1, batch_size=2)
    op.model_cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000), MODEL=SimpleNamespace(IMAGE_SIZE=256))
    with profile_session(True) as profile:
        batches = list(op.prepare_batches({'frame_start': 0}, 'unused', {}))
    assert [len(batch['metadata']) for batch in batches] == [2, 1]
    assert profile.stats['phase_crop_collate_calls'] == 2
    assert profile.stats['phase_crop_collate_wall_seconds'] == 2.5


@pytest.mark.parametrize('field,value', [('split_stage_bytes', 0), ('split_execution', 1), ('scheduling_policy', 'auto')])
def test_split_configuration_rejects_invalid_bounds(field, value):
    config = PipelineConfig(); setattr(config.execution, field, value)
    with pytest.raises(ValueError): config.resolved()


def test_cost_policy_worker_feasibility_wait_and_ablation(tmp_path, monkeypatch):
    from egoworld.diagnostics import calibration as target
    config = asdict(PipelineConfig().resolved())
    config['execution'].update(scheduling_policy='cost', scheduling_profile=str(tmp_path/'profile.json'), max_stage_wait_s=60.)
    config['operators']['fast3r'] = {'enabled': True, 'params': {'model_name_or_path': 'fixture'}}
    def load(terms):
        config['execution']['cost_terms'] = terms
        doc = {'schema_version': 1, 'source_tree_hash': target.source_identity(), 'stages': {}}
        for stage, cost in [('sam2', 1), ('fast3r', 5)]:
            doc['stages'][stage] = {'parameter_hash': target.parameter_hash(config, stage), 'assets': {},
                'load_seconds': 100., 'service_seconds_per_unit': cost, 'sample_units': 10, 'result_bytes': 128}
        doc['calibration_hash'] = target.digest(doc)
        Path(config['execution']['scheduling_profile']).write_text(json.dumps(doc))
        return target.CostPolicy(config)
    backend = SimpleNamespace(slots=[{'last_stage': 'fast3r'}, {'last_stage': 'sam2'}],
                              available=lambda kind, stage: [0])  # slot 1 is busy, cannot select it
    jobs = [{'stage': s, 'clip': {'clip_id': s, 'frame_start': 0, 'frame_end': 10}, '_ready_since': 90.} for s in ('sam2','fast3r')]
    picked, record = load(['load', 'service']).choose(jobs, backend, 100.)
    assert picked['stage'] == 'fast3r' and picked['_preferred_slot'] == 0
    assert len(record['candidates']) == 2
    picked, _ = load(['service']).choose(jobs, backend, 100.)
    assert picked['stage'] == 'sam2'
    jobs[0]['_ready_since'] = 1.
    picked, record = load(['load','service']).choose(jobs, backend, 100.)
    assert picked['stage'] == 'sam2' and record['reason'] == 'max_wait'
    config['operators']['sam2']['params']['precision'] = 'bf16'
    with pytest.raises(ValueError, match='incompatible'): target.CostPolicy(config)


@pytest.mark.ray_integration
def test_ray_split_overlap_retries_writer_drain_and_resume(tmp_path, monkeypatch):
    import os
    import subprocess
    import time
    if os.getenv('EGOWORLD_RAY_TEST') != '1': pytest.skip('Set EGOWORLD_RAY_TEST=1')
    import ray
    import egoworld.pipeline.backend as target
    from egoworld.pipeline.stages import ModelActor
    from egoworld.pipeline.artifacts import WriterActor
    from egoworld.pipeline.driver import run_pipeline
    from egoworld.utils.errors import TransientIOError
    from test_pipeline_logic import full_packet
    class PhaseFixture(ModelActor):
        def process(self, job, prepared, inputs):
            op, cid = job['_operation'], job['clip']['clip_id']
            overlap = self.config['operators']['sam2']['params'].get('failure') == 'none'
            folder = Path(self.config['operators']['sam2']['params']['markers']); folder.mkdir(exist_ok=True)
            def wait_for(name):
                deadline = time.monotonic() + 12
                while not (folder/name).exists():
                    if time.monotonic() >= deadline: raise TimeoutError(name)
                    time.sleep(.01)
            scratch = Path(job['work_dir'])/'.phases'/job['stage']; scratch.mkdir(parents=True, exist_ok=True)
            if op == 'input':
                (scratch/'temporary').write_text('scratch')
                if overlap and cid == 'c1': wait_for('c0-infer')
            else:
                assert prepared['_phase_input']['token'] == cid
            (folder/f'{cid}-{op}').write_text('running')
            if overlap and cid == 'c0' and op == 'infer': wait_for('c1-input')
            if overlap and cid == 'c0' and op == 'post': wait_for('c1-infer')
            if overlap and cid == 'c1' and op == 'infer': wait_for('c0-post')
            failure = self.config['operators']['sam2']['params'].get('failure')
            if cid == 'c0' and op == failure and not (folder/'failed').exists():
                (folder/'failed').touch(); raise TransientIOError('one phase failure')
            if op != 'post':
                return {'_handoff': True, 'token': cid, 'stats': {'result_bytes': 64, 'work_units': 10}}
            packet = full_packet(job['stage'], job['clip']); packet.setdefault('stats', {})['result_bytes'] = 512
            return packet
    class SlowWriter(WriterActor):
        def write_stage(self, job, packet):
            time.sleep(.03)
            folder = Path(self.config['operators']['sam2']['params']['markers'])
            if self.config['operators']['sam2']['params'].get('failure') == 'write' and not (folder/'failed').exists():
                (folder/'failed').touch(); raise TransientIOError('slow writer transient')
            return super().write_stage(job, packet)
    monkeypatch.setattr(target, 'ModelActor', PhaseFixture)
    monkeypatch.setattr(target, 'WriterActor', SlowWriter)
    video = tmp_path/'clip.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=duration=1:size=32x32:rate=10','-pix_fmt','yuv420p',str(video)],check=True)
    vm, cm = tmp_path/'videos.jsonl', tmp_path/'clips.jsonl'
    vm.write_text(json.dumps({'video_id':'v','path':str(video),'fps':10})+'\n')
    cm.write_text(''.join(json.dumps({'clip_id':f'c{i}','video_id':'v','start_s':0.,'end_s':1.,'frame_start':0,'frame_end':10})+'\n' for i in range(3)))
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ew-phase-ray-', dir='/tmp') as ray_dir:
        ray.init(num_cpus=8, num_gpus=1, include_dashboard=False, object_store_memory=100*1024**2, _temp_dir=ray_dir)
        try:
            for failure in ('none', 'input', 'infer', 'post', 'write', 'interrupt'):
                config = asdict(PipelineConfig().resolved()); config['run_id'] = failure
                config['paths'].update(output_root=str(tmp_path/'out'), state_db_path=str(tmp_path/'state.db'))
                config['execution'].update(split_execution=True, cpu_workers=1, write_workers=1, phase_workers=2,
                                           split_stage_bytes=2048, split_block_bytes=1024, split_total_bytes=6144)
                config['backpressure'].update(max_in_flight_write=3, max_in_flight_bytes=3072, estimated_result_bytes=1024)
                config['retry'].update(max_retries=1, base_delay_s=0)
                config['operators']['sam2']['params'] = {'markers':str(tmp_path/f'markers-{failure}'), 'failure': failure}
                cfg=tmp_path/'config.json'; cfg.write_text(json.dumps(config))
                if failure == 'interrupt':
                    class InterruptBackend(target.RayBackend):
                        def poll(self, timeout):
                            events = super().poll(timeout)
                            if any(e[0] == 'input' and not e[3] for e in events):
                                raise KeyboardInterrupt('injected after CPU handoff')
                            return events
                    with pytest.raises(KeyboardInterrupt):
                        run_pipeline(str(cfg),str(vm),str(cm),backend_factory=InterruptBackend)
                    interrupted = tmp_path/'out'/f'run_id={failure}'
                    assert json.loads((interrupted/'run_summary.json').read_text())['status'] == 'interrupted'
                    assert not list(interrupted.glob('.work/*/.phases'))
                run_pipeline(str(cfg),str(vm),str(cm),backend_factory=target.RayBackend)
                root=tmp_path/'out'/f'run_id={failure}'
                summary=json.loads((root/'run_summary.json').read_text())
                assert summary['completed_clips']==3 and summary['memory_budget']['remaining_bytes']==0
                assert summary['queue_peaks']['uncommitted_results']<=3
                assert summary['queue_peaks']['split_reserved_bytes']<=6144
                assert not list(root.glob('.work/*/.phases'))
                events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
                assert any(e.get('phase')=='input_done' for e in events)
                assert sum(e['status']=='Retry' for e in events)==int(failure not in ('none', 'interrupt'))
                run_pipeline(str(cfg),str(vm),str(cm),backend_factory=target.RayBackend)
                assert json.loads((root/'run_summary.json').read_text())['skipped_clips']==3
        finally:
            ray.shutdown()


def test_sam_phase_groups_bounded_masks_and_keeps_tail(tmp_path):
    from egoworld.pipeline.partitioned import run_phase
    class Predictor:
        params = {"mask_block_frames": 16}
        def run(self, video, start, end, prepared_prompts, mask_sink):
            for frame in range(33):
                mask = np.ones((1, 3, 3), np.uint8)
                mask_sink({"frame_index": frame, "fps": 10., "object_ids": [7],
                    "objects": {"7": {"label": "right hand", "detection_score": .9}},
                    "shape": list(mask.shape), "packed": np.packbits(mask.reshape(-1))})
            return {"processed_frames":33, "empty_mask_rate":0.}
    job = {"stage":"sam2", "_operation":"infer", "clip":{"clip_id":"c","frame_start":10,"fps":10.}, "work_dir":str(tmp_path)}
    execution = asdict(PipelineConfig().execution)
    prepared = {"video_path":"unused", "_phase_input":{"blocks":[],"stats":{"work_units":33}}}
    middle = run_phase(Predictor(),job,prepared,{},execution)
    assert len(middle["blocks"]) == 3
    final = run_phase(Predictor(),dict(job,_operation="post"),dict(prepared,_phase_input=middle),{},execution)
    assert [r["frame_index"] for r in final["tables"]["instances"]] == list(range(10,43))
    assert len(final["tables"]["masks"]) == 33 and final["stats"]["processed_frames"] == 33
    assert not list(tmp_path.glob('.phases/sam2/*/*'))


def test_contiguous_spool_preserves_scalar_empty_noncontiguous_and_uses_two_files(tmp_path):
    spool=Spool(tmp_path,100000,50000)
    data={'odd':np.ones(3,np.uint8),'scalar':np.array(3.25,np.float64),
          'empty':np.empty((0,3),np.float32),'view':np.arange(32,dtype=np.float32).reshape(4,8)[:,::-1]}
    data.update({str(i):np.arange(5,dtype=np.int32) for i in range(50)})
    ref=spool.put(data)
    assert {p.name for p in (tmp_path/ref['block']).iterdir()}=={'arrays.dat','tree.json'}
    result=spool.get(ref)
    for key,value in data.items():
        np.testing.assert_array_equal(result[key],value)
        assert result[key].dtype==value.dtype and result[key].shape==value.shape
    assert spool.used<=spool.limit


def test_calibration_retains_source_and_rejects_changed_asset(tmp_path,monkeypatch):
    from egoworld.diagnostics import calibration as target
    monkeypatch.setattr(target,'source_identity',lambda:'fixture-source')
    config=asdict(PipelineConfig().resolved())
    asset=tmp_path/'weights';asset.write_bytes(b'original')
    config['operators']['sam2']['params']={'checkpoint':str(asset)}
    (tmp_path/'config.supplied.json').write_text(json.dumps(config))
    def trial(phase,seconds):
        return {'phase':phase,'status':'passed','stats':{'result_bytes':128,'model_load_seconds':3. if phase=='cold' else 0.},
                'diagnostics':{'synchronized_stage_wall_seconds':seconds,'device_uuid':'fixture-GPU'}}
    report={'status':'passed','stage':'sam2','recommendations':{'warm_model_reuse_verified':True},
        'model_identity':{'source_tree_hash':'fixture-source','local_assets':{'sam2.checkpoint':{'path':str(asset),'sha256':target.sha256_file(str(asset))}}},
        'trials':[trial('cold',5),trial('hot',2),trial('hot',4),trial('hot',3)],
        'input_specification':{'frame_count':10},'observed_identity_hash':'fixture-observed'}
    source=tmp_path/'report.json';source.write_text(json.dumps(report));output=tmp_path/'costs.json'
    built=target.build_calibration([source],output)
    assert built['stages']['sam2']['service_seconds_per_unit']==.3
    assert built['reports']==[{'path':str(source),'sha256':target.sha256_file(str(source))}]
    config['execution'].update(scheduling_policy='cost',scheduling_profile=str(output))
    target.CostPolicy(config)
    asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset identity'):target.CostPolicy(config)
    with pytest.raises(ValueError,match='exactly one'):target.build_calibration([source,source],output)


def test_local_spool_isolates_worktrees_and_only_releases_owned_namespace(tmp_path):
    from egoworld.pipeline.spool import spool_path,release_spool
    local=tmp_path/'local';local.mkdir();sentinel=local/'user-file';sentinel.write_text('keep')
    cfg={'split_directory':str(local)}
    a=spool_path(cfg,tmp_path/'run-A'/'work','hamer')
    b=spool_path(cfg,tmp_path/'run-B'/'work','hamer')
    first=Spool(a,10000,5000);second=Spool(b,10000,5000)
    first.put(np.ones(10));ref=second.put(np.arange(10))
    release_spool(cfg,tmp_path/'run-A'/'work','hamer')
    assert not a.exists() and not a.parent.exists()
    np.testing.assert_array_equal(second.get(ref),np.arange(10))
    assert sentinel.read_text()=='keep'
    release_spool(cfg,tmp_path/'run-B'/'work','hamer')
    assert list(local.iterdir())==[sentinel]


@pytest.mark.parametrize('extra_on', ['reference', 'trial'])
def test_native_comparison_rejects_missing_or_extra_correspondences(tmp_path, extra_on):
    import importlib.util
    script = Path(__file__).parents[1]/'scripts'/'compare_throughput.py'
    spec = importlib.util.spec_from_file_location('throughput_comparison_fixture', script)
    comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
    for side in ('reference', 'trial'):
        folder = tmp_path/side/'outputs'; folder.mkdir(parents=True)
        np.savez(folder/'reconstruction-pointcloud.npz', points=np.ones((2, 3)))
    np.savez(tmp_path/extra_on/'outputs'/'reconstruction-correspondences.npz', source_pixel_xy=np.ones((2, 2)))
    result = comparison.compare_outputs({'directory':str(tmp_path/'reference')}, {'directory':str(tmp_path/'trial')})
    assert not result['passed'] and result['reason']=='array artifact set differs'
