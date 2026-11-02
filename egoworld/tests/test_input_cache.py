"""Real file locks, memmaps, subprocesses and video; no model downloads."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest

from egoworld.config import PipelineConfig
from egoworld.pipeline.input_cache import (InputCache, cache_session, clear_cache, identity,
                                           options, prepare_cache, release_prepared_cache, cache_root)
from test_pipeline_logic import pipeline_case


def spec(root, **limits):
    return {"root": str(root), "namespace": identity("clip-a"),
            "policy": options({"input_cache": {"enabled": True, **limits}})}


def disk_bytes(root):
    return sum(p.stat().st_size for p in root.iterdir() if p.suffix in (".dat", ".json", ".tmp") and p.name != "policy.json")


def test_memmaps_close_and_consumers_cannot_mutate_cached_data(tmp_path):
    cache = InputCache(spec(tmp_path))
    value = {"frame": np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
             "size": np.float32(2.5), "id": 4, "zero": np.array(7)}
    cache.get("frame", "x", lambda: value)
    def forbidden():
        raise AssertionError("Unexpected decode")
    for _ in range(10):
        result = cache.get("frame", "x", forbidden)
        for key in value:
            assert type(result[key]) is type(value[key])
            np.testing.assert_array_equal(result[key], value[key])
        result["frame"][0] = 255
        assert not isinstance(result["frame"], np.memmap)
    maps = Path('/proc/self/maps')
    if maps.exists():
        assert str(tmp_path) not in maps.read_text()
    assert cache.stats["input_cache_hits"] == 10
    assert cache.stats["input_cache_peak_mapped_bytes"] > 0
    assert clear_cache(tmp_path) > 0 and disk_bytes(tmp_path) == 0


def test_budget_eviction_oversize_bypass_and_namespace_cleanup(tmp_path):
    settings = spec(tmp_path, max_disk_bytes=2400, max_entry_bytes=800, max_entries=2)
    one = InputCache(settings)
    other = InputCache(dict(settings, namespace=identity("clip-b")))
    for key in range(12):
        one.get("frame", key, lambda: {"a": np.zeros(700, dtype=np.uint8)})
        assert disk_bytes(tmp_path) <= 2400
        assert len(list(tmp_path.glob("*.dat"))) <= 2
    assert one.stats["input_cache_evictions"] >= 10
    one.get("frame", "large", lambda: {"a": np.zeros(801, dtype=np.uint8)})
    assert one.stats["input_cache_bypasses"] == 1
    other.get("crop", "other", lambda: {"a": np.ones(50)})
    release_prepared_cache({"input_cache": settings})
    assert len(list(tmp_path.glob("*.dat"))) == 1
    assert all(p.name.startswith(other.namespace) for p in tmp_path.glob("*.dat"))


@pytest.mark.parametrize("damage", ["data", "metadata", "truncated", "missing"])
def test_damaged_cache_rebuilds_from_original_factory(tmp_path, damage):
    cache = InputCache(spec(tmp_path))
    calls = []
    def make():
        calls.append(1)
        return {"a": np.arange(100)}
    expected = cache.get("frame", "x", make)
    dat = next(tmp_path.glob("*.dat"))
    if damage == "metadata":
        path = dat.with_suffix('.json'); doc = json.loads(path.read_text())
        doc['fields']['a']['shape'] = [10, 10]; path.write_text(json.dumps(doc))
    elif damage == "missing": dat.unlink()
    elif damage == "truncated": dat.write_bytes(b"short")
    else:
        with dat.open("r+b") as f: f.write(b"wrong")
    actual = cache.get("frame", "x", make)
    np.testing.assert_array_equal(actual["a"], expected["a"])
    assert len(calls) == 2 and cache.stats["input_cache_corruptions"] == 1


def concurrent_read(settings):
    cache = InputCache(settings)
    def make():
        with (cache.root / 'builds').open('a') as f:
            f.write(str(os.getpid()) + '\n')
        return {"frame": np.arange(10000, dtype=np.int32)}
    return cache.get('frame', 'same', make)['frame'].sum().item()


def test_concurrent_processes_build_only_once(tmp_path):
    settings = spec(tmp_path)
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(concurrent_read, [settings] * 8))
    assert results == [sum(range(10000))] * 8
    assert len((tmp_path / 'builds').read_text().splitlines()) == 1


def concurrent_fill(args):
    settings, index = args
    cache = InputCache(settings)
    result = cache.get('frame', index, lambda: {'frame': np.full(700, index, dtype=np.uint8)})
    return int(result['frame'][0]), cache.stats['input_cache_peak_disk_bytes']


def test_budget_is_global_across_concurrent_workers(tmp_path):
    settings = spec(tmp_path, max_disk_bytes=2400, max_entry_bytes=800, max_entries=2)
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        result = list(pool.map(concurrent_fill, [(settings, i) for i in range(20)]))
    assert [item[0] for item in result] == list(range(20))
    assert all(item[1] <= 2400 for item in result) and disk_bytes(tmp_path) <= 2400


def test_killed_publication_releases_locks_and_reclaims_orphans(tmp_path):
    # Simulate death at the two publication boundaries using real OS locks.
    script = '''import sys,time
from pathlib import Path
from egoworld.pipeline.input_cache import locked
root=Path(sys.argv[1])
with locked(root/'.lock'):
    (root/'abandoned.tmp').write_bytes(b'partial')
    (root/'orphan.dat').write_bytes(b'uncommitted')
    print('ready',flush=True)
    time.sleep(60)
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        child.send_signal(signal.SIGKILL); child.wait(timeout=5)
        cache = InputCache(spec(tmp_path))
        assert disk_bytes(tmp_path) == 0
        cache.get('frame', 'x', lambda: {'x': np.ones(4)})
    finally:
        if child.poll() is None: child.kill(); child.wait()


def test_shared_frame_reader_is_lazy_and_preserves_decode(tmp_path, monkeypatch):
    import cv2
    from egoworld.operators.common import frame_reader, read_frame
    video = tmp_path / 'video.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=duration=1:size=64x48:rate=5',
                    '-pix_fmt', 'yuv420p', str(video)], check=True)
    cap = frame_reader(video)
    expected = [read_frame(cap, i) for i in [0, 2, 4]]; cap.release()
    settings = prepare_cache({'extra': {'input_cache': {'enabled': True}}},
                             {'work_dir': str(tmp_path / 'clip'), 'clip': {'clip_id': 'a'}}, video)
    for stage in ['hamer', 'foundationpose', 'fast3r']:
        with cache_session(settings) as cache:
            if stage != 'hamer':
                monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: pytest.fail('Cache hit must not open decoder'))
            cap = frame_reader(video)
            for i, frame in zip([0, 2, 4], expected):
                np.testing.assert_array_equal(read_frame(cap, i), frame)
            cap.release()
            assert cache.stats['input_cache_hits' if stage != 'hamer' else 'input_cache_misses'] == 3
    video.touch()
    with cache_session(settings), pytest.raises(ValueError, match='changed'):
        frame_reader(video)


class OfficialDatasetFixture:
    def __init__(self):
        self.cfg = 'crop-config-v1'
        self.img_cv2 = np.zeros((20, 20, 3), dtype=np.uint8)
        self.center = np.array([[10., 10.]])
        self.scale = np.array([[1., 1.]])
        self.right = np.array([1.])
        self.personid = np.array([3])
        self.calls = 0
    def __len__(self): return 1
    def __getitem__(self, index):
        self.calls += 1
        return {'img': np.full((3, 4, 4), self.scale[index, 0], dtype=np.float32),
                'right': self.right[index], 'personid': int(self.personid[index])}


def test_crop_key_covers_image_bbox_side_config_and_transform(tmp_path, monkeypatch):
    from egoworld.operators.cached_inputs import cached_hamer_dataset
    import inspect
    code = tmp_path / 'source' / 'vitdet.py'; code.parent.mkdir(); code.write_text('v1')
    monkeypatch.setattr(inspect, 'getfile', lambda cls: str(code))
    data = OfficialDatasetFixture()
    with cache_session(spec(tmp_path / 'cache')) as cache:
        for _ in range(2):
            cached = cached_hamer_dataset(data)
            assert cached[0]['personid'] == 3
        assert data.calls == 1
        for change in ['bbox', 'side', 'config', 'image', 'source']:
            if change == 'bbox': data.scale[0, 0] = 2
            elif change == 'side': data.right[0] = 0
            elif change == 'config': data.cfg = 'new-config'
            elif change == 'image': data.img_cv2[0] = 1
            else: code.write_text('v2')
            cached_hamer_dataset(data)[0]
        assert data.calls == 6
        assert cache.stats['input_cache_crop_hits'] == 1


@pytest.mark.parametrize('value', [{'enabled': 'yes'}, {'max_entries': 0}, {'max_entry_bytes': True},
                                  {'max_disk_bytes': 1}, {'unknown': 1}, {'directory': 'relative'}])
def test_invalid_cache_policy_rejected(value):
    config = PipelineConfig(extra={'input_cache': value})
    with pytest.raises(ValueError, match='input_cache'): config.validate()


@pytest.mark.parametrize('outcome', ['success', 'failure', 'interrupt'])
def test_pipeline_releases_local_cache_on_exit(pipeline_case, tmp_path, monkeypatch, outcome):
    from egoworld.pipeline import stages
    from egoworld.pipeline.engine import Engine
    case = pipeline_case
    case['config']['extra'] = {'input_cache': {'enabled': True, 'directory': str(tmp_path / 'local-cache')}}
    work = tmp_path / 'output/run_id=A/.work'
    local = cache_root(case['config'], work)
    def prepare(self, job):
        settings = {'root': str(local), 'namespace': identity(job['clip']['clip_id']),
                    'policy': options(case['config']['extra'])}
        InputCache(settings).get('frame', 0, lambda: {'frame': np.zeros((20, 20, 3), dtype=np.uint8)})
        return {'video_path': 'unused', 'frames': 10, 'input_cache': settings}
    monkeypatch.setattr(stages.PrepareActor, 'prepare', prepare)
    if outcome == 'failure':
        def fail(*args): raise ValueError('fixture failure')
        monkeypatch.setattr(stages.Sam2Operator, 'run', fail)
    if outcome == 'interrupt':
        def interrupt(self, clips):
            prepare(None, {'clip': next(clips)})
            raise KeyboardInterrupt('fixture')
        monkeypatch.setattr(Engine, 'run', interrupt)
    if outcome == 'success': case['run']()
    else:
        with pytest.raises(KeyboardInterrupt if outcome == 'interrupt' else RuntimeError): case['run']()
    assert disk_bytes(local) == 0
    assert local != cache_root(case['config'], tmp_path / 'another-run/.work')
