#!/usr/bin/env python3
"""Reproducible CPU-only probes. These are not model or server benchmarks."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import platform
import hashlib
import statistics
import sys
import tempfile
import time

from egoworld.config import PipelineConfig
from egoworld.pipeline.stages import ModelActor


def median_seconds(function, repeats):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def parquet_probe(folder, repeats, count=512):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from egoworld.operators.common import rows
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    mask = json.dumps({'size': [720, 1280], 'counts': [0] + [128] * 7200})
    data = [dict(frame_index=i, object_id=1, label='right hand', area_px=460800,
                 bbox_xyxy=[0., 0., 1280., 720.], mask_rle=mask) for i in range(count)]
    pq.write_table(pa.Table.from_pylist(data), folder / 'instances.parquet', compression='zstd')
    inputs = {'sam2': {'directory': str(folder)}}
    columns = ['frame_index', 'object_id', 'label', 'area_px', 'bbox_xyxy']
    def full(): return rows(inputs, 'sam2', 'instances')
    def projected(): return rows(inputs, 'sam2', 'instances', columns)
    assert [{key: row[key] for key in columns} for row in full()] == projected()
    file = pq.ParquetFile(folder / 'instances.parquet')
    from egoworld.utils.performance import packet_bytes
    payload = {'tables': {'instances': full()}}
    return {'payload_size_estimation_seconds': median_seconds(lambda: packet_bytes(payload), repeats),
            'rows': count, 'mask_utf8_bytes_per_row': len(mask.encode()),
            'full_read_seconds': median_seconds(full, repeats),
            'projected_read_seconds': median_seconds(projected, repeats),
            'full_arrow_bytes': file.read().nbytes,
            'projected_arrow_bytes': file.read(columns=columns).nbytes,
            'scope': 'Synthetic stripe-mask table; same requested columns and values; timings include conversion to Python rows.'}


def runner_probe(folder, repeats, count=6):
    runner = folder / 'fixture.py'
    runner.write_text('''import sys
from egoworld.pipeline import stages
from egoworld import worker
class Model:
    def __init__(self, **params): pass
    def run(self, *args):
        return {'frames': [], 'instances': [], 'processed_frames': 1}
stages.OPERATORS['sam2'] = Model
worker.main()
''')
    report = {}
    for mode in ('per_task', 'persistent'):
        hits = []
        def run():
            config = asdict(PipelineConfig().resolved())
            config['execution']['external_runner_mode'] = mode
            config['operators']['sam2']['params']['runner_command'] = [sys.executable, str(runner)]
            actor = ModelActor(config)
            try:
                cached = 0
                for i in range(count):
                    packet = actor.process({'stage': 'sam2', 'clip': {'clip_id': str(i)},
                        'work_dir': str(folder / mode / str(i))}, {'video_path': 'unused'}, {})
                    cached += packet['stats']['model_cache_hit']
                hits.append(cached)
            finally:
                actor.close()
        report[mode] = {'seconds': median_seconds(run, repeats), 'tasks': count, 'cache_hits': hits}
    report['scope'] = 'Real runner/guard/file protocol with a zero-inference fixture. Includes Python imports and process cleanup; no model weights or CUDA.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    with tempfile.TemporaryDirectory(prefix='egoworld-cpu-probe-') as folder:
        root = Path(folder)
        import numpy, pyarrow, cv2
        digest = hashlib.sha256()
        package = Path(__file__).resolve().parents[1] / 'src/egoworld'
        for source in sorted(package.rglob('*.py')):
            digest.update(str(source.relative_to(package)).encode())
            digest.update(source.read_bytes())
        report = {'benchmark_script_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'threads': 1, 'source_tree_hash': digest.hexdigest(), 'versions': {'numpy':numpy.__version__, 'pyarrow':pyarrow.__version__, 'opencv':cv2.__version__}, 'platform': platform.platform(), 'python': platform.python_version(),
                  'repeats': args.repeats, 'parquet': parquet_probe(root, args.repeats),
                  'external_runner': runner_probe(root, args.repeats)}
    from egoworld.io.writers import write_json
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
