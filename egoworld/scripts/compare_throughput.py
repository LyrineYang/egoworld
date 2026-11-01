"""Compare all retained hot trials against the first baseline, including variance.

Thresholds are adapter regression bounds, not dataset-level accuracy claims.
HaMeR: <= 0.1 mm joints, <= 1 mm camera, <= 1e-3 MANO components.
FP: <= 1e-3 transform component. Fast3R: <= 1e-4 relative points, <= 1e-3
pose and <= 0.1 px intrinsics. SAM2: exact decoded masks and instance metadata.
"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import pyarrow.parquet as pq
from egoworld.operators.common import decode_rle
from egoworld.utils.hashing import sha256_file
from egoworld.io.writers import write_json

THRESHOLDS = {'joints_3d':1e-4, 'camera_translation':1e-3, 'global_orient':1e-3, 'mano_pose':1e-3,
    'betas':1e-3, 'transform':1e-3, 'camera_to_world':1e-3, 'intrinsics':.1,
    'points':1e-4, 'confidence':1e-4, 'source_pixel_xy':0, 'view_index':0}


def compare_outputs(reference, trial):
    a, b = Path(reference['directory'])/'outputs', Path(trial['directory'])/'outputs'
    differences, passed = {}, True
    names = {p.name for p in a.glob('*.parquet')}
    if names != {p.name for p in b.glob('*.parquet')}: return {'passed':False,'reason':'table set differs'}
    arrays = {p.name for p in a.glob('*.npz')}
    if arrays != {p.name for p in b.glob('*.npz')}: return {'passed':False,'reason':'array artifact set differs'}
    for name in sorted(names):
        left, right = pq.ParquetFile(a/name).read().to_pylist(), pq.ParquetFile(b/name).read().to_pylist()
        if len(left)!=len(right):
            differences[name]={'rows': [len(left),len(right)]}; passed=False; continue
        for x,y in zip(left,right):
            if set(x)!=set(y): passed=False; differences[name+'.fields']=False; continue
            for key in x:
                label=name+'.'+key
                if key in THRESHOLDS:
                    av,bv=np.asarray(x[key]),np.asarray(y[key])
                    difference=float(np.max(np.abs(av-bv))) if av.shape==bv.shape and av.size else 0. if av.shape==bv.shape else float('inf')
                    differences[label]=max(differences.get(label,0),difference)
                    passed &= difference <= THRESHOLDS[key]
                elif key=='mask_rle':
                    equal=np.array_equal(decode_rle(x[key]),decode_rle(y[key]));passed &= equal
                    differences[label]=differences.get(label,True) and equal
                elif x[key]!=y[key]:
                    differences[label]=False;passed=False
    for name in sorted(arrays):
        with np.load(a/name,allow_pickle=False) as left,np.load(b/name,allow_pickle=False) as right:
            if set(left.files)!=set(right.files): differences[name+'.fields']=False;passed=False;continue
            for key in left.files:
                av,bv=left[key],right[key]
                difference=float(np.max(np.abs(av-bv))) if av.shape==bv.shape and av.size else 0. if av.shape==bv.shape else float('inf')
                differences[name+'.'+key]=difference
                passed &= difference<=THRESHOLDS.get(key,0)
    return {'passed':bool(passed),'differences':differences}


def summarize(directory):
    root=Path(directory); result={'thresholds':THRESHOLDS,'models':{}}
    for index in sorted(root.glob('*-index.json')):
        stage=index.name.removesuffix('-index.json'); variants={}; reference=None
        entries=json.loads(index.read_text())
        for entry in entries:
            if entry['variant']=='baseline':
                baseline=json.loads(Path(entry['report_path']).read_text())
                reference=next((t for t in baseline['trials'] if t['phase']=='hot' and t['status']=='passed'),None)
        if reference is None:raise ValueError(f'{stage}: no passed baseline hot trial')
        for entry in entries:
            report=json.loads(Path(entry['report_path']).read_text())
            trials=report['trials'];hot=[t for t in trials if t['phase']=='hot' and t['status']=='passed']
            timing=[t['diagnostics']['synchronized_stage_wall_seconds'] for t in hot]
            gpu=[t['diagnostics'].get('gpu_step_wall_seconds',t['diagnostics']['synchronized_stage_wall_seconds']) for t in hot]
            variants[entry['variant']]={'report_path':entry['report_path'],'report_sha256':sha256_file(entry['report_path']),
                'status':report['status'],'error':report.get('error'),'hot_seconds':timing,
                'hot_median_seconds':statistics.median(timing) if timing else None,
                'gpu_step_hot_median_seconds':statistics.median(gpu) if gpu else None,
                'device_peak_bytes':max((t.get('diagnostics',{}).get('device_used_peak_bytes') or 0 for t in trials),default=0),
                'cpu_runner_peak_bytes':max((t.get('diagnostics',{}).get('cpu_process_peak_rss_bytes') or 0 for t in trials),default=0),
                'gpu_runner_peak_rss_bytes':max((t.get('diagnostics',{}).get('process_tree_peak_rss_bytes') or 0 for t in trials),default=0),
                'spool_peak_bytes':max((t.get('stats',{}).get('spool_peak_bytes',0) for t in trials),default=0),
                'hot_stats': [t['stats'] for t in hot],
                'output_comparisons':[compare_outputs(reference,t) for t in hot] if reference else [],
                'other_gpu_pids':report.get('recommendations',{}).get('non_runner_compute_pids_observed')}
        result['models'][stage]=variants
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--output',required=True)
    args=parser.parse_args();report=summarize(args.directory);write_json(args.output,report)
    if not report['models'] or any(v['status']!='passed' or not v['output_comparisons'] or
            not all(c['passed'] for c in v['output_comparisons'])
            for variants in report['models'].values() for v in variants.values()):raise SystemExit(1)
