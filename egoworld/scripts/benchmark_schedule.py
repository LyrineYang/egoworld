"""Real single-device scheduling matrix; explicit configs, repetitions and telemetry.

All variants keep the same videos, sampling, precision and logical outputs. The
caller chooses enabled models and exact input windows in a dedicated config.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import math
import json
from pathlib import Path
import statistics
import os
import subprocess
import threading
import time
from uuid import uuid4

from egoworld.config import load_config
from egoworld.io.writers import write_json
from egoworld.pipeline.driver import run_pipeline
from egoworld.diagnostics.observations import process_tree_rss
from egoworld.utils.hashing import sha256_file
from egoworld.utils.video import get_video_info


class Monitor:
    def __init__(self, root, gpu_uuid, external_spools=()):
        import pynvml
        self.nvml = pynvml; pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode())
        self.root = root; self.external_spools = external_spools
        self.stop = threading.Event(); self.samples = []; self.errors = []
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            try:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                work_bytes = sum(p.stat().st_size for p in self.root.glob('.work/**/*') if p.is_file())
                phase_bytes = sum(p.stat().st_size for root in self.external_spools for p in root.rglob('*') if p.is_file())
                import psutil
                own = {os.getpid()} | {p.pid for p in psutil.Process().children(recursive=True)}
                foreign = sorted(p.pid for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle) if p.pid not in own)
                self.samples.append({'monotonic':time.monotonic(),'rss_bytes':process_tree_rss(),
                    'device_used_bytes':int(memory.used),'scratch_bytes':work_bytes+phase_bytes,
                    'work_scratch_bytes':work_bytes,'external_phase_bytes':phase_bytes,'foreign_compute_pids':foreign})
            except FileNotFoundError:
                pass  # a consumer may delete scratch between glob and stat
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop.wait(.2)

    def finish(self):
        self.stop.set(); self.thread.join(); self.nvml.nvmlShutdown()
        return {'sampling_interval_s':.2, 'scope':'coordinator process tree; entire CUDA device; run .work plus this run external phase namespaces',
            'peak_rss_bytes':max((s['rss_bytes'] or 0 for s in self.samples),default=0),
            'device_peak_bytes':max((s['device_used_bytes'] for s in self.samples),default=0),
            'scratch_peak_bytes':max((s['scratch_bytes'] for s in self.samples),default=0),
            'external_phase_peak_bytes':max((s['external_phase_bytes'] for s in self.samples),default=0),
            'foreign_compute_pids':sorted({pid for s in self.samples for pid in s['foreign_compute_pids']}),
            'errors':self.errors,'samples':self.samples}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--clips',type=int,default=8);p.add_argument('--clip-frames',type=int,default=10)
    p.add_argument('--step-frames',type=int,default=50);p.add_argument('--repeats',type=int,default=2)
    variants=['baseline-w4','heuristic-w1','heuristic-w4','heuristic-w8','cost-all','cost-no-load','cost-no-service','cost-no-release']
    p.add_argument('--variants',nargs='+',choices=variants,default=variants)
    args=p.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    base=asdict(load_config(args.config).resolved()); original=sha256_file(args.config)
    video=Path(args.video).resolve();info=get_video_info(str(video));checksum=sha256_file(str(video))
    video_id='schedule-'+checksum[:12]
    if min(args.clips,args.clip_frames,args.step_frames,args.repeats)<1 or args.step_frames<args.clip_frames:
        raise ValueError('Benchmark windows must be positive and nonoverlapping')
    clips=[]
    for i in range(args.clips):
        start=i*args.step_frames;end=start+args.clip_frames
        if end>info.frame_count:raise ValueError('Requested benchmark windows exceed video')
        clips.append({'clip_id':f'c{i:03d}','video_id':video_id,'start_s':start/info.fps,'end_s':end/info.fps,
            'frame_start':start,'frame_end':end,'scenedetect_failed':False})
    vm,cm=root/'videos.jsonl',root/'clips.jsonl'
    vm.write_text(json.dumps({'video_id':video_id,'path':str(video),'checksum':checksum,**asdict(info)})+'\n')
    cm.write_text(''.join(json.dumps(c)+'\n' for c in clips))
    index_path=root/'index.json';index=json.loads(index_path.read_text()) if index_path.exists() else []
    import ray
    import tempfile
    for repetition in range(args.repeats):
        # Reverse order on alternate repetitions to reduce monotonic warm-cache bias.
        for variant in (args.variants if repetition%2==0 else list(reversed(args.variants))):
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy: raise RuntimeError('GPU busy before benchmark: '+busy)
            config=json.loads(json.dumps(base));identifier=f'{variant}-r{repetition}-{uuid4().hex[:6]}'
            config['run_id']=identifier
            config['paths'].update(output_root=str(root/'runs'),state_db_path=str(root/'states'/f'{identifier}.db'))
            execution=config['execution'];execution.update(phase_profiling=True,split_execution=variant!='baseline-w4',
                external_runner_mode='persistent',
                scheduling_policy='cost' if variant.startswith('cost-') else 'heuristic',scheduling_profile=str(Path(args.calibration).resolve()),
                model_residency_tasks=int(variant[-1]) if '-w' in variant else 8)
            execution['cost_terms']=[term for term in ('load','service','release') if variant!='cost-no-'+term]
            config_path=root/'configs'/f'{identifier}.json';write_json(config_path,config)
            run_root=root/'runs'/f'run_id={identifier}'
            record={'variant':variant,'repeat':repetition,'run_id':identifier,'config_path':str(config_path),'run_dir':str(run_root),
                'unique_video_seconds':args.clips*args.clip_frames/info.fps,'source_sha256':checksum,'status':'running'}
            index.append(record);write_json(index_path,index)
            from egoworld.pipeline.spool import spool_path
            external_spools = [spool_path(execution, run_root/'.work'/c['clip_id'], 'sam2').parent for c in clips] if execution.get('split_directory') else []
            monitor=Monitor(run_root,args.gpu_uuid,external_spools);monitor.thread.start();started=time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix='ew-bench-ray-',dir='/tmp') as ray_dir:
                    ray.init(num_cpus=12,num_gpus=1,include_dashboard=False,object_store_memory=256*1024**2,_temp_dir=ray_dir)
                    from egoworld.pipeline.backend import RayBackend
                    run_pipeline(str(config_path),str(vm),str(cm),backend_factory=RayBackend)
                    record['summary']=json.loads((run_root/'run_summary.json').read_text())
                    if record['summary']['status'] != 'passed' or record['summary']['completed_clips'] != args.clips:
                        raise RuntimeError('Pipeline did not complete every requested clip')
                    events=[json.loads(line) for line in (run_root/'events.jsonl').read_text().splitlines()]
                    waits=[e['stats'].get('phase_compute_queue_wall_seconds',e['stats'].get('queue_wait_seconds',0)) for e in events if e.get('status')=='Done' and e.get('stats')]
                    record['queue_wait_p95_seconds']=sorted(waits)[min(len(waits)-1,int(.95*len(waits)))] if waits else None
                    record['decision_count']=sum(e.get('status')=='decision' for e in events)
                    clip_times={}
                    for event in events:
                        cid=event.get('clip_id')
                        if not cid:continue
                        value=datetime.fromisoformat(event['ts']).timestamp()
                        times=clip_times.setdefault(cid,{})
                        if event.get('stage')=='prepare' and event.get('status')=='Running':times.setdefault('start',value)
                        if event.get('stage') not in ('prepare','pipeline','scheduler') and event.get('status')=='Done':times['last_commit']=value
                    latency=sorted(t['last_commit']-t['start'] for t in clip_times.values() if 'last_commit' in t and 'start' in t)
                    record['clip_model_commit_latency_seconds']=latency
                    record['clip_model_commit_latency_p95_seconds']=latency[math.ceil(.95*len(latency))-1] if latency else None
                    record['status']='passed'
                    ray.shutdown()
            except BaseException as exc:
                record.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
                ray.shutdown()
                if isinstance(exc,KeyboardInterrupt):raise
            finally:
                record['elapsed_seconds']=time.perf_counter()-started
                telemetry=monitor.finish();write_json(root/'telemetry'/f'{identifier}.json',telemetry)
                record['telemetry']={k:v for k,v in telemetry.items() if k!='samples'}
                record['telemetry_path']=str(root/'telemetry'/f'{identifier}.json')
                write_json(index_path,index);print(json.dumps({k:v for k,v in record.items() if k not in ('summary','telemetry')}),flush=True)
    if sha256_file(args.config)!=original:raise RuntimeError('Input config changed')
    if any(record['status'] != 'passed' for record in index):raise SystemExit(1)


if __name__=='__main__':main()
