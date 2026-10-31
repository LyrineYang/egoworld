"""One-stage replay and serial cold/warm probes using the existing guarded runner."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from egoworld.config import load_config
from egoworld.contracts import GPU_STAGES, STAGE_DEPENDENCIES, STAGE_ARTIFACTS
from egoworld.diagnostics.inputs import freeze_inputs, model_identity, digest
from egoworld.diagnostics.observations import source_identity
from egoworld.io.writers import write_json
from egoworld.pipeline.artifacts import WriterActor
from egoworld.pipeline.external import RunnerSession
from egoworld.pipeline.records import new_record
from egoworld.pipeline.stages import PrepareActor
from egoworld.utils.hashing import sha256_file
from egoworld.utils.video import get_video_info


def _device_environment(index, gpu):
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError('gpu_device must be a nonnegative visible device index')
    if not gpu:
        return {'CUDA_VISIBLE_DEVICES': ''}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        devices = [item.strip() for item in visible.split(',') if item.strip() and item.strip() != '-1']
        if index >= len(devices):
            raise ValueError('gpu_device is outside CUDA_VISIBLE_DEVICES')
        return {'CUDA_VISIBLE_DEVICES': devices[index]}
    return {'CUDA_VISIBLE_DEVICES': str(index)}


def recommendations(trials, stage):
    completed = [trial for trial in trials if trial['status'] == 'passed']
    hot = [trial for trial in completed if trial['phase'] == 'hot']
    gpu = stage in GPU_STAGES
    warmth = bool(hot) and all(trial.get('stats', {}).get('model_cache_hit') == 1 for trial in hot)
    results = [trial.get('stats', {}).get('result_bytes', 0) for trial in completed]
    devices = [trial.get('diagnostics', {}) for trial in completed]
    free = [item['device_free_min_bytes'] for item in devices if item.get('device_free_min_bytes') is not None]
    totals = [item['device_total_bytes'] for item in devices if item.get('device_total_bytes')]
    complete_device_observation = bool(completed) and all(item.get('device_observation') == 'nvml' for item in devices)
    margin = max(512 * 1024**2, math.ceil(max(totals) * .15)) if totals else None
    return {'automatic_resource_plan': False, 'configuration_modified': False,
            'warm_model_reuse_verified': warmth,
            'empty_output_trial_count': sum(trial.get('output_rows', 0) == 0 for trial in completed),
            'sample_review': 'Check nonempty target coverage and input size before applying resource suggestions.',
            'recommended_gpu_workers_per_device': 1 if gpu else None,
            'gpu_colocation': 'not_evaluated',
            'result_payload_reservation_bytes_min': math.ceil(max(results) * 1.25) if results else None,
            'result_reservation_scope': 'Payload only; excludes model/decoder/process heaps and device memory. Do not treat this as a complete stage working-set estimate.',
            'device_headroom_bytes_suggested': margin,
            'device_headroom_rule': 'Heuristic: max(512 MiB, 15% of capacity); requires validation on representative inputs.',
            'device_headroom_status': ('not_applicable' if not gpu else 'unknown' if len(completed) != len(trials) or not complete_device_observation or not margin or len(free) != len(completed)
                                       else 'low' if min(free) < margin else 'observed_for_this_sample'),
            'hot_stage_wall_seconds_median': statistics.median([trial['diagnostics']['synchronized_stage_wall_seconds'] for trial in hot])
                if hot and all('synchronized_stage_wall_seconds' in trial.get('diagnostics', {}) for trial in hot) else None,
            'scope': 'One fixed sample, one serial worker. No automatic precision, sampling, batch-size, worker-count or configuration changes; no OOM guarantee or multi-GPU throughput claim.'}


def _freeze_depth(config, clip, directory, inputs):
    """FoundationPose reads external depths; freeze every requested source frame."""
    from egoworld.operators.common import rgbd_assets, rows
    import shutil
    params = config['operators']['foundationpose']['params']
    assets = rgbd_assets(params['assets_manifest'])
    spec = dict(assets[clip['video_id']])
    hashes = {}
    suffix = Path(spec['depth_pattern'].format(frame_index=clip['frame_start'])).suffix
    target = directory / 'depth'; target.mkdir()
    frames = {row['frame_index'] for row in rows(inputs, 'sam2', 'instances', ['frame_index', 'label', 'area_px'])
              if row['label'] in params['objects'] and row['area_px']}
    for frame in sorted(frames):
        source = Path(spec['depth_pattern'].format(frame_index=frame))
        path = target / f'{frame}{suffix}'
        expected = sha256_file(str(source))
        shutil.copyfile(source, path)
        if sha256_file(str(path)) != expected:
            raise ValueError('Depth changed during copy')
        hashes[str(frame)] = expected
    spec['depth_pattern'] = str(target / ('{frame_index}' + suffix))
    snapshot = directory / 'rgbd-assets.json'
    write_json(snapshot, {clip['video_id']: spec})
    params['assets_manifest'] = str(snapshot)
    return hashes


def run_diagnostic(config_path, source_run, clip_id, stage, output_dir, *, mode='replay',
                   warmup=1, repeats=3, gpu_device=0, timeout_s=None,
                   video_manifest=None, clip_manifest=None):
    if stage not in STAGE_DEPENDENCIES or mode not in ('replay', 'probe'):
        raise ValueError('Unknown diagnostic mode or stage')
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0 or not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('warmup must be nonnegative and repeats must be positive')
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError('timeout_s must be finite and positive')
    source_root, output_root = Path(source_run).resolve(), Path(output_dir).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError('Diagnostic output must be outside the source run')
    environment = _device_environment(gpu_device, stage in GPU_STAGES)
    identifier, root = new_record(output_root)
    report = {'version': 1, 'invocation_id': identifier, 'mode': mode, 'stage': stage,
              'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'report_path': str(root / 'report.json'), 'trials': [],
              'automatic_resource_plan': False, 'configuration_modified': False}
    write_json(root / 'report.json', report)
    session = None
    started = time.perf_counter()
    try:
        config = load_config(str(config_path)).resolved()
        if not getattr(config.operators, stage).enabled:
            raise ValueError('Selected stage must be enabled in the supplied configuration')
        # Preserve virtual-environment interpreter symlinks, including on older configs.
        import yaml
        raw = yaml.safe_load(Path(config_path).read_text())
        original_command = raw.get('operators', {}).get(stage, {}).get('params', {}).get('runner_command')
        if original_command:
            executable = original_command[0]
            getattr(config.operators, stage).params['runner_command'][0] = (
                os.path.abspath(os.path.expanduser(executable)) if '/' in executable else executable)
        supplied = asdict(config)
        write_json(root / 'config.supplied.json', supplied)
        params = supplied['operators'][stage]['params']
        command = params.get('runner_command') or [sys.executable, '-m', 'egoworld.worker']
        builtin = any(command[i:i + 2] == ['-m', 'egoworld.worker'] for i in range(len(command) - 1))
        if not builtin and config.execution.external_runner_mode != 'persistent':
            raise ValueError('Custom diagnostic runners require explicit external_runner_mode=persistent and the --serve protocol')
        timeout = timeout_s if timeout_s is not None else config.execution.stage_timeout_s
        inputs_dir = root / 'inputs'; inputs_dir.mkdir()
        clip, inputs, lineage = freeze_inputs(source_root, clip_id, stage, inputs_dir,
                                              video_manifest=video_manifest, clip_manifest=clip_manifest)
        identity = model_identity(config, stage, inputs_dir)
        snapshot = json.loads(json.dumps(supplied))
        snapshot['execution'].update(measure_gpu_memory=True, external_runner_mode='persistent')
        snapshot['paths'].update(output_root=str(root / 'outputs'), state_db_path=str(root / 'state.db'),
                                 manifest_path=str(inputs_dir), runlog_path=str(root / 'runlog.md'))
        selected = snapshot['operators'][stage]['params']
        if stage in GPU_STAGES:
            if not str(selected.get('device', 'cuda')).startswith('cuda'):
                raise ValueError('A GPU probe/replay requires a CUDA device in the selected stage')
            selected['device'] = 'cuda:0'
            if stage == 'sam2':
                if not str(selected.get('prompting', {}).get('gd_device', 'cuda')).startswith('cuda'):
                    raise ValueError('SAM2 diagnostic requires CUDA GroundingDINO')
                selected.setdefault('prompting', {})['gd_device'] = 'cuda:0'
        selected['debug_dir'] = str(root / 'debug')
        if stage == 'foundationpose':
            lineage['depth_sha256_by_frame'] = _freeze_depth(snapshot, clip, inputs_dir, inputs)
        job = {'clip': clip, 'stage': stage, 'work_dir': str(root / 'prepared'), 'directory': str(root / 'outputs')}
        prepare_started = time.perf_counter()
        prepared = PrepareActor(snapshot).prepare(job)
        if sha256_file(clip['video_path']) != lineage['video']['checksum']:
            raise ValueError('Source video changed during preparation')
        report.update(prepare_seconds=time.perf_counter() - prepare_started,
                      prepared_sha256=sha256_file(prepared['video_path']),
                      input_specification=asdict(get_video_info(prepared['video_path'])),
                      lineage=lineage, model_identity=identity, gpu_environment=environment,
                      diagnostic_overrides={'measure_gpu_memory': True, 'external_runner_mode': 'persistent',
                                            'selected_device': selected.get('device'), 'isolated_output_paths': True})
        report['input_specification']['upstream_rows'] = {name: {file: info.get('rows') for file, info in receipt['files'].items()}
                                                        for name, receipt in inputs.items()}
        report['measurement_fingerprint'] = digest({'identity': identity, 'lineage': lineage,
            'prepared_sha256': report['prepared_sha256'], 'config': supplied, 'gpu_environment': environment})
        job['fingerprint'] = report['measurement_fingerprint']
        write_json(root / 'config.execution.json', snapshot)
        write_json(root / 'report.json', report)
        # Ensure guard and model interpreter both import this worktree's package.
        package_root = str(Path(__file__).resolve().parents[2])
        environment['PYTHONPATH'] = os.pathsep.join(filter(None, [package_root, os.environ.get('PYTHONPATH')]))
        phases = ['cold'] if mode == 'replay' else ['cold'] + ['warmup'] * warmup + ['hot'] * repeats
        writer = WriterActor(snapshot)
        for number, phase in enumerate(phases):
            trial_dir = root / 'trials' / f'{number:03d}-{phase}'; trial_dir.mkdir(parents=True)
            record = {'index': number, 'phase': phase, 'status': 'running', 'directory': str(trial_dir)}
            report['trials'].append(record)
            write_json(root / 'report.json', report)
            trial_job = dict(job, directory=str(trial_dir / 'outputs'), work_dir=str(trial_dir / 'work'))
            request, response = trial_dir / 'request.json', trial_dir / 'response.json'
            write_json(request, {'config': snapshot, 'job': trial_job, 'prepared': prepared,
                                 'inputs': inputs, 'diagnostics': True})
            trial_started = time.perf_counter()
            try:
                if session is None:
                    session = RunnerSession(command, stage, root / 'logs', config.execution.worker_threads,
                                            environment=environment, working_dir=params.get('working_dir'))
                packet = session.run(request, response, timeout)
                record['runner_roundtrip_seconds'] = time.perf_counter() - trial_started
                for name in ('diagnostics', 'runtime'):
                    if name in packet:
                        record[name] = packet[name]
                if 'error' in packet:
                    record['runner_error'] = packet['error']
                    raise RuntimeError(f"{packet['error']['type']}: {packet['error']['message']}")
                if builtin and packet.get('runtime', {}).get('source_tree_hash') != source_identity():
                    raise ValueError('Runner source differs from the diagnostic coordinator')
                receipt = writer.write_stage(trial_job, packet)
                record.update(status='passed', stats=receipt['stats'], files=receipt['files'],
                              output_rows=sum(info.get('rows', 0) for info in receipt['files'].values()))
                if phase == 'hot' and record['stats'].get('model_cache_hit') != 1:
                    raise ValueError('Hot trial did not report model reuse; no warm measurement claim is valid')
                if record.get('diagnostics', {}).get('synchronization_error'):
                    raise RuntimeError(record['diagnostics']['synchronization_error'])
            except (Exception, KeyboardInterrupt) as exc:
                record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              error={'type': type(exc).__name__, 'message': str(exc)},
                              runner_roundtrip_seconds=time.perf_counter() - trial_started)
                raise
            finally:
                record['elapsed_seconds'] = time.perf_counter() - trial_started
                write_json(trial_dir / 'record.json', record)
                write_json(root / 'report.json', report)
        report['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                report.update(status='failed', cleanup_error=str(exc))
        report.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started,
                      recommendations=recommendations(report['trials'], stage))
        report['observed_identity_hash'] = digest({'measurement_fingerprint': report.get('measurement_fingerprint'),
            'runtimes': [trial.get('runtime') for trial in report['trials']],
            'devices': [trial.get('diagnostics', {}).get('device_uuid') for trial in report['trials']]})
        report['retained_artifact_bytes'] = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
        write_json(root / 'report.json', report)
    return report
