"""Verify source identities and freeze upstream commits for one explicit replay."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from egoworld.contracts import OUTPUT_SCHEMA_VERSION, STAGE_DEPENDENCIES, validate_rows
from egoworld.io.writers import write_json
from egoworld.pipeline.artifacts import load_stage
from egoworld.pipeline.driver import load_manifest, _build_clip_tasks, _clip_to_dict
from egoworld.utils.hashing import sha256_file
from egoworld.utils.provenance import provenance


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ancestors(stage):
    result = []
    def visit(name):
        for parent in STAGE_DEPENDENCIES[name]:
            visit(parent)
            if parent not in result:
                result.append(parent)
    visit(stage)
    return result


def _manifest_file(root, manifest, name, explicit):
    expected = manifest['provenance']['input_manifests'][name]
    paths = [Path(explicit)] if explicit else []
    if not explicit:
        # New runs retain exact original bytes for each invocation.
        paths += sorted((root / 'invocations').glob(f'*/{name}.*'))
        if manifest.get('input_manifest_paths', {}).get(name):
            paths.append(Path(manifest['input_manifest_paths'][name]))
        # The older single_gpu_smoke layout.
        paths.append(root.parent.parent / f'{name}.jsonl')
        config = manifest.get('config', {})
        folder = config.get('paths', {}).get('manifest_path')
        if folder:
            filename = 'video_manifest.jsonl' if name == 'videos' else 'clip_manifest.jsonl'
            paths.extend([Path(folder) / filename, Path(folder) / manifest['run_id'] / filename])
    for path in paths:
        if path.is_file() and sha256_file(str(path)) == expected:
            return path.resolve()
    option = '--video-manifest' if name == 'videos' else '--clip-manifest'
    raise ValueError(f'Cannot verify original {name} manifest; supply {option} matching the source run hash')


def freeze_inputs(source_run, clip_id, stage, destination, *, video_manifest=None, clip_manifest=None):
    import pyarrow.parquet as pq
    root, destination = Path(source_run).resolve(), Path(destination)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    if manifest['output_schema_version'] != OUTPUT_SCHEMA_VERSION:
        raise ValueError('Source output schema is incompatible')
    fingerprint = manifest['config_fingerprint']
    copies = {}
    for name, explicit in (('videos', video_manifest), ('clips', clip_manifest)):
        original = _manifest_file(root, manifest, name, explicit)
        target = destination / (name + original.suffix)
        shutil.copyfile(original, target)
        if sha256_file(str(target)) != manifest['provenance']['input_manifests'][name]:
            raise ValueError('Source manifest changed during copy')
        copies[name] = target
    videos = load_manifest(str(copies['videos']))
    clips = [row for row in load_manifest(str(copies['clips'])) if row['clip_id'] == clip_id]
    if len(clips) != 1:
        raise ValueError('Replay requires exactly one matching clip_id')
    selected_videos = [row for row in videos if row['video_id'] == clips[0]['video_id']]
    if len(selected_videos) != 1:
        raise ValueError('Replay requires exactly one matching source video')
    video = selected_videos[0]
    clip = _clip_to_dict(_build_clip_tasks(clips, {video['video_id']: video})[0])
    clip['fps'] = float(video['fps'])
    for field in ('clip_id', 'video_id'):
        if not clip[field] or Path(clip[field]).name != clip[field] or '\\' in clip[field] or clip[field] in ('.', '..'):
            raise ValueError('Invalid source identifier')
    if abs(clip['start_s'] - clip['frame_start'] / clip['fps']) > 1e-5 or abs(clip['end_s'] - clip['frame_end'] / clip['fps']) > 1e-5:
        raise ValueError('Source frame/time alignment mismatch')
    if not video.get('checksum') or sha256_file(clip['video_path']) != video['checksum']:
        raise ValueError('Source video checksum missing or changed; rebuild a traceable source run')
    folder = root / f"video_id={clip['video_id']}" / f"clip_id={clip_id}"
    inputs, lineage = {}, {}
    for dependency in ancestors(stage):
        receipt = load_stage(folder, dependency, fingerprint, verify_hashes=True)
        if receipt is None or receipt.get('stage') != dependency:
            raise ValueError(f'Missing or corrupt upstream commit: {dependency}')
        target = destination / 'upstream' / dependency
        target.mkdir(parents=True)
        for name, info in receipt['files'].items():
            shutil.copyfile(folder / name, target / name)
            if sha256_file(str(target / name)) != info['sha256']:
                raise ValueError(f'Upstream changed during copy: {dependency}/{name}')
            if name.endswith('.parquet'):
                validate_rows(Path(name).stem, pq.ParquetFile(target / name).read().to_pylist(), clip)
        # Bind reads to the verified copy, never the directory embedded in the old receipt.
        inputs[dependency] = dict(receipt, directory=str(target.resolve()))
        write_json(target / f'stage_{dependency}.json', inputs[dependency])
        lineage[dependency] = {'fingerprint': fingerprint, 'files': receipt['files']}
    write_json(destination / 'source_run_manifest.json', manifest)
    return clip, inputs, {'source_run': str(root), 'source_fingerprint': fingerprint,
        'input_manifest_hashes': manifest['provenance']['input_manifests'],
        'video': video, 'upstream': lineage}


def model_identity(config, stage, inputs_dir):
    """Hash configured assets plus native adapters' implicit local assets."""
    selected = copy.deepcopy(config)
    for name, operator in vars(selected.operators).items():
        operator.enabled = name == stage
    identity = provenance(selected, str(next(inputs_dir.glob('videos.*'))), str(next(inputs_dir.glob('clips.*'))))
    params = getattr(config.operators, stage).params
    extra = []
    if stage == 'hamer':
        working = Path(params.get('working_dir', Path.cwd()))
        extra += [working / '_DATA/data/mano/MANO_RIGHT.pkl', working / '_DATA/data/mano_mean_params.npz',
                  Path(params['checkpoint']).parent.parent / 'model_config.yaml']
    elif stage == 'foundationpose':
        extra += list((Path(params.get('repo_path', '')) / 'weights').glob('*/*'))
    elif stage == 'dex_retarget':
        import yaml
        spec = yaml.safe_load(Path(params['config']).read_text())['retargeting']
        urdf = Path(spec['urdf_path'])
        extra += [urdf if urdf.is_absolute() else Path(params.get('urdf_dir', '.')) / urdf]
    identity['implicit_assets'] = {str(path.resolve()): sha256_file(str(path)) for path in extra if path.is_file()}
    identity['upstream_sources'] = {}
    for key in ('working_dir', 'repo_path'):
        if not params.get(key):
            continue
        path = Path(params[key])
        try:
            commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            diff = subprocess.check_output(['git', '-C', str(path), 'diff', 'HEAD', '--', '*.py'])
            untracked = subprocess.check_output(['git', '-C', str(path), 'ls-files', '--others', '--exclude-standard', '--', '*.py'], text=True).splitlines()
            identity['upstream_sources'][key] = {'commit': commit, 'python_diff_sha256': hashlib.sha256(diff).hexdigest(),
                'untracked_python': {name: sha256_file(str(path / name)) for name in untracked}}
        except (OSError, subprocess.CalledProcessError):
            identity['upstream_sources'][key] = {'commit': 'unavailable'}
    identity['identity_hash'] = digest(identity)
    return identity
