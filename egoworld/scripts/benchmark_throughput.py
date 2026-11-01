"""Controlled stage ablations using existing replay/probe; never edits input config.

Run from a frozen source snapshot for long experiment matrices. Each failed
attempt remains in the index; no fastest-trial selection or implicit retries.
"""
import argparse
import json
from pathlib import Path
from egoworld.diagnostics.replay import run_diagnostic
from egoworld.io.writers import write_json
from egoworld.utils.hashing import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--source-run', required=True)
    p.add_argument('--clip-id', required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variants', nargs='+', default=['baseline', 'profile', 'split'], choices=['baseline', 'profile', 'split'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hamer-batch-size', type=int)
    args = p.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    original_hash = sha256_file(args.config)
    entries = []
    for variant in args.variants:
        snapshot = json.loads(json.dumps(config))
        snapshot.setdefault('execution', {}).update(phase_profiling=variant != 'baseline', split_execution=variant == 'split')
        if args.hamer_batch_size is not None:
            snapshot['operators']['hamer']['params']['batch_size'] = args.hamer_batch_size
        target = root/f'{args.stage}-{variant}.json'; write_json(target, snapshot)
        report = run_diagnostic(str(target), args.source_run, args.clip_id, args.stage, str(root/f'{args.stage}-{variant}'),
                                mode='probe', warmup=1, repeats=args.repeats, random_seed=args.seed)
        entries.append({'variant': variant, 'report_path': report['report_path'], 'status': report['status'],
                        'report_sha256': sha256_file(report['report_path']), 'error': report.get('error')})
        write_json(root/f'{args.stage}-index.json', entries)
        print(json.dumps(entries[-1]), flush=True)
    if sha256_file(args.config) != original_hash:
        raise RuntimeError('Input config changed during experiment')
    if any(e['status'] != 'passed' for e in entries):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
