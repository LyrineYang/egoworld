"""CLI entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from egoworld.config import load_config
from egoworld.manifests.build_manifest import build_manifests, write_manifest_json
from egoworld.pipeline.driver import run_pipeline


def make_manifest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    video_paths = [str(p) for p in Path(args.input_dir).glob(args.glob)]
    video_rows, clip_rows = build_manifests(video_paths, split=args.split, scenedetect=config.scenedetect)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_json(str(output_dir / "video_manifest.jsonl"), video_rows)
    write_manifest_json(str(output_dir / "clip_manifest.jsonl"), clip_rows)


def run(args: argparse.Namespace) -> None:
    run_pipeline(args.config, args.video_manifest, args.clip_manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="egoworld pipeline")
    sub = parser.add_subparsers(dest="command")

    manifest = sub.add_parser("make-manifest", help="Build manifests")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--input-dir", required=True)
    manifest.add_argument("--glob", default="**/*.mp4")
    manifest.add_argument("--output-dir", required=True)
    manifest.add_argument("--split", default="train")
    manifest.set_defaults(func=make_manifest)

    run_cmd = sub.add_parser("run", help="Run pipeline")
    run_cmd.add_argument("--config", required=True)
    run_cmd.add_argument("--video-manifest", required=True)
    run_cmd.add_argument("--clip-manifest", required=True)
    run_cmd.set_defaults(func=run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
