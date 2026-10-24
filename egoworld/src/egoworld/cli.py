"""CLI entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoworld.config import load_config
from egoworld.manifests.build_manifest import build_manifests, write_manifest_json


def make_manifest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config.validate()
    video_paths = sorted(str(p.resolve()) for p in Path(args.input_dir).glob(args.glob) if p.is_file())
    if not video_paths:
        raise ValueError(f"No videos matched {args.glob} in {args.input_dir}")
    video_rows, clip_rows = build_manifests(video_paths, split=args.split, scenedetect=config.scenedetect)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_json(str(output_dir / "video_manifest.jsonl"), video_rows)
    write_manifest_json(str(output_dir / "clip_manifest.jsonl"), clip_rows)
    print(json.dumps({"videos": len(video_rows), "clips": len(clip_rows), "output_dir": str(output_dir)}))


def run(args: argparse.Namespace) -> None:
    from egoworld.pipeline.driver import run_pipeline, check_inputs

    if args.check_only:
        load_config(args.config).validate()
        print(json.dumps(check_inputs(args.video_manifest, args.clip_manifest)))
        return
    run_id = run_pipeline(args.config, args.video_manifest, args.clip_manifest, run_id=args.run_id)
    print(json.dumps({"run_id": run_id}))


def doctor(args: argparse.Namespace) -> None:
    from egoworld.preflight import runtime_report

    report = runtime_report(args.config, gpu=args.gpu, stages=args.stages.split(",") if args.stages else None, assets=args.assets)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def inspect(args: argparse.Namespace) -> None:
    from egoworld.acceptance import inspect_run

    print(json.dumps(inspect_run(Path(args.run_dir)), indent=2))


def smoke(args: argparse.Namespace) -> None:
    from egoworld.acceptance import single_gpu_smoke

    print(json.dumps(single_gpu_smoke(args.config, args.input_video, args.output_dir,
                                     args.seconds, args.prompt, args.stages.split(",") if args.stages else None,
                                     profile=args.profile), indent=2))


def plan(args):
    from dataclasses import asdict
    from egoworld.contracts import enabled_stages, STAGE_ARTIFACTS, GPU_STAGES
    config = load_config(args.config).resolved()
    graph = enabled_stages(config)
    print(json.dumps({"stages": [{"name": name, "after": list(dependencies),
        "resource": "gpu" if name in GPU_STAGES else "cpu", "outputs": list(STAGE_ARTIFACTS[name])}
        for name, dependencies in graph.items()], "execution": asdict(config.execution),
        "limits": asdict(config.backpressure)}, indent=2))


def schema(args):
    from egoworld.contracts import OUTPUT_SCHEMA_VERSION, schemas
    print(json.dumps({"version": OUTPUT_SCHEMA_VERSION,
        "artifacts": {name: {field.name: str(field.type) for field in definition}
                      for name, definition in schemas().items()}}, indent=2))


def launch(args):
    from egoworld.launch import launch as launch_pipeline
    result = launch_pipeline(args.config, args.input_dir, run_id=args.run_id,
                             pattern=args.glob, check_only=args.check_only)
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


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
    run_cmd.add_argument("--run-id", help="Reuse an existing ID to resume; use a new ID after changing models")
    run_cmd.add_argument("--check-only", action="store_true", help="Validate inputs without Ray, CUDA or state changes")
    run_cmd.set_defaults(func=run)

    doctor_cmd = sub.add_parser("doctor", help="Check CPU prerequisites, or add --gpu for model readiness")
    doctor_cmd.add_argument("--config", default="egoworld/configs/example.json")
    doctor_cmd.add_argument("--gpu", action="store_true")
    doctor_cmd.add_argument("--assets", action="store_true", help="Check enabled model/data assets without loading tensors")
    doctor_cmd.add_argument("--stages", help="Only check these enabled stages in this Python environment")
    doctor_cmd.set_defaults(func=doctor)

    plan_cmd = sub.add_parser("plan", help="Print the complete dependency graph and resource limits without model imports")
    plan_cmd.add_argument("--config", default="egoworld/configs/example.json")
    plan_cmd.set_defaults(func=plan)

    schema_cmd = sub.add_parser("schema", help="Print output schemas directly from the runtime contracts")
    schema_cmd.set_defaults(func=schema)

    launch_cmd = sub.add_parser("launch", help="Preflight, build/reuse manifests, run or resume from an input directory")
    launch_cmd.add_argument("--config", required=True)
    launch_cmd.add_argument("--input-dir", required=True)
    launch_cmd.add_argument("--run-id", help="Use the same ID to resume")
    launch_cmd.add_argument("--glob", default="**/*.mp4")
    launch_cmd.add_argument("--check-only", action="store_true", help="CPU dependency/assets/input-directory checks; no Ray/CUDA or writes")
    launch_cmd.set_defaults(func=launch)

    inspect_cmd = sub.add_parser("inspect", help="Summarize completed clip outputs and QC")
    inspect_cmd.add_argument("--run-dir", required=True)
    inspect_cmd.set_defaults(func=inspect)

    smoke_cmd = sub.add_parser("smoke", help="Single-GPU acceptance on a short real video, including resume")
    smoke_cmd.add_argument("--config", default="egoworld/configs/example.json")
    smoke_cmd.add_argument("--input-video", required=True)
    smoke_cmd.add_argument("--output-dir", default="./single-gpu-check")
    smoke_cmd.add_argument("--seconds", type=float, default=3.0)
    smoke_cmd.add_argument("--prompt", default="left hand . right hand . cup .")
    smoke_cmd.add_argument("--stages", help="Comma-separated target stages; dependencies are enabled automatically")
    smoke_cmd.add_argument("--profile", choices=("default", "16gb"), default="default",
                           help="16gb: isolated model processes, batch=1, two Fast3R views; requires <=3s sample; memory fit still needs measurement")
    smoke_cmd.set_defaults(func=smoke)

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
