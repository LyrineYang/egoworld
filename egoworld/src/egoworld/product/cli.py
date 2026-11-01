"""User-facing data selection; scheduling and probe commands remain separate."""
import json


def execute(args):
    try:
        if args.data_command == "catalog":
            from egoworld.product.registry import catalog
            result = catalog()
        elif args.data_command == "plan":
            from egoworld.product.recipes import compile_recipe
            _, _, result = compile_recipe(args.recipe)
        elif args.data_command == "run":
            from egoworld.product.recipes import run_recipe
            result = run_recipe(args.recipe, run_id=args.run_id)
        elif args.data_command == "calibrate":
            from egoworld.product.world import calibrate_rgbd
            result = calibrate_rgbd(args.clip_dir, args.rgbd_assets, args.output,
                                   threshold_m=args.threshold_m, min_inlier_fraction=args.min_inlier_fraction,
                                   confidence_quantile=args.confidence_quantile)
        elif args.data_command == "calibrate-hand":
            from egoworld.product.world import calibrate_hand
            result = calibrate_hand(args.clip_dir, args.calibration, args.measurements, args.output,
                                    threshold_m=args.threshold_m)
        else:
            from egoworld.product.world import fuse_world
            result = fuse_world(args.clip_dir, args.calibration, args.outputs.split(","), args.output_dir)
        print(json.dumps(result, indent=2))
        if isinstance(result, dict) and result.get("status") in ("failed", "interrupted", "partial"):
            raise SystemExit(1)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        raise SystemExit(1) from None


def add_parser(sub):
    parser = sub.add_parser("data", help="Choose data types, compose kernels and reuse verified outputs")
    commands = parser.add_subparsers(dest="data_command", required=True)
    catalog = commands.add_parser("catalog", help="List selectable data types and fields")
    catalog.set_defaults(func=execute)
    for name in ("plan", "run"):
        command = commands.add_parser(name, help="Explain a recipe without GPU work" if name == "plan" else "Execute a recipe with stage caching")
        command.add_argument("--recipe", required=True)
        if name == "run":
            command.add_argument("--run-id", help="Reuse only for an unchanged recipe; new recipes can reuse the stage cache")
        command.set_defaults(func=execute)
    command = commands.add_parser("calibrate", help="Fit a metric scene alignment using aligned RGB-D correspondences")
    command.add_argument("--clip-dir", required=True)
    command.add_argument("--rgbd-assets", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--threshold-m", type=float, default=.03)
    command.add_argument("--min-inlier-fraction", type=float, default=.5)
    command.add_argument("--confidence-quantile", type=float, default=.5,
                         help="Discard lower-confidence source points before fitting; default keeps the upper half")
    command.set_defaults(func=execute)
    command = commands.add_parser("calibrate-hand", help="Fit per-observation metric hand alignment from measured camera joints")
    command.add_argument("--clip-dir", required=True)
    command.add_argument("--calibration", required=True)
    command.add_argument("--measurements", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--threshold-m", type=float, default=.02)
    command.set_defaults(func=execute)
    command = commands.add_parser("fuse", help="Export calibrated world data from a completed clip")
    command.add_argument("--clip-dir", required=True)
    command.add_argument("--calibration", required=True)
    command.add_argument("--outputs", default="world_camera,world_scene")
    command.add_argument("--output-dir", required=True)
    command.set_defaults(func=execute)
