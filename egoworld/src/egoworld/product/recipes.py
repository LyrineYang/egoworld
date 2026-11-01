"""Output-driven recipes compiled onto the existing Ray pipeline."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

from egoworld.config import load_config
from egoworld.contracts import OUTPUT_SCHEMA_VERSION, GPU_STAGES
from egoworld.io.writers import write_json, write_json_lines
from egoworld.product.cache import StageCache, stage_identity, runtime_identity, file_signature
from egoworld.product.registry import DATA_TYPES, resolve_outputs
from egoworld.utils.hashing import sha256_file


def compile_recipe(path):
    path = Path(path).resolve()
    recipe = yaml.safe_load(path.read_text())
    allowed = {"version", "config", "inputs", "outputs", "output_dir", "cache", "operators", "calibration"}
    if not isinstance(recipe, dict) or recipe.get("version") != 1 or set(recipe) - allowed:
        raise ValueError("Expected recipe version 1 and documented recipe fields")
    def absolute(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (path.parent / p).resolve())
    selections, graph = resolve_outputs(recipe.get("outputs"))
    config = load_config(absolute(recipe["config"]))
    overrides = recipe.get("operators", {})
    if not isinstance(overrides, dict) or set(overrides) - set(vars(config.operators)):
        raise ValueError("Recipe operators must name existing kernels")
    def merge(target, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    for stage, operator in vars(config.operators).items():
        operator.enabled = stage in graph
        if stage in overrides:
            if not isinstance(overrides[stage], dict):
                raise ValueError("Kernel overrides must be parameter objects")
            merge(operator.params, overrides[stage])
    inputs = recipe.get("inputs")
    if not isinstance(inputs, list) or not inputs or any(not isinstance(p, str) for p in inputs):
        raise ValueError("Recipe inputs must list video paths")
    inputs = [absolute(p) for p in inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate input videos")
    for video in inputs:
        if not Path(video).is_file():
            raise ValueError(f"Input video not found: {video}")
    cache = recipe.get("cache", {})
    if not isinstance(cache, dict) or set(cache) - {"root", "read", "write"}:
        raise ValueError("Cache options are root, read, write")
    if any(not isinstance(cache.get(flag, True), bool) for flag in ("read", "write")):
        raise ValueError("Cache read/write options must be boolean")
    root = absolute(recipe.get("output_dir", "./egoworld-results"))
    cache = {"root": absolute(cache.get("root", str(Path(root) / "cache"))),
             "read": cache.get("read", True), "write": cache.get("write", True)}
    calibration = recipe.get("calibration")
    world = [name for name in selections if name.startswith("world_")]
    requirements = []
    if world:
        if not calibration:
            requirements.append("World outputs need calibration: {method: rgbd, assets_manifest: ...} or {path: ...}")
        elif not isinstance(calibration, dict):
            raise ValueError("Calibration must be an object")
        elif calibration.get("method") == "rgbd":
            if set(calibration) - {"method", "assets_manifest", "threshold_m", "min_inlier_fraction", "confidence_quantile"}:
                raise ValueError("Unknown RGB-D calibration option")
            calibration = {**calibration, "assets_manifest": absolute(calibration["assets_manifest"])}
            config.operators.fast3r.params["save_correspondences"] = True
            if not Path(calibration["assets_manifest"]).is_file():
                requirements.append("RGB-D calibration assets manifest is missing")
            if "world_hand" in world:
                requirements.append("world_hand needs measured hand_camera_alignments; RGB-D scene calibration alone is insufficient")
        elif set(calibration) == {"path"}:
            calibration = {"path": absolute(calibration["path"])}
            if not Path(calibration["path"]).is_file():
                requirements.append("World calibration file is missing")
        else:
            raise ValueError("Calibration must specify a path or method=rgbd")
    config.validate()
    compiled = {"version": 1, "config": asdict(config), "inputs": inputs,
                "outputs": {k: list(v) for k, v in selections.items()}, "output_dir": root,
                "cache": cache, "calibration": calibration}
    plan = {"outputs": list(selections), "kernels": [{"name": stage, "requires": list(deps),
            "resource": "gpu" if stage in GPU_STAGES else "cpu"} for stage, deps in graph.items()],
            "derived_kernels": ["world"] if world else [], "requirements": requirements,
            "output_dir": root, "cache": cache,
            "note": "Selection changes exported fields, not model workload. Dependencies remain available as raw artifacts."}
    return config, compiled, plan


def export_selected(folder, selections, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, fields in selections.items():
        spec = DATA_TYPES[name]
        if spec.kernel == "world":
            continue
        source = Path(folder) / (spec.artifact if spec.artifact.endswith(".npz") else spec.artifact + ".parquet")
        if source.suffix == ".npz":
            target = destination / f"{name}.npz"
            with np.load(source, allow_pickle=False) as data:
                np.savez_compressed(target, **{field: data[field] for field in fields})
                count = len(data["points"])
            result[name] = {"file": str(target), "fields": list(fields), "points": count}
        else:
            parquet = pq.ParquetFile(source)
            # Field selection cannot discard the coordinate/unit context or
            # the names and units needed to interpret robot joint values.
            context_fields = ("frame_index", "timestamp_s", "object_id", "handedness", "coord_frame",
                              "length_unit", "scale_status", "joint_order", "robot_id", "joint_names",
                              "joint_units", "source_scale_status")
            identity = [key for key in context_fields if key in parquet.schema_arrow.names]
            columns = list(dict.fromkeys(identity + list(fields)))
            target = destination / f"{name}.parquet"
            selected = parquet.read(columns=columns)
            pq.write_table(selected, target, compression="zstd")
            result[name] = {"file": str(target), "fields": columns, "rows": selected.num_rows}
        result[name]["sha256"] = sha256_file(str(target))
        result[name]["source_sha256"] = sha256_file(str(source))
    return result


def run_recipe(path, *, run_id=None, backend_factory=None):
    from egoworld.manifests.build_manifest import build_manifests
    from egoworld.pipeline.driver import make_run_id, run_pipeline, _build_clip_tasks, _clip_to_dict
    from egoworld.pipeline.artifacts import load_stage, WriterActor
    from egoworld.preflight import runtime_report
    from egoworld.utils.provenance import provenance
    from egoworld.product.world import calibrate_rgbd, fuse_world
    config, compiled, plan = compile_recipe(path)
    if plan["requirements"]:
        raise ValueError("; ".join(plan["requirements"]))
    run_id = run_id or make_run_id()
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid recipe run ID")
    root = Path(compiled["output_dir"]) / f"run_id={run_id}"
    recipe_file = root / "recipe.json"
    if recipe_file.is_file() and json.loads(recipe_file.read_text()) != compiled:
        raise ValueError("Recipe changed; use a new run ID and reuse the stage cache")
    root.mkdir(parents=True, exist_ok=True)
    write_json(str(recipe_file), compiled)
    config.run_id = run_id
    config.paths.output_root = str(root / "model-output")
    config.paths.state_db_path = str(root / "state.db")
    config.paths.manifest_path = str(root / "manifests")
    config.paths.runlog_path = str(root / "runlog.md")
    cfg_file = root / "config.json"
    write_json(str(cfg_file), asdict(config))
    config = load_config(str(cfg_file)).resolved()
    videos, clips = build_manifests(compiled["inputs"], scenedetect=config.scenedetect)
    video_file, clip_file = root / "videos.jsonl", root / "clips.jsonl"
    write_json_lines(str(video_file), videos)
    write_json_lines(str(clip_file), clips)
    index = {v["video_id"]: v for v in videos}
    jobs = [_clip_to_dict(task) for task in _build_clip_tasks(clips, index)]
    for clip in jobs:
        clip["fps"] = float(index[clip["video_id"]]["fps"])
        clip["video_sha256"] = index[clip["video_id"]]["checksum"]
    identity = provenance(config, str(video_file), str(clip_file))
    fingerprint = identity["fingerprint"]
    model_root = Path(config.paths.output_root) / f"run_id={run_id}"
    if (model_root / "run_manifest.json").exists():
        previous = json.loads((model_root / "run_manifest.json").read_text())
        if previous["config_fingerprint"] != fingerprint:
            raise ValueError("Source/model/input identity changed; use a new recipe run ID")
    model_root.mkdir(parents=True, exist_ok=True)
    graph = {k["name"]: k["requires"] for k in plan["kernels"]}
    runtimes = {}
    for stage in graph:
        command = getattr(config.operators, stage).params.get("runner_command", [sys.executable])
        key = tuple(command)
        if key not in runtimes:
            runtimes[key] = runtime_identity(command)
        runtimes[stage] = runtimes[key]
    cache = StageCache(compiled["cache"]["root"])
    result = {"version": 1, "status": "running", "run_id": run_id, "plan": plan,
              "run_dir": str(root), "cache_hits": [], "computed": [], "exports": {}}
    manifest = {**config.to_run_manifest(), "config_fingerprint": fingerprint,
                "provenance": identity, "config": asdict(config), "stages": list(graph),
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat()}
    try:
        missing = set()
        for clip in jobs:
            folder = model_root / f"video_id={clip['video_id']}" / f"clip_id={clip['clip_id']}"
            receipts = {}
            for stage, deps in graph.items():
                if any(d not in receipts for d in deps):
                    missing.add(stage)
                    continue
                key = stage_identity(config, identity, stage, clip,
                    {d: file_signature(receipts[d]) for d in deps}, runtimes[stage])
                receipt = load_stage(folder, stage, fingerprint, True)
                origin = "same_run"
                if receipt is None and compiled["cache"]["read"]:
                    receipt = cache.restore(key, folder, fingerprint)
                    origin = "stage_cache"
                if receipt:
                    receipts[stage] = receipt
                    result["cache_hits"].append({"clip_id": clip["clip_id"], "kernel": stage, "origin": origin})
                else:
                    missing.add(stage)
            if len(receipts) == len(graph):
                WriterActor(asdict(config)).finalize({"clip": clip, "directory": str(folder), "fingerprint": fingerprint}, receipts, {})
        if missing:
            if backend_factory is None:
                readiness = runtime_report(str(cfg_file), gpu=bool(missing & GPU_STAGES), stages=sorted(missing), assets=True)
                write_json(str(root / "readiness.json"), readiness)
                if not readiness["ok"]:
                    raise ValueError("Required kernels are not ready; see readiness.json")
            run_pipeline(str(cfg_file), str(video_file), str(clip_file), run_id=run_id, backend_factory=backend_factory)
        else:
            write_json(str(model_root / "run_manifest.json"), manifest)
        for clip in jobs:
            folder = model_root / f"video_id={clip['video_id']}" / f"clip_id={clip['clip_id']}"
            receipts = {}
            for stage, deps in graph.items():
                receipt = load_stage(folder, stage, fingerprint, True)
                if receipt is None:
                    raise ValueError(f"Kernel {stage} did not commit verified outputs")
                key = stage_identity(config, identity, stage, clip,
                    {d: file_signature(receipts[d]) for d in deps}, runtimes[stage])
                if compiled["cache"]["write"]:
                    cache.publish(key, folder, fingerprint)
                receipts[stage] = receipt
                if not any(h["clip_id"] == clip["clip_id"] and h["kernel"] == stage for h in result["cache_hits"]):
                    result["computed"].append({"clip_id": clip["clip_id"], "kernel": stage})
            exported = export_selected(folder, compiled["outputs"], root / "exports" / clip["clip_id"])
            world_outputs = [n for n in compiled["outputs"] if n.startswith("world_")]
            if world_outputs:
                calibration = compiled["calibration"]
                if calibration.get("method") == "rgbd":
                    calibration_path = root / "calibrations" / f"{clip['clip_id']}.json"
                    calibrate_rgbd(folder, calibration["assets_manifest"], calibration_path,
                        threshold_m=calibration.get("threshold_m", .03),
                        min_inlier_fraction=calibration.get("min_inlier_fraction", .5),
                        confidence_quantile=calibration.get("confidence_quantile", .5))
                else:
                    calibration_path = calibration["path"]
                world = fuse_world(folder, calibration_path, world_outputs, root / "exports" / clip["clip_id"] / "world")
                exported["world"] = world
            result["exports"][clip["clip_id"]] = exported
        empty_outputs = [{"clip_id": clip_id, "type": name} for clip_id, exported in result["exports"].items()
                         for name, info in exported.items() if info.get("rows", 1) == 0 or info.get("points", 1) == 0]
        result["empty_outputs"] = empty_outputs
        result["status"] = "partial" if empty_outputs or any(x.get("world", {}).get("status") == "partial" for x in result["exports"].values()) else "passed"
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        write_json(str(root / "data_manifest.json"), result)
    return result
