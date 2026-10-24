"""Documented input-directory entry point, with durable manifests for resume."""
from dataclasses import asdict
from pathlib import Path
import json

from egoworld.config import load_config
from egoworld.contracts import enabled_stages
from egoworld.io.writers import write_json, write_json_lines
from egoworld.manifests.build_manifest import build_manifests
from egoworld.pipeline.driver import make_run_id, run_pipeline, check_inputs, load_manifest
from egoworld.preflight import runtime_report, asset_report


def launch(config_path, input_dir, *, run_id=None, pattern="**/*.mp4", check_only=False):
    config = load_config(config_path).resolved()
    run_id = run_id or config.run_id or make_run_id()
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id may contain only letters, digits, hyphens and underscores")
    if not enabled_stages(config):
        raise ValueError("Enable at least one model stage")
    sources = sorted(p.resolve() for p in Path(input_dir).glob(pattern) if p.is_file())
    if not sources:
        raise ValueError(f"No videos matched {pattern} in {input_dir}")
    startup = runtime_report(config_path, gpu=not check_only, assets=True)
    result = {"status": "ready" if startup["ok"] else "failed", "run_id": run_id,
              "stages": list(enabled_stages(config)), "videos": len(sources), "startup": startup}
    if check_only or not startup["ok"]:
        return result
    folder = Path(config.paths.manifest_path) / run_id
    video_path, clip_path, source_path = (folder / name for name in ("video_manifest.jsonl", "clip_manifest.jsonl", "sources.json"))
    identity = {"sources": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources],
                "scenedetect": asdict(config.scenedetect)}
    if source_path.exists():
        if json.loads(source_path.read_text()) != identity:
            raise ValueError("Input files or clip settings changed; use a new run_id")
        check_inputs(str(video_path), str(clip_path))
        videos, clips = load_manifest(str(video_path)), load_manifest(str(clip_path))
    else:
        if (Path(config.paths.output_root) / f"run_id={run_id}" / "run_manifest.json").exists():
            raise ValueError("Existing run is missing its launch manifests; restore them or use a new run_id")
        videos, clips = build_manifests([str(p) for p in sources], scenedetect=config.scenedetect)
        write_json_lines(str(video_path), videos)
        write_json_lines(str(clip_path), clips)
        write_json(str(source_path), identity)  # commit marker for this pair of manifests
    coverage = asset_report(config, videos=videos, clips=clips)
    result["startup"]["checks"].extend(coverage["checks"])
    result["startup"]["ok"] = startup["ok"] and coverage["ok"]
    write_json(str(folder / "startup.json"), result["startup"])
    if not coverage["ok"]:
        result["status"] = "failed"
        return result
    config.run_id = run_id
    snapshot = folder / "config.resolved.json"
    write_json(str(snapshot), asdict(config))
    run_pipeline(str(snapshot), str(video_path), str(clip_path), run_id=run_id)
    result.update(status="passed", clips=len(clips), manifest_dir=str(folder),
                  run_dir=str(Path(config.paths.output_root) / f"run_id={run_id}"))
    return result
