"""Ray pipeline driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4
import json
import time

from egoworld.config import load_config
from egoworld.io.paths import run_dir
from egoworld.io.writers import write_json, write_run_manifest
from egoworld.contracts import OUTPUT_SCHEMA_VERSION



@dataclass
class ClipTask:
    clip_id: str
    video_id: str
    video_path: str
    start_s: float
    end_s: float
    frame_start: int
    frame_end: int
    scenedetect_failed: bool
    retry_count: int = 0


def _load_json_lines(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_parquet(path: str) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for parquet manifests") from exc
    table = pq.read_table(path)
    return table.to_pylist()


def load_manifest(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".parquet"):
        return _load_parquet(path)
    return _load_json_lines(path)


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def _build_clip_tasks(clips: List[Dict[str, Any]], video_manifest: Dict[str, Dict[str, Any]]) -> List[ClipTask]:
    tasks: List[ClipTask] = []
    seen = set()
    for clip in clips:
        if clip["clip_id"] in seen:
            raise ValueError(f"Duplicate clip_id: {clip['clip_id']}")
        seen.add(clip["clip_id"])
        if not (0 <= float(clip["start_s"]) < float(clip["end_s"])) or not (0 <= int(clip["frame_start"]) < int(clip["frame_end"])):
            raise ValueError(f"Invalid time/frame interval: {clip['clip_id']}")
        video = video_manifest.get(clip["video_id"])
        if not video:
            raise RuntimeError(f"clip_manifest references missing video_id: {clip['video_id']}")
        if not video.get("path"):
            raise RuntimeError(f"video_manifest has empty path for video_id: {clip['video_id']}")
        tasks.append(
            ClipTask(
                clip_id=clip["clip_id"],
                video_id=clip["video_id"],
                video_path=video["path"],
                start_s=float(clip["start_s"]),
                end_s=float(clip["end_s"]),
                frame_start=int(clip["frame_start"]),
                frame_end=int(clip["frame_end"]),
                scenedetect_failed=bool(clip.get("scenedetect_failed", False)),
                retry_count=int(clip.get("retry_count", 0)),
            )
        )
    return tasks


def check_inputs(video_manifest_path: str, clip_manifest_path: str) -> Dict[str, Any]:
    videos = _load_video_index(video_manifest_path)
    clips = load_manifest(clip_manifest_path)
    if not videos or not clips:
        raise ValueError("No input videos or clips; check the input directory and glob")
    tasks = _build_clip_tasks(clips, videos)
    for task in tasks:
        if not Path(task.video_path).is_file():
            raise ValueError(f"Input video not found: {task.video_path}")
    return {"videos": len(videos), "clips": len(tasks),
            "scheduled_seconds": sum(t.end_s - t.start_s for t in tasks)}


def _load_video_index(video_manifest_path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_manifest(video_manifest_path)
    return {row["video_id"]: row for row in rows}


def _clip_to_dict(task: ClipTask) -> Dict[str, Any]:
    return {
        "clip_id": task.clip_id,
        "video_id": task.video_id,
        "video_path": task.video_path,
        "start_s": task.start_s,
        "end_s": task.end_s,
        "frame_start": task.frame_start,
        "frame_end": task.frame_end,
        "scenedetect_failed": task.scenedetect_failed,
        "retry_count": task.retry_count,
    }


def iter_clips(path):
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    else:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def run_pipeline(config_path, video_manifest_path, clip_manifest_path, *, run_id=None, backend_factory=None):
    from egoworld.contracts import enabled_stages, GPU_STAGES
    from egoworld.pipeline.backend import RayBackend
    from egoworld.pipeline.engine import Engine
    from egoworld.pipeline.state_store import init_stage_db, run_counts
    from egoworld.utils.provenance import provenance

    config = load_config(config_path).resolved()
    config.run_id = run_id or config.run_id or make_run_id()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in config.run_id):
        raise ValueError("run_id may contain only letters, digits, hyphens and underscores")
    video_index = _load_video_index(video_manifest_path)
    if not video_index:
        raise ValueError("No input videos; build nonempty manifests first")
    root = run_dir(config.paths.output_root, config.run_id)
    identity = provenance(config, video_manifest_path, clip_manifest_path)
    fingerprint = identity["fingerprint"]
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("config_fingerprint") != fingerprint:
        raise ValueError("Input/model/code settings changed; choose a new run_id")
    root.mkdir(parents=True, exist_ok=True)
    init_stage_db(config.paths.state_db_path)
    manifest = config.to_run_manifest() | {"created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint, "config": asdict(config), "provenance": identity,
        "output_schema_version": OUTPUT_SCHEMA_VERSION, "stages": enabled_stages(config)}
    write_run_manifest(str(manifest_path), manifest)

    def tasks():
        seen = set()
        for row in iter_clips(clip_manifest_path):
            task = _build_clip_tasks([row], video_index)[0]
            if task.clip_id in seen:
                raise ValueError(f"Duplicate clip_id: {task.clip_id}")
            seen.add(task.clip_id)
            for identifier in (task.clip_id, task.video_id):
                if "/" in identifier or "\\" in identifier or identifier in (".", ".."):
                    raise ValueError("Manifest identifiers cannot contain path separators")
            result = _clip_to_dict(task)
            video = video_index[task.video_id]
            result["fps"] = float(video.get("fps") or ((task.frame_end - task.frame_start) / (task.end_s - task.start_s)))
            if abs(task.start_s - task.frame_start / result["fps"]) > 1e-5 or abs(task.end_s - task.frame_end / result["fps"]) > 1e-5:
                raise ValueError(f"Frame/time alignment mismatch: {task.clip_id}")
            yield result

    source = iter(tasks())
    try:
        first = next(source)
    except StopIteration:
        raise ValueError("No input clips; build nonempty manifests first")
    import itertools
    backend, ray = None, None
    started = time.perf_counter()
    summary = {"status": "failed", "run_id": config.run_id}
    events_path = root / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as events:
        def emit(**event):
            event.update(ts=datetime.now(timezone.utc).isoformat(), run_id=config.run_id)
            events.write(json.dumps(event, ensure_ascii=False) + "\n"); events.flush()
        try:
            if backend_factory is None:
                import ray
                ray.init(ignore_reinit_error=True)
                required = config.num_gpus if set(enabled_stages(config)) & GPU_STAGES else 0
                if ray.cluster_resources().get("GPU", 0) < required:
                    raise RuntimeError(f"Requested {required} GPU(s), but Ray sees fewer; run doctor --gpu")
                backend = RayBackend(asdict(config))
            else:
                backend = backend_factory(asdict(config))
            engine = Engine(config, backend, fingerprint, emit)
            summary.update(engine.run(itertools.chain([first], source)))
            summary["status"] = "failed" if summary["failed_clips"] else "passed"
        except BaseException as exc:
            summary["error"] = str(exc)
            summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            raise
        finally:
            if backend:
                backend.close()
            if ray:
                ray.shutdown()
            summary.update(elapsed_seconds=round(time.perf_counter() - started, 3), states=run_counts(config.paths.state_db_path, config.run_id))
            summary["scheduled_frames_per_wall_second"] = summary.get("successful_scheduled_frames", 0) / max(.001, summary["elapsed_seconds"])
            write_json(str(root / "run_summary.json"), summary)
    if summary["failed_clips"]:
        raise RuntimeError(f"Pipeline {config.run_id} failed for {summary['failed_clips']} clip(s); see {root / 'events.jsonl'}")
    return config.run_id
