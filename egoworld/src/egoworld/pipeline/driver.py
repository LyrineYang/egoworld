"""Ray pipeline driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import time

from egoworld.config import PipelineConfig, load_config
from egoworld.io.paths import clip_dir, run_dir
from egoworld.io.writers import write_json, write_parquet_table, write_run_manifest
from egoworld.manifests.schema import FIELD_SPECS
from egoworld.pipeline.queues import enforce_in_flight
from egoworld.pipeline.scheduler import sort_clips_by_duration
from egoworld.pipeline.state_store import (
    bulk_insert_pending,
    init_db,
    mark_dead_letter,
    upsert_clip_status,
)
from egoworld.utils.errors import classify_error
from egoworld.operators.sam2_op import Sam2Operator
from egoworld.operators.hamer_op import HamerOperator
from egoworld.operators.foundationpose_op import FoundationPoseOperator
from egoworld.operators.dex_retarget_op import DexRetargetOperator


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
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _mask_schema():  # pragma: no cover - optional dependency
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp_s", pa.float64()),
            pa.field("mask_rle", pa.string()),
        ]
    )


def _pose_schema():  # pragma: no cover - optional dependency
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp_s", pa.float64()),
            pa.field("pose", pa.list_(pa.float32())),
        ]
    )


def _build_clip_tasks(clips: List[Dict[str, Any]], video_manifest: Dict[str, Dict[str, Any]]) -> List[ClipTask]:
    tasks: List[ClipTask] = []
    for clip in clips:
        video = video_manifest.get(clip["video_id"], {})
        tasks.append(
            ClipTask(
                clip_id=clip["clip_id"],
                video_id=clip["video_id"],
                video_path=video.get("path", ""),
                start_s=float(clip["start_s"]),
                end_s=float(clip["end_s"]),
                frame_start=int(clip["frame_start"]),
                frame_end=int(clip["frame_end"]),
                scenedetect_failed=bool(clip.get("scenedetect_failed", False)),
                retry_count=int(clip.get("retry_count", 0)),
            )
        )
    return tasks


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


class _ActorInitMixin:
    def __init__(self, config: Dict[str, Any]):
        self.config = config


class Sam2Actor(_ActorInitMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from egoworld.config import ParquetConfig

        self.parquet = ParquetConfig(**config.get("parquet", {}))
        self.sam2 = Sam2Operator()
        self.hamer = HamerOperator()
        self.foundation = FoundationPoseOperator()
        self.retarget = DexRetargetOperator()

    def process(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        masks = self.sam2.run(clip["video_path"], clip["start_s"], clip["end_s"])
        hand_pose = self.hamer.run(clip["video_path"], clip["start_s"], clip["end_s"])
        object_pose = self.foundation.run(clip["video_path"], clip["start_s"], clip["end_s"])
        mapping = self.retarget.run(hand_pose)
        return {
            "clip": clip,
            "masks": masks,
            "hand_pose": hand_pose,
            "object_pose": object_pose,
            "mapping": mapping,
        }


class WriterActor(_ActorInitMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def write(self, result: Dict[str, Any]) -> Dict[str, Any]:
        clip = result["clip"]
        run_id = self.config["run_id"]
        out_dir = clip_dir(self.config["paths"]["output_root"], run_id, clip["video_id"], clip["clip_id"])
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "clip": clip,
            "field_specs": FIELD_SPECS,
            "mask_encoding": self.config["coordinates"]["mask_encoding"],
            "time_base": self.config["coordinates"]["time_base"],
        }
        write_json(str(out_dir / "meta.json"), meta)

        masks_rows = result.get("masks", {}).get("frames", [])
        write_parquet_table(
            str(out_dir / "masks.parquet"),
            masks_rows,
            schema=_mask_schema(),
            parquet=self.parquet,
        )
        hand_rows = result.get("hand_pose", {}).get("hand_pose", [])
        write_parquet_table(
            str(out_dir / "hand_pose.parquet"),
            hand_rows,
            schema=_pose_schema(),
            parquet=self.parquet,
        )
        obj_rows = result.get("object_pose", {}).get("object_pose", [])
        write_parquet_table(
            str(out_dir / "object_pose.parquet"),
            obj_rows,
            schema=_pose_schema(),
            parquet=self.parquet,
        )
        map_rows = result.get("mapping", {}).get("mapping", [])
        write_parquet_table(
            str(out_dir / "mapping.parquet"),
            map_rows,
            schema=_pose_schema(),
            parquet=self.parquet,
        )

        return {"clip_id": clip["clip_id"], "status": "written"}


def run_pipeline(
    config_path: str,
    video_manifest_path: str,
    clip_manifest_path: str,
) -> None:
    config = load_config(config_path).resolved()
    run_id = config.run_id or make_run_id()
    config.run_id = run_id

    state_db = config.paths.state_db_path
    init_db(state_db)

    video_index = _load_video_index(video_manifest_path)
    clip_rows = load_manifest(clip_manifest_path)
    bulk_insert_pending(state_db, clip_rows)
    clip_tasks = _build_clip_tasks(clip_rows, video_index)
    clip_tasks = sort_clips_by_duration([_clip_to_dict(t) for t in clip_tasks])
    clip_tasks = [ClipTask(**task) for task in clip_tasks]

    run_manifest = config.to_run_manifest()
    run_manifest["created_at"] = datetime.utcnow().isoformat() + "Z"
    run_root = run_dir(config.paths.output_root, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    write_run_manifest(str(run_root / "run_manifest.json"), run_manifest)

    try:
        import ray  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ray is required to run the pipeline") from exc

    ray.init(ignore_reinit_error=True)

    config_dict = asdict(config)
    writer = ray.remote(WriterActor).options(num_cpus=1).remote(config_dict)
    gpu_actors = [
        ray.remote(Sam2Actor).options(num_gpus=1).remote(config_dict)
        for _ in range(config.num_gpus)
    ]

    pending_gpu: List[Any] = []
    pending_write: List[Any] = []
    ref_to_meta: Dict[Any, Tuple[ClipTask, int]] = {}
    write_ref_to_meta: Dict[Any, Tuple[Dict[str, Any], int]] = {}
    actor_index = 0

    def submit_clip(task: ClipTask, attempt: int) -> None:
        nonlocal actor_index
        upsert_clip_status(state_db, task.clip_id, task.video_id, "Running", "", attempt)
        clip_payload = _clip_to_dict(task)
        actor = gpu_actors[actor_index % len(gpu_actors)]
        actor_index += 1
        ref = actor.process.remote(clip_payload)
        pending_gpu.append(ref)
        ref_to_meta[ref] = (task, attempt)

    for task in clip_tasks:
        submit_clip(task, task.retry_count)

        done_refs, pending_gpu = enforce_in_flight(
            pending_gpu, config.backpressure.max_in_flight_gpu
        )
        for done_ref in done_refs:
            task, attempt = ref_to_meta.pop(done_ref)
            try:
                result = ray.get(done_ref)
                upsert_clip_status(state_db, task.clip_id, task.video_id, "Writing", "", attempt)
                write_ref = writer.write.remote(result)
                pending_write.append(write_ref)
                write_ref_to_meta[write_ref] = (result, 0)
            except Exception as exc:
                classification = classify_error(exc)
                if classification.retryable and attempt < config.retry.max_retries:
                    delay = config.retry.next_delay(attempt + 1)
                    time.sleep(delay)
                    task.retry_count = attempt + 1
                    submit_clip(task, task.retry_count)
                else:
                    upsert_clip_status(
                        state_db,
                        task.clip_id,
                        task.video_id,
                        "Failed",
                        str(exc),
                        attempt,
                    )
                    mark_dead_letter(state_db, task.clip_id, task.video_id, str(exc))

        done_write, pending_write = enforce_in_flight(
            pending_write, config.backpressure.max_in_flight_write
        )
        for write_ref in done_write:
            result, attempt = write_ref_to_meta.pop(write_ref)
            clip = result["clip"]
            try:
                ray.get(write_ref)
                upsert_clip_status(state_db, clip["clip_id"], clip["video_id"], "Done", "", attempt)
            except Exception as exc:
                classification = classify_error(exc)
                if classification.retryable and attempt < config.retry.max_retries:
                    delay = config.retry.next_delay(attempt + 1)
                    time.sleep(delay)
                    retry_ref = writer.write.remote(result)
                    pending_write.append(retry_ref)
                    write_ref_to_meta[retry_ref] = (result, attempt + 1)
                else:
                    upsert_clip_status(
                        state_db,
                        clip["clip_id"],
                        clip["video_id"],
                        "Failed",
                        str(exc),
                        attempt,
                    )
                    mark_dead_letter(state_db, clip["clip_id"], clip["video_id"], str(exc))

    # Drain remaining
    while pending_gpu:
        done_refs, pending_gpu = enforce_in_flight(pending_gpu, 1)
        for done_ref in done_refs:
            task, attempt = ref_to_meta.pop(done_ref)
            try:
                result = ray.get(done_ref)
                upsert_clip_status(state_db, task.clip_id, task.video_id, "Writing", "", attempt)
                write_ref = writer.write.remote(result)
                pending_write.append(write_ref)
                write_ref_to_meta[write_ref] = (result, 0)
            except Exception as exc:
                classification = classify_error(exc)
                if classification.retryable and attempt < config.retry.max_retries:
                    delay = config.retry.next_delay(attempt + 1)
                    time.sleep(delay)
                    task.retry_count = attempt + 1
                    submit_clip(task, task.retry_count)
                else:
                    upsert_clip_status(
                        state_db,
                        task.clip_id,
                        task.video_id,
                        "Failed",
                        str(exc),
                        attempt,
                    )
                    mark_dead_letter(state_db, task.clip_id, task.video_id, str(exc))

    while pending_write:
        done_write, pending_write = enforce_in_flight(pending_write, 1)
        for write_ref in done_write:
            result, attempt = write_ref_to_meta.pop(write_ref)
            clip = result["clip"]
            try:
                ray.get(write_ref)
                upsert_clip_status(state_db, clip["clip_id"], clip["video_id"], "Done", "", attempt)
            except Exception as exc:
                classification = classify_error(exc)
                if classification.retryable and attempt < config.retry.max_retries:
                    delay = config.retry.next_delay(attempt + 1)
                    time.sleep(delay)
                    retry_ref = writer.write.remote(result)
                    pending_write.append(retry_ref)
                    write_ref_to_meta[retry_ref] = (result, attempt + 1)
                else:
                    upsert_clip_status(
                        state_db,
                        clip["clip_id"],
                        clip["video_id"],
                        "Failed",
                        str(exc),
                        attempt,
                    )
                    mark_dead_letter(state_db, clip["clip_id"], clip["video_id"], str(exc))

    ray.shutdown()
