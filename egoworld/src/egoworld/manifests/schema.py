"""Manifest schemas and field specifications."""

from __future__ import annotations

from typing import Dict, Any

SPEC_VERSION = "v1"
TIME_BASE = "seconds"
MASK_ENCODING = "rle"
COORD_FRAME = "camera"
LENGTH_UNIT = "meters"
QUAT_ORDER = "wxyz"

FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "mask": {
        "encoding": MASK_ENCODING,
        "resolution": "same_as_source",
        "alignment": "mask applies to frame_index and timestamp_s",
    },
    "time": {
        "time_base": TIME_BASE,
        "frame_index_base": 0,
        "relation": "timestamp_s = frame_index / fps",
    },
    "pose_6d": {
        "coord_frame": COORD_FRAME,
        "handedness": "right",
        "length_unit": LENGTH_UNIT,
        "quat_order": QUAT_ORDER,
    },
    "pose_3d": {
        "coord_frame": COORD_FRAME,
        "handedness": "right",
        "length_unit": LENGTH_UNIT,
    },
}


def _pa():  # pragma: no cover - optional dependency
    import pyarrow as pa

    return pa


def video_manifest_schema():
    pa = _pa()
    return pa.schema(
        [
            pa.field("video_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("duration_s", pa.float64()),
            pa.field("fps", pa.float64()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("audio", pa.bool_()),
            pa.field("checksum", pa.string()),
            pa.field("split", pa.string()),
        ]
    )


def clip_manifest_schema():
    pa = _pa()
    return pa.schema(
        [
            pa.field("clip_id", pa.string()),
            pa.field("video_id", pa.string()),
            pa.field("start_s", pa.float64()),
            pa.field("end_s", pa.float64()),
            pa.field("frame_start", pa.int64()),
            pa.field("frame_end", pa.int64()),
            pa.field("overlap_s", pa.float64()),
            pa.field("scenedetect_failed", pa.bool_()),
            pa.field("status", pa.string()),
            pa.field("last_error", pa.string()),
            pa.field("retry_count", pa.int32()),
        ]
    )


def run_manifest_schema():
    pa = _pa()
    return pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("config_path", pa.string()),
            pa.field("code_git_hash", pa.string()),
            pa.field("model_versions", pa.string()),
            pa.field("dataset_hash", pa.string()),
            pa.field("parquet_params", pa.string()),
            pa.field("coordinate_spec_version", pa.string()),
            pa.field("mask_encoding", pa.string()),
            pa.field("time_base", pa.string()),
        ]
    )
