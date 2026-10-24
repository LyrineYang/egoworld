"""Output contracts shared by adapters, writers, recovery and tests."""
from __future__ import annotations

import math
from typing import Any
import numpy as np

OUTPUT_SCHEMA_VERSION = 3
STAGE_DEPENDENCIES = {
    "sam2": (), "hamer": ("sam2",), "foundationpose": ("sam2",),
    "fast3r": (), "dex_retarget": ("hamer",),
}
GPU_STAGES = frozenset({"sam2", "hamer", "foundationpose", "fast3r"})
STAGE_ARTIFACTS = {
    "sam2": ("masks", "instances", "tracks"), "hamer": ("hand_pose",),
    "foundationpose": ("object_pose",), "dex_retarget": ("mapping",),
    "fast3r": ("reconstruction",),
}


def schemas():
    import pyarrow as pa
    i, f, s = pa.int64(), pa.float64(), pa.string()
    fields = {
        "masks": [("frame_index", i), ("timestamp_s", f), ("mask_rle", s)],
        "instances": [("frame_index", i), ("timestamp_s", f), ("object_id", i), ("label", s),
                      ("detection_score", f), ("mask_rle", s), ("area_px", i), ("bbox_xyxy", pa.list_(f))],
        "tracks": [("frame_index", i), ("timestamp_s", f), ("object_id", i), ("label", s),
                   ("center_xy", pa.list_(f)), ("area_px", i)],
        "hand_pose": [("frame_index", i), ("timestamp_s", f), ("object_id", i), ("handedness", s),
                      ("joints_3d", pa.list_(f)), ("global_orient", pa.list_(f)),
                      ("mano_pose", pa.list_(f)), ("betas", pa.list_(f)),
                      ("camera_translation", pa.list_(f)), ("coord_frame", s),
                      ("length_unit", s), ("scale_status", s), ("joint_order", s)],
        "object_pose": [("frame_index", i), ("timestamp_s", f), ("object_id", i), ("label", s),
                        ("transform", pa.list_(f)), ("coord_frame", s), ("length_unit", s), ("mesh_path", s)],
        "mapping": [("frame_index", i), ("timestamp_s", f), ("object_id", i), ("handedness", s),
                    ("robot_id", s), ("joint_names", pa.list_(s)), ("joint_positions", pa.list_(f)),
                    ("joint_units", pa.list_(s)), ("source_scale_status", s)],
        "reconstruction": [("frame_index", i), ("timestamp_s", f), ("view_index", i),
                           ("camera_to_world", pa.list_(f)), ("intrinsics", pa.list_(f)),
                           ("image_width", i), ("image_height", i), ("pointcloud_path", s), ("coord_frame", s), ("length_unit", s), ("scale_status", s)],
    }
    return {name: pa.schema(cols) for name, cols in fields.items()}


def validate_rows(name: str, rows: list[dict], clip: dict | None = None) -> None:
    schema = schemas()[name]
    required = set(schema.names)
    seen = set()
    for row in rows:
        if missing := required - row.keys():
            raise ValueError(f"{name}: missing fields {sorted(missing)}")
        if unexpected := row.keys() - required:
            raise ValueError(f"{name}: unexpected fields {sorted(unexpected)}")
        optional = {"instances": {"bbox_xyxy"}, "tracks": {"center_xy"}}
        if any(row[field] is None for field in required - optional.get(name, set())):
            raise ValueError(f"{name}: null required field")
        frame, ts = row["frame_index"], row["timestamp_s"]
        if not isinstance(frame, (int, np.integer)) or frame < 0 or not math.isfinite(ts) or ts < 0:
            raise ValueError(f"{name}: invalid frame/time")
        if clip:
            if not clip["frame_start"] <= frame < clip["frame_end"]:
                raise ValueError(f"{name}: frame outside clip")
            if clip.get("fps") and abs(ts - frame / clip["fps"]) > 1e-5:
                raise ValueError(f"{name}: timestamp is not aligned to source fps")
        key = (frame, row.get("object_id", row.get("view_index", 0)))
        if key in seen:
            raise ValueError(f"{name}: duplicate frame/object key {key}")
        seen.add(key)
        for field in schema:
            value = row[field.name]
            if isinstance(value, (float, np.floating)) and not math.isfinite(value):
                raise ValueError(f"{name}: nonfinite {field.name}")
            if isinstance(value, list) and value and isinstance(value[0], (float, int)) and not np.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite {field.name}")
        lengths = {"hand_pose": {"joints_3d": 63, "global_orient": 9, "mano_pose": 135,
                                   "betas": 10, "camera_translation": 3},
                   "object_pose": {"transform": 16},
                   "reconstruction": {"camera_to_world": 16, "intrinsics": 9}}
        for field, length in lengths.get(name, {}).items():
            if len(row[field]) != length:
                raise ValueError(f"{name}: {field} must have {length} values")
        if name in ("object_pose", "reconstruction"):
            matrix = np.asarray(row["transform" if name == "object_pose" else "camera_to_world"]).reshape(4, 4)
            if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=.05) or np.linalg.det(matrix[:3, :3]) < .9:
                raise ValueError(f"{name}: invalid rigid transform")
        if name == "mapping" and not (len(row["joint_names"]) == len(row["joint_positions"]) == len(row["joint_units"])):
            raise ValueError("mapping: joint arrays must have equal length")
        if name == "mapping" and (not row["joint_names"] or len(set(row["joint_names"])) != len(row["joint_names"]) or set(row["joint_units"]) - {"radians", "meters"}):
            raise ValueError("mapping: invalid joint names/units")
        if name in ("hand_pose", "mapping") and row["handedness"] not in ("left", "right"):
            raise ValueError(f"{name}: unknown handedness")
        if name == "hand_pose" and (row["coord_frame"], row["length_unit"], row["scale_status"], row["joint_order"]) != ("hand_root", "meters", "model_prior", "openpose21"):
            raise ValueError("hand_pose: unsupported coordinate/unit/scale/joint order")
        if name == "object_pose" and (row["coord_frame"], row["length_unit"]) != ("object_to_camera", "meters"):
            raise ValueError("object_pose: unsupported coordinate/unit")
        if name == "reconstruction":
            if (row["coord_frame"] != "first_view_world" or (row["length_unit"], row["scale_status"]) not in {("relative", "unscaled"), ("meters", "calibrated")}):
                raise ValueError("reconstruction: inconsistent coordinate/unit/scale")
            k = np.asarray(row["intrinsics"]).reshape(3, 3)
            if row["image_width"] < 1 or row["image_height"] < 1 or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
                raise ValueError("reconstruction: invalid camera intrinsics/dimensions")
        if name == "instances" and (row["area_px"] < 0 or (row["area_px"] > 0 and (row["bbox_xyxy"] is None or len(row["bbox_xyxy"]) != 4))):
            raise ValueError("instances: invalid area/bounding box")


def enabled_stages(config) -> dict[str, tuple[str, ...]]:
    enabled = {name for name in STAGE_DEPENDENCIES if getattr(config.operators, name).enabled}
    for name in enabled:
        missing = set(STAGE_DEPENDENCIES[name]) - enabled
        if missing:
            raise ValueError(f"{name} requires enabled operators: {sorted(missing)}")
    return {name: STAGE_DEPENDENCIES[name] for name in STAGE_DEPENDENCIES if name in enabled}
