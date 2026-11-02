"""Streaming diagnostics; original model observations are retained unchanged."""
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def inspect_clip(folder, clip, receipts):
    result = {"version": 1, "flags": [], "branch_filters": [], "observations": {}}
    for stage, receipt in receipts.items():
        stats = receipt.get("stats", {})
        if stats.get("branch_filtered"):
            result["branch_filters"].append({"stage": stage, "reason": stats.get("filter_reason")})
    hand_file = Path(folder) / "hand_pose.parquet"
    if hand_file.is_file():
        previous, rows, frames, peak_speed, peak_frame = {}, 0, set(), 0., None
        scales = set()
        for batch in pq.ParquetFile(hand_file).iter_batches(batch_size=256, columns=["frame_index", "timestamp_s", "object_id", "handedness", "camera_translation", "scale_status"]):
            for row in batch.to_pylist():
                rows += 1; frames.add(row["frame_index"]); scales.add(row["scale_status"])
                key = (row["object_id"], row["handedness"])
                point = np.asarray(row["camera_translation"], float)
                if key in previous:
                    time, last = previous[key]
                    dt = row["timestamp_s"] - time
                    if dt > 0:
                        speed = float(np.linalg.norm(point - last) / dt)
                        if speed > peak_speed:
                            peak_speed, peak_frame = speed, row["frame_index"]
                    else:
                        result["flags"].append("nonmonotonic_hand_observations")
                previous[key] = row["timestamp_s"], point
        result["observations"]["hand"] = {"rows": rows, "source_frames": len(frames),
            "peak_camera_wrist_speed": peak_speed, "peak_speed_frame": peak_frame,
            "speed_scale_status": sorted(scales), "note": "Model-prior speeds are diagnostic estimates, not measured physical velocity"}
        if "model_prior" in scales:
            result["flags"].append("hand_scale_is_model_prior")
    result["status"] = "filtered" if result["branch_filters"] else "review" if result["flags"] else "passed"
    return result
