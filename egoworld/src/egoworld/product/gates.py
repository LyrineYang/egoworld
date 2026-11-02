"""CPU branch admission using committed SAM instances, before model loading."""
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egoworld.contracts import STAGE_ARTIFACTS
from egoworld.io.writers import write_json
from egoworld.utils.performance import packet_bytes


def branch_admission(config, job, prepared, inputs):
    stage, clip = job["stage"], job["clip"]
    policy = config.get("extra", {}).get("branch_cleaning")
    if policy is None or stage not in ("hamer", "foundationpose", "dex_retarget"):
        return None, inputs
    from egoworld.operators.common import iter_rows as rows, rgbd_assets, decode_rle
    params = config["operators"][stage]["params"]
    reason = None
    if stage == "dex_retarget":
        available = any(r["handedness"] == params.get("handedness") for r in rows(inputs, "hamer", "hand_pose", ["handedness"]))
        if not available:
            reason = "no_source_hand_for_robot"
    else:
        original = pq.ParquetFile(Path(inputs["sam2"]["directory"]) / "instances.parquet")
        accepted_count, candidate_count, frames = 0, 0, set()
        columns = ["frame_index", "object_id", "label", "area_px", "bbox_xyxy"] if stage == "hamer" else None
        schema = pa.schema([original.schema_arrow.field(c) for c in columns]) if columns else original.schema_arrow
        destination = Path(job["work_dir"]) / "branch-inputs" / stage
        if Path(inputs["sam2"]["directory"]).resolve() == destination.resolve():
            return None, inputs  # external runner consuming the parent's validated private view
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "instances.parquet"
        writer = pq.ParquetWriter(target, schema, compression="zstd")
        assets = rgbd_assets(params["assets_manifest"])[clip["video_id"]] if stage == "foundationpose" else None
        depth_cache, depth_frame = None, None
        try:
            for batch in original.iter_batches(batch_size=512, columns=columns):
                accepted = []
                for row in batch.to_pylist():
                    if stage == "hamer":
                        if not row["area_px"] or not any(t in row["label"].lower() for t in ("hand", "glove")):
                            continue
                        if params.get("object_ids") and row["object_id"] not in params["object_ids"]:
                            continue
                        if (row["frame_index"] - clip["frame_start"]) % int(params.get("frame_stride", 5)):
                            continue
                        candidate_count += 1
                        if row["area_px"] < policy.get("min_hand_area_px", 0):
                            continue
                        side = params.get("handedness_by_id", {}).get(str(row["object_id"])) or ("left" if "left" in row["label"].lower() else "right" if "right" in row["label"].lower() else params.get("unknown_handedness"))
                        if side not in ("left", "right"):
                            continue
                    else:
                        if row["label"] not in params["objects"] or not row["area_px"]:
                            continue
                        candidate_count += 1
                        if row["frame_index"] != depth_frame:
                            import cv2
                            path = assets["depth_pattern"].format(frame_index=row["frame_index"])
                            depth_cache = np.load(path, allow_pickle=False) if str(path).endswith(".npy") else cv2.imread(path, cv2.IMREAD_UNCHANGED)
                            depth_frame = row["frame_index"]
                        if depth_cache is None:
                            continue
                        mask = decode_rle(row["mask_rle"])
                        if mask.shape != depth_cache.shape or not np.any(mask):
                            continue
                        fraction = float((np.isfinite(depth_cache[mask]) & (depth_cache[mask] > 0)).mean())
                        if fraction <= 0 or fraction < policy.get("min_depth_fraction", 0):
                            continue
                    accepted.append(row)
                    frames.add(row["frame_index"])
                if accepted:
                    writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
                    accepted_count += len(accepted)
        finally:
            writer.close()
        if stage == "hamer":
            expected = math.ceil((clip["frame_end"] - clip["frame_start"]) / int(params.get("frame_stride", 5)))
            if len(frames) / max(1, expected) < policy.get("min_hand_coverage", 0):
                accepted_count = 0; reason = "hand_coverage_below_policy"
        if not accepted_count:
            reason = reason or ("no_valid_hand_instances" if stage == "hamer" else "no_valid_rgbd_object_instances")
        elif accepted_count < candidate_count:
            # Filter a private view. Committed SAM data and other branches keep
            # their original rows, identities and receipts.
            inputs = {**inputs, "sam2": {**inputs["sam2"], "directory": str(destination)}}
        else:
            target.unlink(missing_ok=True)
    if reason is None:
        return None, inputs
    report = {"stage": stage, "reason": reason, "policy": policy, "clip_id": clip["clip_id"], "model_loaded": False}
    write_json(str(Path(job["work_dir"]) / f"filtered_{stage}.json"), report)
    operation = job.get("_operation")
    packet = {"blocks": [], "_handoff": True} if operation in ("input", "infer") else {"tables": {a: [] for a in STAGE_ARTIFACTS[stage]}}
    packet["stats"] = {"branch_filtered": 1, "filter_reason": reason,
                       "inference_seconds": 0., "stage_wall_seconds": 0., "model_load_seconds": 0.,
                       "process_cpu_seconds": 0., "process_peak_rss_bytes": 0, "work_units": 0}
    packet["stats"]["result_bytes"] = packet_bytes(packet)
    return packet, inputs
