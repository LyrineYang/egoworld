"""Calibrated, clip-scoped world exports alongside immutable model outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egoworld.io.writers import write_json, write_parquet_table
from egoworld.product.geometry import rigid, similarity, transform_points, robust_similarity
from egoworld.utils.hashing import sha256_file


def table(folder, name):
    for batch in pq.ParquetFile(Path(folder) / f"{name}.parquet").iter_batches(batch_size=512):
        yield from batch.to_pylist()


def context(folder):
    meta = json.loads((Path(folder) / "meta.json").read_text())
    if meta["status"] != "Done":
        raise ValueError("World export requires a completed source clip")
    return meta["clip"]


def calibrate_hand(folder, calibration_path, measurements_path, output, *, threshold_m=.02):
    folder = Path(folder)
    clip = context(folder)
    calibration = json.loads(Path(calibration_path).read_text())
    measurements = json.loads(Path(measurements_path).read_text())
    if (measurements.get("version"), measurements.get("video_id"), measurements.get("clip_id"),
            measurements.get("joint_order"), measurements.get("length_unit")) != (1, clip["video_id"], clip["clip_id"], "openpose21", "meters"):
        raise ValueError("Hand measurements must match this clip, OpenPose21 order and meters")
    if (calibration.get("video_id"), calibration.get("clip_id")) != (clip["video_id"], clip["clip_id"]):
        raise ValueError("World calibration belongs to another clip")
    source_file = folder / "hand_pose.parquet"
    hands = {(r["frame_index"], r["object_id"]): r for r in table(folder, "hand_pose")}
    samples = measurements.get("samples", [])
    if not samples:
        raise ValueError("No measured hand samples")
    alignments = dict(calibration.get("hand_camera_alignments", {}))
    seen = set()
    for sample in samples:
        key = (sample["frame_index"], sample["object_id"])
        if key not in hands or key in seen or not sample.get("evidence"):
            raise ValueError("Measured hands need unique matching source rows and evidence")
        seen.add(key)
        row = hands[key]
        xyz = np.asarray(row["joints_3d"]).reshape(21, 3) + row["camera_translation"]
        target = np.asarray(sample["joints_camera_meters"], float).reshape(21, 3)
        alignment = robust_similarity(xyz, target, threshold_m)
        alignments[f"{key[0]}:{key[1]}"] = {**alignment,
            "source_hand_pose_sha256": sha256_file(str(source_file)),
            "evidence": {"description": sample["evidence"],
                         "measurements_sha256": sha256_file(str(measurements_path))}}
    calibration["hand_camera_alignments"] = alignments
    write_json(str(output), calibration)
    return calibration


def calibrate_rgbd(folder, assets_path, output, *, threshold_m=.03, min_inlier_fraction=.5, confidence_quantile=.5):
    import cv2
    folder, assets_path, output = Path(folder), Path(assets_path).resolve(), Path(output)
    clip = context(folder)
    assets = json.loads(assets_path.read_text())[clip["video_id"]]
    k = np.asarray(assets["intrinsics"], float).reshape(3, 3)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("RGB-D calibration requires valid source-image intrinsics")
    factor = float(assets["depth_scale_to_meters"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Depth scale must be positive")
    reconstruction = table(folder, "reconstruction")
    first = min(reconstruction, key=lambda r: r["frame_index"])
    pattern = Path(assets["depth_pattern"])
    if not pattern.is_absolute():
        pattern = assets_path.parent / pattern
    depth_path = Path(str(pattern).format(frame_index=first["frame_index"]))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError("Expected aligned, single-channel metric depth")
    correspondence = folder / "reconstruction-correspondences.npz"
    with np.load(correspondence, allow_pickle=False) as data:
        selected = data["view_index"] == first["view_index"]
        xyz, xy, confidence = data["points"][selected], data["source_pixel_xy"][selected], data["confidence"][selected]
    if not 0 <= confidence_quantile < 1:
        raise ValueError("confidence_quantile must be in [0, 1)")
    pixels = np.rint(xy).astype(int)
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < depth.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < depth.shape[0])
    xyz, xy, pixels, confidence = xyz[valid], xy[valid], pixels[valid], confidence[valid]
    z = depth[pixels[:, 1], pixels[:, 0]].astype(float) * factor
    valid = np.isfinite(z) & (z > 0)
    xyz, xy, z, confidence = xyz[valid], xy[valid], z[valid], confidence[valid]
    if len(xyz) < 12:
        raise ValueError("Too few valid depth correspondences")
    cutoff = float(np.quantile(confidence, confidence_quantile))
    keep = confidence >= cutoff
    xyz, xy, z = xyz[keep], xy[keep], z[keep]
    rays = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(k).T
    target = rays * z[:, None]
    alignment = robust_similarity(xyz, target, threshold_m, min_inlier_fraction)
    calibration = {"version": 1, "video_id": clip["video_id"], "clip_id": clip["clip_id"],
        "world_id": f"{clip['video_id']}/{clip['clip_id']}/camera-{first['frame_index']}",
        "length_unit": "meters", "scene_alignment": alignment,
        "bound_files": {name: sha256_file(str(folder / name)) for name in
                        ("reconstruction.parquet", "reconstruction-pointcloud.npz", "reconstruction-correspondences.npz")},
        "evidence": {"method": "RANSAC similarity from exact Fast3R source pixels and aligned metric depth",
            "depth_path": str(depth_path), "depth_sha256": sha256_file(str(depth_path)),
            "assets_sha256": sha256_file(str(assets_path)), "intrinsics": k.reshape(-1).tolist(),
            "anchor_frame": first["frame_index"],
            "confidence_quantile": confidence_quantile, "minimum_confidence": cutoff,
            "scope": "World anchored to one RGB-D camera frame; later camera poses remain Fast3R estimates. "
                     "Held-out correspondence residuals are not ground-truth trajectory accuracy."},
        "camera_overrides": {str(first["frame_index"]): np.eye(4).reshape(-1).tolist()},
        "hand_camera_alignments": {}}
    write_json(str(output), calibration)
    return calibration


def fuse_world(folder, calibration_path, outputs, destination):
    folder, calibration_path, destination = Path(folder), Path(calibration_path), Path(destination)
    clip = context(folder)
    calibration = json.loads(calibration_path.read_text())
    if (calibration.get("version"), calibration.get("video_id"), calibration.get("clip_id"), calibration.get("length_unit")) != (1, clip["video_id"], clip["clip_id"], "meters"):
        raise ValueError("Calibration must match this exact clip and use meters")
    if not calibration.get("world_id") or not calibration.get("evidence", {}).get("method"):
        raise ValueError("World calibration needs a named reference frame and evidence")
    required = {"reconstruction.parquet", "reconstruction-pointcloud.npz"}
    if not required <= calibration.get("bound_files", {}).keys():
        raise ValueError("Calibration must bind the source camera and scene artifacts")
    for name, expected in calibration["bound_files"].items():
        if Path(name).name != name or sha256_file(str(folder / name)) != expected:
            raise ValueError("Calibration source artifact identity changed")
    scale, alignment = similarity(calibration["scene_alignment"])
    reconstruction = table(folder, "reconstruction")
    cameras = {}
    for row in reconstruction:
        pose = rigid(row["camera_to_world"])
        transformed = np.eye(4)
        transformed[:3, :3] = alignment[:3, :3] @ pose[:3, :3]
        transformed[:3, 3] = transform_points(pose[:3, 3], alignment, scale)
        cameras[row["frame_index"]] = rigid(transformed)
    # External calibrated trajectories may provide every frame explicitly.
    for frame, pose in calibration.get("camera_overrides", {}).items():
        frame = int(frame)
        if not clip["frame_start"] <= frame < clip["frame_end"]:
            raise ValueError("Camera calibration frame outside source clip")
        cameras[frame] = rigid(pose)
    selected = set(outputs)
    if selected - {"world_camera", "world_scene", "world_object", "world_hand"}:
        raise ValueError("Unknown world output")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "world_manifest.json").unlink(missing_ok=True)
    common = {"world_id": calibration["world_id"], "length_unit": "meters", "scale_status": "calibrated"}
    report = {"version": 1, "status": "passed", "world_id": calibration["world_id"],
        "calibration_sha256": sha256_file(str(calibration_path)), "source_clip": str(folder.resolve()),
        "outputs": {}, "skipped_frames": {}, "evidence": calibration["evidence"]}
    base = [("frame_index", pa.int64()), ("timestamp_s", pa.float64())]
    units = [("world_id", pa.string()), ("length_unit", pa.string()), ("scale_status", pa.string())]
    def save(name, rows, fields):
        if not rows:
            raise ValueError(f"No time-aligned rows for {name}")
        schema = pa.schema(base + fields + units)
        write_parquet_table(str(destination / f"{name}.parquet"), rows, schema=schema)
        report["outputs"][name] = {"rows": len(rows)}
    if "world_camera" in selected:
        save("world_camera", [{"frame_index": f, "timestamp_s": f / clip["fps"],
            "camera_to_world": t.reshape(-1).tolist(), **common} for f, t in sorted(cameras.items())],
            [("camera_to_world", pa.list_(pa.float64()))])
    if "world_scene" in selected:
        with np.load(folder / "reconstruction-pointcloud.npz", allow_pickle=False) as content:
            keep = content["confidence"] >= calibration["evidence"].get("minimum_confidence", -np.inf)
            points = transform_points(content["points"][keep], alignment, scale)
            if not len(points) or not np.isfinite(points).all():
                raise ValueError("Empty or nonfinite world point cloud")
            np.savez_compressed(destination / "world_scene.npz", points=points,
                                confidence=content["confidence"][keep], view_index=content["view_index"][keep])
        report["outputs"]["world_scene"] = {"points": len(points)}
    for kind, source in (("world_object", "object_pose"), ("world_hand", "hand_pose")):
        if kind not in selected:
            continue
        rows, skipped = [], []
        for row in table(folder, source):
            frame = row["frame_index"]
            if frame not in cameras:
                skipped.append(frame)
                continue  # never silently use another frame's camera pose
            record = {"frame_index": frame, "timestamp_s": row["timestamp_s"],
                      "object_id": row["object_id"], **common}
            if kind == "world_object":
                if (row["coord_frame"], row["length_unit"]) != ("object_to_camera", "meters"):
                    raise ValueError("Object pose is not metric object-to-camera")
                record.update(label=row["label"], transform=(cameras[frame] @ rigid(row["transform"])).reshape(-1).tolist())
            else:
                key = f"{frame}:{row['object_id']}"
                hand = calibration.get("hand_camera_alignments", {}).get(key)
                if not hand or hand.get("source_hand_pose_sha256") != sha256_file(str(folder / "hand_pose.parquet")) or not hand.get("evidence"):
                    raise ValueError(f"Metric world hand requires measured camera alignment for {key}; model_prior is insufficient")
                hs, ht = similarity(hand)
                xyz = np.asarray(row["joints_3d"]).reshape(21, 3) + row["camera_translation"]
                xyz = transform_points(transform_points(xyz, ht, hs), cameras[frame])
                record.update(handedness=row["handedness"], joints_3d=xyz.reshape(-1).tolist(),
                              joint_order=row["joint_order"])
            rows.append(record)
        report["skipped_frames"][kind] = sorted(set(skipped))
        fields = [("object_id", pa.int64())]
        fields += ([("label", pa.string()), ("transform", pa.list_(pa.float64()))] if kind == "world_object" else
                   [("handedness", pa.string()), ("joints_3d", pa.list_(pa.float64())), ("joint_order", pa.string())])
        save(kind, rows, fields)
    report["files"] = {p.name: sha256_file(str(p)) for p in destination.iterdir() if p.is_file() and p.suffix in (".parquet", ".npz")}
    if any(report["skipped_frames"].values()):
        report["status"] = "partial"
    write_json(str(destination / "world_manifest.json"), report)
    return report
