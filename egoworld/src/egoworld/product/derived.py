"""Identity and construction of measured world derivatives."""
from pathlib import Path
import json

from egoworld.utils.hashing import sha256_file


def world_identity(folder, calibration, outputs, clip, receipts):
    from egoworld.product.cache import file_signature
    root = Path(__file__).parent
    evidence = {}
    if calibration.get("method") == "rgbd":
        from egoworld.operators.common import rgbd_assets
        asset = rgbd_assets(calibration["assets_manifest"])[clip["video_id"]]
        import pyarrow.parquet as pq
        first_frame = min(pq.ParquetFile(Path(folder) / "reconstruction.parquet").read(columns=["frame_index"])["frame_index"].to_pylist())
        evidence = {"assets": sha256_file(calibration["assets_manifest"]),
                    "depth": sha256_file(asset["depth_pattern"].format(frame_index=first_frame))}
    else:
        evidence["calibration"] = sha256_file(calibration["path"])
        document = json.loads(Path(calibration["path"]).read_text())
        for name, expected in document.get("bound_files", {}).items():
            if Path(name).name != name or sha256_file(str(Path(folder) / name)) != expected:
                raise ValueError("Calibration source artifact identity changed")
    required = {"reconstruction.parquet", "reconstruction-pointcloud.npz"}
    if "world_hand" in outputs:
        required.add("hand_pose.parquet")
    if "world_object" in outputs:
        required.add("object_pose.parquet")
    if calibration.get("method") == "rgbd":
        required.add("reconstruction-correspondences.npz")
    import numpy as np
    import pyarrow as pa
    import cv2
    return {"version": 1, "kind": "world", "calibration": calibration, "outputs": outputs,
            "clip": {k: clip[k] for k in ("video_id", "clip_id", "frame_start", "frame_end", "fps")},
            "evidence": evidence, "sources": {s: file_signature(r) for s, r in receipts.items()},
            "actual_sources": {name: sha256_file(str(Path(folder) / name)) for name in sorted(required)},
            "runtime": {"numpy": np.__version__, "pyarrow": pa.__version__, "opencv": cv2.__version__},
            "code": {name: sha256_file(str(root / name)) for name in ("world.py", "geometry.py", "derived.py")}}


def build_world(folder, calibration, outputs, destination):
    from egoworld.product.world import calibrate_rgbd, fuse_world
    destination.mkdir(parents=True, exist_ok=True)
    if calibration.get("method") == "rgbd":
        path = destination / "calibration.json"
        calibrate_rgbd(folder, calibration["assets_manifest"], path,
            threshold_m=calibration.get("threshold_m", .03),
            min_inlier_fraction=calibration.get("min_inlier_fraction", .5),
            confidence_quantile=calibration.get("confidence_quantile", .5))
    else:
        path = calibration["path"]
    return fuse_world(folder, path, outputs, destination)
