"""FoundationPose RGB-D tracking with explicit meshes, calibration and metric scale."""
from __future__ import annotations
from egoworld.utils.performance import timed_load
from pathlib import Path
import sys
import numpy as np
from egoworld.utils.timing import phase, timed
from egoworld.operators.common import frame_reader, read_frame, iter_rows as rows, decode_rle, rgbd_assets


class FoundationPoseOperator:
    name = "foundationpose"

    def __init__(self, **params):
        self.params = params
        self.scorer = self.refiner = self.glctx = None

    @timed_load
    def _load(self):
        if self.params.get("repo_path") and self.params["repo_path"] not in sys.path:
            sys.path.insert(0, self.params["repo_path"])
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr
        if self.scorer is None:
            self.scorer, self.refiner = ScorePredictor(), PoseRefinePredictor()
            self.glctx = dr.RasterizeCudaContext()
        return FoundationPose

    def prepare_frames(self, clip, video_path, inputs):
        assets = rgbd_assets(self.params["assets_manifest"])
        video_assets = assets.get(clip["video_id"])
        if not video_assets:
            raise ValueError(f"Missing RGB-D assets for video {clip['video_id']}")
        intrinsics = np.asarray(video_assets["intrinsics"], dtype=float).reshape(3, 3)
        if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0, 0, 1]):
            raise ValueError("Invalid camera intrinsics")
        depth_scale = float(video_assets["depth_scale_to_meters"])
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale_to_meters must be positive")
        selected = [row for row in rows(inputs, "sam2", "instances")
                    if row["label"] in self.params["objects"] and row["area_px"]]
        from itertools import groupby
        cap = frame_reader(video_path)
        try:
            for frame, detections in groupby(sorted(selected, key=lambda r: (r["frame_index"], r["object_id"])), key=lambda r: r["frame_index"]):
                with phase("rgbd_prepare"):
                    depth_path = video_assets["depth_pattern"].format(frame_index=frame)
                    if Path(depth_path).suffix == ".npy":
                        depth = np.load(depth_path, allow_pickle=False).astype(np.float32) * depth_scale
                    else:
                        import cv2
                        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        if depth is None:
                            raise ValueError(f"Missing depth frame {depth_path}")
                        depth = depth.astype(np.float32) * depth_scale
                    image = read_frame(cap, frame - clip["frame_start"])[..., ::-1].copy()
                    if depth.shape != image.shape[:2] or not np.isfinite(depth).all() or np.any(depth < 0):
                        raise ValueError("Depth must be finite and aligned to RGB resolution")
                yield {"intrinsics": intrinsics, "image": image, "depth": depth, "rows": list(detections)}
        finally:
            cap.release()

    def infer_frames(self, batches):
        import trimesh
        from itertools import chain
        batches = iter(batches)
        first = next(batches, None)  # validates calibration before any GPU load
        if first is None:
            return {"tables": {"object_pose": []}}
        batches = chain([first], batches)
        cls = self._load()
        trackers, output, meshes = {}, [], {}
        for data in batches:
            intrinsics, image, depth = data["intrinsics"], data["image"], data["depth"]
            for row in data["rows"]:
                frame, spec = row["frame_index"], self.params["objects"][row["label"]]
                if row["object_id"] not in trackers:
                    mesh_scale = float(spec["mesh_scale_to_meters"])
                    if not np.isfinite(mesh_scale) or mesh_scale <= 0:
                        raise ValueError("mesh_scale_to_meters must be finite and positive")
                    key = (spec["mesh_path"], mesh_scale)
                    if key not in meshes:
                        with phase("mesh_load"):
                            mesh = trimesh.load(spec["mesh_path"], force="mesh")
                        mesh.apply_scale(mesh_scale)
                        meshes[key] = mesh
                    mesh = meshes[key].copy()  # trackers may mutate meshes; never share their state
                    tracker = cls(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                  scorer=self.scorer, refiner=self.refiner, glctx=self.glctx,
                                  debug=0, debug_dir=self.params.get("debug_dir", "/tmp/egoworld-foundationpose"))
                    mask = decode_rle(row["mask_rle"])
                    if mask.shape != depth.shape or not np.any((depth > 0) & mask):
                        raise ValueError("Registration mask must align with RGB-D and contain valid depth")
                    with phase("fp_register_including_render", cuda=True):
                        pose = tracker.register(K=intrinsics, rgb=image.copy(), depth=depth.copy(), ob_mask=mask,
                                                iteration=int(self.params.get("register_iterations", 5)))
                    trackers[row["object_id"]] = tracker
                else:
                    with phase("fp_track_including_render", cuda=True):
                        pose = trackers[row["object_id"]].track_one(rgb=image.copy(), depth=depth.copy(), K=intrinsics,
                                                            iteration=int(self.params.get("track_iterations", 2)))
                output.append({"frame_index": frame, "timestamp_s": row["timestamp_s"],
                    "object_id": row["object_id"], "label": row["label"], "transform": np.asarray(pose).reshape(-1).tolist(),
                    "coord_frame": "object_to_camera", "length_unit": "meters", "mesh_path": spec["mesh_path"]})
        return {"tables": {"object_pose": output}}

    def run(self, clip, video_path, inputs):
        return self.infer_frames(self.prepare_frames(clip, video_path, inputs))
