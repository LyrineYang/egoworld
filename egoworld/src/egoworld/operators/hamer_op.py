"""HaMeR adapter using SAM2 hand boxes and the official crop/model API."""
from __future__ import annotations
from egoworld.utils.performance import timed_load
import numpy as np
from egoworld.operators.common import array, frame_reader, read_frame, iter_rows as rows, frame_fields


class HamerOperator:
    name = "hamer"

    def __init__(self, **params):
        self.params = params
        self.model = self.model_cfg = None

    @timed_load
    def _load(self):
        if self.model is None:
            from hamer.models import load_hamer
            self.model, self.model_cfg = load_hamer(self.params["checkpoint"])
            self.model = self.model.to(self.params.get("device", "cuda")).eval()

    def run(self, clip, video_path, inputs):
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        stride = int(self.params.get("frame_stride", 5))
        batch_size = int(self.params.get("batch_size", 8))
        if stride < 1 or batch_size < 1:
            raise ValueError("HaMeR stride and batch size must be positive")
        selected_ids = self.params.get("object_ids")
        if selected_ids is not None:
            if (not isinstance(selected_ids, list) or not selected_ids or
                    any(not isinstance(i, int) or isinstance(i, bool) or i < 1 for i in selected_ids)):
                raise ValueError("HaMeR object_ids must be a nonempty list of positive tracked IDs")
            selected_ids = set(selected_ids)
        by_frame = {}
        for row in rows(inputs, "sam2", "instances", ["frame_index", "object_id", "label", "area_px", "bbox_xyxy"]):
            if (selected_ids is None or row["object_id"] in selected_ids) and row["area_px"] and any(word in row["label"].lower() for word in ("hand", "glove")) and (row["frame_index"] - clip["frame_start"]) % stride == 0:
                by_frame.setdefault(row["frame_index"], []).append(row)
        if selected_ids is not None and not by_frame:
            raise ValueError("No nonempty hand instances match object_ids")
        self._load()
        cap = frame_reader(video_path)
        output = []
        try:
            for index, detections in sorted(by_frame.items()):
                sides = []
                for detection in detections:
                    label = detection["label"].lower()
                    side = self.params.get("handedness_by_id", {}).get(str(detection["object_id"]))
                    side = side or ("left" if "left" in label else "right" if "right" in label else self.params.get("unknown_handedness", "error"))
                    if side not in ("left", "right"):
                        raise ValueError("HaMeR needs left/right hand labels or handedness_by_id; refusing to guess")
                    sides.append(side)
                image = read_frame(cap, index - clip["frame_start"])
                dataset = ViTDetDataset(self.model_cfg, image, np.asarray([d["bbox_xyxy"] for d in detections]),
                                       np.asarray([int(s == "right") for s in sides]),
                                       rescale_factor=float(self.params.get("rescale_factor", 2.0)))
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                offset = 0
                for batch in loader:
                    batch = recursive_to(batch, self.params.get("device", "cuda"))
                    with torch.inference_mode():
                        prediction = self.model(batch)
                    joints = array(prediction["pred_keypoints_3d"])
                    mano = prediction["pred_mano_params"]
                    camera = array(prediction["pred_cam"]).copy()
                    centers, sizes, image_sizes = array(batch["box_center"]), array(batch["box_size"]), array(batch["img_size"])
                    for j in range(len(joints)):
                        detection, side = detections[offset + j], sides[offset + j]
                        sign = 1 if side == "right" else -1
                        xyz = joints[j].copy(); xyz[:, 0] *= sign
                        wrist = xyz[0].copy()
                        xyz -= wrist
                        focal = float(self.params.get("focal_length_px") or self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * image_sizes[j].max())
                        scale, tx, ty = camera[j]; tx *= sign
                        z = 2 * focal / (float(sizes[j]) * float(scale) + 1e-9)
                        # The upstream camera translation applies to MANO's
                        # origin. Our joints are wrist-relative, so preserve
                        # camera-space points by including the mirrored wrist.
                        translation = np.asarray([
                            tx + 2 * (centers[j, 0] - image_sizes[j, 0] / 2) / (sizes[j] * scale + 1e-9),
                            ty + 2 * (centers[j, 1] - image_sizes[j, 1] / 2) / (sizes[j] * scale + 1e-9), z]) + wrist
                        output.append({**frame_fields(clip, index - clip["frame_start"]),
                            "object_id": detection["object_id"], "handedness": side,
                            "joints_3d": xyz.reshape(-1).tolist(),
                            "global_orient": array(mano["global_orient"])[j].reshape(-1).tolist(),
                            "mano_pose": array(mano["hand_pose"])[j].reshape(-1).tolist(),
                            "betas": array(mano["betas"])[j].reshape(-1).tolist(),
                            "camera_translation": list(map(float, translation)), "coord_frame": "hand_root",
                            "length_unit": "meters", "scale_status": "model_prior", "joint_order": "openpose21"})
                    offset += len(joints)
        finally:
            cap.release()
        return {"tables": {"hand_pose": output}, "stats": {"sampled_frames": len(by_frame)}}
