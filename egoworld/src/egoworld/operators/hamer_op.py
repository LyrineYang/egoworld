"""HaMeR adapter using SAM2 hand boxes and the official crop/model API."""
from __future__ import annotations
from egoworld.utils.performance import timed_load
import numpy as np
from egoworld.utils.timing import phase, timed
from egoworld.operators.common import array, frame_reader, read_frame, rows, frame_fields


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

    def _load_config(self):
        if self.model_cfg is None:
            from pathlib import Path
            from hamer.configs import get_config
            cfg = get_config(str(Path(self.params["checkpoint"]).parent.parent / "model_config.yaml"), update_cachedir=True)
            if cfg.MODEL.BACKBONE.TYPE == 'vit' and 'BBOX_SHAPE' not in cfg.MODEL:
                if cfg.MODEL.IMAGE_SIZE != 256:
                    raise ValueError("Unsupported HaMeR ViT crop geometry")
                cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
            if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
                cfg.defrost(); cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS'); cfg.freeze()
            self.model_cfg = cfg

    def prepare_batches(self, clip, video_path, inputs, cross_frame=False):
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from egoworld.operators.cached_inputs import cached_hamer_dataset
        from egoworld.utils.performance import packet_bytes
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
        self._load_config()
        limit = int(self.params.get("max_crop_bytes", 32 * 1024**2))
        max_crops = int(self.params.get("max_batch_crops", batch_size))
        if limit < 1 or max_crops < 1:
            raise ValueError("HaMeR crop byte/count limits must be positive")
        batch_size = min(batch_size, max_crops)
        cap, pending, metadata, used = frame_reader(video_path), [], [], 0
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
                with phase("crop_prepare"):
                    dataset = cached_hamer_dataset(ViTDetDataset(self.model_cfg, image,
                        np.asarray([d["bbox_xyxy"] for d in detections]), np.asarray([int(s == "right") for s in sides]),
                        rescale_factor=float(self.params.get("rescale_factor", 2.0))))
                if not cross_frame:
                    offset = 0
                    loader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))
                    # Dataset construction is lazy: the official crop and
                    # default collation actually run when advancing the loader.
                    for _ in range((len(detections) + batch_size - 1) // batch_size):
                        with phase("crop_collate"):
                            batch = next(loader)
                        count = len(batch["right"])
                        yield {"batch": batch, "metadata": [{"index": index, "detection": d, "side": side}
                            for d, side in zip(detections[offset:offset+count], sides[offset:offset+count])],
                            "focal_ratio": float(self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE)}
                        offset += count
                    continue
                for j in range(len(detections)):
                    with phase("crop_prepare"):
                        item = dataset[j]
                        size = packet_bytes(item)
                    if size > limit:
                        raise ValueError("One HaMeR crop exceeds max_crop_bytes")
                    if pending and (len(pending) >= batch_size or used + size > limit):
                        yield self._collate(pending, metadata)
                        pending, metadata, used = [], [], 0
                    pending.append(item); used += size
                    metadata.append({"index": index, "detection": detections[j], "side": sides[j]})
            if pending:
                yield self._collate(pending, metadata)
        finally:
            cap.release()

    def _collate(self, samples, metadata):
        from torch.utils.data import default_collate
        with phase("crop_collate"):
            return {"batch": default_collate(samples), "metadata": metadata,
                    "focal_ratio": float(self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE)}

    def infer_batch(self, data):
        import torch
        from hamer.utils import recursive_to
        self._load()
        # Spool transport is NumPy, including scalar tensors. Preserve dtypes.
        def tensors(x):
            if isinstance(x, np.ndarray): return torch.from_numpy(x) if hasattr(torch, "from_numpy") else x
            if isinstance(x, dict): return {k: tensors(v) for k, v in x.items()}
            return x
        with phase("h2d", cuda=True):
            batch = recursive_to(tensors(data["batch"]), self.params.get("device", "cuda"))
        with phase("forward", cuda=True), torch.inference_mode():
            prediction = self.model(batch)
        with phase("d2h", cuda=True):
            return {"joints": array(prediction["pred_keypoints_3d"]),
                "mano": {k: array(v) for k, v in prediction["pred_mano_params"].items()},
                "camera": array(prediction["pred_cam"]).copy(),
                "centers": array(batch["box_center"]), "sizes": array(batch["box_size"]),
                "image_sizes": array(batch["img_size"]), "metadata": data["metadata"], "focal_ratio": data["focal_ratio"]}

    @timed("hand_postprocess")
    def post_batch(self, clip, data):
        joints, mano, camera = data["joints"], data["mano"], data["camera"]
        centers, sizes, image_sizes = data["centers"], data["sizes"], data["image_sizes"]
        metadata, focal_ratio, output = data["metadata"], data["focal_ratio"], []
        for j in range(len(joints)):
            index = metadata[j]["index"]
            detection, side = metadata[j]["detection"], metadata[j]["side"]
            sign = 1 if side == "right" else -1
            xyz = joints[j].copy(); xyz[:, 0] *= sign
            wrist = xyz[0].copy()
            xyz -= wrist
            focal = float(self.params.get("focal_length_px") or focal_ratio * image_sizes[j].max())
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
                "global_orient": mano["global_orient"][j].reshape(-1).tolist(),
                "mano_pose": mano["hand_pose"][j].reshape(-1).tolist(),
                "betas": mano["betas"][j].reshape(-1).tolist(),
                "camera_translation": list(map(float, translation)), "coord_frame": "hand_root",
                "length_unit": "meters", "scale_status": "model_prior", "joint_order": "openpose21"})
        return output

    def run(self, clip, video_path, inputs):
        # Baseline preserves per-frame batch boundaries. Split execution opts in
        # to bounded cross-frame batching with the same configured batch_size.
        output = []
        for batch in self.prepare_batches(clip, video_path, inputs):
            output.extend(self.post_batch(clip, self.infer_batch(batch)))
        return {"tables": {"hand_pose": output}, "stats": {"sampled_frames": len({r["frame_index"] for r in output})}}
