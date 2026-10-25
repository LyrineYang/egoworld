"""Fast3R multiview inference; explicit relative scale and non-fabricated PnP poses."""
from __future__ import annotations
from pathlib import Path
import tempfile
import time
import numpy as np
from egoworld.operators.common import array, frame_reader, read_frame, frame_fields


class Fast3ROperator:
    name = "fast3r"

    def __init__(self, **params):
        self.params = params
        self.model = None

    def run(self, clip, video_path, inputs):
        import cv2
        import torch
        from fast3r.models.fast3r import Fast3R
        from fast3r.dust3r.inference_multiview import inference
        from fast3r.dust3r.utils.image import load_images
        from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
        device = torch.device(self.params.get("device", "cuda"))
        load_started = time.perf_counter()
        if self.model is None:
            model = Fast3R.from_pretrained(self.params["model_name_or_path"])
            implementation = self.params.get("attn_implementation")
            if implementation:
                if implementation not in ("pytorch_auto", "pytorch_naive", "flash_attention"):
                    raise ValueError("Unsupported Fast3R attention implementation")
                changed = 0
                for module in model.modules():
                    if hasattr(module, "attn_implementation"):
                        module.attn_implementation = implementation
                        changed += 1
                if not changed:
                    raise ValueError("Fast3R version exposes no configurable attention modules")
            self.model = model.to(device).eval()
        self._model_load_seconds = getattr(self, "_model_load_seconds", 0.) + time.perf_counter() - load_started
        count = clip["frame_end"] - clip["frame_start"]
        views = min(int(self.params.get("max_views", 8)), count)
        if views < 2:
            raise ValueError("Fast3R requires at least two frames")
        indices = np.unique(np.linspace(0, count - 1, views).astype(int))
        cap = frame_reader(video_path)
        try:
            with tempfile.TemporaryDirectory(prefix="egoworld-fast3r-") as folder:
                files = []
                for index in indices:
                    path = str(Path(folder) / f"{index:09d}.jpg")
                    if not cv2.imwrite(path, read_frame(cap, int(index))):
                        raise IOError(f"Cannot write sampled view {path}")
                    files.append(path)
                images = load_images(files, size=int(self.params.get("image_size", 512)), verbose=False)
                # Upstream loss_of_one_batch interprets precision strings; a
                # float32 dtype object does not disable its default autocast.
                precision = {"fp32": "32", "fp16": "16-mixed", "bf16": "bf16-mixed"}[self.params.get("dtype", "fp32")]
                with torch.inference_mode():
                    result = inference(images, self.model, device, dtype=precision, verbose=False, profiling=False)
                # Current upstream returns a dict when profiling=False. Some
                # earlier snapshots return (dict, profiling_info).
                if isinstance(result, tuple):
                    result = result[0]
        finally:
            cap.release()
        scale = float(self.params.get("metric_scale", 1.0))
        calibrated = "metric_scale" in self.params
        if not np.isfinite(scale) or scale <= 0 or (calibrated and not self.params.get("metric_scale_source")):
            raise ValueError("Metric scaling requires a positive factor and metric_scale_source")
        output, points, confidences, point_views = [], [], [], []
        budget = int(self.params.get("max_points", 100000))
        if budget < views:
            raise ValueError("max_points must be at least max_views")
        if len(result["preds"]) != len(indices):
            raise ValueError("Fast3R returned an incomplete set of views")
        for j, (index, pred) in enumerate(zip(indices, result["preds"])):
            xyz = array(pred["pts3d_in_other_view"])[0]
            confidence = array(pred["conf"])[0]
            valid = np.isfinite(xyz).all(-1) & np.isfinite(confidence) & (confidence > float(self.params.get("min_confidence", 1.0)))
            if valid.sum() < 8:
                raise ValueError(f"Fast3R view {j} has too few confident points for pose estimation")
            focal, pose = fast_pnp(torch.from_numpy(xyz), None, torch.from_numpy(valid), "cpu",
                                   pp=None, niter_PnP=int(self.params.get("pnp_iterations", 10)))
            if focal is None or pose is None:
                raise ValueError(f"Fast3R PnP failed for view {j}; no identity fallback is written")
            pose = array(pose).copy(); pose[:3, 3] *= scale
            height, width = xyz.shape[:2]
            output.append({**frame_fields(clip, int(index)), "view_index": j,
                "camera_to_world": pose.reshape(-1).tolist(),
                "intrinsics": [float(focal), 0., width / 2, 0., float(focal), height / 2, 0., 0., 1.],
                "image_width": width, "image_height": height,
                "pointcloud_path": "reconstruction-pointcloud.npz", "coord_frame": "first_view_world",
                "length_unit": "meters" if calibrated else "relative", "scale_status": "calibrated" if calibrated else "unscaled"})
            selected = np.flatnonzero(valid.ravel())
            step = max(1, int(np.ceil(len(selected) / max(1, budget // views))))
            selected = selected[::step]
            points.append(xyz.reshape(-1, 3)[selected] * scale)
            confidences.append(confidence.ravel()[selected])
            point_views.append(np.full(len(selected), j, dtype=np.int32))
        return {"tables": {"reconstruction": output}, "arrays": {"reconstruction-pointcloud.npz": {
            "points": np.concatenate(points), "confidence": np.concatenate(confidences), "view_index": np.concatenate(point_views)}},
            "stats": {"views": len(output), "points": sum(map(len, points))}}
