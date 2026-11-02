"""Fast3R multiview inference; explicit relative scale and non-fabricated PnP poses."""
from __future__ import annotations
from pathlib import Path
import tempfile
import time
import numpy as np
from egoworld.utils.timing import phase, timed
from egoworld.utils.performance import timed_load
from egoworld.operators.common import array, frame_reader, read_frame, frame_fields


class Fast3ROperator:
    name = "fast3r"

    def __init__(self, **params):
        self.params = params
        self.model = None

    @timed_load
    def _load(self):
        import torch
        from fast3r.models.fast3r import Fast3R
        device = torch.device(self.params.get("device", "cuda"))
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

    @timed("view_prepare")
    def prepare(self, clip, video_path):
        import cv2
        from fast3r.dust3r.utils.image import load_images
        count = clip["frame_end"] - clip["frame_start"]
        views = min(int(self.params.get("max_views", 8)), count)
        if views < 2:
            raise ValueError("Fast3R requires at least two frames")
        indices = np.unique(np.linspace(0, count - 1, views).astype(int))
        cap = frame_reader(video_path)
        try:
            with tempfile.TemporaryDirectory(prefix="egoworld-fast3r-") as folder:
                files, source_sizes = [], []
                for index in indices:
                    path = str(Path(folder) / f"{index:09d}.jpg")
                    frame = read_frame(cap, int(index))
                    source_sizes.append((frame.shape[1], frame.shape[0]))
                    with phase("jpeg_encode"):
                        if not cv2.imwrite(path, frame):
                            raise IOError(f"Cannot write sampled view {path}")
                    files.append(path)
                with phase("official_preprocess"):
                    images = load_images(files, size=int(self.params.get("image_size", 512)), verbose=False)
        finally:
            cap.release()
        return {"images": images, "indices": indices, "source_sizes": source_sizes}

    def infer(self, data):
        import torch
        from fast3r.dust3r.inference_multiview import inference
        self._load()
        images = data["images"]
        # Only img was a Torch tensor in the official loader; true_shape is NumPy.
        for view in images:
            if isinstance(view, dict) and isinstance(view.get("img"), np.ndarray):
                view["img"] = torch.from_numpy(view["img"])
        precision = {"fp32": "32", "fp16": "16-mixed", "bf16": "bf16-mixed"}[self.params.get("dtype", "fp32")]
        # Official API includes H2D and to_cpu. Name the combined span honestly.
        with phase("forward_and_transfers", cuda=True), torch.inference_mode():
            result = inference(images, self.model, torch.device(self.params.get("device", "cuda")),
                               dtype=precision, verbose=False, profiling=False)
        if isinstance(result, tuple):
            result = result[0]
        with phase("prediction_arrays"):
            preds = [{key: array(pred[key]) for key in ("pts3d_in_other_view", "conf")} for pred in result["preds"]]
        return {"preds": preds, "indices": data["indices"], "source_sizes": data["source_sizes"]}

    @timed("reconstruction_postprocess")
    def postprocess(self, clip, data):
        import torch
        from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
        indices, source_sizes = data["indices"], data["source_sizes"]
        views, result = len(indices), data
        scale = float(self.params.get("metric_scale", 1.0))
        calibrated = "metric_scale" in self.params
        if not np.isfinite(scale) or scale <= 0 or (calibrated and not self.params.get("metric_scale_source")):
            raise ValueError("Metric scaling requires a positive factor and metric_scale_source")
        output, points, confidences, point_views, source_pixels = [], [], [], [], []
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
            with phase("pnp"):
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
            if self.params.get("save_correspondences", False):
                from egoworld.product.geometry import fast3r_source_pixels
                pixels = np.stack([selected % width, selected // width], axis=1)
                source_pixels.append(fast3r_source_pixels(pixels, source_sizes[j],
                    int(self.params.get("image_size", 512)), (width, height)))
        arrays = {"reconstruction-pointcloud.npz": {
            "points": np.concatenate(points), "confidence": np.concatenate(confidences), "view_index": np.concatenate(point_views)}}
        if source_pixels:
            arrays["reconstruction-correspondences.npz"] = {
                **arrays["reconstruction-pointcloud.npz"], "source_pixel_xy": np.concatenate(source_pixels)}
        return {"tables": {"reconstruction": output}, "arrays": arrays,
            "stats": {"views": len(output), "points": sum(map(len, points))}}

    def run(self, clip, video_path, inputs):
        return self.postprocess(clip, self.infer(self.prepare(clip, video_path)))
