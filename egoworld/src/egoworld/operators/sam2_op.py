"""SAM2 operator for video mask propagation with optional GroundingDINO prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import subprocess
import tempfile

import numpy as np

from egoworld.operators.base import Operator
from egoworld.operators.groundingdino_op import GroundingDINOOperator
from egoworld.utils.mask import encode_mask_rle
from egoworld.utils.video import get_video_info, iter_frames, seconds_from_frames


@dataclass
class PromptConfig:
    source: str = "groundingdino"
    prompt_text: str = "hand ."
    prompt_interval_s: float = 2.0
    max_prompts_per_clip: int = 60
    max_boxes_per_frame: int = 6
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_iou: float = 0.5
    min_box_area: float = 256.0
    gd_config: str = "./models/groundingdino/GroundingDINO_SwinT_OGC.py"
    gd_checkpoint: str = "./models/groundingdino/groundingdino_swint_ogc.pth"
    gd_device: str = "cuda"


class Sam2Operator(Operator):
    name = "sam2"

    def __init__(self, **params: Any):
        self.params = params
        self._predictor = None
        self._gd = None

    def _ensure_predictor(self) -> None:
        if self._predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        except Exception as exc:
            raise RuntimeError("SAM2 is not installed or not on PYTHONPATH") from exc

        checkpoint = self.params.get("checkpoint")
        model_cfg = _resolve_model_cfg(self.params.get("config"))
        device = self.params.get("device", "cuda")
        vos_optimized = bool(self.params.get("vos_optimized", False))

        if not checkpoint or not model_cfg:
            raise RuntimeError("SAM2 requires 'checkpoint' and 'config' in params")

        try:
            self._predictor = build_sam2_video_predictor(
                model_cfg,
                checkpoint,
                device=device,
                vos_optimized=vos_optimized,
            )
        except TypeError:
            self._predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    def _ensure_groundingdino(self, prompt_cfg: PromptConfig) -> GroundingDINOOperator:
        if self._gd is None:
            self._gd = GroundingDINOOperator(
                prompt_cfg.gd_config,
                prompt_cfg.gd_checkpoint,
                device=prompt_cfg.gd_device,
            )
        return self._gd

    def run(
        self,
        video_path: str,
        start_s: float,
        end_s: float,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        params = params or self.params
        self._ensure_predictor()
        predictor = self._predictor

        precision = params.get("precision", "bf16")
        device = params.get("device", "cuda")
        clip_path = _extract_clip(video_path, start_s, end_s)
        video_info = get_video_info(clip_path)
        fps = video_info.fps or 30.0

        prompt_cfg = _load_prompt_config(params.get("prompting", {}))
        prompt_frames = _collect_prompt_frames(
            clip_path,
            prompt_cfg.prompt_interval_s,
            prompt_cfg.max_prompts_per_clip,
        )

        if not prompt_frames:
            return _empty_result(video_path, start_s, end_s)

        gd = None
        if prompt_cfg.source == "groundingdino":
            gd = self._ensure_groundingdino(prompt_cfg)

        import torch

        autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        device_type = "cuda" if "cuda" in device else "cpu"

        try:
            with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                state = _init_state_with_fallback(predictor, clip_path)
                obj_id = 1
                tracked_boxes: Dict[int, Tuple[float, float, float, float]] = {}

                for frame_idx, _, frame_rgb in prompt_frames:
                    detections: List[Tuple[float, float, float, float]] = []
                    if gd is not None:
                        results = gd.predict(
                            frame_rgb,
                            prompt_cfg.prompt_text,
                            box_threshold=prompt_cfg.box_threshold,
                            text_threshold=prompt_cfg.text_threshold,
                            max_boxes=prompt_cfg.max_boxes_per_frame,
                        )
                        detections = [d.box_xyxy for d in results]

                    detections = _filter_boxes(detections, prompt_cfg.min_box_area, prompt_cfg.nms_iou)

                    for box in detections:
                        matched_id = _match_box(tracked_boxes, box, iou_threshold=0.5)
                        if matched_id is None:
                            matched_id = obj_id
                            obj_id += 1
                        tracked_boxes[matched_id] = box
                        _add_box_prompt(predictor, state, frame_idx, matched_id, box)

                if not tracked_boxes:
                    return _empty_result(video_path, start_s, end_s)

                frames: List[Dict[str, Any]] = []
                empty_count = 0
                total_count = 0

                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(state):
                    total_count += 1
                    mask = _union_masks(out_mask_logits)
                    if mask is None:
                        empty_count += 1
                        continue
                    rle = encode_mask_rle(mask)
                    frame_index = out_frame_idx + int(round(start_s * fps))
                    frames.append(
                        {
                            "frame_index": int(frame_index),
                            "timestamp_s": float(seconds_from_frames(frame_index, fps)),
                            "mask_rle": rle,
                        }
                    )

                empty_rate = empty_count / max(1, total_count)
                return {
                    "frames": frames,
                    "mask_encoding": "rle",
                    "empty_mask_rate": float(empty_rate),
                    "start_s": start_s,
                    "end_s": end_s,
                    "video_path": video_path,
                }
        finally:
            if clip_path != video_path:
                try:
                    os.remove(clip_path)
                except OSError:
                    pass


def _load_prompt_config(raw: Dict[str, Any]) -> PromptConfig:
    return PromptConfig(
        source=raw.get("source", "groundingdino"),
        prompt_text=raw.get("prompt_text", "hand ."),
        prompt_interval_s=float(raw.get("prompt_interval_s", 2.0)),
        max_prompts_per_clip=int(raw.get("max_prompts_per_clip", 60)),
        max_boxes_per_frame=int(raw.get("max_boxes_per_frame", 6)),
        box_threshold=float(raw.get("box_threshold", 0.35)),
        text_threshold=float(raw.get("text_threshold", 0.25)),
        nms_iou=float(raw.get("nms_iou", 0.5)),
        min_box_area=float(raw.get("min_box_area", 256)),
        gd_config=raw.get("gd_config", "./models/groundingdino/GroundingDINO_SwinT_OGC.py"),
        gd_checkpoint=raw.get(
            "gd_checkpoint", "./models/groundingdino/groundingdino_swint_ogc.pth"
        ),
        gd_device=raw.get("gd_device", "cuda"),
    )


def _collect_prompt_frames(
    video_path: str,
    interval_s: float,
    max_prompts: int,
) -> List[Tuple[int, float, np.ndarray]]:
    info = get_video_info(video_path)
    fps = info.fps or 30.0
    stride = max(1, int(round(interval_s * fps)))
    frames = []
    for frame_idx, timestamp_s, frame_rgb in iter_frames(video_path, 0.0, 1e9, stride):
        frames.append((frame_idx, timestamp_s, frame_rgb))
        if len(frames) >= max_prompts:
            break
    return frames


def _extract_clip(video_path: str, start_s: float, end_s: float) -> str:
    if start_s <= 0 and end_s <= 0:
        return video_path
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, start_s)),
        "-to",
        str(max(0.0, end_s)),
        "-i",
        video_path,
        "-c",
        "copy",
        "-an",
        tmp.name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp.name
    except Exception:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        return video_path


def _init_state_with_fallback(predictor: Any, clip_path: str) -> Any:
    try:
        return predictor.init_state(clip_path)
    except Exception:
        return predictor.init_state(video_path=clip_path)


def _add_box_prompt(predictor: Any, state: Any, frame_idx: int, obj_id: int, box: Tuple[float, float, float, float]) -> None:
    try:
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=np.array(box, dtype=np.float32),
        )
    except Exception:
        predictor.add_new_points_or_box(state, frame_idx, obj_id, box)


def _union_masks(mask_logits: Any) -> np.ndarray | None:
    if mask_logits is None:
        return None
    masks = mask_logits
    if hasattr(mask_logits, "sigmoid"):
        masks = (mask_logits > 0).detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.size == 0:
        return None
    if masks.ndim == 2:
        return masks.astype(np.uint8)
    union = np.any(masks > 0, axis=0)
    return union.astype(np.uint8)


def _filter_boxes(
    boxes: List[Tuple[float, float, float, float]],
    min_area: float,
    nms_iou: float,
) -> List[Tuple[float, float, float, float]]:
    boxes = [b for b in boxes if _box_area(b) >= min_area]
    if not boxes:
        return []
    return _nms(boxes, nms_iou)


def _box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _nms(boxes: List[Tuple[float, float, float, float]], iou_threshold: float) -> List[Tuple[float, float, float, float]]:
    if not boxes:
        return []
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(tuple(boxes_np[i]))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def _match_box(
    tracked: Dict[int, Tuple[float, float, float, float]],
    box: Tuple[float, float, float, float],
    iou_threshold: float,
) -> int | None:
    best_id = None
    best_iou = 0.0
    for obj_id, prev in tracked.items():
        iou = _iou(prev, box)
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_id = obj_id
    return best_id


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _empty_result(video_path: str, start_s: float, end_s: float) -> Dict[str, Any]:
    return {
        "frames": [],
        "mask_encoding": "rle",
        "empty_mask_rate": 1.0,
        "start_s": start_s,
        "end_s": end_s,
        "video_path": video_path,
    }


def _resolve_model_cfg(config_path: str | None) -> str | None:
    if config_path and os.path.isfile(config_path):
        return config_path
    if config_path:
        base_name = os.path.basename(config_path)
    else:
        base_name = ""
    try:
        import sam2  # type: ignore

        root = os.path.dirname(sam2.__file__)
        candidates = []
        if base_name:
            candidates.append(os.path.join(root, "configs", "sam2.1", base_name))
            candidates.append(os.path.join(root, "configs", base_name))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        return config_path
    return config_path
