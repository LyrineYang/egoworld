"""GroundingDINO operator wrapper for text-conditioned detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import tempfile
import os

import numpy as np


@dataclass
class Detection:
    box_xyxy: Tuple[float, float, float, float]
    score: float
    phrase: str


class GroundingDINOOperator:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._use_model_class = False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from groundingdino.util.inference import Model  # type: ignore

            self._model = Model(self.config_path, self.checkpoint_path, device=self.device)
            self._use_model_class = True
            return
        except Exception:
            self._use_model_class = False

        from groundingdino.util.inference import load_model  # type: ignore

        self._model = load_model(self.config_path, self.checkpoint_path, device=self.device)

    def predict(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        max_boxes: int = 20,
    ) -> List[Detection]:
        self._ensure_model()
        if self._use_model_class:
            return self._predict_with_model(image_rgb, prompt, box_threshold, text_threshold, max_boxes)
        return self._predict_with_functions(image_rgb, prompt, box_threshold, text_threshold, max_boxes)

    def _predict_with_model(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        max_boxes: int,
    ) -> List[Detection]:
        model = self._model
        if not hasattr(model, "predict_with_caption"):
            return self._predict_with_functions(image_rgb, prompt, box_threshold, text_threshold, max_boxes)
        boxes, scores, phrases = model.predict_with_caption(  # type: ignore[attr-defined]
            image_rgb,
            prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return _to_detections(boxes, scores, phrases, image_rgb.shape[:2], max_boxes)

    def _predict_with_functions(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        max_boxes: int,
    ) -> List[Detection]:
        from groundingdino.util.inference import load_image, predict  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            tmp_path = handle.name
        try:
            _write_image(tmp_path, image_rgb)
            image_source, image = load_image(tmp_path)
            boxes, logits, phrases = predict(
                model=self._model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            return _to_detections(boxes, logits, phrases, image_source.shape[:2], max_boxes)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _write_image(path: str, image_rgb: np.ndarray) -> None:
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def _to_detections(
    boxes: Any,
    scores: Any,
    phrases: Any,
    image_hw: Tuple[int, int],
    max_boxes: int,
) -> List[Detection]:
    h, w = image_hw
    boxes = np.asarray(boxes)
    scores = np.asarray(scores).reshape(-1)
    phrases = list(phrases)

    if boxes.size == 0:
        return []

    boxes = _normalize_boxes(boxes, w, h)
    order = np.argsort(scores)[::-1]
    detections: List[Detection] = []
    for idx in order[:max_boxes]:
        x1, y1, x2, y2 = boxes[idx]
        detections.append(
            Detection(
                box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                score=float(scores[idx]),
                phrase=str(phrases[idx]) if idx < len(phrases) else "",
            )
        )
    return detections


def _normalize_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = boxes.astype(np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    # Heuristic: if x2 < x1 for most rows, assume cxcywh
    if np.mean(boxes[:, 2] < boxes[:, 0]) > 0.5:
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    # If normalized, scale to pixel
    if boxes.max() <= 1.5:
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes
