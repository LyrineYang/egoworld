"""SAM2 operator stub."""

from __future__ import annotations

from typing import Any, Dict, List

from egoworld.operators.base import Operator


class Sam2Operator(Operator):
    name = "sam2"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def run(self, video_path: str, start_s: float, end_s: float) -> Dict[str, Any]:
        return {
            "frames": [],
            "mask_encoding": "rle",
            "empty_mask_rate": 1.0,
            "start_s": start_s,
            "end_s": end_s,
            "video_path": video_path,
        }
