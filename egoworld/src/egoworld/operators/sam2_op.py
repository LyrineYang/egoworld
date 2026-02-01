"""SAM2 operator stub."""

from __future__ import annotations

from typing import Any, Dict

from egoworld.operators.base import Operator


class Sam2Operator(Operator):
    name = "sam2"

    def __init__(self, **params: Any):
        self.params = params

    def run(
        self,
        video_path: str,
        start_s: float,
        end_s: float,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        params = params or self.params
        return {
            "frames": [],
            "mask_encoding": "rle",
            "empty_mask_rate": 1.0,
            "start_s": start_s,
            "end_s": end_s,
            "video_path": video_path,
            "params_used": params,
        }
