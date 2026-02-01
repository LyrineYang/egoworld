"""Fast3R operator stub."""

from __future__ import annotations

from typing import Any, Dict

from egoworld.operators.base import Operator


class Fast3ROperator(Operator):
    name = "fast3r"

    def __init__(self, model_name_or_path: str | None = None):
        self.model_name_or_path = model_name_or_path

    def run(self, video_path: str, start_s: float, end_s: float, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "camera_poses": [],
            "pointcloud_path": "",
            "video_path": video_path,
            "start_s": start_s,
            "end_s": end_s,
        }
