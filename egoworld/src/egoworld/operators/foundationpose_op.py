"""FoundationPose operator stub."""

from __future__ import annotations

from typing import Any, Dict

from egoworld.operators.base import Operator


class FoundationPoseOperator(Operator):
    name = "foundationpose"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def run(self, video_path: str, start_s: float, end_s: float) -> Dict[str, Any]:
        return {"object_pose": []}
