"""Scene detection operator (wrapper)."""

from __future__ import annotations

from typing import List, Tuple

from egoworld.config import SceneDetectConfig
from egoworld.manifests.build_manifest import detect_scenes
from egoworld.operators.base import Operator


class SceneDetectOperator(Operator):
    name = "scenedetect"

    def __init__(self, config: SceneDetectConfig | None = None):
        self.config = config or SceneDetectConfig()

    def run(self, video_path: str, duration_s: float) -> Tuple[List[Tuple[float, float]], bool]:
        return detect_scenes(video_path, duration_s, self.config)
