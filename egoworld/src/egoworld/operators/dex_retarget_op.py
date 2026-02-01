"""DexRetargeting operator stub."""

from __future__ import annotations

from typing import Any, Dict

from egoworld.operators.base import Operator


class DexRetargetOperator(Operator):
    name = "dex_retarget"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def run(self, hand_pose: Dict[str, Any]) -> Dict[str, Any]:
        return {"mapping": []}
