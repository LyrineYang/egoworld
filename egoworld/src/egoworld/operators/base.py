"""Base operator interfaces."""

from __future__ import annotations

from typing import Any, Dict


class Operator:
    name = "operator"

    def run(self, *args, **kwargs) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError
