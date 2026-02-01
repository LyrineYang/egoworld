"""Quality control checks for output distributions."""

from __future__ import annotations

from typing import Dict, Iterable, List


def empty_mask_rate(masks: Iterable[Dict[str, object]]) -> float:
    masks = list(masks)
    if not masks:
        return 1.0
    empty = sum(1 for m in masks if not m)
    return empty / max(1, len(masks))


def distribution_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    total = sum(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": total / len(values),
    }
