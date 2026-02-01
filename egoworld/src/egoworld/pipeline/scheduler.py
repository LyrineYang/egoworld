"""Clip scheduling utilities."""

from __future__ import annotations

from typing import Iterable, List


def sort_clips_by_duration(clips: Iterable[dict]) -> List[dict]:
    return sorted(clips, key=lambda c: (c.get("end_s", 0) - c.get("start_s", 0)), reverse=True)
