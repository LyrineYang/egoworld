"""Video time utilities."""

from __future__ import annotations


def frames_from_seconds(seconds: float, fps: float) -> int:
    if fps <= 0:
        return 0
    return int(round(seconds * fps))


def seconds_from_frames(frame_index: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame_index) / float(fps)


def validate_time_alignment(start_s: float, end_s: float, frame_start: int, frame_end: int, fps: float) -> bool:
    return (
        abs(start_s - seconds_from_frames(frame_start, fps)) < 1e-6
        and abs(end_s - seconds_from_frames(frame_end, fps)) < 1e-6
    )
