"""Video time utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


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


def get_video_info(path: str) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def iter_frames(
    path: str,
    start_s: float,
    end_s: float,
    stride: int,
) -> Generator[Tuple[int, float, "np.ndarray"], None, None]:
    import cv2
    import numpy as np

    if stride <= 0:
        stride = 1

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    start_frame = frames_from_seconds(start_s, fps)
    end_frame = frames_from_seconds(end_s, fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_s = seconds_from_frames(frame_idx, fps)
            yield frame_idx - start_frame, timestamp_s, frame_rgb
        frame_idx += 1
    cap.release()
