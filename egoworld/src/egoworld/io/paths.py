"""Output path helpers."""

from __future__ import annotations

from pathlib import Path


def run_dir(output_root: str, run_id: str) -> Path:
    return Path(output_root) / f"run_id={run_id}"


def clip_dir(output_root: str, run_id: str, video_id: str, clip_id: str) -> Path:
    return run_dir(output_root, run_id) / f"video_id={video_id}" / f"clip_id={clip_id}"
