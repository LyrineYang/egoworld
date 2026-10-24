"""Build video and clip manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import math
import subprocess

from egoworld.config import SceneDetectConfig
from egoworld.utils.hashing import sha256_file, sha256_text
from egoworld.utils.video import frames_from_seconds, seconds_from_frames


@dataclass
class VideoMeta:
    video_id: str
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    audio: bool
    checksum: str
    split: str


def run_ffprobe(path: str) -> Dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _parse_fps(fps_text: str, path: str) -> float:
    if "/" in fps_text:
        try:
            num, den = fps_text.split("/", 1)
            num_f = float(num)
            den_f = float(den)
        except Exception as exc:
            raise ValueError(f"Invalid r_frame_rate '{fps_text}' for {path}") from exc
        if den_f == 0:
            raise ValueError(f"Invalid r_frame_rate '{fps_text}' for {path} (division by zero)")
        fps = num_f / den_f
    else:
        try:
            fps = float(fps_text)
        except Exception as exc:
            raise ValueError(f"Invalid r_frame_rate '{fps_text}' for {path}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps {fps} from r_frame_rate '{fps_text}' for {path}")
    return fps


def parse_video_meta(path: str, split: str = "train") -> VideoMeta:
    data = run_ffprobe(path)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    fps_text = video_stream.get("r_frame_rate", "0/1")
    fps = _parse_fps(fps_text, path)
    # Container duration may include a longer audio stream. For CFR input,
    # the video frame count is the authoritative processing boundary.
    count = video_stream.get("nb_frames")
    if count not in (None, "N/A"):
        duration_s = int(count) / fps
    elif video_stream.get("duration") not in (None, "N/A"):
        duration_s = float(video_stream["duration"])
    elif video_stream.get("duration_ts") is not None and video_stream.get("time_base"):
        num, den = video_stream["time_base"].split("/")
        duration_s = int(video_stream["duration_ts"]) * float(num) / float(den)
    else:
        # Some containers omit video duration/count. Count once rather than
        # guessing from audio; this scan is only needed for those inputs.
        counted = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", path],
            check=True, capture_output=True, text=True)
        duration_s = int(json.loads(counted.stdout)["streams"][0]["nb_read_frames"]) / fps
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"Invalid video duration for {path}")
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    audio = any(s.get("codec_type") == "audio" for s in streams)
    checksum = sha256_file(path)
    video_id = f"{Path(path).stem}-{checksum[:8]}"
    return VideoMeta(
        video_id=video_id,
        path=str(Path(path).resolve()),
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        audio=audio,
        checksum=checksum,
        split=split,
    )


def detect_scenes(path: str, duration_s: float, config: SceneDetectConfig) -> Tuple[List[Tuple[float, float]], bool]:
    """Return list of (start_s, end_s) and a flag if fallback was used."""
    if config.method != "scenedetect":
        if config.fallback_full_clip:
            return [(0.0, duration_s)], True
        return [], False
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception:
        if config.fallback_full_clip:
            return [(0.0, duration_s)], True
        return [], False

    try:
        video_manager = VideoManager([path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        clips = []
        for start, end in scene_list:
            clips.append((start.get_seconds(), end.get_seconds()))
        if config.min_scene_len_s > 0:
            clips = [c for c in clips if (c[1] - c[0]) >= config.min_scene_len_s]
        if not clips:
            if config.fallback_full_clip:
                return [(0.0, duration_s)], True
            return [], False
        return clips, False
    except Exception:
        if config.fallback_full_clip:
            return [(0.0, duration_s)], True
        return [], False
    finally:
        try:
            video_manager.release()
        except Exception:
            pass


def build_manifests(
    video_paths: Iterable[str],
    split: str = "train",
    scenedetect: SceneDetectConfig | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scenedetect = scenedetect or SceneDetectConfig()
    if scenedetect.max_clip_duration_s <= 0 or scenedetect.overlap_s < 0:
        raise ValueError("Clip duration must be positive and overlap nonnegative")
    video_manifest: List[Dict[str, object]] = []
    clip_manifest: List[Dict[str, object]] = []

    for path in video_paths:
        meta = parse_video_meta(path, split=split)
        video_manifest.append(meta.__dict__)
        clips, used_fallback = detect_scenes(path, meta.duration_s, scenedetect)
        overlap = scenedetect.overlap_s
        bounded_clips = []
        max_frames = max(1, int(math.floor(scenedetect.max_clip_duration_s * meta.fps)))
        total_frames = frames_from_seconds(meta.duration_s, meta.fps)
        overlap_frames = int(round(overlap * meta.fps))
        if overlap_frames >= max_frames:
            raise ValueError("overlap_s must be shorter than max_clip_duration_s")
        for start_s, end_s in clips:
            first = max(0, frames_from_seconds(start_s, meta.fps) - overlap_frames)
            last = min(total_frames, frames_from_seconds(end_s, meta.fps))
            while first < last:
                end = min(first + max_frames, last)
                bounded_clips.append((first, end))
                if end == last:
                    break
                first = end - overlap_frames
        for frame_start, frame_end in sorted(set(bounded_clips)):
            start_s_aligned = seconds_from_frames(frame_start, meta.fps)
            end_s_aligned = seconds_from_frames(frame_end, meta.fps)
            clip_hash = sha256_text(f"{meta.video_id}:{frame_start}:{frame_end}")[:8]
            clip_id = f"{meta.video_id}-{frame_start:09d}-{frame_end:09d}-{clip_hash}"
            clip_manifest.append(
                {
                    "clip_id": clip_id,
                    "video_id": meta.video_id,
                    "start_s": float(start_s_aligned),
                    "end_s": float(end_s_aligned),
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end),
                    "overlap_s": float(overlap),
                    "scenedetect_failed": bool(used_fallback),
                    "status": "Pending",
                    "last_error": "",
                    "retry_count": 0,
                }
            )

    return video_manifest, clip_manifest


def write_manifest_json(path: str, rows: List[Dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
