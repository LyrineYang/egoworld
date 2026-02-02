import pytest

from egoworld.config import SceneDetectConfig
from egoworld.manifests import build_manifest as bm
from egoworld.utils.video import seconds_from_frames


def _fake_meta(path: str, split: str = "train") -> bm.VideoMeta:
    return bm.VideoMeta(
        video_id="video-abc",
        path=path,
        duration_s=10.0,
        fps=30.0,
        width=1920,
        height=1080,
        audio=False,
        checksum="deadbeef",
        split=split,
    )


def _fake_scenes(path: str, duration_s: float, config: SceneDetectConfig):
    return [(0.5, 2.0), (8.9, 9.8)], False


def test_build_manifests_clip_id_deterministic_and_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "parse_video_meta", _fake_meta)
    monkeypatch.setattr(bm, "detect_scenes", _fake_scenes)

    config = SceneDetectConfig(overlap_s=1.0)
    _, clips_first = bm.build_manifests(["/tmp/a.mp4"], scenedetect=config)
    _, clips_second = bm.build_manifests(["/tmp/a.mp4"], scenedetect=config)

    assert [c["clip_id"] for c in clips_first] == [c["clip_id"] for c in clips_second]

    for clip in clips_first:
        assert 0.0 <= clip["start_s"] <= clip["end_s"] <= 10.0
        assert clip["start_s"] == seconds_from_frames(clip["frame_start"], 30.0)
        assert clip["end_s"] == seconds_from_frames(clip["frame_end"], 30.0)
        assert clip["scenedetect_failed"] is False


def test_detect_scenes_non_scenedetect_fallback() -> None:
    config = SceneDetectConfig(method="none", fallback_full_clip=True)
    clips, used_fallback = bm.detect_scenes("/tmp/a.mp4", 10.0, config)
    assert clips == [(0.0, 10.0)]
    assert used_fallback is True


def test_detect_scenes_non_scenedetect_no_fallback() -> None:
    config = SceneDetectConfig(method="none", fallback_full_clip=False)
    clips, used_fallback = bm.detect_scenes("/tmp/a.mp4", 10.0, config)
    assert clips == []
    assert used_fallback is False


def test_parse_video_meta_rejects_invalid_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_ffprobe(path: str) -> dict:
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "r_frame_rate": "0/0",
                    "width": 1920,
                    "height": 1080,
                }
            ],
            "format": {"duration": "10.0"},
        }

    monkeypatch.setattr(bm, "run_ffprobe", _fake_ffprobe)
    with pytest.raises(ValueError):
        bm.parse_video_meta("/tmp/a.mp4")
