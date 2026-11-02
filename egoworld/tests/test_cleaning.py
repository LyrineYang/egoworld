import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from egoworld.product.cleaning import screen_inputs, settings
from egoworld.utils.hashing import hash_session


def video(path, moving=False):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
    for i in range(12):
        frame = np.zeros((48, 64, 3), np.uint8)
        if moving:
            frame[8:32, 2 + i * 2:18 + i * 2] = 255
        writer.write(frame)
    writer.release()
    return path


def test_corrupt_duplicate_and_motion_policy(tmp_path):
    static = video(tmp_path / "static.mp4")
    moving = video(tmp_path / "moving.mp4", True)
    duplicate = tmp_path / "copy.mp4"; shutil.copyfile(moving, duplicate)
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a video")
    with hash_session():
        report = screen_inputs([str(p) for p in (static, moving, duplicate, bad)], {"min_motion": .00001, "reject_blank": False})
    assert report["accepted"] == [str(moving)]
    assert report["records"][0]["reasons"] == ["min_motion"]
    assert report["records"][2]["reasons"] == ["duplicate_content"]
    assert report["records"][3]["reasons"] == ["invalid_input"]
    assert report["records"][1]["statistics"]["decoded_frames"] == 12
    assert all(p.exists() for p in (static, moving, duplicate, bad))


def test_quality_cache_reuses_stats_but_rechecks_thresholds(tmp_path, monkeypatch):
    from egoworld.product import cleaning
    path = video(tmp_path / "stationary.mp4")
    first = screen_inputs([str(path)], {"reject_blank": False}, cache_root=tmp_path / "cache")
    assert first["accepted"] == [str(path)]
    monkeypatch.setattr(cleaning, "_statistics", lambda *a: pytest.fail("Reuse statistics, not decisions"))
    second = screen_inputs([str(path)], {"min_motion": .01, "reject_blank": False}, tmp_path / "cache")
    assert second["cache_hits"] == 1 and second["rejected"] == [str(path)]
    assert second["records"][0]["reasons"] == ["min_motion"]


@pytest.mark.parametrize("policy", [{"min_motion": float("nan")}, {"max_samples": 0}, {"decode": "guess"},
    {"min_brightness": 2}, {"min_motion": 2, "max_motion": 1}, {"unknown": True}])
def test_invalid_cleaning_policies_fail_before_input_reads(policy):
    with pytest.raises(ValueError):
        settings(policy)


def test_sample_mode_reports_limited_validation(tmp_path):
    path = video(tmp_path / "sample.mp4", True)
    report = screen_inputs([str(path)], {"decode": "sample", "max_samples": 2})
    stat = report["records"][0]["statistics"]
    assert stat["validation_scope"] == "sample" and stat["decoded_frames"] == 2


def test_default_rejects_uniform_blank_but_not_stationary_texture(tmp_path):
    blank = video(tmp_path / "blank.mp4")
    assert screen_inputs([str(blank)])["records"][0]["reasons"] == ["blank_video"]
    textured = tmp_path / "textured.mp4"
    writer = cv2.VideoWriter(str(textured), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
    frame = np.zeros((48, 64, 3), np.uint8); frame[8:32, 16:40] = 255
    for _ in range(12): writer.write(frame)
    writer.release()
    assert screen_inputs([str(textured)])["accepted"] == [str(textured)]


def test_adaptive_reads_match_seek_pixels(tmp_path):
    from egoworld.operators.frame_sampling import read_index
    path = video(tmp_path / "source.mp4", True)
    cap = cv2.VideoCapture(str(path)); expected = []
    for i in (0, 5, 10, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, frame = cap.read(); assert ok
        expected.append(frame)
    cap.release(); cap = cv2.VideoCapture(str(path))
    try:
        for i, frame in zip((0, 5, 10, 2), expected):
            assert np.array_equal(frame, read_index(cap, i))
    finally:
        cap.release()
