import numpy as np
import json
import shutil
import subprocess
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest

from egoworld.operators.sam2_op import _filter_boxes, _load_prompt_config, _union_masks
from egoworld.operators import sam2_op
from egoworld.utils.errors import DecodeError
from egoworld.utils.mask import encode_mask_rle


def test_prompt_config_defaults() -> None:
    cfg = _load_prompt_config({})
    assert cfg.source == "groundingdino"
    assert cfg.prompt_text == "hand ."
    assert cfg.prompt_interval_s == 2.0
    assert cfg.max_prompts_per_clip == 60
    assert cfg.max_boxes_per_frame == 6


@pytest.mark.parametrize("mask,counts", [
    (np.array([[0, 1, 0], [0, 1, 1]]), [2, 2, 1, 1]),
    (np.zeros((2, 3)), [6]), (np.ones((2, 3)), [0, 6]),
])
def test_fallback_rle_matches_coco_column_order(mask, counts):
    from egoworld.utils.mask import _simple_rle

    assert _simple_rle(mask) == {"size": [2, 3], "counts": counts}


def test_filter_boxes_min_area() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 30.0, 30.0)]
    filtered = _filter_boxes(boxes, min_area=200.0, nms_iou=1.0)
    assert filtered == [(0.0, 0.0, 30.0, 30.0)]


def test_union_masks() -> None:
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1
    masks[1, 1, 1] = 1
    union = _union_masks(masks)
    assert union is not None
    assert union[0, 0] == 1
    assert union[1, 1] == 1


@pytest.mark.parametrize("shape", [(4, 4), (2, 4, 4), (2, 1, 4, 4)])
def test_union_masks_encodes_real_predictor_shapes(shape):
    logits = np.full(shape, -1.0)
    logits[..., 1, 2] = 1.0
    union = _union_masks(logits)
    assert union.shape == (4, 4)
    assert union.sum() == 1
    assert union[1, 2] == 1
    assert json.loads(encode_mask_rle(union))["size"] == [4, 4]


def test_clip_extraction_failure_raises_and_removes_temp_file(monkeypatch, tmp_path):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(sam2_op.subprocess, "run", fail)
    with pytest.raises(DecodeError, match="Failed to extract clip"):
        sam2_op._extract_clip("source.mp4", 1.0, 2.0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (1, 1), (-1, -1)])
def test_invalid_clip_interval_fails(start, end):
    with pytest.raises(DecodeError, match="Invalid clip interval"):
        sam2_op._extract_clip("source.mp4", start, end)


def test_empty_prompt_frames_remove_extracted_clip(monkeypatch, tmp_path):
    clip = tmp_path / "extracted.mp4"
    clip.touch()
    monkeypatch.setattr(sam2_op, "_extract_clip", lambda *args: str(clip))
    monkeypatch.setattr(sam2_op, "get_video_info", lambda *args: type("Info", (), {"fps": 30.0})())
    monkeypatch.setattr(sam2_op, "_collect_prompt_frames", lambda *args: [])
    op = sam2_op.Sam2Operator()
    op._predictor = object()
    assert op.run("source.mp4", 1, 2)["frames"] == []
    assert not clip.exists()


def test_real_clip_extraction_preserves_frame_interval(tmp_path):
    cv2 = pytest.importorskip("cv2")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    source = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=3:size=64x64:rate=10", "-c:v", "libx264",
         "-crf", "0", "-g", "30", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True,
    )
    extracted = Path(sam2_op._extract_clip(str(source), 1.1, 2.1))
    try:
        original = cv2.VideoCapture(str(source))
        original.set(cv2.CAP_PROP_POS_FRAMES, 11)
        ok, expected = original.read()
        original.release()
        assert ok
        clip = cv2.VideoCapture(str(extracted))
        try:
            assert int(clip.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
            ok, actual = clip.read()
            assert ok
            np.testing.assert_array_equal(actual, expected)
        finally:
            clip.release()
    finally:
        extracted.unlink()


def test_operator_encodes_four_dimensional_masks_and_source_timestamps(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    logits = np.full((2, 1, 4, 4), -1.0)
    logits[0, 0, 1, 2] = 1.0
    predictor = SimpleNamespace(
        init_state=lambda path, **kwargs: object(),
        add_new_points_or_box=lambda *args, **kwargs: None,
        propagate_in_video=lambda state: iter([(0, [1, 2], logits), (1, [1, 2], np.full_like(logits, -1))]),
    )
    torch = SimpleNamespace(
        inference_mode=nullcontext, autocast=lambda **kwargs: nullcontext(),
        bfloat16="bf16", float16="fp16",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sam2_op, "_extract_clip", lambda *args: str(clip))
    monkeypatch.setattr(sam2_op, "get_video_info", lambda *args: SimpleNamespace(fps=30.0))
    monkeypatch.setattr(sam2_op, "_collect_prompt_frames", lambda *args: [(0, 0.0, np.zeros((4, 4, 3)))])
    op = sam2_op.Sam2Operator(prompting={"min_box_area": 0})
    op._predictor = predictor
    op._gd = SimpleNamespace(predict=lambda *args, **kwargs: [
        SimpleNamespace(box_xyxy=(0, 0, 3, 3), phrase="cup", score=.9),
        SimpleNamespace(box_xyxy=(0, 0, 3, 3), phrase="hand", score=.8),
    ])
    result = op.run("source.mp4", 1.0, 2.0)
    assert [frame["frame_index"] for frame in result["frames"]] == [30, 31]
    assert [frame["timestamp_s"] for frame in result["frames"]] == [1.0, 31 / 30]
    assert json.loads(result["frames"][0]["mask_rle"])["size"] == [4, 4]
    assert result["empty_mask_rate"] == 0.5
    assert [row["object_id"] for row in result["instances"]] == [1, 2, 1, 2]
    assert [row["label"] for row in result["instances"]] == ["cup", "hand", "cup", "hand"]
    assert [row["area_px"] for row in result["instances"]] == [1, 0, 0, 0]
    assert result["instances"][0]["detection_score"] == .9
    assert not clip.exists()
