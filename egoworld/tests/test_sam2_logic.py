import numpy as np

from egoworld.operators.sam2_op import _filter_boxes, _load_prompt_config, _union_masks


def test_prompt_config_defaults() -> None:
    cfg = _load_prompt_config({})
    assert cfg.source == "groundingdino"
    assert cfg.prompt_text == "hand ."
    assert cfg.prompt_interval_s == 2.0
    assert cfg.max_prompts_per_clip == 60
    assert cfg.max_boxes_per_frame == 6


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
