"""Mask encoding utilities."""

from __future__ import annotations

from typing import Any, Dict
import json

import numpy as np


def encode_mask_rle(mask: np.ndarray) -> str:
    mask = mask.astype(np.uint8)
    try:
        from pycocotools import mask as mask_utils  # type: ignore

        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        return json.dumps(rle, ensure_ascii=True)
    except Exception:
        return json.dumps(_simple_rle(mask), ensure_ascii=True)


def _simple_rle(mask: np.ndarray) -> Dict[str, Any]:
    h, w = mask.shape
    counts = []
    flat = mask.flatten(order="F")
    prev = 0
    run = 0
    for val in flat:
        if val == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = val
    counts.append(run)
    return {"size": [h, w], "counts": counts}
