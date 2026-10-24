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
    flat = (mask.ravel(order="F") != 0).astype(np.uint8)
    boundaries = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": [h, w], "counts": counts}
