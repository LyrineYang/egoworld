"""Small shared helpers; no model dependency is imported at module load."""
from pathlib import Path
import json
import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rows(receipts, stage, artifact):
    import pyarrow.parquet as pq
    receipt = receipts[stage]
    return pq.ParquetFile(Path(receipt["directory"]) / f"{artifact}.parquet").read().to_pylist()


def rgbd_assets(path):
    path = Path(path).resolve()
    content = json.loads(path.read_text())
    for item in content.values():
        if "depth_pattern" not in item:
            continue  # report calibration errors before attempting to read depth
        pattern = Path(item["depth_pattern"]).expanduser()
        item["depth_pattern"] = str(pattern if pattern.is_absolute() else path.parent / pattern)
    return content


def frame_reader(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def read_frame(cap, index):
    import cv2
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}")
    return frame


def frame_fields(clip, local_index):
    index = clip["frame_start"] + int(local_index)
    return {"frame_index": index, "timestamp_s": index / clip["fps"]}


def decode_rle(text):
    import json
    rle = json.loads(text)
    if isinstance(rle["counts"], str):
        from pycocotools import mask as mask_utils
        return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)
    total = int(np.prod(rle["size"]))
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if np.any(counts < 0) or counts.sum() != total:
        raise ValueError("Invalid RLE counts")
    return np.repeat(np.arange(len(counts)) % 2, counts).reshape(rle["size"], order="F").astype(bool)
