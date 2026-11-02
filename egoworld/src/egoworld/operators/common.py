"""Small shared helpers; no model dependency is imported at module load."""
from pathlib import Path
import json
import numpy as np
from egoworld.utils.timing import timed, phase


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()  # NumPy has no native torch.bfloat16 representation.
        value = value.numpy()
    return np.asarray(value)


def rows(receipts, stage, artifact, columns=None):
    """Compatibility helper; production adapters consume iter_rows instead."""
    return list(iter_rows(receipts, stage, artifact, columns))


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
    from egoworld.pipeline.input_cache import CURRENT
    cache = CURRENT.get()
    if cache is not None and str(Path(path).resolve()) == cache.spec["video_path"]:
        from egoworld.operators.cached_inputs import CachedFrameReader
        return CachedFrameReader(path, cache)
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def read_frame(cap, index):
    if hasattr(cap, "read_index"):
        return cap.read_index(index)
    from egoworld.operators.frame_sampling import read_index
    try:
        from egoworld.utils.timing import phase
    except ModuleNotFoundError:
        phase = None
    return read_index(cap, int(index), phase_factory=phase)


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


def iter_rows(receipts, stage, artifact, columns=None):
    import pyarrow.parquet as pq
    from contextlib import nullcontext
    try:
        from egoworld.utils.timing import phase
    except ModuleNotFoundError:
        phase = lambda name: nullcontext()
    receipt = receipts[stage]
    iterator = iter(pq.ParquetFile(Path(receipt["directory"]) / f"{artifact}.parquet").iter_batches(batch_size=512, columns=columns))
    while True:
        with phase("upstream_read"):
            try:
                data = next(iterator).to_pylist()
            except StopIteration:
                return
        yield from data
