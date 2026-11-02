"""Cache adapters that preserve OpenCV decoding and official HaMeR transforms."""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path

from egoworld.pipeline.input_cache import CURRENT, identity


class CachedFrameReader:
    def __init__(self, path, cache):
        import cv2
        self.path, self.cache, self.cap = str(path), cache, None
        self.decoder = identity(["opencv-bgr-uint8", cv2.__version__, cv2.getBuildInformation()])
        info = Path(path).stat()
        if [info.st_size, info.st_mtime_ns] != cache.spec["video_stat"]:
            raise ValueError("Prepared video changed after cache identity was recorded")

    def read_index(self, index):
        import cv2
        if index < 0:
            raise ValueError("Frame index must be nonnegative")
        def decode():
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.path)
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open video: {self.path}")
            from egoworld.operators.frame_sampling import read_index
            from egoworld.utils.timing import phase
            return {"frame": read_index(self.cap, int(index), phase_factory=phase)}
        return self.cache.get("frame", [self.decoder, int(index)], decode)["frame"]

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def cached_hamer_dataset(dataset):
    """Cache the official __getitem__ result, including its normalization metadata.

    Crops belong to their exact transform/config, never to a generic bbox alone.
    Only the per-frame dataset wrapper lives in memory; arrays live in the bounded
    shared store. DataLoader remains num_workers=0 and keeps its original batches.
    """
    cache = CURRENT.get()
    if cache is None:
        return dataset
    location = Path(inspect.getfile(type(dataset))).resolve().parent
    source = hashlib.sha256()
    # Include all dataset helpers; official cropping uses functions in utils.py.
    for path in sorted(location.rglob("*.py")):
        source.update(str(path.relative_to(location)).encode())
        source.update(path.read_bytes())
    versions = {}
    for name in ("numpy", "scikit-image", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    import cv2
    transform = identity([source.hexdigest(), str(dataset.cfg), versions, cv2.getBuildInformation()])
    frame = hashlib.sha256(memoryview(dataset.img_cv2).cast("B")).hexdigest()

    class Dataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            key = [transform, frame, dataset.center[index].tolist(), dataset.scale[index].tolist(),
                   float(dataset.right[index])]
            def crop():
                item = dict(dataset[index])
                item.pop("personid")
                return item
            item = cache.get("crop", key, crop)
            item["personid"] = int(dataset.personid[index])
            return item
    return Dataset()
