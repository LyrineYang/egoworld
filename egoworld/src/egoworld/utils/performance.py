"""Low-cost process observations; byte estimates are not OS memory limits."""
from __future__ import annotations

from functools import wraps
import resource
import sys
import time


def timed_load(method):
    @wraps(method)
    def load(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._model_load_seconds = getattr(self, "_model_load_seconds", 0.) + time.perf_counter() - start
    return load


def packet_bytes(packet):
    """Estimate live Python/NumPy payload size, counting shared values once."""
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            total += sum(map(size, value))
        elif hasattr(value, "nbytes") and not getattr(getattr(value, "flags", None), "owndata", False):
            total += value.nbytes
        return total
    return size(packet)


def model_load_seconds(operator):
    return getattr(operator, "_model_load_seconds", 0.) + getattr(getattr(operator, "_gd", None), "_model_load_seconds", 0.)


def process_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def cuda_metrics(reset=False):
    """Observe only an already-imported CUDA runtime; never initialize the driver."""
    torch = sys.modules.get("torch")
    if not torch or not torch.cuda.is_initialized():
        return {}
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return {}
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_name": properties.name, "gpu_total_bytes": properties.total_memory}
