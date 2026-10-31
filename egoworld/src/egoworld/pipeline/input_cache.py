"""Opt-in, run-local input cache. Entries are disposable, never stage commits.

One global file lock protects the short map/copy/close and publish/evict sections.
Striped locks coalesce concurrent misses without holding the global lock during
decoding/cropping. No memmap escapes the cache: callers receive mutable copies.
This deliberately trades a copy for bounded mappings and simple crash recovery.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

CURRENT = ContextVar("egoworld_input_cache", default=None)
DEFAULTS = {"enabled": False, "max_disk_bytes": 512 * 1024**2,
            "max_entry_bytes": 32 * 1024**2, "max_entries": 2048, "directory": None}


def options(extra):
    value = extra.get("input_cache", {})
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("extra.input_cache has unknown options or is not an object")
    result = DEFAULTS | value
    if not isinstance(result["enabled"], bool):
        raise ValueError("input_cache.enabled must be boolean")
    directory = result["directory"]
    if directory is not None and (not isinstance(directory, str) or not directory or not Path(directory).is_absolute()):
        raise ValueError("input_cache.directory must be an absolute local path or null")
    for key in ("max_disk_bytes", "max_entry_bytes", "max_entries"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"input_cache.{key} must be a positive integer")
    if result["max_entry_bytes"] > result["max_disk_bytes"]:
        raise ValueError("input_cache.max_entry_bytes exceeds max_disk_bytes")
    return result


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def prepare_cache(config, job, video_path):
    policy = options(config.get("extra", {}))
    if not policy["enabled"]:
        return None
    from egoworld.utils.hashing import sha256_file
    source = sha256_file(str(video_path))
    return {"root": str(cache_root(config, Path(job["work_dir"]).resolve().parent)),
            "namespace": identity([job["clip"]["clip_id"], source]),
            "source_sha256": source, "video_path": str(Path(video_path).resolve()),
            "video_stat": [Path(video_path).stat().st_size, Path(video_path).stat().st_mtime_ns],
            "policy": policy, "version": 1}


def cache_root(config, work_root):
    """An optional local cache location still has a distinct namespace per run."""
    policy = options(config.get("extra", {}))
    work_root = Path(work_root).resolve()
    if policy["directory"] is not None:
        return Path(policy["directory"]).resolve() / identity(str(work_root))
    return work_root / ".input-cache"


@contextmanager
def locked(path):
    # Supported deployment: Linux/WSL local filesystems, not a shared NFS cache.
    import fcntl
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class InputCache:
    def __init__(self, spec):
        self.spec = spec
        self.root = Path(spec["root"])
        self.namespace = spec["namespace"]
        if len(self.namespace) != 64 or any(c not in "0123456789abcdef" for c in self.namespace):
            raise ValueError("Invalid input cache namespace")
        self.policy = options({"input_cache": spec["policy"]})
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = {"input_cache_hits": 0, "input_cache_misses": 0, "input_cache_bypasses": 0,
                      "input_cache_corruptions": 0, "input_cache_evictions": 0,
                      "input_cache_peak_disk_bytes": 0, "input_cache_peak_mapped_bytes": 0,
                      "input_cache_read_seconds": 0., "input_cache_build_seconds": 0.,
                      "input_cache_write_seconds": 0.}
        with locked(self.root / ".lock"):
            path = self.root / "policy.json"
            if path.exists() and json.loads(path.read_text()) != self.policy:
                raise ValueError("Input cache policy changed within a run; use a new run")
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.policy))
                temporary.replace(path)
            self.stats["input_cache_peak_disk_bytes"] = sum(item[2] for item in self._inventory())

    def _delete(self, stem):
        for suffix in (".dat", ".json", ".tmp"):
            (self.root / (stem + suffix)).unlink(missing_ok=True)

    def _inventory(self):
        """Called only under the global lock; interrupted publications are garbage."""
        for path in self.root.glob("*.tmp"):
            path.unlink()
        result = []
        for path in self.root.glob("*.dat"):
            meta = path.with_suffix(".json")
            if not meta.exists():
                path.unlink()
            else:
                result.append((meta.stat().st_mtime_ns, path.stem,
                               path.stat().st_size + meta.stat().st_size))
        for path in self.root.glob("*-*.json"):
            if not path.with_suffix(".dat").exists():
                path.unlink()
        return sorted(result)

    def _read(self, stem, key):
        path, meta = self.root / (stem + ".dat"), self.root / (stem + ".json")
        if not meta.exists():
            return None
        mapped = None
        started = time.perf_counter()
        try:
            document = json.loads(meta.read_text())
            signature = document.pop("metadata_sha256")
            if identity(document) != signature or document["key"] != key:
                raise ValueError("Cache metadata identity differs")
            size = path.stat().st_size
            if size > self.policy["max_entry_bytes"] or size != document["bytes"]:
                raise ValueError("Cache data size differs")
            mapped = np.memmap(path, mode="r", dtype=np.uint8)
            self.stats["input_cache_peak_mapped_bytes"] = max(self.stats["input_cache_peak_mapped_bytes"], size)
            if hashlib.sha256(mapped).hexdigest() != document["sha256"]:
                raise ValueError("Cache data checksum differs")
            result, offset = {}, 0
            for name, field in document["fields"].items():
                dtype, shape = np.dtype(field["dtype"]), tuple(field["shape"])
                if dtype.hasobject or dtype.fields or len(shape) > 8 or any(n < 0 for n in shape):
                    raise ValueError("Invalid cached dtype/shape")
                value = np.ndarray(shape, dtype=dtype, buffer=mapped, offset=offset).copy()
                offset += value.nbytes
                kind = field["kind"]
                result[name] = value.item() if kind == "python" else value[()] if kind == "scalar" else value
            if offset != size:
                raise ValueError("Cached fields differ from data size")
            os.utime(meta, None)
            return result
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            self.stats["input_cache_corruptions"] += 1
            return None
        finally:
            if mapped is not None:
                mapped._mmap.close()
            self.stats["input_cache_read_seconds"] += time.perf_counter() - started

    def get(self, kind, key, factory):
        key = identity([1, self.namespace, kind, key])
        stem = self.namespace + "-" + key
        # Fixed number of lock files, even for millions of requested frames.
        with locked(self.root / f".stripe-{int(key[:4], 16) % 64:02d}"):
            with locked(self.root / ".lock"):
                value = self._read(stem, key)
                if value is not None:
                    self.stats["input_cache_hits"] += 1
                    self.stats[f"input_cache_{kind}_hits"] = self.stats.get(f"input_cache_{kind}_hits", 0) + 1
                    return value
                self._delete(stem)
            self.stats["input_cache_misses"] += 1
            self.stats[f"input_cache_{kind}_misses"] = self.stats.get(f"input_cache_{kind}_misses", 0) + 1
            started = time.perf_counter()
            value = factory()
            self.stats["input_cache_build_seconds"] += time.perf_counter() - started
            arrays, fields, size = {}, {}, 0
            for name, item in sorted(value.items()):
                array = np.asarray(item)
                if array.dtype.hasobject or array.dtype.fields or array.ndim > 8:
                    raise ValueError("Input cache accepts only plain numeric arrays/scalars")
                arrays[name] = array
                fields[name] = {"dtype": array.dtype.str, "shape": array.shape,
                                "kind": "array" if isinstance(item, np.ndarray) else "scalar" if isinstance(item, np.generic) else "python"}
                size += array.nbytes
            if not size or size > self.policy["max_entry_bytes"]:
                self.stats["input_cache_bypasses"] += 1
                return value
            started = time.perf_counter()
            with locked(self.root / ".lock"):
                inventory = self._inventory()
                digest = hashlib.sha256()
                arrays = {name: np.ascontiguousarray(array) for name, array in arrays.items()}
                for array in arrays.values():
                    digest.update(memoryview(array).cast("B"))
                document = {"key": key, "bytes": size, "fields": fields, "sha256": digest.hexdigest()}
                document["metadata_sha256"] = identity(document)
                encoded = json.dumps(document, sort_keys=True).encode()
                needed = size + len(encoded)
                if needed > self.policy["max_disk_bytes"]:
                    self.stats["input_cache_bypasses"] += 1
                    return value
                used = sum(item[2] for item in inventory)
                while inventory and (used + needed > self.policy["max_disk_bytes"] or len(inventory) >= self.policy["max_entries"]):
                    _, old, count = inventory.pop(0)
                    self._delete(old)
                    used -= count
                    self.stats["input_cache_evictions"] += 1
                temporary = self.root / (stem + ".tmp")
                try:
                    with temporary.open("wb") as stream:
                        for array in arrays.values():
                            stream.write(memoryview(array).cast("B"))
                    temporary.replace(self.root / (stem + ".dat"))
                    temporary.write_bytes(encoded)
                    temporary.replace(self.root / (stem + ".json"))
                except BaseException:
                    self._delete(stem)
                    raise
                self.stats["input_cache_peak_disk_bytes"] = max(self.stats["input_cache_peak_disk_bytes"], used + needed)
            self.stats["input_cache_write_seconds"] += time.perf_counter() - started
            return value


@contextmanager
def cache_session(spec):
    cache = InputCache(spec) if spec else None
    token = CURRENT.set(cache)
    try:
        yield cache
    finally:
        CURRENT.reset(token)


def clear_cache(root, namespace=None):
    root = Path(root)
    if not root.exists():
        return 0
    removed = 0
    with locked(root / ".lock"):
        for path in root.iterdir():
            if path.name == "policy.json" or path.suffix not in (".dat", ".json", ".tmp"):
                continue
            if namespace is None or path.name.startswith(namespace + "-"):
                removed += path.stat().st_size
                path.unlink()
    return removed


def release_prepared_cache(prepared):
    spec = (prepared or {}).get("input_cache")
    return clear_cache(spec["root"], spec["namespace"]) if spec else 0
