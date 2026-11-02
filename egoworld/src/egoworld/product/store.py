"""Locked, verifiable derived artifacts and conservative cache maintenance."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import time
from uuid import uuid4

from egoworld.io.writers import write_json
from egoworld.utils.hashing import sha256_file

_HELD = ContextVar("egoworld_cache_locks", default=frozenset())


@contextmanager
def guard(root):
    root = Path(root).resolve()
    key = str(root)
    if key in _HELD.get():
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    token = _HELD.set(_HELD.get() | {key})
    try:
        with (root / ".egoworld-cache.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        _HELD.reset(token)


def locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with guard(self.root):
            return method(self, *args, **kwargs)
    return wrapped


def initialize(root):
    with guard(root):
        marker = Path(root) / ".egoworld-cache.json"
        if not marker.exists():
            write_json(str(marker), {"version": 1, "owner": "egoworld"})


class DerivedCache:
    def __init__(self, root):
        self.root = Path(root)
        initialize(root)

    def compute(self, identity, destination, builder, *, read=True, write=True):
        from egoworld.product.cache import digest
        key = digest(identity)
        entry = self.root / "derived" / key
        destination = Path(destination)
        if read:
            result = self.restore(identity, entry, destination)
            if result is not None:
                return result, True
        result = builder(destination)
        if write:
            self.publish(identity, entry, destination, result)
        return result, False

    @locked
    def restore(self, identity, entry, destination):
        receipt = None
        try:
            candidate = json.loads((entry / "derived.json").read_text())
            if candidate["identity"] != identity:
                raise ValueError("Derived identity mismatch")
            for name, info in candidate["files"].items():
                file = entry / name
                if Path(name).name != name or file.is_symlink() or file.stat().st_size != info["bytes"] or sha256_file(str(file)) != info["sha256"]:
                    raise ValueError("Corrupt derived artifact")
            receipt = candidate
        except (ValueError, OSError, KeyError, TypeError):
            pass
        if receipt is not None:
            destination.mkdir(parents=True, exist_ok=True)
            for name in receipt["files"]:
                temporary = destination / f".{name}.{uuid4().hex}.tmp"
                shutil.copyfile(entry / name, temporary)
                os.replace(temporary, destination / name)
            os.utime(entry / "derived.json", None)
            return receipt["result"]
        return None

    @locked
    def publish(self, identity, entry, destination, result):
        temporary = entry.parent / f".{entry.name}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        try:
            files = {}
            for file in destination.iterdir():
                if file.is_file() and not file.is_symlink() and not file.name.startswith("."):
                    shutil.copyfile(file, temporary / file.name)
                    files[file.name] = {"bytes": file.stat().st_size, "sha256": sha256_file(str(file))}
            write_json(str(temporary / "derived.json"), {"identity": identity, "result": result, "files": files})
            if entry.exists():
                shutil.rmtree(entry)
            temporary.rename(entry)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def inspect_cache(root, *, max_bytes=None, max_age_seconds=None, delete=False):
    root = Path(root).resolve()
    if max_bytes is not None and max_bytes < 0 or max_age_seconds is not None and max_age_seconds < 0:
        raise ValueError("Cache budgets must be nonnegative")
    with guard(root):
        marker = root / ".egoworld-cache.json"
        if not marker.is_file() or json.loads(marker.read_text()) != {"version": 1, "owner": "egoworld"}:
            raise ValueError("Cache maintenance requires an EgoWorld-managed cache root")
        entries = []
        for kind in ("sam2", "hamer", "foundationpose", "fast3r", "dex_retarget", "derived", "quality"):
            parent = root / kind
            if not parent.is_dir() or parent.is_symlink():
                continue
            for directory in parent.iterdir():
                if directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{64}|\.[0-9a-f]{64}\.invalid\.[0-9a-f]{32}", directory.name):
                    continue
                receipt = directory / ({"derived": "derived.json", "quality": "statistics.json"}.get(kind, "entry.json"))
                if not receipt.is_file() or receipt.is_symlink():
                    continue
                entries.append({"kind": kind, "key": directory.name, "path": str(directory),
                    "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and not p.is_symlink()),
                    "age_seconds": max(0, time.time() - receipt.stat().st_mtime)})
        total = sum(e["bytes"] for e in entries)
        remaining = total
        removed = []
        for entry in sorted(entries, key=lambda e: e["age_seconds"], reverse=True):
            if (max_age_seconds is not None and entry["age_seconds"] > max_age_seconds) or (max_bytes is not None and remaining > max_bytes):
                removed.append(entry)
                remaining -= entry["bytes"]
                if delete:
                    shutil.rmtree(entry["path"])
        return {"root": str(root), "entries": entries, "total_bytes": total,
                "selected_for_removal": removed, "remaining_bytes": remaining,
                "dry_run": not delete, "scope": "Managed stage/derived/quality entries; excludes models, run outputs and input scratch"}
