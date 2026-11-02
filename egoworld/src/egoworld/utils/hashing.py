"""Hash helpers."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import time


_SESSION = ContextVar("egoworld_hash_session", default=None)


def file_stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


@contextmanager
def hash_session(registry=None):
    """Share verified digests within one invocation. Cross-run trust is opt-in.

    A registry is an explicitly trusted, locally managed asset inventory. Its
    metadata shortcut is never used by the default strict mode. Changed stamps
    fall back to reading the full file. No global memo survives an invocation.
    """
    if _SESSION.get() is not None:
        yield _SESSION.get()["stats"]
        return
    entries = {}
    if registry:
        content = json.loads(Path(registry).read_text())
        if content.get("version") != 1 or not isinstance(content.get("files"), dict):
            raise ValueError("Invalid trusted asset registry")
        entries = content["files"]
    stats = {"bytes_hashed": 0, "files_hashed": 0, "memo_hits": 0,
             "registered_hits": 0, "hash_seconds": 0.0}
    token = _SESSION.set({"memo": {}, "registered": entries, "stats": stats})
    try:
        yield stats
    finally:
        _SESSION.reset(token)


def register_assets(paths, output):
    """Explicitly inventory trusted assets after reading their complete bytes."""
    result = {"version": 1, "trust": "Local asset contents and registry must be controlled; strict mode remains available", "files": {}}
    if str(Path(output).resolve()) in {str(Path(p).resolve()) for p in paths}:
        raise ValueError("Asset registry must not overwrite a registered asset")
    for path in paths:
        path = str(Path(path).resolve())
        before = file_stamp(path)
        value = sha256_file(path)
        if file_stamp(path) != before:
            raise ValueError(f"Asset changed while registering: {path}")
        result["files"][path] = {"sha256": value, "stamp": before}
    from egoworld.io.writers import write_json
    write_json(str(output), result)
    return {"registry": str(output), "files": len(result["files"]),
            "bytes": sum(v["stamp"][2] for v in result["files"].values())}


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    session = _SESSION.get()
    canonical = str(Path(path).resolve())
    before = file_stamp(path)
    if session:
        key = (canonical, tuple(before))
        if key in session["memo"]:
            session["stats"]["memo_hits"] += 1
            return session["memo"][key]
        registered = session["registered"].get(canonical, {})
        recorded = registered.get("sha256", "")
        if registered.get("stamp") == before and isinstance(recorded, str) and len(recorded) == 64 and all(c in "0123456789abcdef" for c in recorded):
            value = registered["sha256"]
            session["memo"][key] = value
            session["stats"]["registered_hits"] += 1
            return value
    started = time.perf_counter()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    if file_stamp(path) != before:
        raise ValueError(f"File changed while hashing: {path}")
    value = digest.hexdigest()
    if session:
        session["memo"][key] = value
        session["stats"]["bytes_hashed"] += before[2]
        session["stats"]["files_hashed"] += 1
        session["stats"]["hash_seconds"] += time.perf_counter() - started
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
