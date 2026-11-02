"""Bounded CPU screening before any model/runtime admission.

Rejected inputs are excluded from manifests, never deleted. Thresholds operate
on documented statistics; default screening rejects corruption and duplicates,
not legitimate stationary, dark or small-hand scenes.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from egoworld.io.writers import write_json
from egoworld.utils.hashing import sha256_file


DEFAULTS = {"enabled": True, "deduplicate": True, "reject_blank": True, "decode": "full",
            "sample_fps": 2.0, "max_samples": 64, "max_side": 320,
            "motion_method": "farneback", "min_duration_s": 0.0,
            "max_duration_s": None, "min_width": 1, "min_height": 1,
            "min_blur": None, "min_brightness": None, "max_brightness": None,
            "max_dark_fraction": None, "min_motion": None, "max_motion": None}


def settings(value=None):
    if value is False:
        value = {"enabled": False}
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError(f"Unknown cleaning options; allowed: {list(DEFAULTS)}")
    result = {**DEFAULTS, **value}
    for key in ("enabled", "deduplicate", "reject_blank"):
        if not isinstance(result[key], bool):
            raise ValueError(f"cleaning.{key} must be boolean")
    if result["decode"] not in ("full", "sample") or result["motion_method"] not in ("farneback", "difference"):
        raise ValueError("cleaning decode=full|sample; motion_method=farneback|difference")
    for key in ("max_samples", "max_side", "min_width", "min_height"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"cleaning.{key} must be a positive integer")
    for key, item in result.items():
        if key in ("enabled", "deduplicate", "reject_blank", "decode", "motion_method") or item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError(f"cleaning.{key} must be finite and nonnegative")
    if result["sample_fps"] <= 0:
        raise ValueError("cleaning.sample_fps must be positive")
    for low, high in (("min_duration_s", "max_duration_s"), ("min_motion", "max_motion"), ("min_brightness", "max_brightness")):
        if result[low] is not None and result[high] is not None and result[low] > result[high]:
            raise ValueError(f"cleaning.{low} exceeds {high}")
    for key in ("min_brightness", "max_brightness", "max_dark_fraction"):
        if result[key] is not None and result[key] > 1:
            raise ValueError(f"cleaning.{key} must be in [0,1]")
    return result


def _statistics(path, meta, cfg):
    import cv2
    total = max(1, round(meta.duration_s * meta.fps))
    n = min(total, cfg["max_samples"], max(2, math.ceil(meta.duration_s * cfg["sample_fps"])))
    selected = set(np.linspace(0, total - 1, n).astype(int).tolist())
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("decoder_open_failed")
    decoder_fps = cap.get(cv2.CAP_PROP_FPS)
    if decoder_fps > 0 and abs(decoder_fps - meta.fps) / meta.fps > .001:
        cap.release()
        raise ValueError("variable_or_inconsistent_timebase; normalize to constant FPS")
    brightness, blur, dark, motion, gray_ranges = [], [], [], [], []
    previous = None
    count = 0
    try:
        indices = range(total + 1) if cfg["decode"] == "full" else sorted(selected)
        for index in indices:
            if cfg["decode"] == "sample":
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                if cfg["decode"] == "full" and index == total:
                    break
                raise ValueError(f"decode_failed_at_frame:{index}")
            count += 1
            if index == total:
                raise ValueError("decoded_frame_count_exceeds_metadata")
            if frame.shape[:2] != (meta.height, meta.width):
                raise ValueError(f"decoded_dimensions_changed:{index}")
            if index not in selected:
                continue
            scale = min(1.0, cfg["max_side"] / max(frame.shape[:2]))
            if scale < 1:
                frame = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_ranges.append(int(np.ptp(gray)))
            brightness.append(float(gray.mean() / 255))
            dark.append(float((gray <= 8).mean()))
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if previous is not None:
                if cfg["motion_method"] == "farneback":
                    flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                    motion.append(float(np.linalg.norm(flow, axis=-1).mean() / math.hypot(*gray.shape)))
                else:
                    motion.append(float(np.abs(gray.astype(float) - previous).mean() / 255))
            previous = gray
    finally:
        cap.release()
    if not brightness:
        raise ValueError("no_decodable_samples")
    return {"brightness": float(np.mean(brightness)), "blur": float(np.mean(blur)),
            "dark_fraction": float(np.mean(dark)), "motion": float(np.mean(motion)) if motion else 0.0,
            "motion_unit": "mean_flow_per_frame_diagonal_between_samples" if cfg["motion_method"] == "farneback" else "mean_absolute_gray_difference_0_1",
            "sampled_frames": len(brightness), "decoded_frames": count,
            "max_sample_gray_range": max(gray_ranges),
            "validation_scope": cfg["decode"], "sample_indices": sorted(selected)}


def screen_inputs(paths, config=None, cache_root=None):
    from egoworld.manifests.build_manifest import parse_video_meta
    cfg = settings(config)
    if cache_root:
        from egoworld.product.store import initialize
        initialize(cache_root)
    report = {"version": 1, "policy": cfg, "records": [], "accepted": [], "rejected": [], "cache_hits": 0}
    seen = {}
    import cv2
    algorithm = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parameters = {k: cfg[k] for k in ("decode", "sample_fps", "max_samples", "max_side", "motion_method")}
    for value in paths:
        path = str(Path(value).resolve())
        started = time.perf_counter()
        record = {"path": path, "status": "accepted", "reasons": [], "stats_cache_hit": False}
        try:
            checksum = sha256_file(path)
            record["sha256"] = checksum
            if cfg["deduplicate"] and checksum in seen:
                record.update(status="rejected", reasons=["duplicate_content"], duplicate_of=seen[checksum])
            else:
                meta = parse_video_meta(path)
                record["metadata"] = asdict(meta)
                reasons = []
                for metric, limit, reverse in ((meta.duration_s, cfg["min_duration_s"], False), (meta.duration_s, cfg["max_duration_s"], True),
                        (meta.width, cfg["min_width"], False), (meta.height, cfg["min_height"], False)):
                    if limit is not None and ((metric > limit) if reverse else (metric < limit)):
                        reasons.append("metadata_outside_policy")
                if reasons:
                    record.update(status="rejected", reasons=sorted(set(reasons)))
                elif cfg["enabled"]:
                    key = hashlib.sha256(json.dumps({"input": checksum, "algorithm": algorithm, "parameters": parameters,
                        "opencv_build": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()}, sort_keys=True).encode()).hexdigest()
                    cache_file = Path(cache_root) / "quality" / key / "statistics.json" if cache_root else None
                    stats = None
                    if cache_file and cache_file.is_file():
                        try:
                            from egoworld.product.store import guard
                            with guard(cache_root):
                                entry = json.loads(cache_file.read_text())
                            if entry["key"] == key and hashlib.sha256(json.dumps(entry["stats"], sort_keys=True).encode()).hexdigest() == entry["sha256"]:
                                stats = entry["stats"]; record["stats_cache_hit"] = True; report["cache_hits"] += 1
                        except (ValueError, KeyError, OSError):
                            pass
                    if stats is None:
                        stats = _statistics(path, meta, cfg)
                        if cache_file:
                            from egoworld.product.store import guard
                            with guard(cache_root):
                                try:
                                    write_json(str(cache_file), {"key": key, "stats": stats, "sha256": hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()})
                                except OSError as exc:
                                    record["cache_warning"] = str(exc)  # cache storage failure does not make a valid video bad
                    record["statistics"] = stats
                    if cfg["reject_blank"] and stats["max_sample_gray_range"] == 0:
                        record["reasons"].append("blank_video")
                    for name, field, reverse in (("min_blur", "blur", False), ("min_brightness", "brightness", False),
                            ("max_brightness", "brightness", True), ("max_dark_fraction", "dark_fraction", True),
                            ("min_motion", "motion", False), ("max_motion", "motion", True)):
                        limit = cfg[name]
                        if limit is not None and ((stats[field] > limit) if reverse else (stats[field] < limit)):
                            record["reasons"].append(name)
                    if record["reasons"]:
                        record["status"] = "rejected"
                if record["status"] == "accepted":
                    seen[checksum] = path
        except Exception as exc:
            # A broken video must not abort an otherwise valid batch. Preserve
            # a machine-readable reason and the original error for review.
            record.update(status="rejected", reasons=["invalid_input"], error=f"{type(exc).__name__}: {exc}")
        record["elapsed_seconds"] = time.perf_counter() - started
        report["records"].append(record)
        report["accepted" if record["status"] == "accepted" else "rejected"].append(path)
    report["summary"] = {"inputs": len(paths), "accepted": len(report["accepted"]), "rejected": len(report["rejected"]),
                         "reasons": {reason: sum(reason in r["reasons"] for r in report["records"]) for reason in sorted({x for r in report["records"] for x in r["reasons"]})}}
    report["status"] = "passed" if report["accepted"] else "rejected"
    return report
