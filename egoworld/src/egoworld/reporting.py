"""Bounded, read-only run inspection and a self-contained offline HTML report."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from egoworld.contracts import OUTPUT_SCHEMA_VERSION
from egoworld.pipeline.artifacts import load_stage


EVENT_TAIL_BYTES = 2 * 1024 * 1024


def _read_json(path, issues, required=False):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except FileNotFoundError:
        if required:
            issues.append(f"Missing {path.name}")
    except (OSError, ValueError) as exc:
        issues.append(f"{path.name}: {exc}")
    return {}


def _event_tail(root, issues):
    """The log spans invocations; retain only each clip/stage's latest event."""
    path = root / "events.jsonl"
    if not path.is_file():
        return {}, False
    events = {}
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - EVENT_TAIL_BYTES)
        handle.seek(start)
        if start:
            handle.readline()  # a truncated first line is not an event
        for line in handle:
            try:
                event = json.loads(line)
                if (isinstance(event, dict) and isinstance(event.get("clip_id"), str)
                        and isinstance(event.get("stage", "pipeline"), str)):
                    events.setdefault(event["clip_id"], {})[event.get("stage", "pipeline")] = event
            except (ValueError, UnicodeError):
                issues.append("Ignored an incomplete or invalid event log line")
    return events, bool(start)


def _clip_report(directory, manifest, events):
    issues = []
    meta = _read_json(directory / "meta.json", issues, required=True)
    qc = _read_json(directory / "qc.json", issues, required=True)
    graph = manifest.get("stages", {})
    fingerprint = manifest.get("config_fingerprint")
    stages = []
    valid = bool(fingerprint)
    committed = set()
    for name in graph:
        receipt = None
        if fingerprint:
            try:
                receipt = load_stage(directory, name, fingerprint, verify_hashes=False)
            except Exception as exc:
                issues.append(f"{name}: cannot inspect commit metadata: {exc}")
        event = events.get(name, {})
        if receipt and any(dependency not in committed for dependency in graph[name]):
            receipt = None
            issues.append(f"{name}: upstream commit is unavailable; resume will invalidate this result")
        if receipt is None:
            valid = False
            state = "invalid" if (directory / f"stage_{name}.json").is_file() else event.get("status", "missing")
            issues.append(f"{name}: no valid stage commit")
        else:
            state = "committed"
            committed.add(name)
        stages.append({"name": name, "status": state, "files": (receipt or {}).get("files", {})})
    if meta and (meta.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
                 or meta.get("fingerprint") != fingerprint
                 or set(meta.get("stages", [])) != set(graph)
                 or meta.get("quality") != qc):
        valid = False
        issues.append("Clip metadata does not match the current run, stages or QC")
    errors = dict(meta.get("errors", {}))
    if not meta:
        errors.update({name: e["error"] for name, e in events.items()
                       if e.get("error") and e.get("status") in ("Failed", "Blocked", "Retry")})
    if errors or meta.get("status") == "Failed":
        status = "failed"
    elif not meta or not qc or not valid or meta.get("status") != "Done":
        status = "incomplete"
    else:
        status = qc.get("status", "incomplete")
        if status not in ("pass", "review", "skipped"):
            status = "incomplete"
            issues.append("Unrecognized QC status")
    # Coverage describes SAM2 only; an independent geometry run has no such observation.
    coverage = qc.get("frame_coverage") if "sam2" in graph else None
    return {"video_id": directory.parent.name.removeprefix("video_id="),
            "clip_id": directory.name.removeprefix("clip_id="), "directory": str(directory),
            "status": status, "coverage": coverage, "objects": qc.get("object_count"),
            "empty_mask_rate": qc.get("empty_mask_rate") if "sam2" in graph else None,
            "empty_stages": qc.get("empty_stages", []), "clip": meta.get("clip", {}),
            "stages": stages, "errors": errors, "issues": issues}


def collect_report(directory, max_clips=1000):
    if not isinstance(max_clips, int) or isinstance(max_clips, bool) or max_clips < 1:
        raise ValueError("max_clips must be a positive integer")
    root = Path(directory).resolve()
    issues = []
    manifest = _read_json(root / "run_manifest.json", issues, required=True)
    if not manifest:
        raise ValueError(f"Cannot read run_manifest.json in {root}: {'; '.join(issues)}")
    if manifest.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError(f"Report requires output schema {OUTPUT_SCHEMA_VERSION}")
    summary = _read_json(root / "run_summary.json", issues)
    events, tail_limited = _event_tail(root, issues)
    clips, discovered = [], 0
    # Stream directory discovery instead of materializing every Parquet row or clip path.
    for path in root.glob("video_id=*/clip_id=*"):
        if not path.is_dir():
            continue
        discovered += 1
        records = events.pop(path.name.removeprefix("clip_id="), {})
        if len(clips) < max_clips:
            clips.append(_clip_report(path, manifest, records))
    for clip_id, records in events.items():
        discovered += 1
        if len(clips) < max_clips:
            errors = {stage: event["error"] for stage, event in records.items()
                      if event.get("error") and event.get("status") in ("Failed", "Blocked", "Retry")}
            failed = any(event.get("status") == "Failed" for event in records.values())
            clips.append({"video_id": "—", "clip_id": clip_id, "directory": "",
                          "status": "failed" if failed else "incomplete", "coverage": None,
                          "objects": None, "empty_mask_rate": None, "empty_stages": [],
                          "clip": {}, "stages": [], "errors": errors,
                          "issues": ["Only log evidence available; no clip output directory"]})
    clips.sort(key=lambda c: ({"failed": 0, "incomplete": 1, "review": 2}.get(c["status"], 3), c["video_id"], c["clip_id"]))
    stages = []
    for name, dependencies in manifest.get("stages", {}).items():
        metrics = summary.get("stage_metrics", {}).get(name, {})
        calls = metrics.get("calls", 0)
        stages.append({"name": name, "after": dependencies, "calls": calls,
                       "wall": metrics.get("stage_wall_seconds", metrics.get("inference_seconds", 0)) / calls if calls else None,
                       "queue": metrics.get("queue_wait_seconds", 0) / calls if calls else None,
                       "write": metrics.get("write_seconds", 0) / calls if calls else None,
                       "cache_hits": metrics.get("model_cache_hit", 0) if calls else None})
    return {"run_id": summary.get("run_id", manifest.get("run_id", root.name)),
            "generated_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
            "schema_version": OUTPUT_SCHEMA_VERSION, "fingerprint": manifest.get("config_fingerprint"),
            "summary": summary, "stages": stages, "clips": clips,
            "counts": dict(Counter(c["status"] for c in clips)), "discovered_clips": discovered,
            "max_clips": max_clips, "truncated": discovered > len(clips),
            "event_tail_limited": tail_limited, "issues": list(dict.fromkeys(issues))}


def write_report(directory, output=None, max_clips=1000):
    from egoworld.report_template import HTML

    data = collect_report(directory, max_clips)
    target = Path(output).resolve() if output else Path(data["run_dir"]) / "report.html"
    if target.suffix.lower() != ".html":
        raise ValueError("Report output must have an .html extension")
    # Prevent embedded data (including labels and exception strings) escaping the JSON script.
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = HTML.replace("__REPORT_DATA__", payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(page)
        os.replace(temporary, target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return target
