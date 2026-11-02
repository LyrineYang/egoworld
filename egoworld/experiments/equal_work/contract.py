"""Frozen inputs and a shared, conservative hand-output validation rule."""
import hashlib
import copy
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DJ_REVISION = "09178e1e0ac76cfbfe11232d66244e1ccc5add6f"
RULE = "egoworld-hand-contract-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def records(path):
    with open(path) as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def source_identity():
    files = list(Path(__file__).parent.glob("*.py"))
    files += [ROOT / "src/egoworld/contracts.py", ROOT / "scripts/benchmark_equal_work.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def evaluate(sample):
    """One decision per input; retain the full original row without coercion.

    This is contract validation, not a pose-accuracy or temporal-motion filter.
    Cross-row duplicates are rejected when freezing the corpus, not silently
    dropped here. The same function is called by both executors and the auditor.
    """
    from egoworld.contracts import validate_rows

    payload = sample["payload"]
    row = json.loads(payload)
    reasons = []
    if not isinstance(row, dict):
        reasons.append("row_not_object")
    else:
        # Reject bool/numeric-string inputs rather than relying on NumPy casts.
        for key in ("frame_index", "object_id"):
            if type(row.get(key)) is not int or row[key] < 0:
                reasons.append("invalid_" + key)
        ts = row.get("timestamp_s")
        if type(ts) not in (int, float) or not math.isfinite(ts) or ts < 0:
            reasons.append("invalid_timestamp_s")
        for key in ("joints_3d", "global_orient", "mano_pose", "betas", "camera_translation"):
            values = row.get(key)
            if not isinstance(values, list) or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in values
            ):
                reasons.append("invalid_" + key)
        if not reasons:
            try:
                validate_rows("hand_pose", [row], sample["clip"])
            except ValueError as exc:
                reasons.append(str(exc))
    return {"sample_id": sample["sample_id"], "accepted": not reasons,
            "reasons": reasons, "payload_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def accepted_record(sample):
    return {"sample_id": sample["sample_id"], "row": json.loads(sample["payload"])}


def verify_corpus(folder, *, verify_sources=False):
    folder = Path(folder).resolve()
    contract = json.loads((folder / "contract.json").read_text())
    if contract["rule"] != RULE or contract["source_identity"] != source_identity():
        raise ValueError("Experiment/rule source changed: freeze a new corpus")
    if contract["corpus_sha256"] != sha256(folder / "samples.jsonl"):
        raise ValueError("Corpus content changed")
    if verify_sources:
        for entry in contract["inputs"].values():
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Source content changed: " + entry["path"])
    return contract


def freeze_corpus(hand_pose, meta, provenance, output, *, fault_fixtures=False):
    import pyarrow.parquet as pq
    import sqlite3

    meta = Path(meta).resolve()
    metadata = json.loads(meta.read_text())
    clip = metadata["clip"]
    if not (type(clip["frame_start"]) is int and type(clip["frame_end"]) is int
            and 0 <= clip["frame_start"] < clip["frame_end"]
            and math.isfinite(clip["fps"]) and clip["fps"] > 0):
        raise ValueError("Invalid source-frame range/fps")
    paths = {"hand_pose": Path(hand_pose).resolve(), "meta": meta,
             "provenance": Path(provenance).resolve(), "video": Path(clip["video_path"]).resolve()}
    inputs = {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()}
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    for relative in identity:
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    count = 0
    first = None
    # Disk-backed uniqueness guard: no unbounded set for long recordings.
    with sqlite3.connect(folder / "freeze.sqlite") as db, (folder / "samples.jsonl").open("w") as out:
        db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
        for batch in pq.ParquetFile(hand_pose).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                sid = canonical([clip["video_id"], clip["clip_id"], row["frame_index"], row["object_id"]])
                try:
                    db.execute("INSERT INTO ids VALUES (?)", (sid,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate frame/object identity in corpus") from exc
                sample = {"sample_id": sid, "payload": canonical(row), "clip": clip}
                if first is None:
                    first = row
                out.write(canonical(sample) + "\n")
                count += 1
        if fault_fixtures and first is not None:
            # Explicitly labeled integration fixtures, never independent model outputs.
            for index, (field, value) in enumerate((("length_unit", "mm"), ("joints_3d", [0.]),
                                                     ("handedness", "unknown"), ("timestamp_s", float("nan")))):
                row = copy.deepcopy(first)
                row[field] = value
                payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
                sample = {"sample_id": canonical(["synthetic-contract-fault", index]), "payload": payload, "clip": clip}
                out.write(canonical(sample) + "\n")
                count += 1
    (folder / "freeze.sqlite").unlink()
    if count == 0:
        raise ValueError("Empty corpus cannot establish throughput")
    if identity != source_identity() or any(sha256(entry["path"]) != entry["sha256"] for entry in inputs.values()):
        raise ValueError("Source changed while freezing; create a new corpus")
    contract = {"version": 1, "scope": "hand-artifact-contract-cleaning-only", "rule": RULE,
                "rows": count, "inputs": inputs, "corpus_sha256": sha256(folder / "samples.jsonl"),
                "source_identity": identity, "datajuicer_revision": DJ_REVISION,
                "synthetic_fault_rows": 4 if fault_fixtures else 0,
                "quality_limit": "Contract-valid does not imply accurate or useful pose",
                "timing_scope": "Fresh process through persisted normalized outputs; no model inference"}
    write_json(folder / "contract.json", contract)
    return contract
