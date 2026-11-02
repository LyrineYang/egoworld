"""Exact equality and full coverage gates, using bounded disk-backed indexes."""
import json
import math
from pathlib import Path
import sqlite3
import statistics
import tempfile

from equal_work.contract import (accepted_record, canonical, evaluate, records,
                                 sha256, verify_corpus)


def audit_trial(corpus, trial):
    corpus, trial = Path(corpus), Path(trial)
    contract = verify_corpus(corpus)
    worker = json.loads((trial / "worker.json").read_text())
    if (worker.get("status") != "ok" or worker.get("contract_sha256") != sha256(corpus / "contract.json")
            or worker.get("corpus_sha256") != contract["corpus_sha256"]):
        raise ValueError("Worker failed or used a different contract/corpus")
    for name in ("decisions.jsonl", "accepted.jsonl"):
        if worker["output_sha256"][name] != sha256(trial / name):
            raise ValueError("Output changed after invocation: " + name)
    with tempfile.TemporaryDirectory(prefix="equal-work-audit-") as tmp:
        with sqlite3.connect(Path(tmp) / "audit.sqlite") as db:
            db.execute("CREATE TABLE expected (id TEXT PRIMARY KEY, decision TEXT, accepted TEXT, d_seen INTEGER DEFAULT 0, a_seen INTEGER DEFAULT 0)")
            total = kept = 0
            for sample in records(corpus / "samples.jsonl"):
                decision = evaluate(sample)
                accepted = canonical(accepted_record(sample)) if decision["accepted"] else None
                try:
                    db.execute("INSERT INTO expected (id, decision, accepted) VALUES (?, ?, ?)",
                               (sample["sample_id"], canonical(decision), accepted))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate corpus sample ID") from exc
                total += 1
                kept += decision["accepted"]
            for filename, column, seen in (("decisions.jsonl", "decision", "d_seen"),
                                           ("accepted.jsonl", "accepted", "a_seen")):
                for row in records(trial / filename):
                    expected = db.execute(f"SELECT {column}, {seen} FROM expected WHERE id=?", (row["sample_id"],)).fetchone()
                    if expected is None or expected[0] is None:
                        raise ValueError("Unexpected output ID: " + row["sample_id"])
                    if expected[1]:
                        raise ValueError("Duplicate output ID: " + row["sample_id"])
                    if expected[0] != canonical(row):
                        raise ValueError("Output mismatch: " + row["sample_id"])
                    db.execute(f"UPDATE expected SET {seen}=1 WHERE id=?", (row["sample_id"],))
            missing = db.execute("SELECT COUNT(*) FROM expected WHERE d_seen=0 OR (accepted IS NOT NULL AND a_seen=0)").fetchone()[0]
            if missing or total != contract["rows"] or total == 0:
                raise ValueError("Missing outputs, empty or changed corpus")
            if worker["rows"] != total or worker["accepted"] != kept:
                raise ValueError("Worker counters disagree with audited outputs")
    return {"passed": True, "rows": total, "accepted": kept, "rejected": total - kept,
            "comparison": "exact canonical JSON, keyed by sample_id, all decisions and all accepted fields"}


def distribution(values):
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values), "min": values[0],
            "max": values[-1], "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1]}


def summarize(trials, repeats, *, environment_stable=True):
    """No ratio if any invocation failed, disappeared, or failed the quality gate."""
    backends = ("stream_reference", "datajuicer")
    expected = {(r, b) for r in range(repeats) for b in backends}
    actual = [(t["repeat"], t["backend"]) for t in trials]
    complete = len(actual) == len(expected) and set(actual) == expected
    passed = environment_stable and complete and all(t.get("status") == "ok" and t.get("audit", {}).get("passed")
                              and math.isfinite(t["wall_seconds"]) and t["wall_seconds"] > 0 for t in trials)
    result = {"all_trials_valid": passed, "environment_stable": environment_stable,
              "expected_trials": len(expected), "observed_trials": len(trials),
              "wall_seconds": {}, "datajuicer_over_stream_reference_wall_ratio": None,
              "interpretation": "CPU artifact-cleaning pilot; not a production EgoWorld or GPU inference comparison"}
    if passed:
        result["wall_seconds"] = {b: distribution([t["wall_seconds"] for t in trials if t["backend"] == b]) for b in backends}
        result["datajuicer_over_stream_reference_wall_ratio"] = (
            result["wall_seconds"]["datajuicer"]["median"] / result["wall_seconds"]["stream_reference"]["median"])
    return result
