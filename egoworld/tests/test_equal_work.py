"""Acceptance gates for the experimental artifact-cleaning benchmark."""
import copy
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from equal_work.audit import audit_trial, summarize
from equal_work.contract import (accepted_record, canonical, evaluate, freeze_corpus,
                                 sha256, source_identity, verify_corpus, write_json, RULE)
from equal_work.worker import run_worker


def valid_row():
    return {"frame_index": 2, "timestamp_s": .2, "object_id": 1, "handedness": "right",
            "joints_3d": [.01] * 63, "global_orient": [0.] * 9, "mano_pose": [0.] * 135,
            "betas": [0.] * 10, "camera_translation": [0., 0., 1.], "coord_frame": "hand_root",
            "length_unit": "meters", "scale_status": "model_prior", "joint_order": "openpose21"}


def sample(row=None, sid="a"):
    return {"sample_id": sid, "payload": canonical(valid_row() if row is None else row),
            "clip": {"frame_start": 0, "frame_end": 10, "fps": 10.}}


@pytest.mark.parametrize("field,value", [
    ("frame_index", True), ("timestamp_s", "0.2"), ("object_id", -1),
    ("joints_3d", [None] * 63), ("joints_3d", [0.] * 62), ("joints_3d", ["NaN"] * 63),
    ("camera_translation", [0., 0.]), ("handedness", "unknown"), ("length_unit", "mm"),
    ("coord_frame", "camera"), ("timestamp_s", .9), ("frame_index", 20),
])
def test_shared_rule_rejects_contract_errors(field, value):
    row = valid_row()
    row[field] = value
    decision = evaluate(sample(row))
    assert not decision["accepted"] and decision["reasons"]


def test_no_pose_accuracy_claim_and_no_silent_numeric_coercion():
    value = sample()
    assert evaluate(value)["accepted"]
    assert accepted_record(value)["row"] == valid_row()
    assert not evaluate(sample([]))["accepted"]
    # Non-standard JSON may appear in third-party inputs: never count NaN as valid.
    value["payload"] = value["payload"].replace('"timestamp_s":0.2', '"timestamp_s":NaN')
    assert not evaluate(value)["accepted"]


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    invalid = valid_row()
    invalid["length_unit"] = "mm"
    samples = [sample(), sample(invalid, "b")]
    (folder / "samples.jsonl").write_text("".join(canonical(s) + "\n" for s in samples))
    contract = {"rule": RULE, "rows": 2, "inputs": {}, "source_identity": source_identity(),
                "corpus_sha256": sha256(folder / "samples.jsonl")}
    write_json(folder / "contract.json", contract)
    return folder


def trial_for(corpus, tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    run_worker("stream_reference", corpus, trial)
    return trial


def reseal(trial):
    """Test semantic gates independently of simple artifact checksums."""
    worker = json.loads((trial / "worker.json").read_text())
    worker["output_sha256"] = {name: sha256(trial / name) for name in ("decisions.jsonl", "accepted.jsonl")}
    write_json(trial / "worker.json", worker)


def test_streaming_roundtrip_and_source_invalidation(corpus, tmp_path):
    trial = trial_for(corpus, tmp_path)
    assert audit_trial(corpus, trial)["accepted"] == 1
    with pytest.raises(FileExistsError):
        run_worker("stream_reference", corpus, trial)
    (corpus / "samples.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Corpus content changed"):
        verify_corpus(corpus)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "numerical", "reason", "selection", "contract"])
def test_equal_outputs_gate_rejects_false_speedup(corpus, tmp_path, change):
    trial = trial_for(corpus, tmp_path)
    decisions = trial / "decisions.jsonl"
    accepted = trial / "accepted.jsonl"
    if change == "missing":
        accepted.write_text("")
    elif change == "duplicate":
        accepted.write_text(accepted.read_text() * 2)
    elif change == "extra":
        row = json.loads(accepted.read_text())
        row["sample_id"] = "extra"
        accepted.write_text(accepted.read_text() + canonical(row) + "\n")
    elif change == "numerical":
        row = json.loads(accepted.read_text())
        row["row"]["joints_3d"][0] += 1e-9
        accepted.write_text(canonical(row) + "\n")
    elif change in ("reason", "selection"):
        rows = [json.loads(line) for line in decisions.read_text().splitlines()]
        if change == "reason":
            rows[1]["reasons"] = ["a different rule"]
        else:
            rows[1]["accepted"] = True
        decisions.write_text("".join(canonical(r) + "\n" for r in rows))
    else:
        worker = json.loads((trial / "worker.json").read_text())
        worker["contract_sha256"] = "changed"
        write_json(trial / "worker.json", worker)
    reseal(trial)
    with pytest.raises(ValueError):
        audit_trial(corpus, trial)


def test_all_trials_must_pass_not_only_fastest():
    trials = [{"repeat": r, "backend": b, "status": "ok", "audit": {"passed": True}, "wall_seconds": t}
              for r in range(3) for b, t in (("stream_reference", 1 + r), ("datajuicer", 2 + r))]
    result = summarize(trials, 3)
    assert result["all_trials_valid"] and result["wall_seconds"]["datajuicer"]["n"] == 3
    assert result["datajuicer_over_stream_reference_wall_ratio"] == 1.5
    assert summarize(trials, 3, environment_stable=False)["datajuicer_over_stream_reference_wall_ratio"] is None
    for changed in (trials[:-1], trials + [trials[0]], copy.deepcopy(trials)):
        if len(changed) == 6:
            changed[-1]["status"] = "failed"
        assert summarize(changed, 3)["datajuicer_over_stream_reference_wall_ratio"] is None


def test_freeze_pins_video_and_provenance_and_rejects_duplicates(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"corpus-freezer fixture, not a decodable video")
    meta = tmp_path / "meta.json"
    write_json(meta, {"clip": {"video_path": str(video), "video_id": "v", "clip_id": "c",
                              "frame_start": 0, "frame_end": 10, "fps": 10.}})
    provenance = tmp_path / "run_manifest.json"
    write_json(provenance, {"models": "fixture"})
    parquet = tmp_path / "hand_pose.parquet"
    pq.write_table(pa.Table.from_pylist([valid_row()]), parquet)
    freeze_corpus(parquet, meta, provenance, tmp_path / "frozen")
    assert verify_corpus(tmp_path / "frozen", verify_sources=True)["rows"] == 1
    frozen = tmp_path / "fault-fixtures"
    freeze_corpus(parquet, meta, provenance, frozen, fault_fixtures=True)
    assert verify_corpus(frozen)["synthetic_fault_rows"] == 4
    trial = trial_for(frozen, tmp_path)
    assert audit_trial(frozen, trial)["rejected"] == 4
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source content changed"):
        verify_corpus(tmp_path / "frozen", verify_sources=True)
    pq.write_table(pa.Table.from_pylist([valid_row(), valid_row()]), parquet)
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_corpus(parquet, meta, provenance, tmp_path / "duplicate")
