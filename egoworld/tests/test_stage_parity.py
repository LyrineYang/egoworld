"""Reject misleading matched-stage results even when their files are valid."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from equal_work.stage_parity import audit_receipts, input_identity
from egoworld.config import PipelineConfig
from egoworld.pipeline.artifacts import WriterActor
from test_pipeline_logic import full_packet


def commit(root, packet):
    config = asdict(PipelineConfig())
    clip = {"clip_id": "c", "video_id": "v", "frame_start": 0, "frame_end": 10, "fps": 10}
    job = {"stage": "dex_retarget", "clip": clip, "directory": str(root), "fingerprint": "same"}
    return WriterActor(config).write_stage(job, packet)


def packet():
    return full_packet("dex_retarget", {"frame_start": 0, "fps": 10})


@pytest.mark.parametrize("change", ["numeric", "missing", "extra", "corrupt", "empty", "identity"])
def test_parity_rejects_changes(tmp_path, change):
    first, second = packet(), packet()
    if change == "numeric":
        second["tables"]["mapping"][0]["joint_positions"] = [.2]
    elif change == "missing":
        second["tables"]["mapping"] = []
    elif change == "extra":
        second["tables"]["mapping"].append(dict(second["tables"]["mapping"][0], object_id=2))
    elif change == "empty":
        first["tables"]["mapping"] = second["tables"]["mapping"] = []
    receipts = [commit(tmp_path / "a", first), commit(tmp_path / "b", second)]
    if change == "corrupt":
        (tmp_path / "b/mapping.parquet").write_bytes(b"corrupt")
    elif change == "identity":
        receipts[1]["fingerprint"] = "different"
    with pytest.raises(ValueError):
        audit_receipts(receipts)


def test_parity_matches_full_fields_and_checks_input_mutation(tmp_path):
    receipts = [commit(tmp_path / name, packet()) for name in ("a", "b")]
    assert audit_receipts(receipts)["rows"] == {"mapping": 1}
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fixture")
    request = {"job": {"stage": "dex_retarget"}, "prepared": {"video_path": str(video)},
               "config": {"operators": {"dex_retarget": {"params": {}}}}, "inputs": {"dex": receipts[0]}}
    before = input_identity(request)
    video.write_bytes(b"changed")
    assert input_identity(request) != before
    (tmp_path / "a/mapping.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Upstream artifact changed"):
        input_identity(request)


def test_parity_refuses_partial_phase_request():
    with pytest.raises(ValueError, match="full-stage"):
        input_identity({"job": {"_operation": "infer"}, "prepared": {}})
