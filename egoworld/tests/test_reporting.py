"""Check reports against real commits, partial outputs and hostile log strings."""
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from test_pipeline_logic import pipeline_case, full_packet, enable_all
from egoworld.pipeline import stages
from egoworld.reporting import collect_report, write_report


def test_report_reads_commits_and_keeps_qc_distinct_from_execution(pipeline_case):
    case = pipeline_case
    case["run"]()
    root = case["root"] / "output/run_id=A"
    report = collect_report(root)
    assert report["summary"]["status"] == "passed"
    assert report["counts"] == {"review": 3}  # empty outputs executed successfully
    assert report["stages"][0]["calls"] == 3
    assert all(s["status"] == "committed" for c in report["clips"] for s in c["stages"])
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    output = write_report(root)
    assert output == root / "report.html"
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items())


def test_missing_or_corrupt_commit_is_not_reported_as_qc_pass(pipeline_case, monkeypatch):
    case = pipeline_case
    monkeypatch.setattr(stages.ModelActor, "process", lambda self, job, *_: full_packet(job["stage"], job["clip"]))
    case["run"]()
    root = case["root"] / "output/run_id=A"
    assert collect_report(root)["counts"] == {"pass": 3}
    (root / "video_id=v/clip_id=c0/instances.parquet").write_bytes(b"corrupt")
    (root / "video_id=v/clip_id=c1/meta.json").unlink()
    report = collect_report(root)
    assert report["counts"] == {"incomplete": 2, "pass": 1}
    assert report["clips"][0]["stages"][0]["status"] == "invalid"


def test_invalid_upstream_is_not_treated_as_a_committed_downstream(pipeline_case, monkeypatch):
    case = pipeline_case
    enable_all(case)
    monkeypatch.setattr(stages.ModelActor, "process", lambda self, job, *_: full_packet(job["stage"], job["clip"]))
    case["run"]()
    root = case["root"] / "output/run_id=A"
    (root / "video_id=v/clip_id=c0/instances.parquet").write_bytes(b"corrupt")
    clip = next(c for c in collect_report(root)["clips"] if c["clip_id"] == "c0")
    states = {s["name"]: s["status"] for s in clip["stages"]}
    assert states == {"sam2": "invalid", "hamer": "invalid", "foundationpose": "invalid",
                      "fast3r": "committed", "dex_retarget": "invalid"}


def test_failed_prepare_without_output_directory_is_visible_and_old_errors_clear(pipeline_case):
    case = pipeline_case
    case["run"]()
    root = case["root"] / "output/run_id=A"
    with (root / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({"clip_id": "early-failure", "stage": "prepare", "status": "Failed", "error": "ffmpeg failed"}) + "\n")
        handle.write(json.dumps({"clip_id": "restarted", "stage": "prepare", "status": "Failed", "error": "old error"}) + "\n")
        handle.write(json.dumps({"clip_id": "restarted", "stage": "prepare", "status": "Running", "error": ""}) + "\n")
        handle.write('{"truncated":')
    report = collect_report(root)
    failure = next(c for c in report["clips"] if c["clip_id"] == "early-failure")
    restarted = next(c for c in report["clips"] if c["clip_id"] == "restarted")
    assert failure["status"] == "failed" and failure["errors"] == {"prepare": "ffmpeg failed"}
    assert restarted["status"] == "incomplete" and restarted["errors"] == {}
    assert report["issues"]


def test_report_escapes_embedded_json_and_does_not_import_models(pipeline_case):
    case = pipeline_case
    case["run"]()
    root = case["root"] / "output/run_id=A"
    attack = '</script><script>alert("x")</script>'
    summary = json.loads((root / "run_summary.json").read_text())
    summary["error"] = attack
    (root / "run_summary.json").write_text(json.dumps(summary))
    program = '''
import sys
sys.modules['torch'] = None
sys.modules['ray'] = None
from egoworld.cli import main
main()
'''
    result = subprocess.run([sys.executable, "-c", program, "report", "--run-dir", str(root)],
                            check=True, capture_output=True, text=True)
    page = Path(json.loads(result.stdout)["report"]).read_text()
    assert attack not in page and "https://" not in page
    payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S)[1]
    assert json.loads(payload)["summary"]["error"] == attack


def test_resumed_calls_do_not_invent_timing(pipeline_case):
    case = pipeline_case
    case["run"]()
    case["run"]()
    report = collect_report(case["root"] / "output/run_id=A")
    assert report["stages"][0]["calls"] == 0 and report["stages"][0]["wall"] is None
    assert report["summary"]["skipped_clips"] == 3
    assert len(report["clips"]) == 3


def test_partial_and_geometry_only_runs_do_not_show_false_segmentation_coverage(tmp_path):
    root = tmp_path / "run_id=geometry"
    clip = root / "video_id=v/clip_id=c"
    clip.mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps({"output_schema_version": 3, "stages": {"fast3r": []}}))
    (clip / "qc.json").write_text(json.dumps({"frame_coverage": 0, "status": "pass"}))
    report = collect_report(root)
    assert report["clips"][0]["coverage"] is None
    assert report["clips"][0]["status"] == "incomplete"
    assert not report["summary"]
    assert write_report(root).is_file()


def test_export_limit_and_invalid_input_are_explicit(pipeline_case):
    case = pipeline_case
    case["run"]()
    root = case["root"] / "output/run_id=A"
    report = collect_report(root, max_clips=1)
    assert len(report["clips"]) == 1 and report["discovered_clips"] == 3
    assert report["truncated"]
    with pytest.raises(ValueError, match="positive integer"):
        collect_report(root, max_clips=0)
    with pytest.raises(ValueError, match=".html"):
        write_report(root, root / "qc.json")
    with pytest.raises(ValueError, match="run_manifest"):
        collect_report(root / "missing")
