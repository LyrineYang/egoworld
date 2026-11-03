"""Sequential paired product study with exact-model and independent quality gates.

Plans declare all groups, orders, inputs and quality policies before execution.
No failure is retried/overwritten automatically. Native CPU and CUDA references
remain separate from DefaultExecutor's explicitly shared-model configuration.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / "scripts")]

from product_comparison import run as benchmark, sha
from matched_audit import audit_recipe_pair
from output_quality import egoworld_predictions, datajuicer_predictions, evaluate, canonical, CONTRACT
from bind_native_quality_frames import bind


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frozen_files(plan_path, plan):
    """Keep identities fixed across groups, as well as within each trial."""
    paths = {Path(plan_path).resolve(), Path(__file__).resolve()}
    paths.update((ROOT / "src/egoworld").rglob("*.py"))
    for group in plan["groups"]:
        spec_path = Path(group["spec"])
        spec = json.loads(spec_path.read_text())
        paths.add(spec_path.resolve())
        paths.update(Path(p).resolve() for p in spec.get("inputs", []))
        paths.update(Path(p).resolve() for p in spec.get("dependencies", []))
        paths.update(Path(j["config"]).resolve() for j in spec["jobs"] if "config" in j)
    return {str(p): sha(p) for p in sorted(paths)}


def verify_frozen(identities):
    changed = [p for p, digest in identities.items() if not Path(p).is_file() or sha(p) != digest]
    if changed:
        raise ValueError(f"Frozen study files changed across groups: {changed}")


def native_common_predictions(export, source_indices, samples, focal_ratio):
    """Normalize every native hand slot, including repeated source frames.

    Selection on GT frames is only for scoring. The common product artifact
    must not discard other predictions or collapse duplicate sampled frames.
    """
    export = Path(export)
    main = [json.loads(s) for s in export.read_text().splitlines() if s.strip()]
    sidecar = export.with_name(export.stem + "_stats.jsonl")
    rows = [json.loads(s) for s in sidecar.read_text().splitlines() if s.strip()]
    if len(rows) != 1:
        raise ValueError("Expected one native montage metadata row")
    meta = rows[0]["__dj__meta__"]
    meta = json.loads(meta) if isinstance(meta, str) else meta
    sample = meta["input_id"]
    if len(main) != 1 or main[0]["sample_id"] != sample or sample not in samples:
        raise ValueError("Unmapped full native product identity")
    size = samples[sample]["image_size"]
    tags = meta["hand_reconstruction_tags"]
    if any(len(tags[k]) != len(source_indices) for k in ("joints", "if_right_hand", "camera_translation")):
        raise ValueError("Full native frame metadata coverage mismatch")
    predictions = []
    for slot, (frame, joints, sides, cameras) in enumerate(zip(source_indices,
            tags["joints"], tags["if_right_hand"], tags["camera_translation"])):
        if not len(joints) == len(sides) == len(cameras):
            raise ValueError("Full native hand field coverage mismatch")
        for hand, (xyz, side, camera) in enumerate(zip(joints, sides, cameras)):
            if side not in (0, 1):
                raise ValueError("Invalid native side")
            predictions.append(canonical(sample, frame, f"slot-{slot}-hand-{hand}",
                "right" if side else "left", xyz, camera, focal_ratio * max(size), size))
    return predictions


def quality(group, output, spec):
    corpus_path = Path(group["corpus"])
    corpus = json.loads(corpus_path.read_text())
    result = {"contract": CONTRACT, "corpus": str(corpus_path), "corpus_sha256": sha(corpus_path),
        "policy": group["quality_policy"], "exact_frames": {}, "identical_image_content": {},
        "image_content_policy": "Supplementary static-image comparison declared before these runs; identical decoded repeated source pixels verified; never select on pose error"}
    for job in spec["jobs"]:
        folder = output / job["name"]
        raw = json.loads((folder / "trial.json").read_text())
        if job["kind"] in ("egoworld_recipe", "datajuicer_matched_recipe"):
            normalized_started = time.perf_counter()
            path = folder / "worker/runs/run_id=c1/data_manifest.json"
            predictions, evidence = egoworld_predictions(path, corpus["samples"], group["focal_ratio"])
            common_path = folder / "common-output-predictions.json"
            write(common_path, {"contract": CONTRACT, "predictions": predictions})
            normalization_seconds = time.perf_counter() - normalized_started
            score = evaluate(predictions, corpus["ground_truth"], corpus["samples"], group["quality_policy"])
            score.update(native_wall_seconds=raw["wall_seconds"], raw_trial_status=raw["status"], source_outputs=evidence,
                common_output_file=str(common_path), common_output_sha256=sha(common_path),
                common_output_rows=len(predictions), normalization_seconds=normalization_seconds,
                common_output_ready_seconds=raw["wall_seconds"] + normalization_seconds)
            result["exact_frames"][job["name"]] = score
            result["identical_image_content"][job["name"]] = score
            write(folder / "quality-predictions.json", {"contract": CONTRACT, "predictions": predictions})
        elif job["kind"] == "datajuicer_native":
            cfg = json.loads(Path(job["config"]).read_text())
            op = next(p["video_hand_reconstruction_mapper"] for p in cfg["process"] if "video_hand_reconstruction_mapper" in p)
            if len(corpus["samples"]) != 1:
                raise ValueError("Native image task requires one frozen montage")
            video = next(iter(corpus["samples"].values()))["video_path"]
            times = folder / "native-sample-times.json"
            alignment_started = time.perf_counter()
            env = dict(os.environ, UV_OFFLINE="1", PIP_NO_INDEX="1", PYTHONDONTWRITEBYTECODE="1")
            with (folder / "native-sample-times.log").open("w") as log:
                subprocess.run([job["python"], str(ROOT / "scripts/native_quality_sample_times.py"),
                    "--video", video, "--frame-num", str(op.get("frame_num", 3)), "--output", str(times)],
                    check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            alignment_seconds = time.perf_counter() - alignment_started
            normalized_started = time.perf_counter()
            from bind_native_quality_frames import frame_indices
            temporal = [r["source_frame"] for r in json.loads(times.read_text())["frames"]]
            frame_folder = Path(op["frame_dir"]) / Path(video).stem
            actual_indices, _ = frame_indices(temporal, frame_folder.iterdir())
            common = native_common_predictions(folder / "worker/datajuicer.jsonl", actual_indices, corpus["samples"], group["focal_ratio"])
            common_path = folder / "common-output-predictions.json"
            write(common_path, {"contract": CONTRACT, "predictions": common})
            normalization_seconds = alignment_seconds + time.perf_counter() - normalized_started
            for mode, key in (("exact", "exact_frames"), ("identical-image", "identical_image_content")):
                view_path = folder / ("corpus-" + mode + ".json")
                bound = bind(corpus_path, op["frame_dir"], times, folder / "worker/datajuicer.jsonl", view_path, mode)
                if bound["ground_truth"] != corpus["ground_truth"]:
                    raise ValueError("Frame binding changed pose labels")
                predictions, evidence = datajuicer_predictions(bound["native_frame_binding"]["score_view_export"], bound["samples"], group["focal_ratio"])
                score = evaluate(predictions, corpus["ground_truth"], corpus["samples"], group["quality_policy"])
                score.update(native_wall_seconds=raw["wall_seconds"], raw_trial_status=raw["status"],
                    source_outputs=evidence, binding_file=str(view_path), binding_sha256=sha(view_path),
                    common_output_file=str(common_path), common_output_sha256=sha(common_path), common_output_rows=len(common),
                    normalization_seconds=normalization_seconds, common_output_ready_seconds=raw["wall_seconds"] + normalization_seconds)
                result[key][job["name"]] = score
                write(folder / (mode + "-quality-predictions.json"), {"contract": CONTRACT, "predictions": predictions})
    write(output / "quality.json", result)
    return result


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output).resolve()
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    state = {"status": "running", "pid": os.getpid(), "plan_sha256": sha(plan_path),
        "started_utc_epoch": time.time(), "groups": [], "active_group": None,
        "same_model_speed_ratio": None, "goal_complete": False}
    write(output / "study-state.json", state)
    try:
        identities = frozen_files(plan_path, plan)
        write(output / "frozen-file-hashes.json", identities)
        for group in plan["groups"]:
            verify_frozen(identities)
            name = group["name"]
            if not name or Path(name).name != name:
                raise ValueError("Invalid study group name")
            state.update(active_group=name, phase="benchmark")
            write(output / "study-state.json", state)
            spec = json.loads(Path(group["spec"]).read_text())
            folder = output / name
            report = benchmark(group["spec"], folder)
            verify_frozen(identities)
            summary = {"name": name, "track": group["track"], "raw_status": report["status"],
                "raw_report": str(folder / "report.json"), "raw_report_sha256": sha(folder / "report.json"),
                "trials": [{k: t.get(k) for k in ("name", "kind", "status", "wall_seconds", "error",
                    "process_tree_peak_rss_bytes", "gpu_whole_device_peak_used_bytes")} for t in report["trials"]]}
            state["groups"].append(summary)
            for key in ("sources_unchanged", "inputs_unchanged", "configs_unchanged", "dependencies_unchanged"):
                if not report[key]:
                    raise ValueError(f"Frozen study identity changed: {key}")
            if report["status"] != "completed":
                # Legitimate empty model predictions remain in the product
                # quality denominator. A failed/incomplete execution does not
                # receive an eligible fixed-model performance ratio.
                allowed = group["track"] != "fixed_models"
                for job, trial in zip(spec["jobs"], report["trials"]):
                    if trial["status"] == "passed":
                        continue
                    manifest = folder / job["name"] / "worker/runs/run_id=c1/data_manifest.json"
                    allowed = allowed and job["kind"] == "egoworld_recipe" and manifest.exists()
                    allowed = allowed and json.loads(manifest.read_text()).get("status") == "partial"
                if not allowed:
                    raise RuntimeError(f"Study group did not complete: {name}; raw failures retained")
            state["phase"] = "audit"
            write(output / "study-state.json", state)
            if group["track"] == "fixed_models":
                if len(spec["jobs"]) < 2:
                    raise ValueError("Fixed model groups require at least two complete pipelines")
                reference = group.get("reference_job", spec["jobs"][0]["name"])
                if reference not in {j["name"] for j in spec["jobs"]}:
                    raise ValueError("Missing fixed-model reference pipeline")
                reference_path = folder / reference / "worker/runs/run_id=c1/data_manifest.json"
                parity = {"passed": True, "reference": reference, "comparisons": {}}
                for job in spec["jobs"]:
                    if job["name"] != reference:
                        path = folder / job["name"] / "worker/runs/run_id=c1/data_manifest.json"
                        parity["comparisons"][job["name"]] = audit_recipe_pair(reference_path, path)
                write(folder / "full-parity.json", parity)
                summary["full_parity"] = parity
            if group.get("corpus"):
                state["phase"] = "quality"
                write(output / "study-state.json", state)
                scored = quality(group, folder, spec)
                summary["quality_file"] = str(folder / "quality.json")
                summary["quality_summary"] = {mode: {name: {k: s[k] for k in
                    ("gt_hands", "matched_hands", "false_positives", "mpjpe_root_mm_detected_only", "quality_hands")}
                    for name, s in scored[mode].items()} for mode in ("exact_frames", "identical_image_content")}
            summary["audit_status"] = "passed"
            verify_frozen(identities)
            write(output / "study-state.json", state)
            print(json.dumps({"group": name, "status": summary["audit_status"],
                "seconds": [t["wall_seconds"] for t in summary["trials"]]}), flush=True)
        state.update(status="completed", active_group=None, phase="done")
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished_or_last_epoch"] = time.time()
        write(output / "study-state.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
