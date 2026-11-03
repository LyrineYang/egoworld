"""Common-output hand quality evaluation, independent of the model/framework.

No prediction is used as ground truth. Failed/missing samples remain in the
coverage denominator; inter-model disagreement is never called accuracy.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CONTRACT = {
    "version": 1, "output": "hand_joints", "joint_order": "openpose21",
    "length_unit": "meters", "coord_frame": "hand_root",
    "axes": "camera_x_right_y_down_z_forward", "root_index": 0,
    "scale_status": "model_prior_not_measured_world_scale",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def points(value, dimensions=3):
    a = np.asarray(value, dtype=float)
    if a.shape != (21, dimensions) or not np.isfinite(a).all():
        raise ValueError(f"Expected finite 21x{dimensions} joints")
    return a


def canonical(sample, frame, identity, side, xyz, translation, focal, size):
    xyz = points(xyz)
    translation = np.asarray(translation, dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Invalid camera translation")
    if side not in ("left", "right"):
        raise ValueError("Unknown hand side")
    size = np.asarray(size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Invalid image dimensions")
    camera = xyz + translation
    if np.any(camera[:, 2] <= 0) or not np.isfinite(focal) or focal <= 0:
        raise ValueError("Projection requires positive camera depth/focal length")
    image = camera[:, :2] / camera[:, 2:] * focal + np.asarray(size) / 2
    return {"sample_id": sample, "frame_index": int(frame), "prediction_id": str(identity),
            "handedness": side, "joints_root_m": (xyz - xyz[0]).tolist(),
            "joints_image_px": image.tolist()}


def egoworld_predictions(manifest_path, samples, focal_ratio):
    """Read checked product exports; explicit sample names bind clip identity."""
    import pyarrow.parquet as pq
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["status"] not in ("passed", "partial"):
        raise ValueError("EgoWorld run has no completed product exports to score")
    predictions, evidence = [], []
    for clip, exports in manifest["exports"].items():
        names = [name for name in samples if clip.startswith(name + "-")]
        if len(names) != 1:
            raise ValueError(f"Ambiguous/unmapped clip {clip}")
        sample = names[0]
        info = exports.get("hand_joints")
        if info is None:
            continue
        path = Path(info["file"])
        if digest(path) != info["sha256"]:
            raise ValueError("EgoWorld export hash changed")
        evidence.append({"path": str(path), "sha256": digest(path)})
        for row in pq.ParquetFile(path).read().to_pylist():
            if any(row[k] != v for k, v in {
                "joint_order": "openpose21", "coord_frame": "hand_root",
                "length_unit": "meters", "scale_status": "model_prior"}.items()):
                raise ValueError("Unsupported EgoWorld joint coordinates")
            size = samples[sample]["image_size"]
            predictions.append(canonical(sample, row["frame_index"], row["object_id"],
                row["handedness"], np.asarray(row["joints_3d"]).reshape(21, 3),
                row["camera_translation"], focal_ratio * max(size), size))
    return predictions, evidence


def datajuicer_predictions(export_path, samples, focal_ratio):
    """Keep native side predictions, roots and all hands (including duplicates).

    WiLoR already mirrors left-hand X in the stock OP. Its joints still have
    MANO origin, so translate only for projection; canonical() subtracts wrist.
    Frame identity must be supplied explicitly; stock tags omit source indices.
    """
    path = Path(export_path)
    main = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    sidecar = path.with_name(path.stem + "_stats.jsonl")
    records = [json.loads(s) for s in sidecar.read_text().splitlines() if s.strip()] if sidecar.exists() else main
    if len(main) != len(records):
        raise ValueError("Native stats coverage mismatch")
    ids = [r["sample_id"] for r in main]
    if len(ids) != len(set(ids)) or set(ids) - set(samples):
        raise ValueError("Duplicate/unmapped native sample identity")
    predictions, seen = [], set()
    for row in records:
        meta = row.get("__dj__meta__", {})
        meta = json.loads(meta) if isinstance(meta, str) else meta
        sample = meta.get("input_id")
        if sample not in ids or sample in seen:
            raise ValueError("Missing/duplicate native metadata identity")
        seen.add(sample)
        tags = meta.get("hand_reconstruction_tags", {})
        joints, sides, translations = (tags.get(k, []) for k in ("joints", "if_right_hand", "camera_translation"))
        indices = samples[sample]["native_frame_indices"]
        if not (len(joints) == len(sides) == len(translations) == len(indices)):
            raise ValueError("Native frame coverage differs from declared sampling")
        size = samples[sample]["image_size"]
        for frame, xyzs, hand_sides, cams in zip(indices, joints, sides, translations):
            if not len(xyzs) == len(hand_sides) == len(cams):
                raise ValueError("Native hand field coverage mismatch")
            for i, (xyz, right, cam) in enumerate(zip(xyzs, hand_sides, cams)):
                if right not in (0, 1):
                    raise ValueError("Invalid native hand side")
                predictions.append(canonical(sample, frame, i, "right" if right else "left",
                    xyz, cam, focal_ratio * max(size), size))
    return predictions, [{"path": str(p), "sha256": digest(p)} for p in (path, sidecar) if p.exists()]


def similarity_align(prediction, target):
    """Least-squares scale+proper rotation+translation; reflections forbidden."""
    x, y = points(prediction), points(target)
    xc, yc = x - x.mean(0), y - y.mean(0)
    variance = np.sum(xc**2)
    if variance < 1e-16 or np.linalg.matrix_rank(xc, tol=1e-9) < 2:
        raise ValueError("Degenerate prediction cannot be similarity-aligned")
    u, singular, vt = np.linalg.svd(xc.T @ yc)
    orientation = np.ones(3)
    orientation[-1] = np.linalg.det(u @ vt)
    rotation = (u * orientation) @ vt
    scale = np.dot(singular, orientation) / variance
    return scale * xc @ rotation + y.mean(0)


def evaluate(predictions, ground_truth, samples, policy):
    """Match by image-space wrist distance, never by best 3D pose error.

    Image-space matching ignores side, enabling independent side-error counts.
    Every other valid prediction in a scored frame is an FP. Error averages are
    conditional on detection; PCK and quality yield keep all GT in denominator.
    """
    matching = float(policy["wrist_match_diagonal_fraction"])
    thresholds = policy["mpjpe_thresholds_mm"]
    if not 0 < matching <= 1 or not thresholds or any(t <= 0 for t in thresholds):
        raise ValueError("Invalid preregistered quality thresholds")
    gt_keys = [(r["sample_id"], r["frame_index"], r["hand_id"]) for r in ground_truth]
    pred_keys = [(r["sample_id"], r["frame_index"], r["prediction_id"]) for r in predictions]
    if len(gt_keys) != len(set(gt_keys)) or len(pred_keys) != len(set(pred_keys)):
        raise ValueError("Duplicate GT/prediction identity")
    if not ground_truth:
        raise ValueError("Accuracy evaluation requires independent ground truth")
    for row in ground_truth:
        points(row["joints_camera_m"])
        points(row["joints_image_px"], 2)
        if row["sample_id"] not in samples or row["handedness"] not in ("left", "right"):
            raise ValueError("Unknown GT sample/side")
    for row in predictions:
        points(row["joints_root_m"])
        points(row["joints_image_px"], 2)
        if row["sample_id"] not in samples or row["handedness"] not in ("left", "right"):
            raise ValueError("Unknown prediction sample/side")
    frames = {(r["sample_id"], r["frame_index"]) for r in ground_truth}
    # Negative frames are explicit; missing GT is never treated as background.
    frames.update((name, i) for name, s in samples.items() for i in s["score_frames"])
    details, total_predictions, tp, fp, side_correct = [], 0, 0, 0, 0
    root_errors, pa_errors, projected_errors = [], [], []
    pck_hits = {str(t): 0 for t in thresholds}
    quality_hits = {str(t): 0 for t in thresholds}
    for sample, frame in sorted(frames):
        gts = [r for r in ground_truth if (r["sample_id"], r["frame_index"]) == (sample, frame)]
        preds = [r for r in predictions if (r["sample_id"], r["frame_index"]) == (sample, frame)]
        diagonal = float(np.linalg.norm(samples[sample]["image_size"]))
        pairs = sorted((float(np.linalg.norm(np.asarray(p["joints_image_px"])[0] -
                    np.asarray(g["joints_image_px"])[0])), gi, pi)
                       for gi, g in enumerate(gts) for pi, p in enumerate(preds))
        matched_gt, matched_pred = set(), set()
        for distance, gi, pi in pairs:
            if distance > matching * diagonal or gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi); matched_pred.add(pi)
            g, p = gts[gi], preds[pi]
            target = points(g["joints_camera_m"]); target -= target[0].copy()
            predicted = points(p["joints_root_m"]); predicted -= predicted[0].copy()
            errors = np.linalg.norm(predicted - target, axis=1) * 1000
            error = float(errors.mean())
            try:
                aligned = float(np.linalg.norm(similarity_align(predicted, target) - target, axis=1).mean() * 1000)
            except ValueError:
                aligned = None
            image_error = float(np.linalg.norm(points(p["joints_image_px"], 2) - points(g["joints_image_px"], 2), axis=1).mean())
            side = p["handedness"] == g["handedness"]
            tp += 1; side_correct += int(side)
            root_errors.append(error); projected_errors.append(image_error)
            if aligned is not None:
                pa_errors.append(aligned)
            for t in thresholds:
                pck_hits[str(t)] += int(np.sum(errors <= t))
                quality_hits[str(t)] += int(error <= t and side)
            details.append({"sample_id": sample, "frame_index": frame, "hand_id": g["hand_id"],
                "prediction_id": p["prediction_id"], "status": "matched", "side_correct": side,
                "mpjpe_root_mm": error, "pa_mpjpe_mm": aligned, "reprojection_mean_px": image_error})
        fp += len(preds) - len(matched_pred)
        total_predictions += len(preds)
        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                details.append({"sample_id": sample, "frame_index": frame, "hand_id": g["hand_id"], "status": "missed"})
    count = len(ground_truth)
    mean = lambda values: float(np.mean(values)) if values else None
    return {"gt_hands": count, "scored_predictions": total_predictions, "matched_hands": tp,
        "false_positives": fp, "missed_hands": count - tp,
        "recall": tp / count, "precision": tp / total_predictions if total_predictions else None,
        "handedness_accuracy_matched": side_correct / tp if tp else None,
        "mpjpe_root_mm_detected_only": mean(root_errors), "pa_mpjpe_mm_detected_only": mean(pa_errors),
        "pa_evaluable_hands": len(pa_errors), "reprojection_mean_px_detected_only": mean(projected_errors),
        "pck_all_gt": {k: v / (21 * count) for k, v in pck_hits.items()},
        "quality_hands": quality_hits, "quality_yield_all_gt": {k: v / count for k, v in quality_hits.items()},
        "source_image_count": len({r.get("source_image_id", r["sample_id"]) for r in ground_truth}),
        "details": details}


def run(spec_path, output):
    spec_path, output = Path(spec_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(spec_path.read_text())
    gt_path = Path(spec["corpus"])
    corpus = json.loads(gt_path.read_text())
    if corpus.get("ground_truth_unit") != "meters" or corpus.get("ground_truth_joint_order") != "openpose21":
        raise ValueError("Ground-truth unit/joint order must match the common contract")
    samples = corpus["samples"]
    for sample in samples.values():
        if digest(sample["video_path"]) != sample["video_sha256"]:
            raise ValueError("Frozen quality input changed")
    report = {"contract": CONTRACT, "policy": spec["policy"], "corpus": str(gt_path),
        "corpus_sha256": digest(gt_path), "spec_sha256": digest(spec_path),
        "comparison_track": "native_models_common_output_quality", "pipelines": {},
        "framework_speed_ratio": None,
        "limits": ["Pilot subset, no general accuracy/speed superiority conclusion.",
            "Source images may be correlated; image count is not independent subject/capture count.",
            "PA alignment removes scale/orientation/translation; it does not validate metric world pose.",
            "Conditional pose errors are accompanied by all-GT coverage and quality yield.",
            "Model/resource/output scopes differ; end-to-end time is a native product observation."]}
    names = [p["name"] for p in spec["pipelines"]]
    if len(names) != len(set(names)) or any(not n or Path(n).name != n for n in names):
        raise ValueError("Pipeline names must be unique plain file names")
    for pipeline in spec["pipelines"]:
        trial = json.loads(Path(pipeline["trial"]).read_text())
        if pipeline["kind"] == "egoworld":
            state = json.loads(Path(pipeline["manifest"]).read_text())["status"]
            if trial["status"] != "passed" and state != "partial":
                raise ValueError("Failed job is not a completed partial product; inspect raw trial")
            predictions, evidence = egoworld_predictions(pipeline["manifest"], samples, pipeline["focal_ratio"])
        elif pipeline["kind"] == "datajuicer":
            if trial["status"] != "passed":
                raise ValueError("Native job did not complete; inspect raw trial")
            predictions, evidence = datajuicer_predictions(pipeline["export"], samples, pipeline["focal_ratio"])
        else:
            raise ValueError("Unsupported native output adapter")
        normalized = output / (pipeline["name"] + "-predictions.json")
        normalized.write_text(json.dumps({"contract": CONTRACT, "predictions": predictions}, indent=2, allow_nan=False) + "\n")
        quality = evaluate(predictions, corpus["ground_truth"], samples, spec["policy"])
        quality.update(raw_trial_status=trial["status"], native_wall_seconds=trial["wall_seconds"],
            quality_hands_per_second={k: v / trial["wall_seconds"] for k, v in quality["quality_hands"].items()},
            native_trial_sha256=digest(pipeline["trial"]), source_outputs=evidence,
            normalized_predictions=str(normalized), normalized_sha256=digest(normalized))
        report["pipelines"][pipeline["name"]] = quality
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
