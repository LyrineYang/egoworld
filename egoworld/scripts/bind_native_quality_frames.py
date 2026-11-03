#!/usr/bin/env python3
"""Bind stock WiLoR stats positions to actual sorted extraction filenames.

The stock OP uses lexical filename order, not numeric video order. Preserve its
outputs and repair only the scorer's frame identity, independently of pose GT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def frame_indices(temporal_indices, files):
    files = sorted(Path(p) for p in files)
    ordinals = []
    for file in files:
        match = re.fullmatch(r"frame_(\d+)\.jpg", file.name)
        if not match or not file.is_file():
            raise ValueError("Unexpected file in native extraction directory")
        ordinals.append(int(match.group(1)))
    if len(ordinals) != len(temporal_indices) or set(ordinals) != set(range(len(temporal_indices))):
        raise ValueError("Native frame files are missing, duplicated or outside the declared sample")
    return [temporal_indices[i] for i in ordinals], files


def bind(corpus_path, frame_root, times_path, native_export, output, matching="exact"):
    import cv2
    import numpy as np
    corpus_path, output = Path(corpus_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    corpus = json.loads(corpus_path.read_text())
    records, selected_positions, samples_count = {}, {}, len(corpus["samples"])
    if samples_count != 1:
        raise ValueError("This pilot binds one montage; provide one native timing record per video")
    times = json.loads(Path(times_path).read_text())
    for name, sample in corpus["samples"].items():
        folder = Path(frame_root) / Path(sample["video_path"]).stem
        if Path(times["video"]).resolve() != Path(sample["video_path"]).resolve():
            raise ValueError("Native timing video identity mismatch")
        actual_indices = [r["source_frame"] for r in times["frames"]]
        indices, files = frame_indices(actual_indices, folder.iterdir())
        cap = cv2.VideoCapture(sample["video_path"])
        verified, selected, score_indices = [], [], []
        try:
            for index, file in zip(indices, files):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, reference = cap.read()
                frame = cv2.imread(str(file))
                if not ok or frame is None or frame.shape != reference.shape:
                    raise ValueError("Native frame shape does not match the declared source frame")
                difference = float(np.abs(frame.astype(float) - reference.astype(float)).mean())
                # JPEG extraction is lossy. This is an identity check for this
                # pilot, not a pose metric or a threshold selected on pose GT.
                if difference > 5:
                    raise ValueError(f"Native frame content mismatch at {file.name}: {difference}")
                verified.append({"file": str(file), "source_frame": index,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "decoded_mean_absolute_difference_0_255": difference})
            for score_frame in sample["score_frames"]:
                candidates = [i for i, index in enumerate(indices) if index == score_frame]
                equivalent_frames = [score_frame]
                if matching == "identical-image":
                    source = next(s for s in corpus["source_images"].values() if score_frame in s["montage_frames"])
                    equivalent_frames = source["montage_frames"]
                    pixel_hashes = []
                    for f in equivalent_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, image = cap.read()
                        if not ok:
                            raise ValueError("Cannot verify repeated-frame image identity")
                        pixel_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    if len(set(pixel_hashes)) != 1:
                        raise ValueError("Image-equivalent scoring requires identical decoded source pixels")
                    candidates = [i for i, index in enumerate(indices) if index in equivalent_frames]
                if candidates:
                    chosen = min(candidates, key=lambda i: (abs(indices[i] - score_frame), indices[i], i))
                    selected.append(chosen); score_indices.append(score_frame)
        finally:
            cap.release()
        sample["native_frame_indices"] = score_indices
        selected_positions[name] = selected
        records[name] = {"actual_lexical_frame_order": indices, "files": verified,
            "selected_raw_stats_positions": selected, "canonical_score_frames": score_indices,
            "strict_missing_score_frames": [f for f in sample["score_frames"] if f not in indices],
            "duplicate_source_frames": sorted({f for f in indices if indices.count(f) > 1})}
    # A separate score-only view, with every field copied from the raw output.
    # Never modify native exports or call the model again.
    native_export = Path(native_export)
    sidecar = native_export.with_name(native_export.stem + "_stats.jsonl")
    main = [json.loads(s) for s in native_export.read_text().splitlines() if s.strip()]
    rows = [json.loads(s) for s in sidecar.read_text().splitlines() if s.strip()]
    if len(main) != 1 or len(rows) != 1:
        raise ValueError("Expected one native montage row")
    meta = rows[0]["__dj__meta__"]
    meta = json.loads(meta) if isinstance(meta, str) else meta
    name = meta["input_id"]
    tags = meta["hand_reconstruction_tags"]
    if any(len(values) != len(times["frames"]) for values in tags.values()):
        raise ValueError("Native prediction/timing cardinality mismatch")
    meta["hand_reconstruction_tags"] = {k: [values[i] for i in selected_positions[name]] for k, values in tags.items()}
    rows[0]["__dj__meta__"] = meta
    derived = output.with_suffix(".native-scored.jsonl")
    derived_sidecar = derived.with_name(derived.stem + "_stats.jsonl")
    if derived.exists() or derived_sidecar.exists():
        raise FileExistsError("Score view already exists")
    corpus["native_frame_binding"] = {"source_corpus": str(corpus_path.resolve()),
        "source_corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "method": "Actual stock sampling PTS and lexical filenames plus decoded pixel comparison; no pose labels used",
        "matching": matching, "records": records,
        "native_export": str(native_export), "native_export_sha256": hashlib.sha256(native_export.read_bytes()).hexdigest(),
        "native_stats_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "timing_record_sha256": hashlib.sha256(Path(times_path).read_bytes()).hexdigest(),
        "score_view_export": str(derived)}
    output.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text("\n".join(json.dumps(r, allow_nan=False) for r in main) + "\n")
    derived_sidecar.write_text("\n".join(json.dumps(r, allow_nan=False) for r in rows) + "\n")
    output.write_text(json.dumps(corpus, indent=2, allow_nan=False) + "\n")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--frame-times", required=True)
    parser.add_argument("--native-export", required=True)
    parser.add_argument("--matching", choices=("exact", "identical-image"), default="exact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bind(args.corpus, args.frames_root, args.frame_times, args.native_export, args.output, args.matching)
