"""CPU-only identity and sparse decode experiments; does not change execution."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2

from egoworld.product.cache import digest, runtime_identity, stage_identity
from egoworld.product.recipes import compile_recipe
from egoworld.utils.provenance import provenance


def read_samples(path, indices, sequential):
    cap = cv2.VideoCapture(str(path))
    wanted = set(indices)
    hashes, reads, seeks = {}, 0, 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        for index in range(max(indices) + 1) if sequential else indices:
            if not sequential and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                seeks += 1
            ok, frame = cap.read()
            reads += 1
            if not ok:
                raise ValueError(f"Cannot read {path} frame {index}")
            if index in wanted:
                hashes[index] = hashlib.sha256(frame.tobytes()).hexdigest()
    finally:
        cap.release()
    return hashes, reads, seeks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config, _, _ = compile_recipe(args.root / "chicken-hands.json")
    config = config.resolved()
    run = args.root / "data-runs/run_id=release_chicken-hands"
    report = {"scope": "CPU microbenchmarks only; not GPU/pipeline/Data-Juicer throughput; page caches not flushed",
              "identity": {"provenance_seconds": [], "runtime_identity_seconds": {}},
              "identity_matrix": [], "decode": []}
    for _ in range(3):
        start = time.perf_counter()
        identity = provenance(config, str(run / "videos.jsonl"), str(run / "clips.jsonl"))
        report["identity"]["provenance_seconds"].append(time.perf_counter() - start)
    paths = {v["path"] for v in identity["local_assets"].values() if v.get("path")}
    report["identity"]["unique_asset_bytes"] = sum(Path(p).stat().st_size for p in paths)
    report["identity"]["source_tree_hash"] = identity["source_tree_hash"]
    for stage in ("sam2", "hamer", "dex_retarget"):
        command = getattr(config.operators, stage).params.get("runner_command", [sys.executable])
        executable = command[0]
        if executable not in report["identity"]["runtime_identity_seconds"]:
            start = time.perf_counter()
            runtime_identity(command)
            report["identity"]["runtime_identity_seconds"][executable] = time.perf_counter() - start
    video = json.loads((run / "videos.jsonl").read_text().splitlines()[0])
    clip = json.loads((run / "clips.jsonl").read_text().splitlines()[0])
    clip.update(fps=video["fps"], video_sha256=video["checksum"])
    # A controlled probe of the actual cache-key implementation: only the
    # named field changes; a fixed runtime makes comparisons deterministic.
    runtime = {"controlled_probe": True}
    base = digest(stage_identity(config, identity, "sam2", clip, {}, runtime))
    variants = []
    cfg = deepcopy(config)
    cfg.operators.dex_retarget.params["research_unrelated"] = True
    variants.append(("change unrelated Dex operator parameter", cfg, identity, clip, False))
    cfg = deepcopy(config)
    cfg.model_versions["dex_retarget"] = "research-different-version"
    variants.append(("change unrelated Dex model version", cfg, identity, clip, True))
    prov = deepcopy(identity)
    prov["source_tree_hash"] = "research-unrelated-module-change"
    variants.append(("change package source hash (including unrelated module)", config, prov, clip, True))
    renamed = deepcopy(clip)
    renamed.update(video_id="renamed-video-same-bytes", clip_id="renamed-clip-same-range")
    variants.append(("rename video/clip IDs, same video SHA and frames", config, identity, renamed, True))
    for label, cfg, prov, item, expected in variants:
        changed = digest(stage_identity(cfg, prov, "sam2", item, {}, runtime)) != base
        assert changed == expected, label
        report["identity_matrix"].append({"change": label, "sam2_key_changed": changed,
            "scope": "Controlled actual key function, not a pipeline launch"})
    candidates = [Path(video["path"])]
    candidates += list((run / "model-output").glob("run_id=*/.work/*/clip.mp4"))
    for path in candidates:
        reader = cv2.VideoCapture(str(path))
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        reader.release()
        for label, indices in (("hamer_stride5", list(range(0, count, 5))),
                               ("fast3r_two_views", [0, count - 1])):
            methods = {}
            for sequential in (False, True):
                times = []
                for _ in range(7):
                    start = time.perf_counter()
                    hashes, reads, seeks = read_samples(path, indices, sequential)
                    times.append(time.perf_counter() - start)
                methods["sequential" if sequential else "seek"] = {
                    "median_ms": statistics.median(times) * 1000,
                    "samples_ms": [t * 1000 for t in times], "read_calls": reads,
                    "seek_calls": seeks, "frame_sha256": hashes}
            assert methods["seek"]["frame_sha256"] == methods["sequential"]["frame_sha256"]
            report["decode"].append({"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pattern": label, "selected_frames": indices, "methods": methods,
                "sampled_pixels_identical": True})
    (args.output / "input-costs.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output / 'input-costs.json'),
                      "identity_seconds": report['identity']['provenance_seconds'],
                      "decode_cases": len(report['decode']), "all_checks_passed": True}))


if __name__ == "__main__":
    main()
