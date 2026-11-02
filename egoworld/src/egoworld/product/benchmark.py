"""Compare explicit workload identities before presenting performance ratios."""
import json
from pathlib import Path


def compare_runs(paths):
    if len(paths) < 2:
        raise ValueError("Compare at least two recipe run directories")
    records = []
    for path in paths:
        root = Path(path)
        manifest = json.loads((root / "data_manifest.json").read_text())
        recipe = json.loads((root / "recipe.json").read_text())
        videos = [json.loads(l) for l in (root / "videos.jsonl").read_text().splitlines() if l.strip()]
        clean = json.loads((root / "cleaning.json").read_text())
        stages = sorted({(r["kernel"], r["key"]) for r in manifest.get("stage_identities", [])})
        peaks = {"process_peak_rss_bytes": 0, "torch_peak_reserved_bytes": 0}
        computed = {(r["clip_id"], r["kernel"]) for r in manifest["computed"]}
        observed = 0
        for receipt in (root / "model-output").glob("run_id=*/video_id=*/clip_id=*/stage_*.json"):
            if (receipt.parent.name.removeprefix("clip_id="), receipt.stem.removeprefix("stage_")) not in computed:
                continue  # cached source telemetry is not a measurement of this invocation
            observed += 1
            stats = json.loads(receipt.read_text()).get("stats", {})
            for field in peaks:
                peaks[field] = max(peaks[field], stats.get(field, 0))
        records.append({"root": str(root.resolve()), "status": manifest["status"],
            "inputs": sorted(v["checksum"] for v in videos), "input_video_seconds": sum(v["duration_s"] for v in videos),
            "outputs": recipe["outputs"], "cleaning": clean["policy"], "retained_inputs": clean["summary"],
            "calibration": recipe.get("calibration"),
            "stage_identities": stages, "cache_hits": len(manifest["cache_hits"]),
            "computed": len(manifest["computed"]), "derived_hits": len(manifest.get("derived_cache_hits", [])),
            "timings": manifest.get("timings", {}), "peaks": peaks, "observed_model_stages": observed,
            "world_coverage": {c: {k: exports["world"].get(k) for k in ("outputs", "skipped_frames", "calibration_sha256", "status")} for c, exports in manifest["exports"].items() if "world" in exports},
            "export_coverage": {c: {n: {k: v[k] for k in ("rows", "points") if k in v} for n, v in exports.items() if n != "world"} for c, exports in manifest["exports"].items()}})
    first = records[0]
    differences = []
    for record in records[1:]:
        changed = [key for key in ("inputs", "outputs", "cleaning", "retained_inputs", "stage_identities", "export_coverage", "calibration", "world_coverage") if record[key] != first[key]]
        if changed or not first["stage_identities"]:
            differences.append({"run": record["root"], "changed": changed or ["missing_stage_identities"]})
    return {"status": "matched" if not differences else "different_workloads", "runs": records,
        "differences": differences,
        "scope": "Identity and coverage comparison, not a quality-accuracy benchmark. Cache/model temperature must be interpreted separately. Peaks only include stages computed in this invocation; zero observed stages means model peaks are unobserved, not zero total RAM. RSS is per-stage lifetime high-water, not total concurrent RAM."}
