"""Full recipe parity, including upstream masks/tracks and all exported fields."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from product_comparison import audit_egoworld


def _canonical(rows):
    return sorted(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows)


def audit_recipe_pair(first, second, *, allow_review=False):
    from egoworld.contracts import STAGE_ARTIFACTS
    from egoworld.pipeline.artifacts import load_stage
    paths = [Path(first), Path(second)]
    manifests = [json.loads(p.read_text()) for p in paths]
    for path in paths:
        audit_egoworld(path)
    if manifests[0]["plan"] != manifests[1]["plan"]:
        # Readiness may include environment-specific notes, but task fields
        # and complete kernel dependencies must match exactly.
        for key in ("outputs", "kernels"):
            if manifests[0]["plan"][key] != manifests[1]["plan"][key]:
                raise ValueError(f"Recipe task mismatch: {key}")
    clips = [set(m["exports"]) for m in manifests]
    if clips[0] != clips[1] or not clips[0]:
        raise ValueError("Recipe clip coverage mismatch/empty")
    if any(m["cache_hits"] or m.get("derived_cache_hits") for m in manifests):
        raise ValueError("Fresh-model comparison cannot reuse prior result caches")
    model_roots = [p.parent / "model-output" / f"run_id={m['run_id']}" for p, m in zip(paths, manifests)]
    run_manifests = [json.loads((p / "run_manifest.json").read_text()) for p in model_roots]
    if list(run_manifests[0]["stages"]) != list(run_manifests[1]["stages"]):
        raise ValueError("Computed stage coverage mismatch")
    for key in ("source_tree_hash", "local_assets"):
        if run_manifests[0]["provenance"][key] != run_manifests[1]["provenance"][key]:
            raise ValueError(f"Model/source identity mismatch: {key}")
    expected = {(clip, stage) for clip in clips[0] for stage in run_manifests[0]["stages"]}
    if allow_review and _canonical(manifests[0].get("filtered", [])) != _canonical(manifests[1].get("filtered", [])):
        raise ValueError("Branch filtering decisions differ")
    for manifest in manifests:
        computed = {(row["clip_id"], row["kernel"]) for row in manifest["computed"]}
        filtered = {(row["clip_id"], row["kernel"]) for row in manifest.get("filtered", [])} if allow_review else set()
        if computed & filtered or computed | filtered != expected:
            raise ValueError("Not all required model stages executed or committed a matched filter decision")
    for stage in run_manifests[0]["stages"]:
        if run_manifests[0]["config"]["operators"][stage] != run_manifests[1]["config"]["operators"][stage]:
            raise ValueError(f"Different kernel parameters/environment: {stage}")
    allowed_qc = {"pass", "review"} if allow_review else {"pass"}
    review_clips = 0
    directories = []
    for root in model_roots:
        folders = list(root.glob("video_id=*/clip_id=*"))
        index = {folder.name.removeprefix("clip_id="): folder for folder in folders}
        if len(index) != len(folders) or not clips[0].issubset(index):
            raise ValueError("Missing or duplicate model clip directory")
        directories.append(index)
    counts = {}
    def equal_tables(a, b, key):
        tables = [pq.ParquetFile(p).read() for p in (a, b)]
        if not tables[0].schema.equals(tables[1].schema) or _canonical(tables[0].to_pylist()) != _canonical(tables[1].to_pylist()):
            raise ValueError(f"Full logical output differs: {key}")
        counts[key] = counts.get(key, 0) + tables[0].num_rows
    for clip in sorted(clips[0]):
        folders = [index[clip] for index in directories]
        stages = list(run_manifests[0]["stages"])
        for stage in stages:
            for folder, run in zip(folders, run_manifests):
                if load_stage(folder, stage, run["config_fingerprint"]) is None:
                    raise ValueError("Unverified model commit")
            for table in STAGE_ARTIFACTS[stage]:
                equal_tables(*(folder / (table + ".parquet") for folder in folders), "stage/" + table)
            arrays = [set(p.name for p in folder.glob("*.npz")) for folder in folders]
            if arrays[0] != arrays[1]:
                raise ValueError("Array coverage differs")
            for filename in arrays[0]:
                with np.load(folders[0] / filename, allow_pickle=False) as a, np.load(folders[1] / filename, allow_pickle=False) as b:
                    if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                        raise ValueError(f"Array differs: {filename}")
        if set(manifests[0]["exports"][clip]) != set(manifests[1]["exports"][clip]):
            raise ValueError("Export type mismatch")
        for output, info in manifests[0]["exports"][clip].items():
            other = manifests[1]["exports"][clip][output]
            if output == "world":
                raise ValueError("World parity needs an explicitly anchored extension to this audit")
            if Path(info["file"]).suffix == ".parquet":
                equal_tables(info["file"], other["file"], "export/" + output)
        qcs = [json.loads((folder / "qc.json").read_text()) for folder in folders]
        if qcs[0] != qcs[1] or qcs[0]["status"] not in allowed_qc:
            raise ValueError("QC mismatch or incomplete product quality")
        review_clips += qcs[0]["status"] == "review"
    if not counts.get("export/hand_joints"):
        raise ValueError("Empty hand product does not prove model parity")
    return {"passed": True, "clips": len(clips[0]), "review_clips": review_clips, "stages": stages, "rows": counts,
        "comparison": "All stage/export logical fields exact, masks/tracks included, row order ignored; arrays exact; same kernel parameters and QC"}
