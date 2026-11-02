"""Acceptance on frozen local reproduction fixtures (real models; no training).

Run only after hands_split/world_split cold cases in the supplied root. Large
artifacts stay beneath that root. This is acceptance evidence, not a broad
throughput comparison with Data-Juicer or independent pose ground truth.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from egoworld.product.recipes import run_recipe
from egoworld.product.benchmark import compare_runs
from egoworld.product.store import inspect_cache
from egoworld.utils.hashing import register_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {"version": 1, "status": "running", "cases": {}, "scope": "Fixed short fixtures; functional acceptance, not general throughput or ground-truth accuracy"}
    output = root / "acceptance.json"
    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")
    def run(label, recipe):
        file = root / f"{label}.json"
        file.write_text(json.dumps(recipe, indent=2) + "\n")
        print("Running", label, flush=True)
        start = time.perf_counter()
        value = run_recipe(file, run_id=label)
        result["cases"][label] = {"status": value["status"], "elapsed_seconds": time.perf_counter() - start,
            "manifest": str(Path(value["run_dir"]) / "data_manifest.json"),
            "computed": value["computed"], "cache_hits": value["cache_hits"],
            "derived_cache_hits": value.get("derived_cache_hits", []), "filtered": value.get("filtered", []),
            "timings": value["timings"], "hashing": value["hashing"]}
        save()
        print(label, value["status"], "computed", value["computed"], flush=True)
        return value
    try:
        assets = set()
        for name in ("hands_split", "world_split"):
            folder = root / f"runs/run_id={name}"
            data = json.loads((folder / "data_manifest.json").read_text())
            assert data["status"] == "passed", name
            manifest = json.loads(next((folder / "model-output").glob("run_id=*/run_manifest.json")).read_text())
            for key, item in manifest["provenance"]["local_assets"].items():
                file = Path(item.get("path", key))
                if file.is_file():
                    assets.add(str(file))
        registry = root / "trusted-model-assets.json"
        result["registry"] = register_assets(sorted(assets), registry)
        save()
        hands = json.loads((root / "chicken-hands.json").read_text())
        world = json.loads((root / "rgbd-world.json").read_text())
        for recipe in (hands, world):
            recipe["cache"].update(verification="registered", asset_registry=str(registry))
        warm_hands = run("hands_registered", hands)
        assert warm_hands["status"] == "passed" and not warm_hands["computed"] and len(warm_hands["cache_hits"]) == 3
        warm_world = run("world_registered", world)
        assert warm_world["status"] == "passed" and not warm_world["computed"]
        assert "world" in {r["kernel"] for r in warm_world["derived_cache_hits"]}
        # Reporting-only recipe changes may rebuild a projection once while
        # leaving the model and measured-world identities reusable.
        warm_world = run("world_registered_repeat", world)
        assert {r["kernel"] for r in warm_world["derived_cache_hits"]} == {"projection", "world"}
        result["comparison"] = compare_runs([str(root / "runs/run_id=world_split"), warm_world["run_dir"]])
        assert result["comparison"]["status"] == "matched"
        # Corrupt/blank/duplicate input rejection must preserve the good clip.
        duplicate = root / "duplicate.mp4"; shutil.copyfile(hands["inputs"][0], duplicate)
        corrupt = root / "corrupt.mp4"; corrupt.write_bytes(b"invalid media fixture")
        blank = root / "blank.mp4"
        writer = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
        for _ in range(8): writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        mix = deepcopy(hands); mix["inputs"] += [str(duplicate), str(corrupt), str(blank)]
        value = run("mixed_quality", mix)
        assert value["status"] == "passed" and value["cleaning"]["accepted"] == 1 and value["cleaning"]["rejected"] == 3 and not value["computed"]
        bad = deepcopy(hands); bad["inputs"] = [str(corrupt), str(blank)]
        value = run("all_bad", bad)
        assert value["status"] == "rejected" and not value["computed"]
        gate = deepcopy(hands); gate["outputs"] = ["boxes", "hand_joints", "robot_joints"]
        gate["branch_cleaning"] = {"min_hand_area_px": 10**9}
        value = run("filtered_branches", gate)
        assert value["status"] == "partial" and not value["computed"] and len(value["filtered"]) == 2
        assert next(iter(value["exports"].values()))["boxes"]["rows"] > 0
        # Prove cached GPU work + changed robot can run on a Ray cluster with 0 GPU.
        left = deepcopy(hands); left["outputs"] = ["robot_joints"]
        left["operators"]["dex_retarget"].update(
            config=str(Path("external/dex-retargeting/src/dex_retargeting/configs/teleop/allegro_hand_left.yml").resolve()),
            handedness="left", robot_id="allegro_left")
        import ray
        ray.init(num_gpus=0, include_dashboard=False)
        try:
            value = run("left_cpu_only", left)
        finally:
            ray.shutdown()
        assert value["status"] == "passed" and [r["kernel"] for r in value["computed"]] == ["dex_retarget"]
        assert value["required_compute_stages"] == ["dex_retarget"]
        for name, side in (("hands_split", "right"), ("left_cpu_only", "left")):
            folder = root / f"runs/run_id={name}/model-output"
            mapping = next(folder.glob("run_id=*/video_id=*/clip_id=*/mapping.parquet"))
            subprocess.run([str(Path(".venv/dex/bin/python").absolute()), "egoworld/scripts/verify_dex_output.py",
                "--mapping", str(mapping), "--urdf", f"external/dex-retargeting/assets/robots/hands/allegro_hand/allegro_hand_{side}.urdf",
                "--output", str(root / f"fk-{side}.json")], check=True)
        result["cache_preview"] = inspect_cache(hands["cache"]["root"], max_bytes=0)
        result["scratch_files_remaining"] = {name: len(list((root / name).rglob("arrays.dat"))) + len(list((root / name).rglob("*.dat"))) for name in ("phase-spool", "input-cache")}
        assert not any(result["scratch_files_remaining"].values())
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
