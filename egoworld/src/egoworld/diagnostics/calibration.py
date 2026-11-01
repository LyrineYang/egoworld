"""Explicit, inspectable scheduling calibration from retained successful probes."""
from pathlib import Path
import json
import math
import statistics
from egoworld.diagnostics.inputs import digest
from egoworld.diagnostics.observations import source_identity
from egoworld.io.writers import write_json
from egoworld.utils.hashing import sha256_file


def parameter_hash(config, stage):
    params = dict(config["operators"][stage]["params"])
    params.pop("debug_dir", None)
    return digest({"split_directory": config["execution"].get("split_directory") if config["execution"].get("split_execution") and stage in config["execution"].get("split_stages", [stage]) else None, "params": params, "version": config.get("model_versions", {}).get(stage),
                   "split_execution": config["execution"].get("split_execution", False) and stage in config["execution"].get("split_stages", [stage])})


def work_units(stage, clip, params, inputs=None):
    frames = clip["frame_end"] - clip["frame_start"]
    if stage == "fast3r":
        return min(frames, int(params.get("max_views", 8)))
    if stage in ("hamer", "foundationpose"):
        rows = (inputs or {}).get("sam2", {}).get("files", {}).get("instances.parquet", {}).get("rows", frames)
        return max(1, rows / int(params.get("frame_stride", 5))) if stage == "hamer" else max(1, rows)
    return frames


def build_calibration(reports, output):
    entries, sources = {}, []
    source = source_identity()
    for report_path in reports:
        path = Path(report_path).resolve()
        report = json.loads(path.read_text())
        if report["status"] != "passed" or not report["recommendations"]["warm_model_reuse_verified"]:
            raise ValueError("Calibration requires passed cold/hot probes with verified model reuse")
        if report["model_identity"]["source_tree_hash"] != source:
            raise ValueError("Probe source identity differs from current code; re-probe explicitly")
        if report["recommendations"].get("non_runner_compute_pids_observed"):
            raise ValueError("Contended probes cannot calibrate isolated service costs")
        stage = report["stage"]
        config = json.loads((path.parent / "config.supplied.json").read_text())
        hot = [t for t in report["trials"] if t["phase"] == "hot"]
        cold = next(t for t in report["trials"] if t["phase"] == "cold")
        units = hot[0]["stats"].get("work_units")
        if units is None:
            table = {"hamer": "hand_pose", "foundationpose": "object_pose", "fast3r": "reconstruction"}.get(stage)
            units = hot[0]["files"].get(f"{table}.parquet", {}).get("rows") if table else report["input_specification"]["frame_count"]
        if not units or any(t["status"] != "passed" for t in hot):
            raise ValueError("Calibration requires nonempty measured work units")
        def service(trial):
            diag = trial["diagnostics"]
            return diag.get("gpu_step_wall_seconds", diag["synchronized_stage_wall_seconds"])
        samples = [service(t) / units for t in hot]
        identity = report["model_identity"]
        assets = {info.get("path", key): info["sha256"] for key, info in identity["local_assets"].items() if info.get("sha256")}
        assets.update(identity.get("implicit_assets", {}))
        entry = {"parameter_hash": parameter_hash(config, stage), "sample_units": units,
                 "input_specification": report["input_specification"], "assets": assets,
                 "load_seconds": max(0., cold["stats"].get("model_load_seconds", service(cold) - statistics.median(map(service, hot)))),
                 "service_seconds_per_unit": statistics.median(samples),
                 "service_samples_per_unit": samples,
                 "result_bytes": max(t["stats"]["result_bytes"] for t in hot),
                 "observed_identity_hash": report["observed_identity_hash"],
                 "device_uuid": cold.get("diagnostics", {}).get("device_uuid")}
        if stage in entries:
            raise ValueError("Supply exactly one probe report per stage; do not silently replace earlier samples")
        entries[stage] = entry
        sources.append({"path": str(path), "sha256": sha256_file(str(path))})
    result = {"schema_version": 1, "source_tree_hash": source, "stages": entries, "reports": sources,
              "scope": "Measured device/sample shapes; linear work-unit estimate, not a capacity or accuracy guarantee",
              "configuration_modified": False}
    result["calibration_hash"] = digest(result)
    write_json(str(output), result)
    return result


class CostPolicy:
    def __init__(self, config):
        self.config = config
        self.profile = json.loads(Path(config["execution"]["scheduling_profile"]).read_text())
        profile = self.profile
        if profile.get("schema_version") != 1 or profile.get("source_tree_hash") != source_identity():
            raise ValueError("Scheduling calibration source/version mismatch")
        if digest({k: v for k, v in profile.items() if k != "calibration_hash"}) != profile["calibration_hash"]:
            raise ValueError("Scheduling calibration hash mismatch")
        checked = {}
        for stage, op in config["operators"].items():
            if not op["enabled"] or stage == "dex_retarget":
                continue
            entry = profile["stages"].get(stage)
            if entry is None or entry["parameter_hash"] != parameter_hash(config, stage):
                raise ValueError(f"Missing or incompatible calibration for {stage}")
            for key in ("load_seconds", "service_seconds_per_unit", "result_bytes", "sample_units"):
                if not isinstance(entry.get(key), (int, float)) or not math.isfinite(entry[key]) or entry[key] < 0:
                    raise ValueError("Invalid scheduling cost")
            for path, expected in entry["assets"].items():
                if path not in checked:
                    checked[path] = sha256_file(path)
                if checked[path] != expected:
                    raise ValueError(f"Scheduling asset identity changed: {path}")
        self.terms = set(config["execution"]["cost_terms"])

    def choose(self, jobs, backend, now):
        candidates = []
        for job in jobs:
            stage = job["stage"]
            entry = self.profile["stages"].get(stage)
            if entry is None:
                continue
            units = job.get("_work_units", work_units(stage, job["clip"], self.config["operators"][stage]["params"]))
            for slot_id in backend.available("compute", stage):
                slot = backend.slots[slot_id]
                load = entry["load_seconds"] if slot["last_stage"] != stage else 0.
                service = entry["service_seconds_per_unit"] * units
                release = job.get("_release_bytes", 0)
                # Estimated bytes freed at this clip's last model stage. Cap its
                # priority bonus so large clips cannot overwhelm waiting fairness.
                bonus = min(1., release / max(1, self.config["execution"]["split_stage_bytes"]))
                cost = (load if "load" in self.terms else 0.) + (service if "service" in self.terms else 0.)
                cost /= 1. + (bonus if "release" in self.terms else 0.)
                wait = now - job["_ready_since"]
                candidates.append({"clip_id": job["clip"]["clip_id"], "stage": stage, "slot": slot_id,
                    "units": units, "load_seconds": load, "service_seconds": service, "release_bytes": release,
                    "cost": cost, "wait_seconds": wait, "urgent": wait >= self.config["execution"]["max_stage_wait_s"]})
        if not candidates:
            raise ValueError("No calibrated feasible GPU candidates")
        picked = min(candidates, key=lambda c: (not c["urgent"], -c["wait_seconds"] if c["urgent"] else c["cost"], -c["wait_seconds"], c["slot"]))
        job = next(j for j in jobs if j["clip"]["clip_id"] == picked["clip_id"] and j["stage"] == picked["stage"])
        job["_preferred_slot"] = picked["slot"]
        return job, {"calibration_hash": self.profile["calibration_hash"], "terms": sorted(self.terms),
                     "reason": "max_wait" if picked["urgent"] else "minimum_estimated_cost", "chosen": picked, "candidates": candidates}
