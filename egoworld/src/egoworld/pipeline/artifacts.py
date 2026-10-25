"""Atomic stage commits, validated resume and final clip metadata."""
from __future__ import annotations
from pathlib import Path
import json
import os
import shutil
import time
import numpy as np
from egoworld.config import ParquetConfig
from egoworld.contracts import OUTPUT_SCHEMA_VERSION, STAGE_ARTIFACTS, schemas, validate_rows
from egoworld.io.writers import write_json, write_parquet_table
from egoworld.utils.hashing import sha256_file


def validate_arrays(name, arrays):
    for value in arrays.values():
        value = np.asarray(value)
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name}: arrays must be finite numeric data without objects")
    if name == "reconstruction-pointcloud.npz":
        if set(arrays) != {"points", "confidence", "view_index"}:
            raise ValueError("Point cloud requires points, confidence and view_index")
        points, confidence, views = (np.asarray(arrays[key]) for key in ("points", "confidence", "view_index"))
        if points.ndim != 2 or points.shape[1] != 3 or confidence.shape != (len(points),) or views.shape != (len(points),):
            raise ValueError("Point cloud arrays have incompatible shapes")
        if not np.issubdtype(views.dtype, np.integer) or np.any(views < 0):
            raise ValueError("Point cloud view indices must be nonnegative integers")


def load_stage(directory, stage, fingerprint, verify_hashes=True):
    import pyarrow.parquet as pq
    root = Path(directory)
    try:
        receipt = json.loads((root / f"stage_{stage}.json").read_text())
        if receipt["fingerprint"] != fingerprint or receipt["schema_version"] != OUTPUT_SCHEMA_VERSION:
            return None
        for name, info in receipt["files"].items():
            if Path(name).name != name:
                return None
            file = root / name
            if not file.is_file() or file.stat().st_size != info["bytes"]:
                return None
            if verify_hashes and sha256_file(str(file)) != info["sha256"]:
                return None
            if file.suffix == ".parquet":
                parquet = pq.ParquetFile(file)
                if parquet.metadata.num_rows != info["rows"] or not parquet.schema_arrow.equals(schemas()[file.stem]):
                    return None
        if not set(f"{name}.parquet" for name in STAGE_ARTIFACTS[stage]).issubset(receipt["files"]):
            return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


class WriterActor:
    def __init__(self, config):
        self.config = config
        self.parquet = ParquetConfig(**config["parquet"])
        import pyarrow as pa
        threads = config["execution"].get("worker_threads", 1)
        pa.set_cpu_count(threads)
        pa.set_io_thread_count(threads)

    def write_stage(self, job, packet):
        started = time.perf_counter()
        root, stage = Path(job["directory"]), job["stage"]
        root.mkdir(parents=True, exist_ok=True)
        receipt = {"stage": stage, "directory": str(root), "schema_version": OUTPUT_SCHEMA_VERSION,
                   "fingerprint": job["fingerprint"], "stats": dict(packet.get("stats", {})), "files": {}}
        if set(packet["tables"]) != set(STAGE_ARTIFACTS[stage]):
            raise ValueError(f"{stage}: output artifact contract mismatch")
        array_names = set(packet.get("arrays", {})) | set(packet.get("array_files", {}))
        for row in packet["tables"].get("reconstruction", []):
            if row["pointcloud_path"] not in array_names:
                raise ValueError("Reconstruction references a missing point cloud")
        for name, rows in packet["tables"].items():
            validate_rows(name, rows, job["clip"])
            file = root / f"{name}.parquet"
            write_parquet_table(str(file), rows, schema=schemas()[name], parquet=self.parquet)
            receipt["files"][file.name] = {"rows": len(rows)}
        for name, arrays in packet.get("arrays", {}).items():
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("Array artifact must be a plain .npz filename")
            validate_arrays(name, arrays)
            with (root / f"{name}.tmp").open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(root / f"{name}.tmp", root / name)
            receipt["files"][name] = {}
        for name, source in packet.get("array_files", {}).items():
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("Invalid array artifact filename")
            with np.load(source, allow_pickle=False) as content:
                validate_arrays(name, {key: content[key] for key in content.files})
            shutil.copyfile(source, root / f"{name}.tmp")
            os.replace(root / f"{name}.tmp", root / name)
            receipt["files"][name] = {}
        for name, info in receipt["files"].items():
            file = root / name
            info.update(bytes=file.stat().st_size, sha256=sha256_file(str(file)))
        receipt["stats"]["write_seconds"] = time.perf_counter() - started
        write_json(str(root / f"stage_{stage}.json"), receipt)
        return receipt

    def finalize(self, job, receipts, errors):
        import pyarrow.parquet as pq
        root = Path(job["directory"]); root.mkdir(parents=True, exist_ok=True)
        sam = receipts.get("sam2", {}).get("stats", {})
        expected = job["clip"]["frame_end"] - job["clip"]["frame_start"]
        processed = sam.get("processed_frames", 0)
        nonempty = 0
        object_ids = set()
        if "sam2" in receipts:
            data = pq.ParquetFile(root / "instances.parquet").read(columns=["area_px", "object_id"]).to_pylist()
            nonempty = sum(row["area_px"] > 0 for row in data)
            object_ids = {row["object_id"] for row in data}
        review = bool(errors) or ("sam2" in receipts and (nonempty == 0 or processed < expected or sam.get("empty_mask_rate", 1) > self.config["metrics"]["empty_mask_rate_max"]))
        empty_stages = [stage for stage, receipt in receipts.items()
                        if not any(f.get("rows", 0) for f in receipt["files"].values())]
        qc = {"status": "review" if review or empty_stages else "pass" if receipts else "skipped",
              "empty_mask_rate": sam.get("empty_mask_rate"), "processed_frames": processed,
              "expected_frames": expected, "frame_coverage": processed / max(1, expected),
              "nonempty_instance_rows": nonempty, "object_count": len(object_ids), "empty_stages": empty_stages}
        write_json(str(root / "qc.json"), qc)
        meta = {"output_schema_version": OUTPUT_SCHEMA_VERSION, "fingerprint": job["fingerprint"],
                "clip": job["clip"], "stages": list(receipts), "errors": errors,
                "status": "Failed" if errors else "Done", "quality": qc}
        write_json(str(root / "meta.json"), meta)
        return meta
