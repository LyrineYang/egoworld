"""Small real-video GPU acceptance; no synthetic inference backend."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import sys

from egoworld.config import load_config, SceneDetectConfig
from egoworld.contracts import OUTPUT_SCHEMA_VERSION, STAGE_ARTIFACTS, enabled_stages
from egoworld.io.writers import write_json


def inspect_run(root: Path) -> dict:
    if not (root / "run_manifest.json").is_file():
        raise ValueError(f"No run_manifest.json in {root}")
    statuses = {"pass": 0, "review": 0, "skipped": 0}
    result = {"run_dir": str(root), "completed_clips": 0, "processed_frames": 0,
              "nonempty_instance_rows": 0, "quality": statuses}
    for path in sorted(root.glob("video_id=*/clip_id=*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
            raise ValueError(f"Old output schema; rerun this clip: {path.parent}")
        if not (path.parent / "qc.json").is_file() or not all((path.parent / f"{name}.parquet").is_file() for stage in meta["stages"] for name in STAGE_ARTIFACTS[stage]):
            raise ValueError(f"Incomplete clip output: {path.parent}")
        qc = json.loads((path.parent / "qc.json").read_text())
        result["completed_clips"] += int(meta["status"] == "Done")
        result["processed_frames"] += qc["processed_frames"]
        result["nonempty_instance_rows"] += qc["nonempty_instance_rows"]
        statuses[qc["status"]] += 1
    summary = root / "run_summary.json"
    if summary.is_file():
        result["run"] = json.loads(summary.read_text())
    return result


def prepare_smoke_config(cfg, stages=None, profile="default"):
    """Apply explicit smoke limits without changing the user's config file."""
    from egoworld.contracts import STAGE_DEPENDENCIES, GPU_STAGES
    if profile not in ("default", "16gb"):
        raise ValueError("Unknown smoke profile")
    if stages:
        selected = set(stages)
        if selected - STAGE_DEPENDENCIES.keys():
            raise ValueError("Unknown smoke stage")
        while True:
            expanded = selected | {dep for name in selected for dep in STAGE_DEPENDENCIES[name]}
            if expanded == selected: break
            selected = expanded
        for name in STAGE_DEPENDENCIES:
            getattr(cfg.operators, name).enabled = name in selected
    cfg.num_gpus = 1
    cfg.execution.gpu_stage_groups = []
    cfg.backpressure.max_in_flight_gpu = 1
    cfg.backpressure.max_in_flight_write = 1
    if profile == "16gb":
        cfg.execution.cpu_workers = cfg.execution.write_workers = cfg.execution.cpu_operator_workers = 1
        cfg.execution.max_active_clips = cfg.execution.max_cached_models = 1
        cfg.execution.keep_prepared = True  # retain logs for the first real trial
        cfg.backpressure.max_in_flight_cpu = 1
        cfg.retry.max_retries = 0
        cfg.operators.hamer.params["batch_size"] = 1
        cfg.operators.fast3r.params.update(max_views=2, dtype="bf16", attn_implementation="pytorch_auto")
        cfg.operators.sam2.params.setdefault("prompting", {})["max_boxes_per_frame"] = 3
        for stage in GPU_STAGES & set(enabled_stages(cfg)):
            getattr(cfg.operators, stage).params.setdefault("runner_command", [sys.executable, "-m", "egoworld.worker"])
    return cfg.resolved()


def single_gpu_smoke(config_path: str, input_video: str, output_dir: str,
                     seconds: float, prompt: str, stages: list[str] | None = None,
                     profile: str = "default") -> dict:
    from egoworld.manifests.build_manifest import build_manifests, write_manifest_json
    from egoworld.pipeline.driver import make_run_id, run_pipeline
    from egoworld.preflight import runtime_report

    if not 0 < seconds <= 15:
        raise ValueError("Acceptance sample duration must be in (0, 15] seconds")
    if profile == "16gb" and seconds > 3:
        raise ValueError("16gb smoke profile requires --seconds <= 3")
    video = Path(input_video).resolve()
    if not video.is_file():
        raise ValueError(f"Input video not found: {video}")
    cfg = prepare_smoke_config(load_config(config_path), stages, profile)
    if not enabled_stages(cfg):
        raise ValueError("Select at least one model stage for acceptance")
    root = Path(output_dir).resolve() / make_run_id()
    root.mkdir(parents=True)
    cfg.run_id = "single_gpu"
    cfg.paths.output_root = str(root / "output")
    cfg.paths.state_db_path = str(root / "state.db")
    params = cfg.operators.sam2.params
    params["device"] = "cuda"
    params["precision"] = "fp16"
    params["vos_optimized"] = False
    params.setdefault("prompting", {})["prompt_text"] = prompt
    params["prompting"]["gd_device"] = "cuda"
    config_file = root / "config.json"
    write_json(str(config_file), asdict(cfg))
    result = {"status": "failed", "profile": profile,
              "stages": list(enabled_stages(cfg)), "config_path": str(config_file),
              "validation_scope": "one GPU, one short video; not a scale benchmark or a 16GB memory guarantee",
              "report_path": str(root / "acceptance.json")}
    try:
        prerequisites = runtime_report(str(config_file), gpu=True, assets=True)
        result["environment"] = prerequisites
        if not prerequisites["ok"]:
            raise RuntimeError("GPU prerequisites failed; see acceptance.json")
        videos, clips = build_manifests([str(video)], scenedetect=SceneDetectConfig(
            method="none", overlap_s=0, max_clip_duration_s=seconds))
        clips = clips[:1]
        if not clips:
            raise RuntimeError("Video contains no usable frames")
        video_manifest, clip_manifest = root / "videos.jsonl", root / "clips.jsonl"
        write_manifest_json(str(video_manifest), videos)
        write_manifest_json(str(clip_manifest), clips)
        started = time.perf_counter()
        run_pipeline(str(config_file), str(video_manifest), str(clip_manifest))
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        run_root = root / "output/run_id=single_gpu"
        result["outputs"] = inspect_run(run_root)
        if cfg.operators.sam2.enabled and result["outputs"]["nonempty_instance_rows"] == 0:
            raise RuntimeError("No nonempty instances; use a video with visible targets or adjust --prompt")
        import pyarrow.parquet as pq

        outputs = list(run_root.glob("video_id=*/clip_id=*/instances.parquet"))
        for path in outputs:
            table = pq.ParquetFile(path).read()
            if not {"object_id", "label", "area_px", "mask_rle"}.issubset(table.column_names):
                raise RuntimeError("Instance output schema is incomplete")
        stage_counts = {}
        for stage in enabled_stages(cfg):
            rows = 0
            for artifact in STAGE_ARTIFACTS[stage]:
                for file in run_root.glob(f"video_id=*/clip_id=*/{artifact}.parquet"):
                    rows += pq.ParquetFile(file).metadata.num_rows
            stage_counts[stage] = rows
            if rows == 0:
                raise RuntimeError(f"Enabled stage {stage} produced no rows on the acceptance sample")
        result["stage_rows"] = stage_counts
        outputs = list(run_root.glob("video_id=*/clip_id=*/*.parquet"))
        mtimes = {path: path.stat().st_mtime_ns for path in outputs}
        run_pipeline(str(config_file), str(video_manifest), str(clip_manifest))
        if any(path.stat().st_mtime_ns != stamp for path, stamp in mtimes.items()):
            raise RuntimeError("Resume rewrote completed instance outputs")
        result.update(status="passed", resume_verified=True, sample_seconds=clips[0]["end_s"])
    except Exception as exc:
        result["error"] = str(exc)
        raise RuntimeError(f"{exc}; report: {root / 'acceptance.json'}") from exc
    finally:
        write_json(str(root / "acceptance.json"), result)
    return result
