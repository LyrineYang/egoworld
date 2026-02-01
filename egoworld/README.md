# egoworld

Offline, TB-scale egocentric video processing pipeline for Embodied AI.

Status
- The pipeline skeleton is implemented (manifests, Ray driver, backpressure, retry/dead-letter, Parquet writer).
- Model operators (SAM2/SAM3, Hamer, FoundationPose, DexRetarget, Fast3R) are currently stubs and require real checkpoints + inference wrappers.

Key capabilities (current)
- Scene detect with fallback to full clip.
- Clip-level scheduling with backpressure and retry policy.
- Atomic Parquet output with fixed compression/row-group/page-size.
- SQLite state store for Pending/Running/Done/Failed + dead-letter.

Project layout
- `src/egoworld/`: core pipeline code.
- `scripts/`: CLI entry points.
- `configs/`: example configs.
- `tests/`: basic validation tests.
- `runlog.md`, `progress.md`: engineering log and progress tracking.

Requirements
- Python 3.10+
- Ray
- PyTorch (GPU builds for your CUDA version)
- FFmpeg/ffprobe
- PyArrow (Parquet IO)
- scenedetect (scene detection)
- prometheus_client (metrics, optional if metrics disabled)

Environment setup (H100 server)
- Follow `docs/env-policy.md` and `docs/env-matrix.md` (scope: Rocky Linux 8.8 + CUDA 12.8).
- Base env must include: Python 3.10, Ray, PyTorch GPU build, pyarrow, scenedetect, ffmpeg.
- Metrics are optional; install `prometheus_client` only if metrics are enabled.
- If you need to build C++/CUDA extensions, install gcc/g++ + cmake + ninja + CUDA toolkit.
- When lock files exist, create envs from `egoworld/env/` and do not upgrade without updating locks.

Quickstart (skeleton)
1) Set Python path:
   `export PYTHONPATH=$PWD/src`
2) Build manifests:
   `python scripts/make_manifest.py make-manifest --config configs/example.json --input-dir ./data --output-dir ./manifests`
3) Run pipeline:
   `python scripts/run_pipeline.py run --config configs/example.json --video-manifest ./manifests/video_manifest.jsonl --clip-manifest ./manifests/clip_manifest.jsonl`

Configuration
- See `configs/example.json` for all supported fields.
- Backpressure controls: `backpressure.max_in_flight_*`
- Retry policy: `retry.max_retries`, `retry.base_delay_s`, `retry.backoff`
- Parquet params: `parquet.compression`, `parquet.row_group_size`, `parquet.data_page_size`
- Coordinate/time spec: `coordinates.*` (mask encoding, time base, coord frame, units)
- Operator toggles and params: `operators.<name>.enabled` + `operators.<name>.params`

Operators (matrix)
- `sam2`: segmentation/masks (GPU). Output: `masks.parquet`
- `hamer`: hand pose (CPU/GPU). Output: `hand_pose.parquet`
- `foundationpose`: object pose (CPU/GPU). Output: `object_pose.parquet`
- `dex_retarget`: hand-to-robot mapping (CPU/GPU). Output: `mapping.parquet`
- `fast3r`: multi-view 3D reconstruction (GPU). Output: `fast3r_pose.parquet` (if enabled)

Model checkpoints (recommended way to provide)
- Keep model code and weights outside git. Store weights under `./models/` (ignored by `.gitignore`).
- Provide paths or model names under `operators.<name>.params` (see example below).
- Record versions in `model_versions` for reproducibility.

Example (operator config + checkpoints)
```
"operators": {
  "sam2": { "enabled": true, "params": { "model_path": "./models/sam2" } },
  "hamer": { "enabled": false, "params": { "model_path": "./models/hamer" } },
  "foundationpose": { "enabled": false, "params": { "model_path": "./models/foundationpose" } },
  "dex_retarget": { "enabled": false, "params": { "model_path": "./models/dex_retarget" } },
  "fast3r": {
    "enabled": true,
    "params": {
      "model_name_or_path": "jedyang97/Fast3R_ViT_Large_512",
      "image_size": 512,
      "dtype": "bf16",
      "max_views": 8,
      "frame_sampling": { "method": "uniform", "num_frames": 8, "stride": 4 }
    }
  }
}
```

Manifests
- `video_manifest`: video-level metadata (duration, fps, size, checksum)
- `clip_manifest`: clip-level schedule and status
- Schema and field specs: `src/egoworld/manifests/schema.py`

Output layout
```
output/
  run_id=YYYYMMDD_HHMMSS/
    video_id=.../clip_id=.../
      masks.parquet
      hand_pose.parquet
      object_pose.parquet
      mapping.parquet
      fast3r_pose.parquet
      meta.json
```

Status tracking
- `progress.md`: milestones and next steps
- `runlog.md`: implementation log

Roadmap
- Integrate real model wrappers + checkpoints.
- Add QC thresholds and Prometheus exporter wiring.
- Add multi-GPU tuning and micro-batching.
- Expand schema with full 3D/6D pose fields.

Documentation
- `plan.md`: offline MVP implementation plan and milestones.
- `techContext.md`: tech stack and model inventory.
- `activeContext.md`: current focus and short-term tasks.
- `progress.md`: progress log and next steps.
- `runlog.md`: implementation log.
- `memory.md`: background notes and decisions.

License
- TBD
