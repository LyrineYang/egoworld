# egoworld

Offline, TB-scale egocentric video processing pipeline for Embodied AI.

Status
- The pipeline skeleton is implemented (manifests, Ray driver, backpressure, retry/dead-letter, Parquet writer).
- Model operators (SAM2/SAM3, Hamer, FoundationPose, DexRetarget) are currently stubs and require real checkpoints + inference wrappers.

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
- Optional: pyarrow, scenedetect, prometheus_client

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

Model checkpoints (recommended way to provide)
- Keep model code and weights outside git. Store weights under `./models/` (ignored by `.gitignore`).
- Provide paths in config via `extra.model_paths`:
  - `sam2`, `hamer`, `foundationpose`, `dex_retarget`
- Record versions in `model_versions` for reproducibility.

Example (local checkpoint paths)
```
"extra": {
  "model_paths": {
    "sam2": "./models/sam2",
    "hamer": "./models/hamer",
    "foundationpose": "./models/foundationpose",
    "dex_retarget": "./models/dex_retarget"
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

License
- TBD
