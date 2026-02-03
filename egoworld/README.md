# egoworld

Offline, TB-scale egocentric video processing pipeline for Embodied AI.

## Status
- The pipeline skeleton is implemented (manifests, Ray driver, backpressure, retry/dead-letter, Parquet writer).
- SAM2 + GroundingDINO integration is implemented (requires installed packages + local checkpoints).
- Default segmentation model: SAM2.1 small.
- Other model operators (Hamer, FoundationPose, DexRetarget, Fast3R) remain stubs and require real checkpoints + inference wrappers.

## Key capabilities (current)
- Scene detect with fallback to full clip.
- Clip-level scheduling with backpressure and retry policy.
- Atomic Parquet output with fixed compression/row-group/page-size.
- SQLite state store for Pending/Running/Done/Failed + dead-letter.

## Project layout
- `src/egoworld/`: core pipeline code.
- `scripts/`: CLI entry points.
- `configs/`: example configs.
- `tests/`: basic validation tests.
- `runlog.md`, `progress.md`: engineering log and progress tracking.

## Requirements
- Python 3.10+
- Ray
- PyTorch (GPU builds for your CUDA version)
- FFmpeg/ffprobe
- PyArrow (Parquet IO)
- scenedetect (scene detection)
- groundingdino (prompt generation for SAM2)
- opencv-python (video frame sampling)
- pycocotools (RLE encoding, optional fallback exists)
- prometheus_client (metrics, optional if metrics disabled)

## Environment setup (H100 server)
- Follow `docs/env-policy.md` and `docs/env-matrix.md` (scope: Rocky Linux 8.8 + CUDA 12.8).
- Base env must include: Python 3.10, Ray, PyTorch GPU build, pyarrow, scenedetect, ffmpeg, SAM2 + GroundingDINO.
- Metrics are optional; install `prometheus_client` only if metrics are enabled.
- If you need to build C++/CUDA extensions, install gcc/g++ + cmake + ninja + CUDA toolkit.
- When lock files exist, create envs from `egoworld/env/` and do not upgrade without updating locks.
- See `egoworld/env/README.md` for base env + SAM2 env templates.
- One-command setup (recommended):
  ```bash
  bash egoworld/scripts/setup_env.sh --weights --smoke
  ```

## Data placement
- Put input videos under a directory such as `./data/` (any readable path works).
- Supported inputs are whatever `ffprobe` + OpenCV can read (e.g., mp4, mov). If `ffprobe` fails, manifest build will fail.
- Use `--glob` in `make-manifest` to filter file types (default: `**/*.mp4`).

## Model checkpoints & paths
- Use `scripts/download_models.sh` to download official weights into `./models/`.
- Set paths under `operators.<name>.params` in your config.
- SAM2 requires both checkpoint + config; GroundingDINO requires config + checkpoint.
- If SAM2 config path is missing locally, it may resolve from the installed `sam2` package.

## Before you run
- Ensure at least one video file exists under `./data/` (or your chosen input dir).
- Ensure `ffmpeg`/`ffprobe` are on PATH.
- Ensure SAM2 + GroundingDINO packages are installed in the active env.

## Minimal runnable (SAM2 pipeline)
1) Set Python path:
   ```bash
   export PYTHONPATH=$PWD/src
   ```
2) Download model weights:
   ```bash
   bash scripts/download_models.sh
   ```
3) Build manifests + run pipeline:
   ```bash
   python scripts/make_manifest.py make-manifest --config configs/example.json --input-dir ./data --output-dir ./manifests
   python scripts/run_pipeline.py run --config configs/example.json --video-manifest ./manifests/video_manifest.jsonl --clip-manifest ./manifests/clip_manifest.jsonl
   ```
Advanced operational details (resume/ops tuning) are kept in internal docs.

## Configuration
- See `configs/example.json` for all supported fields.
- Backpressure controls: `backpressure.max_in_flight_*`
- Retry policy: `retry.max_retries`, `retry.base_delay_s`, `retry.backoff`
- Parquet params: `parquet.compression`, `parquet.row_group_size`, `parquet.data_page_size`
- Coordinate/time spec: `coordinates.*` (mask encoding, time base, coord frame, units)
- Operator toggles and params: `operators.<name>.enabled` + `operators.<name>.params`
  - SAM2 config path may be resolved from the installed `sam2` package if the local file is missing.

## Operators (matrix)
- `sam2`: segmentation/masks (GPU). Output: `masks.parquet`
- `hamer`: hand pose (CPU/GPU). Output: `hand_pose.parquet`
- `foundationpose`: object pose (CPU/GPU). Output: `object_pose.parquet`
- `dex_retarget`: hand-to-robot mapping (CPU/GPU). Output: `mapping.parquet`
- `fast3r`: multi-view 3D reconstruction (GPU). Output: `fast3r_pose.parquet` (if enabled)

## SAM2 prompt policy (default in example config)
- Goal: hand segmentation with object context for egocentric kitchen-style datasets.
- Strategy: low-frequency prompts + video propagation.
- `prompt_interval_s`: 2.0 (keyframe prompts every ~2 seconds).
- `max_prompts_per_clip`: 60 (bounds work on long clips).
- `prompt_text`: hands + common handheld kitchen objects (override as needed).
- Thresholds: `box_threshold=0.35`, `text_threshold=0.25`, `nms_iou=0.5`.

## Model checkpoints (recommended way to provide)
- Keep model code and weights outside git. Store weights under `./models/` (ignored by `.gitignore`).
- Provide paths or model names under `operators.<name>.params` (see example below).
- Record versions in `model_versions` for reproducibility.

## Example (operator config + checkpoints)
```json
"operators": {
  "sam2": {
    "enabled": true,
    "params": {
      "checkpoint": "./models/sam2/sam2.1_hiera_small.pt",
      "config": "./models/sam2/sam2.1_hiera_s.yaml",
      "device": "cuda",
      "precision": "bf16",
      "vos_optimized": true,
      "prompting": {
        "source": "groundingdino",
        "prompt_interval_s": 2.0,
        "max_prompts_per_clip": 60,
        "max_boxes_per_frame": 6,
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "nms_iou": 0.5,
        "min_box_area": 256,
        "gd_config": "./models/groundingdino/GroundingDINO_SwinT_OGC.py",
        "gd_checkpoint": "./models/groundingdino/groundingdino_swint_ogc.pth",
        "gd_device": "cuda",
        "prompt_text": "hand . left hand . right hand . person hand . glove . utensil . knife . spoon . fork . spatula . ladle . tongs . cup . mug . bottle . bowl . plate . pan . pot . lid . cutting board . food . container . jar . can . package . bag . towel . sponge . soap . faucet . sink . stove . microwave . refrigerator . drawer . cabinet . phone . remote . key . pen . scissors"
      }
    }
  },
  "hamer": { "enabled": false, "params": { "model_path": "./models/hamer" } },
  "foundationpose": { "enabled": false, "params": { "model_path": "./models/foundationpose" } },
  "dex_retarget": { "enabled": false, "params": { "model_path": "./models/dex_retarget" } },
  "fast3r": {
    "enabled": false,
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

## Manifests
- `video_manifest`: video-level metadata (duration, fps, size, checksum)
- `clip_manifest`: clip-level schedule and status
- Schema and field specs: `src/egoworld/manifests/schema.py`

## Output layout
```text
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

## How to inspect outputs
- `meta.json`: clip metadata + field specs + time/mask encoding.
- `masks.parquet`: SAM2 masks (RLE, one row per frame).
- `hand_pose.parquet`, `object_pose.parquet`, `mapping.parquet`: stubs unless those models are implemented.
- `fast3r_pose.parquet`: only written when Fast3R is enabled.

## Tests
- Base unit tests: `pytest -q`
- Environment smoke (GPU/ffmpeg/paths): `EGOWORLD_ENV_SMOKE=1 pytest -q egoworld/tests/test_env_smoke.py`
- SAM2 runtime smoke (real weights): `EGOWORLD_SAM2_SMOKE=1 pytest -q egoworld/tests/test_sam2_integration.py`
- Pipeline smoke (end-to-end + SAM2): `EGOWORLD_PIPELINE_SMOKE=1 pytest -q egoworld/tests/test_pipeline_smoke.py`

## Status tracking
- `progress.md`: milestones and next steps
- `runlog.md`: implementation log

## Roadmap
- Integrate real model wrappers + checkpoints.
- Add QC thresholds and Prometheus exporter wiring.
- Add multi-GPU tuning and micro-batching.
- Expand schema with full 3D/6D pose fields.

## What is NOT implemented yet
- Hamer/FoundationPose/DexRetarget/Fast3R real inference (currently stubs).
- QC metrics thresholds + Prometheus exporter wiring (metrics are no-op unless wired).
- Multi-machine execution, Kafka/Redis middleware, exactly-once semantics.
- NCCL/DCGM tooling or GPU utilization alerts (policy exists but not enforced in code).

## Documentation
- `docs/README.md`: documentation map and ownership rules.
- `plan.md`: offline MVP implementation plan and milestones.
- `techContext.md`: tech stack and model inventory.
- `activeContext.md`: current focus and short-term tasks.
- `progress.md`: progress log and next steps.
- `runlog.md`: implementation log.
- `memory.md`: background notes and decisions.

## License
- TBD
