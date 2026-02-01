# Run Log

## 2026-02-01 02:10:42
- Initialized source tree under `src/egoworld/` with config, manifests, pipeline, operators, io, observability, and utils.
- Implemented Ray pipeline driver skeleton with backpressure, retry policy, and dead-letter handling.
- Added manifest builder with SceneDetect fallback to full clip.
- Added Parquet writer with fixed parameters and atomic write behavior.
- Added basic tests for schema alignment, state store, and writer idempotency.

## 2026-02-01 14:25:21
- Added README with project overview, quickstart, configuration, and model checkpoint guidance.
- Added example config in `configs/example.json`.
- Added `.gitignore` for outputs, state, models, and data.

## 2026-02-01 14:41:30
- Fixed WriterActor Parquet config init to avoid AttributeError on first write.
- Enabled resumable runs by filtering clips to Pending/Failed/Writing states from SQLite.
- Made clip_id deterministic (hash of video_id + frame bounds) to preserve idempotency.
- Enforced time/frame alignment by recomputing start_s/end_s from frame indices.
- Applied SceneDetectConfig (method/min_scene_len/fallback) and overlap expansion.
- Aligned run_manifest fields with schema (config_path added, parquet/model versions serialized, created_at treated as string).

## 2026-02-01 14:56:08
- Included `Running` in resumable clip statuses to avoid stranding in-progress clips after crashes.

## 2026-02-01 22:33:25
- Added operator configuration layer (enabled flags + params) and integrated it into pipeline execution.
- Added Fast3R operator stub and optional output (`fast3r_pose.parquet`).
- Updated README with operator matrix and config reference; removed references to missing docs.
- Updated example config to expose operator settings and Fast3R parameters.
