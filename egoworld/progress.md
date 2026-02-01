# Progress

## 2026-02-01
- Added project skeleton under `src/egoworld/` covering config, manifests, pipeline, operators, io, observability, and utils.
- Implemented manifest builder with SceneDetect fallback and time/frame alignment.
- Implemented Ray driver with backpressure limits, retry policy, and dead-letter logging.
- Implemented Parquet writer with fixed compression/row-group/page-size parameters.
- Added basic tests for schema alignment, state store, and writer idempotency.
- Added README and example config for user-facing setup guidance.
- Fixed WriterActor Parquet init and enabled resume by skipping Done clips.
- Made clip_id deterministic and enforced time/frame alignment in manifests.
- Applied SceneDetectConfig settings (method/min_scene_len/fallback/overlap).
- Aligned run_manifest fields with schema expectations.
- Ensured resumable runs also include clips left in Running state.
- Added operator-level configuration (enabled flags + params) and Fast3R stub wiring.
- Expanded README and example config to document operator usage and parameters.

## Next
- Integrate real model checkpoints and inference wrappers.
- Validate end-to-end run on a small dataset and tune backpressure parameters.
- Extend QC metrics with thresholds and integrate Prometheus exporter.
