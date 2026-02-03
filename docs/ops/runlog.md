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

## 2026-02-01 22:51:25
- Ensured fast3r_pose.parquet is always written when Fast3R is enabled, even if no poses are returned.

## 2026-02-01 23:22:04
- Chose SAM2.1 small as default for 1000h-scale hand segmentation on 4×H100: prioritizes throughput while preserving strong quality; larger variants reserved for hard clips.
- Defined prompt strategy for egocentric datasets (EgoProceL/EPIC/HD-EPIC/EgoDex/EgoMimic/Ego4D): low-frequency prompts + video propagation to reduce compute while keeping temporal consistency.
- Set prompt interval to 2s with max 60 prompts/clip to bound long clips; limits per-frame boxes (6) to reduce GPU load and false positives.
- Selected prompt text list centered on hands plus common handheld kitchen objects to maximize recall on egocentric manipulation without exploding open-vocab search space.
- Threshold defaults (box 0.35/text 0.25/NMS 0.5/min area 256) chosen to balance recall for small hands/utensils vs. noise; intended to be tuned on a small validation subset.
- Resource settings: bf16 + vos_optimized enabled to exploit H100 throughput and reduce memory pressure.
- Updated example config + README to expose these operator parameters to users.

## 2026-02-01 23:25:32
- Wired SAM2 operator params through the pipeline so prompt strategy and checkpoint settings are passed into the operator stub.

## 2026-02-02 09:45:25
- Implemented SAM2 video segmentation with GroundingDINO prompt generation; params now control checkpoints, prompting, and optimization flags.
- Added GroundingDINO operator wrapper (text-conditioned detection) and mask RLE encoding utility.
- Added video sampling utilities for prompt frames and integrated prompt-driven propagation output (union masks per frame).
- Updated README and example config to expose GroundingDINO config/checkpoint parameters and required deps.

## 2026-02-02 13:09:08
- Added `scripts/download_models.sh` to pull official weights into `./models/` using stable vendor URLs and repository scripts.
- Defaulted SAM2 config to SAM2.1 small (`sam2.1_hiera_small.pt` + `sam2.1_hiera_s.yaml`) to match throughput goals.
- Wired GroundingDINO config/weights download to match prompt-generation defaults.
- Added Fast3R (HF), FoundationPose (GDrive folder), HaMeR (official repo script) download steps; noted DexRetargeting has no weights.

## 2026-02-02 14:43:06
- Fixed download_models.sh dependency probe to not exit under set -e, so missing modules trigger install as intended.
- Fixed SAM2 config fallback warning line continuation so failure yields a warning instead of aborting.

## 2026-02-03 10:12:00
- Added environment policy and compatibility matrix docs (`docs/env/policy.md`, `docs/env/matrix.md`) for the H100 offline server baseline.
- Added conda env templates under `egoworld/env/` and a helper installer script (`egoworld/scripts/setup_env.sh`).
- Reformatted `README.md` for readability without changing content.

## 2026-02-03 10:35:00
- Unified documentation to state SAM2.1 small as the default segmentation model.
- Updated environment docs to reference SAM2.1 small (Base + model env).
- Added a documentation map and ownership rules to reduce duplication and improve maintenance.

## 2026-02-03 10:48:00
- Updated plan diagram to reflect SAM2.1 small as the segmentation model.

## 2026-02-03 11:18:00
- Moved primary documentation to `README.md` and kept `docs/` as the structured reference.
- Organized docs into `docs/env`, `docs/ops`, and `docs/design` and updated references accordingly.

## 2026-02-03 11:35:00
- Moved the main README content to the repository root and updated docs references.

## 2026-02-03 12:05:00
- Added `docs/ops/active-context.md` and updated doc references to keep a single source of truth for current focus.

## 2026-02-03 12:28:00
- Removed stale `egoworld/activeContext.md` to avoid conflicting active-context sources.
