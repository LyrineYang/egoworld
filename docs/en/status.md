# Implementation and evidence ledger

Reviewed **2026-11-02** for the combined implementation through `248b0c2`, including the replay/GPU-probe branch through `3e2aa5d`. The previously separate cleaning/cache and execution changes are integrated and committed. The historical readiness snapshot does not describe an outstanding uncommitted merge.

The central target is more complete and efficient embodied-data cleaning and processing than Data-Juicer. Core results must compare identical inputs, model behavior, output requirements, and cleaning rules. An advantage has not yet been established.

| Capability | Integrated behavior | Boundary |
| --- | --- | --- |
| Five annotation stages | SAM2/DINO, HaMeR, FoundationPose, DexRetarget, Fast3R | Weights, compatible runtimes and task-specific inputs required |
| Output recipes | 13 selectable types, dependency completion, field selection | Fixed registry, not an arbitrary plugin/DAG framework |
| Video screening | Decode/metadata, exact duplicates, blank input, brightness/blur/motion statistics and optional thresholds | No semantic near-duplicate detector or learned quality guarantee |
| Branch admission | Hand/pose/retarget gates before model load, explicit rejection reasons | Branch-specific usefulness; a missing hand need not discard useful scene data |
| Verified reuse | Stage identities, input preparation reuse, derived world/export caches, inspect/GC and opt-in asset registration | Strict byte checking by default; trusted registry has an explicit trust boundary |
| Execution and recovery | Bounded admission, persistent runners, input reuse, optional CPU/GPU phase splitting, cost scheduling, checked commits | Input/spool bounds are not a whole-machine RAM cap |
| Metric world exports | Calibrated cameras, points and objects; hand calibration interface | Metric evidence required; no automatic metric hands or cross-clip global map |
| Data-Juicer comparison | Real CPU artifact-cleaning pilot and an optional common-runner stage Mapper | Full model/recipe performance and independent quality comparison remain pending |

The catalog contains `masks`, `boxes`, `tracks`, `hand_joints`, `hand_mano`, `object_poses`, `robot_joints`, `camera_poses`, `scene_points`, `world_camera`, `world_scene`, `world_object`, and `world_hand`. Catalog membership does not imply that an arbitrary input can produce every type.

## Acceptance evidence

The integrated regression recorded **297 passed, 11 deselected**, including real Ray and the CPU-pilot acceptance gates. Environment/model smoke markers are excluded from that count; native runs are recorded separately. Production Python source SHA-256 is `0b05f2ca23124447a18a0d3265f825def23d9b447be55c7f652f9030bb818dbc`.

- A real SAM2/DINO → HaMeR → Dex sample produced 12 hand rows (left and right) and 6 right-hand robot rows. All compared numerical fields matched the earlier reproduction sample exactly. A separate CPU-only remaining-Dex run produced 6 left-hand robot rows; left/right strict URDF limits and Pinocchio FK passed.
- A real RGB-D/mesh calibrated sample produced 3 metric cameras, 3 object poses and 5,459 filtered world points. Its held-out inlier RMSE was about 1.71 cm at a 3 cm gate; this is not full-map or trajectory ground truth.
- A combined SAM/HaMeR/Fast3R/Dex run recomputed all model stages and observed one actual shared-frame cache hit. Full result-cache hits, branch rejection, mixed bad-input screening and derived-world reuse were checked separately.
- Final-source cache runs validated committed files and contracts; temporary input/spool arrays and live model workers were absent at completion.

These are short-sample engineering checks on an 8 GB RTX 4060 Laptop, not an annotation-quality or large-scale throughput benchmark. Unseeded Fast3R view embeddings can change a reconstruction; point-count differences are not optimization benefits. See the [implementation ledger](../efficiency-implementation-20261102.md), [integration acceptance](../integration-readiness.md), and [comparison next steps](../comparison-next.md).

## Comparison evidence and remaining claims

The [CPU pilot](../equal-work-benchmark.md) used real Data-Juicer DefaultExecutor and an experimental streaming reference. Five paired repeats of 80 native hand rows matched exactly; a separate four-fault fixture was rejected equally. It excludes video decoding, model inference, and production recipes. Its import-heavy fresh-process timings cannot be reported as an EgoWorld speedup.

Real Data-Juicer left/right Dex stage-interface checks now pass: six robot rows per side, with all logical fields exact and stable inputs/environments. The new preparation gates and existing CPU-pilot tests pass 31 targeted checks. See the [machine-readable acceptance](../benchmarks/comparison-prep-2026-11-02.json). This is a separate engineering gate. The [next-work plan](../comparison-next.md) keeps it distinct from full job timing, useful-output throughput and independent quality review.

Not established: same-quality performance superiority over Data-Juicer, server/multi-GPU scaling, robust long-video annotation, automatic handedness, a metric world for every model/frame/clip, or downstream policy gains. The [research plan](research-plan.md) defines the evidence required.

## Native product pilot and server preparation

The primary target is a single Linux server with 2–16 GPUs; RTX4060 is an additional validation environment. Portable RGB/RGB-D profiles and CPU/RAM/path checks are prepared; physical scaling is unverified. A [native C1 product pilot](../c1-native-comparison-20261102.md) now runs both complete hand pipelines: six finite hand predictions each, EgoWorld auto 100.77 s and unmodified native Data-Juicer CPU 61.18 s. Models, preprocessing, extra exports and handedness inputs differ, so no framework speed ratio is eligible. Strict matched-model/output C1, all corresponding task groups and quality/recovery experiments remain pending. See the [handoff](../handoff-server-comparison-20261102.md).
