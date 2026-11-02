# Implementation and evidence ledger

Reviewed **2026-11-02** for integration branch `codex/stage-replay-gpu-probe`. The branch includes main's English documentation and website commit `529a175`, plus the product/adapter snapshot imported in `94961f2` and replay, probe, cache, phase-execution and scheduling work through `8efd11f`. This describes the integration branch, not a claim that these changes have been published on main. Another task still has uncommitted product cleaning/cache changes in the primary worktree.

## Base capabilities

| Capability | State on this integration branch |
| --- | --- |
| Five annotation stages | SAM2/DINO, HaMeR, FoundationPose, DexRetarget, and Fast3R adapters and configuration |
| Configuration-based CLI | Preflight, processing, smoke, schema, plan, inspection, and offline reports |
| Execution and recovery | Bounded scheduling, reusable runners, stage commits, validation, and dependency-aware recovery |
| Structured outputs | Parquet/NPZ, source-frame identity, coordinate/unit metadata, QC, and run events |
| CPU and Ray validation | Regression records exist; test doubles do not establish CUDA quality |

Model and asset prerequisites still apply. An adapter does not certify every input, device, or scene.

## Additional integrated work and boundaries

| Area | Observed state | Boundary |
| --- | --- | --- |
| Output recipes | Integrated prototype: 13 types, dependency completion, field selection | Fixed registry; no arbitrary DAG or plugin system |
| Cross-run stage cache | Integrated prototype verifies identities and output contracts | Derived world/export work and input preparation are not uniformly cached |
| Calibrated world exports | Integrated cameras, points, objects, and hand interface | Metric evidence required; hands need hand correspondences |
| Adapter/reproduction fixes | Imported hand-coordinate and robot-limit corrections | Review and validate newer product changes separately |
| Replay, probes, input reuse, and scheduling experiments | Integrated; real single-GPU phase and scheduling records | Local small-workload evidence, not target-server or Data-Juicer superiority |
| Equal-work comparison pilot | Shared hand-contract rule, streaming reference, real Data-Juicer executor, exact-output audit | CPU artifact cleaning only; production recipe/model comparison pending |

The development catalog contains `masks`, `boxes`, `tracks`, `hand_joints`, `hand_mano`, `object_poses`, `robot_joints`, `camera_poses`, `scene_points`, `world_camera`, `world_scene`, `world_object`, and `world_hand`. Membership is not unconditional availability: inputs and assets determine which outputs are possible.

## Evidence reported in development

Reviewed development records describe a short RGB chain—SAM2/DINO → HaMeR → Dex, plus Fast3R—running and resuming on an 8 GB RTX 4060 Laptop, and a separate FoundationPose RGB-D/mesh sample. Selected hand examples were compared with the upstream HaMeR API, and robot limits/FK were checked after adapter fixes. These are short-sample engineering checks, not an annotation-quality benchmark.

A calibrated RGB-D prototype exported sampled cameras, object poses, and points under an explicit anchor. It did not establish automatic metric hands, all-frame cameras, or cross-clip global mapping. Raw reproduction artifacts remain in the development environment and are not bundled with this documentation release.

The [throughput validation record](../throughput-validation.md) pins the earlier native-model and scheduling experiments. The [equal-work protocol](../equal-work-benchmark.md) defines the external comparison and its current limits. No new GPU inference is required by the documentation merge or CPU artifact-cleaning pilot; source revisions must remain distinct.

## Claims not established

- Same-input, same-quality speed superiority over Data-Juicer or another framework.
- Server throughput, multi-GPU scaling, or universal VRAM requirements.
- Reliable automatic handedness, robust long-video motion, or all-scene pose accuracy.
- A unified metric world for every model, every frame, and multiple clips.
- Improved robot policies or downstream learning outcomes.

The [research plan](research-plan.md) defines the required evidence. Update this ledger when integration and measurements are reproducible from a published revision.
