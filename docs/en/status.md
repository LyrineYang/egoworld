# Implementation and evidence ledger

Reviewed **2026-11-02**. This presentation update is based on public code revision `e0638a2`, the current development workspace, and the project discussions “同步并审查 egoworld 项目” and “评估调度设计与 GPU 预检”. It changes documentation and website assets; it does not integrate outstanding model, product, or scheduler changes.

## What a public clone contains

| Capability | State on the public baseline |
| --- | --- |
| Five annotation stages | SAM2/DINO, HaMeR, FoundationPose, DexRetarget, and Fast3R adapters and configuration |
| Configuration-based CLI | Preflight, processing, smoke, schema, plan, inspection, and offline reports |
| Execution and recovery | Bounded scheduling, reusable runners, stage commits, validation, and dependency-aware recovery |
| Structured outputs | Parquet/NPZ, source-frame identity, coordinate/unit metadata, QC, and run events |
| CPU and Ray validation | Regression records exist; test doubles do not establish CUDA quality |

Model and asset prerequisites still apply. An adapter does not certify every input, device, or scene.

## Development work awaiting integration

| Area | Observed state | Boundary |
| --- | --- | --- |
| Output recipes | Local prototype: 13 types, dependency completion, field selection | Fixed registry; no arbitrary DAG or plugin system |
| Cross-run stage cache | Local prototype verifies identities and output contracts | Derived world/export work and input preparation are not uniformly cached |
| Calibrated world exports | Local cameras, points, objects, and hand interface | Metric evidence required; hands need hand correspondences |
| Adapter/reproduction fixes | Local hand-coordinate and robot-limit corrections | Development results cannot be attributed to unchanged public code |
| Replay, probes, input reuse, and scheduling experiments | Separate `codex/stage-replay-gpu-probe` branch | Requires integration and validation with product/adapter changes |

The development catalog contains `masks`, `boxes`, `tracks`, `hand_joints`, `hand_mano`, `object_poses`, `robot_joints`, `camera_poses`, `scene_points`, `world_camera`, `world_scene`, `world_object`, and `world_hand`. Membership is not unconditional availability: inputs and assets determine which outputs are possible.

## Evidence reported in development

Reviewed development records describe a short RGB chain—SAM2/DINO → HaMeR → Dex, plus Fast3R—running and resuming on an 8 GB RTX 4060 Laptop, and a separate FoundationPose RGB-D/mesh sample. Selected hand examples were compared with the upstream HaMeR API, and robot limits/FK were checked after adapter fixes. These are short-sample engineering checks, not an annotation-quality benchmark.

A calibrated RGB-D prototype exported sampled cameras, object poses, and points under an explicit anchor. It did not establish automatic metric hands, all-frame cameras, or cross-clip global mapping. Raw reproduction artifacts remain in the development environment and are not bundled with this documentation release.

No new GPU inference was performed for this homepage update. Historical development test counts are not tests of this documentation-only public revision. Consult the applicable revision and its [test records](../testing.md) before reproducing a result.

## Claims not established

- Same-input, same-quality speed superiority over Data-Juicer or another framework.
- Server throughput, multi-GPU scaling, or universal VRAM requirements.
- Reliable automatic handedness, robust long-video motion, or all-scene pose accuracy.
- A unified metric world for every model, every frame, and multiple clips.
- Improved robot policies or downstream learning outcomes.

The [research plan](research-plan.md) defines the required evidence. Update this ledger when integration and measurements are reproducible from a published revision.
