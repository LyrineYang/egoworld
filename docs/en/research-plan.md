# Research plan

**Core objective:** Test whether specialization for embodied data yields more complete cleaning and higher processing efficiency than Data-Juicer under identical inputs, model behavior, output requirements, and acceptance rules. This is the intended central result, not an established claim.

**Question:** How can an egocentric annotation system produce more usable hand, object, and scene data per unit of time and memory while preserving model behavior, geometric meaning, and recoverability?

This plan targets quality-preserving systems improvements. It does not presume a speedup, a new perception model, or downstream learning gains.

## Priorities

Existing annotation families and the developing 13-type catalog are sufficient for the next phase. Prioritize reproducibility, measured bottlenecks, reuse, conservative cleaning, and configuration clarity. Add models or action formats when a concrete consumer needs them.

| Priority | Deliverable | Completion evidence |
| --- | --- | --- |
| P0 | Integrate existing recipe, replay, probe, and input-cache work | Pinned integrated revision, regression and real-model smoke records |
| P0 | Separate entry latency into identity, preparation, inference, verification, and export | Raw invocation records, including failures and resumes |
| P0 | Remove repeated asset verification; explain cache misses | Content changes still invalidate; redundant I/O and verification decline |
| P0 | Add cheap input and branch eligibility checks | Fixed retained set, rejection reasons, false-rejection review |
| P1 | Evaluate bounded frame/crop reuse, mask transfer reuse, and hand batching | Equivalent pixels/crops/keys, quality gates, and bounded memory |
| P1 | Evaluate measured residency and admission | Full-run throughput and tail waiting against simple baselines |
| P1 | Narrow invalidation, stream tables, improve recipe explanations | Relevant changes recompute; unrelated changes reuse; long inputs remain bounded |
| P2 | Cold-cache codecs, continuous trajectories, downstream adapters | Demonstrated demand, quality assessment, asset requirements |

## Hypotheses and counterexamples

| Hypothesis | Test | Evidence against it |
| --- | --- | --- |
| Less repeated identity/preparation work lowers latency | Same inputs/assets; baseline versus deduplicated validation and bounded reuse | Verification/copy cost exceeds saved work, or correctness weakens |
| Measured residency improves useful throughput | Current scheduler, fixed windows, measured policy | Tuned fixed windows match it, memory grows, or tail waiting worsens |
| Cheap branch gates reduce wasted GPU work | Metadata, exact duplicates, post-segmentation eligibility | False rejection removes useful small, slow, occluded, or hand-free scene data |
| Explicit calibration improves geometric usability | Held-out metric correspondences and strict source-frame joins | Residuals, coverage gaps, or scale uncertainty exceed thresholds |

## Workloads

Use independently sourced 10–60 second clips covering one hand, two hands, occlusion, fast motion, multiple objects, no-hand scenes, and malformed input. Separate RGB-D workloads because depth, intrinsics, and meshes change the problem. Short smoke samples are regression fixtures, not performance or accuracy benchmarks.

Freeze source/license, content hash, frame rate, resolution, range, prompts, handedness review, depth units, mesh identity, and output requirements. Repeating one short clip does not increase sample diversity.

Begin on one GPU. Record device, driver, runtime, thread limits, model/source identities, storage, and competing load before extending to multi-GPU.

## Baselines and ablations

1. **Current bounded scheduler.** Preserve recovery and model semantics.
2. **Simple residency.** Test same-model windows such as 1, 2, 4, and 8 clips with waiting protection.
3. **Measured policy.** Add observed switching and stage costs, then memory pressure when telemetry is reliable.
4. **Ablations.** Disable input reuse, switching cost, pressure accounting, or waiting protection independently.

Hold work constant. Resolution, precision, frame sampling, view counts, and model changes belong in a separate quality/cost study, not same-quality execution optimization.

## Measurement protocol

Distinguish cold runner/model startup, warm models with result misses, cross-run hits, input-cache on/off, and interrupted recovery. “Cold” does not imply flushed OS/driver caches. Disclose observable storage-cache conditions.

For short stages, start with one cold call, two warmups, and at least five warm repeats. Alternate A/B order and repeat full invocations. Report distributions, counts, and variability; broaden samples before broad conclusions.

Measure entry-to-final-manifest wall time plus identity, lookup/verification/copying, decode/crop, transfers/inference, CPU postprocessing, writing, QC/geometry/export, and queue wait. Overlapping stage times cannot simply be summed. Use synchronized profiling or CUDA events for device attribution.

| Metric | Meaning |
| --- | --- |
| Accepted hand/object frames per second | Useful throughput under fixed acceptance rules |
| Video seconds per wall second | Processing rate with sampling/coverage disclosed |
| Startup, median and tail completion/wait | User latency and fairness |
| Peak RAM/VRAM and in-flight bytes | Resource cost with method and process scope |
| Hits, misses, invalidations, verification cost | Reuse effectiveness beyond hit rate |
| Failures, OOMs, retries, recovery work | Reliability and operational cost |
| False rejection, missing frames, calibration residuals | Whether speed preserves useful data |

Separate NVML device/process observations, Torch allocation, process-tree RSS/PSS when available, Ray object store, cache mappings, temporary disk, and I/O. Missing telemetry is unknown, not zero.

## Data-Juicer comparison

Pin the upstream revision; the earlier review used `fc0314b17209e5a2c15a2735d26cfa69d7e3f08d`. Reference its [embodied-video recipe](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/demos/ego_hand_action_annotation/configs/vla_pipeline.yaml) and [cache documentation](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/docs/Cache.md).

| Comparison | Hold constant | What it establishes |
| --- | --- | --- |
| Common CPU checks | Exact check and retained-set semantics | Cleaning overhead |
| Matched adapters | Models, weights, inputs, sampling, precision, quality | Execution/caching/scheduling/handoff cost |
| Native workflows | Task and acceptance criteria; disclose different models | Capability, quality, coverage, total cost |

Matched adapters and a complete end-to-end comparison remain future work. Include geometry/export on both sides or exclude them on both. Report executor/fusion/cache settings independently.

## Publication gate

A performance claim needs a pinned revision, reproducible commands, permitted input references, asset/configuration identities, raw distributions, output equivalence or predeclared quality thresholds, and memory/recovery checks. Geometry claims additionally need independent scale evidence and held-out validation. Learning claims need separate learning experiments.

Keep negative results. A slower cache on mounted storage or a fixed window matching a complex scheduler informs design.

Project sources: [original plan](../archive/plan.md), [scheduling design](../design.md), [project review](../overview.md), and development discussions reviewed on 2026-11-02. The [status ledger](status.md) separates intent from publication.

### Executable first step

The [equal-work pilot](../equal-work-benchmark.md) now freezes existing hand annotations and executes the same contract-validation function through a streaming reference and a pinned Data-Juicer DefaultExecutor custom Mapper. It preserves every decision and rejection reason, checks exact accepted outputs, and suppresses performance ratios if any planned invocation fails or diverges. This measures fresh-process CPU artifact cleaning. The streaming reference is experimental, not the production EgoWorld recipe executor; matched model inference and complete end-to-end comparisons remain pending.
