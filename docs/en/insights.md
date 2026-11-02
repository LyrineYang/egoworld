# Insights

**The difficult part is turning a set of predictions into data that someone can interpret and economically reproduce.** EgoWorld investigates that boundary between perception models and usable datasets.

These notes synthesize the implementation discussions and research plans. Observations, hypotheses, and proposed experiments are labeled separately; none is a claim of a new perception algorithm.

## 01 · Ask for data, then choose the work

**Observation.** Hand joints and MANO parameters share one HaMeR invocation. Conversely, selecting boxes alone does not currently make the segmentation kernel a cheap standalone detector.

**Design direction.** Let users select output types, then resolve and deduplicate model stages. Keep configuration visible when it affects output semantics. The developing recipe prototype follows this approach with a fixed registry of 13 output types.

**Implication.** An output catalog is not an arbitrary operator graph. Field selection happens at export and can reduce stored columns without reducing network computation.

**Next experiment.** Compare hand-only, hand-plus-MANO, and hand-plus-robot requests. Record resolved stages, actual model invocations, cache reuse, exported columns, and entry-to-result latency.

## 02 · Cache correctness and usefulness are different

**Observation.** Hashing assets, validating outputs, copying results, and decoding video can dominate a short run even when inference is avoided. A cache on a slow mounted filesystem can lose to recomputation.

**Hypothesis.** Removing repeated identity work and narrowing cache dependencies can improve latency without changing predictions.

**Design direction.** Separate input preparation, model residency, stage results, and derived exports. Give each an explicit identity, lifetime, budget, and invalidation rule. Reuse is valid only while relevant inputs, parameters, assets, environment, and upstream results match.

**Next experiment.** Alternate cold runs, warm-model runs with result misses, and verified cross-run hits. Measure verification and copying separately. Change a relevant dependency, an unrelated model parameter, and file content at the same path; observe invalidation. Ignoring a real change fails correctness even if it is faster.

## 03 · Time alignment is not geometry alignment

**Observation.** A hand estimate, object pose, and scene reconstruction can describe the same source frame using different origins, scales, and transform directions.

**Design direction.** Retain those conventions at the output boundary. A metric world export requires measured scale evidence and explicit transforms. An object pose already in meters must not be multiplied by scene scale again. A hand prior does not become metric because the scene is calibrated.

**Next experiment.** Validate transforms on held-out metric correspondences, retain residuals and coverage, and test missing-frame behavior. Evaluate world-hand calibration separately from scene calibration. Cross-clip mapping is a further problem.

## 04 · Optimize accepted output, not just FPS

**Observation.** Lower resolution, fewer views, aggressive filtering, or skipped detections can make processing faster while changing the dataset.

**Hypothesis.** Cheap, explainable gates and less repeated work can improve useful throughput under fixed quality requirements.

**Design direction.** Begin with corrupt-file checks, metadata, exact duplicates, and branch eligibility. A clip without usable hands can still contain useful scene geometry. Preserve raw predictions and rejection reasons before optional smoothing.

**Next experiment.** Measure accepted hand/object frames per second alongside false rejection, missing outputs, peak RAM/VRAM, and tail latency. Fix the accepted sample set when evaluating pure execution speed.

## 05 · Scheduling should reflect model economics

**Observation.** A short kernel call can sit behind model loading, CPU preparation, I/O, or writer backpressure. GPU occupancy alone misses those costs.

**Hypothesis.** Measured switching costs, stage durations, and intermediate-data pressure can guide useful residency windows without starving branches.

**Design direction.** Start from the bounded scheduler and fixed-window baselines. Profile before adding complexity. Bound prepared inputs and in-flight data; a cache limit is not a whole-process memory limit.

**Next experiment.** Compare fixed windows with a measured policy, then ablate switching cost, memory pressure, and waiting protection individually. Report complete runs. If a simple parameter adjustment explains the benefit, call it engineering tuning.

## Position relative to general processing systems

EgoWorld focuses on a specific hand–object–scene workflow, geometric contracts, recovery, and reproducibility. General systems such as Data-Juicer also cover embodied video; modality support alone is not unique. Its [operator catalog](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/docs/Operators.md) and [cache design](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/docs/Cache.md) provide comparison points.

Running identical models studies execution overhead. Running each system's preferred pipeline studies output quality and cost. Mixing these comparisons cannot establish framework speed. No matched end-to-end comparison has been completed.

Continue with the [research plan](research-plan.md) and [implementation status](status.md).
