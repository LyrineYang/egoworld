"""Completion-driven scheduling with bounded admission and delayed, nonblocking retries."""
from __future__ import annotations
from collections import deque
import heapq
import itertools
import shutil
import time

from egoworld.contracts import OUTPUT_SCHEMA_VERSION, enabled_stages, GPU_STAGES
from egoworld.pipeline.artifacts import load_stage
from egoworld.pipeline.state_store import set_stage, upsert_clip_status, mark_dead_letter, get_clip_state
from egoworld.utils.errors import classify_error
from egoworld.pipeline.scheduling import MemoryBudget, ready_priority

TERMINAL = {"Done", "Failed", "Blocked"}


class Engine:
    def __init__(self, config, backend, fingerprint, emit):
        self.config, self.backend, self.fingerprint, self.emit = config, backend, fingerprint, emit
        self.graph = enabled_stages(config)
        self.db, self.run_id = config.paths.state_db_path, config.run_id
        self.active, self.writes, self.retries = {}, deque(), []
        self.phase_wait = deque()
        self.split_reserved = set()
        self.sequence = itertools.count()
        self.failed, self.completed, self.skipped = set(), 0, 0
        self.successful_frames, self.stage_metrics = 0, {}
        self.memory = MemoryBudget(config.backpressure)
        self.cost_policy = None
        if config.execution.scheduling_policy == "cost":
            from dataclasses import asdict
            from egoworld.diagnostics.calibration import CostPolicy
            self.cost_policy = CostPolicy(asdict(config))
        self.peaks = {"active_clips": 0, "prepare": 0, "compute": 0, "uncommitted_results": 0}

    def job(self, ctx, stage=None):
        return {"clip": ctx["clip"], "stage": stage, "directory": ctx["directory"],
                "work_dir": ctx["work_dir"], "fingerprint": self.fingerprint,
                "_ready_since": ctx["ready_since"].get(stage, time.monotonic()),
                "invocation_id": getattr(self, "invocation_id", None),
                "attempt": ctx["attempts"].get(stage, 0)}

    def split(self, stage):
        return self.config.execution.split_execution and stage in self.config.execution.split_stages

    def release_split(self, job):
        self.split_reserved.discard(self.memory.key(job))
        from egoworld.pipeline.spool import release_spool
        release_spool(vars(self.config.execution), job["work_dir"], job["stage"])

    def choose_cost(self, jobs):
        for job in jobs:
            ctx = self.active[job["clip"]["clip_id"]]
            # Free the prepared clip only when this is its last remaining stage.
            remaining = [s for s, state in ctx["states"].items() if state not in TERMINAL]
            job["_release_bytes"] = (ctx["prepared"] or {}).get("prepared_bytes", 0) if remaining == [job["stage"]] else 0
        job, decision = self.cost_policy.choose(jobs, self.backend, time.monotonic())
        self.emit(stage="scheduler", status="decision", **decision)
        return job

    def record(self, ctx, stage, status, attempt=0, error="", **metrics):
        set_stage(self.db, self.run_id, ctx["clip"]["clip_id"], stage, status, attempt, error)
        self.emit(stage=stage, status=status, clip_id=ctx["clip"]["clip_id"], attempt=attempt,
                  error=error, **metrics)

    def admit(self, clip):
        from egoworld.io.paths import clip_dir, run_dir
        directory = clip_dir(self.config.paths.output_root, self.run_id, clip["video_id"], clip["clip_id"])
        ctx = {"clip": clip, "directory": str(directory),
               "work_dir": str(run_dir(self.config.paths.output_root, self.run_id) / ".work" / clip["clip_id"]),
               "states": {stage: "Pending" for stage in self.graph}, "receipts": {},
               "errors": {}, "prepare": "Pending", "prepared": None, "finalizing": False, "attempts": {}, "ready_since": {}}
        for stage, dependencies in self.graph.items():
            if any(dependency not in ctx["receipts"] for dependency in dependencies):
                continue
            receipt = load_stage(directory, stage, self.fingerprint, self.config.execution.verify_hashes_on_resume)
            if receipt:
                ctx["states"][stage] = "Done"; ctx["receipts"][stage] = receipt
        # Invalidate durable markers before any replacement work can be submitted.
        # Otherwise a second interruption could accept descendants of an old result.
        if len(ctx["receipts"]) != len(self.graph):
            for name in ("meta.json", "qc.json"):
                (directory / name).unlink(missing_ok=True)
            for stage in reversed(self.graph):
                if ctx["states"][stage] != "Done":
                    (directory / f"stage_{stage}.json").unlink(missing_ok=True)
        if self.graph and len(ctx["receipts"]) == len(self.graph):
            import json
            try:
                meta = json.loads((directory / "meta.json").read_text())
                qc = json.loads((directory / "qc.json").read_text())
                if (meta["fingerprint"] == self.fingerprint and meta["status"] == "Done"
                    and meta.get("output_schema_version") == OUTPUT_SCHEMA_VERSION
                    and set(meta.get("stages", [])) == set(self.graph) and qc == meta.get("quality")):
                    state = get_clip_state(self.db, clip["clip_id"], run_id=self.run_id)
                    if state is None or state.status != "Done":
                        upsert_clip_status(self.db, clip["clip_id"], clip["video_id"], "Done", run_id=self.run_id)
                        for stage in ctx["receipts"]:
                            self.record(ctx, stage, "Done", cached=True)
                    self.skipped += 1
                    self.emit(stage="pipeline", status="skipped", clip_id=clip["clip_id"])
                    return
            except (OSError, ValueError, KeyError):
                pass
        upsert_clip_status(self.db, clip["clip_id"], clip["video_id"], "Running", run_id=self.run_id)
        for stage in ctx["receipts"]:
            self.record(ctx, stage, "Done", cached=True)
        self.active[clip["clip_id"]] = ctx

    def retry_or_fail(self, kind, job, ctx, error, result=None):
        stage = job.get("stage") or kind
        if self.split(stage) and (kind in ("input", "compute", "post") or kind == "write" and job.get("_retry_compute")):
            self.memory.release(job)
            self.release_split(job)
            kind, result = "input", None
            job.pop("_operation", None)
            job.pop("_retry_compute", None)
        if kind == "compute" or (kind == "write" and job.get("_retry_compute")):
            self.memory.release(job)
        attempt = ctx["attempts"].get(stage, 0)
        classification = classify_error(error)
        retryable = classification.retryable or isinstance(error, TimeoutError) or "actor died" in str(error).lower() or any(name in type(error).__name__ for name in ("ActorDied", "ActorUnavailable"))
        if retryable and attempt < self.config.retry.max_retries:
            attempt += 1; ctx["attempts"][stage] = attempt
            delay = self.config.retry.next_delay(attempt)
            if kind == "write" and job.pop("_retry_compute", False):
                kind, result = "compute", None
            if kind == "prepare":
                ctx["prepare"] = "Retry"
            elif kind == "finalize":
                ctx["finalizing"] = True
            else:
                ctx["states"][stage] = "Retry"
            heapq.heappush(self.retries, (time.monotonic() + delay, next(self.sequence), kind, job, result))
            self.record(ctx, stage, "Retry", attempt, str(error), delay_s=delay)
        else:
            ctx["errors"][stage] = str(error)
            if kind == "prepare":
                ctx["prepare"] = "Failed"
                for name in ctx["states"]:
                    if ctx["states"][name] != "Done":
                        ctx["states"][name] = "Blocked"
            elif kind == "finalize":
                self.finish(ctx, failed=True)
            else:
                ctx["states"][stage] = "Failed"
                self.memory.release(job)
                if self.split(stage):
                    self.release_split(job)
            self.record(ctx, stage, "Failed", attempt, str(error))

    def finish(self, ctx, failed=False):
        from egoworld.pipeline.input_cache import release_prepared_cache
        release_prepared_cache(ctx["prepared"])
        clip = ctx["clip"]; failed = failed or bool(ctx["errors"])
        error = "; ".join(f"{stage}: {message}" for stage, message in ctx["errors"].items())
        upsert_clip_status(self.db, clip["clip_id"], clip["video_id"], "Failed" if failed else "Done",
                           error, run_id=self.run_id)
        if failed:
            self.failed.add(clip["clip_id"])
            mark_dead_letter(self.db, clip["clip_id"], clip["video_id"], error, run_id=self.run_id)
        else:
            self.completed += 1
            self.successful_frames += clip["frame_end"] - clip["frame_start"]
            if not self.config.execution.keep_prepared:
                shutil.rmtree(ctx["work_dir"], ignore_errors=True)
        del self.active[clip["clip_id"]]

    def submit(self, kind, job, result=None):
        ctx = self.active[job["clip"]["clip_id"]]
        first = kind == "input" or kind == "compute" and not self.split(job["stage"])
        if "_step_ready_since" in job:
            job.setdefault("_phase_waits", {})[kind] = time.monotonic() - job.pop("_step_ready_since")
        if first:
            self.memory.reserve(job)
            if kind == "input":
                self.split_reserved.add(self.memory.key(job))
            job["_queue_wait_seconds"] = time.monotonic() - job["_ready_since"]
        try:
            self.backend.submit(kind, job, prepared=ctx["prepared"], inputs=ctx["receipts"],
                                result=ctx["errors"] if kind == "finalize" else result)
        except BaseException:
            if first:
                self.memory.release(job)
                if kind == "input":
                    self.release_split(job)
            raise

    def prepared_bytes(self):
        return sum((ctx["prepared"] or {}).get("prepared_bytes", 0) for ctx in self.active.values())

    def can_prepare(self):
        limit = self.config.backpressure.max_prepared_bytes
        return (not limit or self.prepared_bytes() < limit) and self.uncommitted() < self.config.backpressure.max_in_flight_write

    def uncommitted(self):
        return len(self.memory.reservations)

    def eligible(self, kind, job):
        if not self.backend.available(kind, job.get("stage") if kind in ("compute", "input", "post") else None):
            return False
        if kind == "prepare":
            return self.can_prepare() and self.backend.count("prepare") < self.config.backpressure.max_in_flight_cpu
        if kind in ("input", "compute"):
            if kind == "compute" and self.split(job["stage"]):
                return self.backend.count("compute") < self.config.backpressure.max_in_flight_gpu
            if kind == "input" and ((len(self.split_reserved) + 1) * self.config.execution.split_stage_bytes > self.config.execution.split_total_bytes):
                return False
            return (self.backend.count("compute") < self.config.backpressure.max_in_flight_gpu or kind == "input") and self.uncommitted() < self.config.backpressure.max_in_flight_write and self.memory.fits(job)
        if kind == "write":
            return True  # draining already-reserved results must never wait on producer limits
        return True

    def run(self, clips):
        source, exhausted = iter(clips), False
        while not exhausted or self.active or self.retries:
            while not exhausted and len(self.active) < self.config.execution.max_active_clips:
                try:
                    self.admit(next(source))
                except StopIteration:
                    exhausted = True
            # Retry timers never sleep the scheduling loop or block independent work.
            delayed = []
            while self.retries and self.retries[0][0] <= time.monotonic():
                due, seq, kind, job, result = heapq.heappop(self.retries)
                if self.eligible(kind, job):
                    self.submit(kind, job, result)
                else:
                    delayed.append((due, seq, kind, job, result))
            for entry in delayed:
                heapq.heappush(self.retries, entry)
            while self.writes and self.backend.available("write"):
                job, result = self.writes.popleft()
                self.submit("write", job, result)
            pending = list(self.phase_wait)
            while pending:
                feasible = [(k, j, v) for k, j, v in pending if self.eligible(k, j)]
                if not feasible:
                    break
                post = [item for item in feasible if item[0] == "post"]
                if post:
                    chosen = post[0]  # drain CPU outputs before producing more
                elif self.cost_policy:
                    selected = self.choose_cost([j for k, j, v in feasible])
                    chosen = next(item for item in feasible if item[1] is selected)
                else:
                    priority = getattr(self.backend, "stage_priority", lambda stage: 1)
                    chosen = min(feasible, key=lambda item: ready_priority(self.active[item[1]["clip"]["clip_id"]],
                        item[1]["stage"], time.monotonic(), self.config.execution.max_stage_wait_s, priority))
                pending.remove(chosen); self.phase_wait.remove(chosen)
                self.submit(*chosen)
            ready = []
            for ctx in list(self.active.values()):
                if ctx["prepare"] == "Pending" and self.backend.available("prepare") and self.backend.count("prepare") < self.config.backpressure.max_in_flight_cpu:
                    if all(value == "Done" for value in ctx["states"].values()) and self.graph:
                        ctx["prepare"] = "Done"  # only final metadata needs rebuilding
                    elif self.can_prepare():
                        ctx["prepare"] = "Running"
                        self.submit("prepare", self.job(ctx))
                        self.record(ctx, "prepare", "Running")
                for stage, dependencies in self.graph.items():
                    if ctx["states"][stage] == "Pending" and any(ctx["states"][d] in ("Failed", "Blocked") for d in dependencies):
                        ctx["states"][stage] = "Blocked"; ctx["errors"][stage] = "upstream stage failed"
                        self.record(ctx, stage, "Blocked", error="upstream stage failed")
                    if (ctx["prepare"] == "Done" and ctx["states"][stage] == "Pending"
                        and all(ctx["states"][d] == "Done" for d in dependencies)):
                        ready.append((ctx, stage))
                        ctx["ready_since"].setdefault(stage, time.monotonic())
                if (ctx["prepare"] in ("Done", "Failed") and all(value in TERMINAL for value in ctx["states"].values())
                    and not ctx["finalizing"] and self.backend.available("finalize")):
                    ctx["finalizing"] = True
                    self.submit("finalize", self.job(ctx))
            priority = getattr(self.backend, "stage_priority", lambda stage: 1)
            while ready:
                candidates = [(ctx, stage) for ctx, stage in ready if self.eligible("input" if self.split(stage) else "compute", self.job(ctx, stage))]
                if not candidates:
                    break
                now = time.monotonic()
                gpu_jobs = [self.job(ctx, stage) for ctx, stage in candidates if stage in GPU_STAGES and not self.split(stage)]
                chosen_job = None
                if self.cost_policy and gpu_jobs:
                    chosen_job = self.choose_cost(gpu_jobs)
                    stage = chosen_job["stage"]; ctx = self.active[chosen_job["clip"]["clip_id"]]
                else:
                    ctx, stage = min(candidates, key=lambda pair: ready_priority(*pair, now, self.config.execution.max_stage_wait_s, priority))
                ready.remove((ctx, stage))
                ctx["states"][stage] = "Running"
                self.submit("input" if self.split(stage) else "compute", chosen_job or self.job(ctx, stage))
                self.record(ctx, stage, "Running")
            self.peaks["active_clips"] = max(self.peaks["active_clips"], len(self.active))
            for name in ("prepare", "compute", "input", "post"):
                self.peaks.setdefault(name, 0)
                self.peaks[name] = max(self.peaks[name], self.backend.count(name))
            self.peaks["uncommitted_results"] = max(self.peaks["uncommitted_results"], self.uncommitted())
            self.peaks["split_reserved_bytes"] = max(self.peaks.get("split_reserved_bytes", 0), len(self.split_reserved) * self.config.execution.split_stage_bytes)
            self.peaks["prepared_bytes"] = max(self.peaks.get("prepared_bytes", 0), self.prepared_bytes())
            if not self.active and exhausted:
                break
            if not self.backend.count("prepare") and not self.backend.count("compute") and not self.backend.count("write") and not self.backend.count("finalize"):
                time.sleep(min(.01, self.config.execution.poll_timeout_s))
            events = self.backend.poll(self.config.execution.poll_timeout_s)
            for kind, job, value, error, duration in events:
                ctx = self.active[job["clip"]["clip_id"]]
                if error:
                    self.retry_or_fail(kind, job, ctx, error, value)
                    continue
                if kind == "prepare":
                    ctx["prepare"], ctx["prepared"] = "Done", value
                    self.record(ctx, "prepare", "Done", duration_s=duration, stats=value.get("stats", {}), profile=value.get("profile"))
                elif kind == "input":
                    job["_step_ready_since"] = time.monotonic()
                    self.phase_wait.append(("compute", job, value))
                    self.record(ctx, job["stage"], "Running", phase="input_done", duration_s=duration)
                elif kind == "compute" and self.split(job["stage"]):
                    job["_step_ready_since"] = time.monotonic()
                    self.phase_wait.append(("post", job, value))
                    self.record(ctx, job["stage"], "Running", phase="infer_done", duration_s=duration)
                elif kind in ("compute", "post"):
                    self.memory.completed(job)
                    ctx["states"][job["stage"]] = "Writing"
                    job["_step_ready_since"] = time.monotonic()
                    self.writes.append((job, value))
                elif kind == "write":
                    self.memory.release(job)
                    if self.split(job["stage"]):
                        self.release_split(job)
                    stage = job["stage"]
                    ctx["states"][stage] = "Done"; ctx["receipts"][stage] = value
                    metrics = self.stage_metrics.setdefault(stage, {"calls": 0, "inference_seconds": 0., "write_seconds": 0., "rows": 0, "bytes": 0})
                    metrics["calls"] += 1
                    for name in ("inference_seconds", "write_seconds"):
                        metrics[name] += value["stats"].get(name, 0.)
                    metrics["rows"] += sum(info.get("rows", 0) for info in value["files"].values())
                    metrics["bytes"] += sum(info["bytes"] for info in value["files"].values())
                    metrics["queue_wait_seconds"] = metrics.get("queue_wait_seconds", 0.) + job.get("_queue_wait_seconds", 0.)
                    for name in ("model_load_seconds", "model_cache_hit", "process_cpu_seconds", "stage_wall_seconds", "runner_roundtrip_seconds", "result_bytes"):
                        metrics[name] = metrics.get(name, 0.) + value["stats"].get(name, 0.)
                    for name in ("process_peak_rss_bytes", "torch_peak_allocated_bytes", "torch_peak_reserved_bytes"):
                        metrics[name] = max(metrics.get(name, 0), value["stats"].get(name, 0))
                    for name, observation in value["stats"].items():
                        if name.startswith(("input_cache_", "phase_", "spool_")):
                            metrics[name] = (max(metrics.get(name, 0), observation) if "_peak_" in name
                                             else metrics.get(name, 0) + observation)
                    metrics["result_peak_bytes"] = max(metrics.get("result_peak_bytes", 0), value["stats"].get("result_bytes", 0))
                    if value["stats"].get("gpu_name"):
                        names = metrics.setdefault("gpu_names", [])
                        if value["stats"]["gpu_name"] not in names:
                            names.append(value["stats"]["gpu_name"])
                    self.record(ctx, stage, "Done", ctx["attempts"].get(stage, 0), duration_s=duration, stats=value["stats"])
                else:
                    self.finish(ctx)
        return self.summary()

    def summary(self):
        """Include committed measurements even when an invocation is interrupted."""
        return {"completed_clips": self.completed, "skipped_clips": self.skipped,
                "failed_clips": len(self.failed), "queue_peaks": self.peaks,
                "successful_scheduled_frames": self.successful_frames, "stage_metrics": self.stage_metrics,
                "memory_budget": {"reserved_peak_bytes": self.memory.peak, "remaining_bytes": self.memory.used,
                                  "stage_estimates_bytes": self.memory.estimates}}
