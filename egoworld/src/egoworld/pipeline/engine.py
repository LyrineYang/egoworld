"""Completion-driven scheduling with bounded admission and delayed, nonblocking retries."""
from __future__ import annotations
from collections import deque
import heapq
import itertools
import shutil
import time

from egoworld.contracts import OUTPUT_SCHEMA_VERSION, enabled_stages
from egoworld.pipeline.artifacts import load_stage
from egoworld.pipeline.state_store import set_stage, upsert_clip_status, mark_dead_letter, get_clip_state
from egoworld.utils.errors import classify_error

TERMINAL = {"Done", "Failed", "Blocked"}


class Engine:
    def __init__(self, config, backend, fingerprint, emit):
        self.config, self.backend, self.fingerprint, self.emit = config, backend, fingerprint, emit
        self.graph = enabled_stages(config)
        self.db, self.run_id = config.paths.state_db_path, config.run_id
        self.active, self.writes, self.retries = {}, deque(), []
        self.sequence = itertools.count()
        self.failed, self.completed, self.skipped = set(), 0, 0
        self.successful_frames, self.stage_metrics = 0, {}
        self.peaks = {"active_clips": 0, "prepare": 0, "compute": 0, "uncommitted_results": 0}

    def job(self, ctx, stage=None):
        return {"clip": ctx["clip"], "stage": stage, "directory": ctx["directory"],
                "work_dir": ctx["work_dir"], "fingerprint": self.fingerprint}

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
               "errors": {}, "prepare": "Pending", "prepared": None, "finalizing": False, "attempts": {}}
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
            self.record(ctx, stage, "Failed", attempt, str(error))

    def finish(self, ctx, failed=False):
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
        self.backend.submit(kind, job, prepared=ctx["prepared"], inputs=ctx["receipts"],
                            result=ctx["errors"] if kind == "finalize" else result)

    def uncommitted(self):
        return self.backend.count("compute") + len(self.writes) + self.backend.count("write") + sum(entry[2] == "write" for entry in self.retries)

    def eligible(self, kind, job):
        if not self.backend.available(kind, job.get("stage") if kind == "compute" else None):
            return False
        if kind == "prepare":
            return self.backend.count("prepare") < self.config.backpressure.max_in_flight_cpu
        if kind == "compute":
            return self.backend.count("compute") < self.config.backpressure.max_in_flight_gpu and self.uncommitted() < self.config.backpressure.max_in_flight_write
        if kind == "write":
            return self.uncommitted() < self.config.backpressure.max_in_flight_write
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
            ready = []
            for ctx in list(self.active.values()):
                if ctx["prepare"] == "Pending" and self.backend.available("prepare") and self.backend.count("prepare") < self.config.backpressure.max_in_flight_cpu:
                    if all(value == "Done" for value in ctx["states"].values()) and self.graph:
                        ctx["prepare"] = "Done"  # only final metadata needs rebuilding
                    else:
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
                if (ctx["prepare"] in ("Done", "Failed") and all(value in TERMINAL for value in ctx["states"].values())
                    and not ctx["finalizing"] and self.backend.available("finalize")):
                    ctx["finalizing"] = True
                    self.submit("finalize", self.job(ctx))
            priority = getattr(self.backend, "stage_priority", lambda stage: 1)
            for ctx, stage in sorted(ready, key=lambda pair: priority(pair[1])):
                if self.eligible("compute", self.job(ctx, stage)):
                    ctx["states"][stage] = "Running"
                    self.submit("compute", self.job(ctx, stage))
                    self.record(ctx, stage, "Running")
            self.peaks["active_clips"] = max(self.peaks["active_clips"], len(self.active))
            for name in ("prepare", "compute"):
                self.peaks[name] = max(self.peaks[name], self.backend.count(name))
            self.peaks["uncommitted_results"] = max(self.peaks["uncommitted_results"], self.uncommitted())
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
                    self.record(ctx, "prepare", "Done", duration_s=duration)
                elif kind == "compute":
                    ctx["states"][job["stage"]] = "Writing"
                    self.writes.append((job, value))
                elif kind == "write":
                    stage = job["stage"]
                    ctx["states"][stage] = "Done"; ctx["receipts"][stage] = value
                    metrics = self.stage_metrics.setdefault(stage, {"calls": 0, "inference_seconds": 0., "write_seconds": 0., "rows": 0, "bytes": 0})
                    metrics["calls"] += 1
                    for name in ("inference_seconds", "write_seconds"):
                        metrics[name] += value["stats"].get(name, 0.)
                    metrics["rows"] += sum(info.get("rows", 0) for info in value["files"].values())
                    metrics["bytes"] += sum(info["bytes"] for info in value["files"].values())
                    self.record(ctx, stage, "Done", ctx["attempts"].get(stage, 0), duration_s=duration, stats=value["stats"])
                else:
                    self.finish(ctx)
        return {"completed_clips": self.completed, "skipped_clips": self.skipped,
                "failed_clips": len(self.failed), "queue_peaks": self.peaks,
                "successful_scheduled_frames": self.successful_frames, "stage_metrics": self.stage_metrics}
