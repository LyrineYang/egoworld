"""Opt-in phase observations. Wall/CPU/CUDA intervals overlap; never add them.

CUDA events cover the current stream only. They are resolved once at the request
boundary, not by synchronizing every operation. CPU-only paths never initialize
CUDA. Aggregates are complete; the detailed event sample is explicitly bounded.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import sys
import time

CURRENT = ContextVar("phase_profile", default=None)


class Profile:
    def __init__(self, identity=None, max_events=2048):
        self.identity = dict(identity or {}, pid=os.getpid())
        self.stats, self.events, self.pending = {}, [], []
        self.max_events = max_events

    def add(self, name, value):
        self.stats[name] = self.stats.get(name, 0) + value

    def finish(self):
        if self.pending:
            # All events are recorded in this worker's current device/stream.
            # Synchronize each recorded end event only at the request boundary.
            for name, start, end, event in self.pending:
                end.synchronize()
                seconds = start.elapsed_time(end) / 1000.
                self.add(f"phase_{name}_cuda_seconds", seconds)
                self.add(f"phase_{name}_cuda_samples", 1)
                if event is not None:
                    event["cuda_seconds"] = seconds
            self.pending.clear()
        return {"identity": self.identity, "stats": self.stats, "events": self.events,
                "semantics": "overlapping wall/process-CPU; current-stream CUDA events; boundary synchronization"}


@contextmanager
def profile_session(enabled, identity=None):
    profile = Profile(identity) if enabled else None
    token = CURRENT.set(profile)
    try:
        yield profile
    finally:
        CURRENT.reset(token)


@contextmanager
def phase(name, cuda=False):
    profile = CURRENT.get()
    if profile is None:
        yield
        return
    wall, cpu = time.perf_counter(), time.process_time()
    start = end = None
    torch = sys.modules.get("torch")
    if (cuda and torch is not None and hasattr(torch, "cuda")
            and torch.cuda.is_initialized() and len(profile.pending) < profile.max_events):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    error = False
    try:
        yield
    except BaseException:
        error = True
        raise
    finally:
        if end is not None:
            end.record()
        elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
        prefix = f"phase_{name}"
        profile.add(prefix + "_wall_seconds", elapsed)
        profile.add(prefix + "_cpu_seconds", cpu_elapsed)
        profile.add(prefix + "_calls", 1)
        profile.add(prefix + "_errors", int(error))
        event = None
        if len(profile.events) < profile.max_events:
            event = {"phase": name, "start_monotonic": wall, "wall_seconds": elapsed,
                     "cpu_seconds": cpu_elapsed, "error": error}
            profile.events.append(event)
        else:
            profile.add("phase_trace_dropped_events", 1)
        if start is not None:
            profile.pending.append((name, start, end, event))
        elif cuda:
            profile.add(prefix + "_cuda_unobserved_calls", 1)


def timed(name, cuda=False):
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            with phase(name, cuda=cuda):
                return fn(*args, **kwargs)
        return call
    return decorate
