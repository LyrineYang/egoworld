"""Prometheus metrics with no-op fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _NoOp:
    def inc(self, *args: Any, **kwargs: Any) -> None:
        return None

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set(self, *args: Any, **kwargs: Any) -> None:
        return None


def _get_metrics():
    try:
        from prometheus_client import Counter, Gauge, Histogram

        return Counter, Gauge, Histogram
    except Exception:
        return _NoOp, _NoOp, _NoOp


@dataclass
class Metrics:
    throughput: Any
    queue_length: Any
    stage_latency: Any
    gpu_util: Any
    failure_count: Any


_def_counter, _def_gauge, _def_hist = _get_metrics()

DEFAULT_METRICS = Metrics(
    throughput=_def_counter("clips_processed_total", "Total processed clips"),
    queue_length=_def_gauge("queue_length", "Queue length", ["stage"]),
    stage_latency=_def_hist("stage_latency_seconds", "Stage latency", ["stage"]),
    gpu_util=_def_gauge("gpu_utilization", "GPU utilization"),
    failure_count=_def_counter("clip_failures_total", "Total clip failures"),
)
