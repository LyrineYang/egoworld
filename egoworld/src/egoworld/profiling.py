"""Summarize measured runs without launching models or changing configuration."""
import json
from pathlib import Path


def profile_run(directory):
    root = Path(directory)
    summary = json.loads((root / "run_summary.json").read_text())
    manifest = json.loads((root / "run_manifest.json").read_text())
    stages = {}
    for name, metrics in summary.get("stage_metrics", {}).items():
        calls = metrics["calls"]
        stages[name] = {
            "completed_calls": calls,
            "mean_stage_wall_seconds": metrics.get("stage_wall_seconds", metrics["inference_seconds"]) / calls,
            "mean_model_load_seconds": metrics.get("model_load_seconds", 0.) / calls,
            "mean_queue_wait_seconds": metrics.get("queue_wait_seconds", 0.) / calls,
            "model_cache_hits": metrics.get("model_cache_hit", 0),
            "process_cpu_seconds": metrics.get("process_cpu_seconds", 0.),
            "process_lifetime_peak_rss_bytes": metrics.get("process_peak_rss_bytes", 0),
            "observed_result_peak_bytes": metrics.get("result_peak_bytes", 0),
            "torch_peak_allocated_bytes": metrics.get("torch_peak_allocated_bytes", 0),
            "torch_peak_reserved_bytes": metrics.get("torch_peak_reserved_bytes", 0),
            "gpu_names": metrics.get("gpu_names", []),
            "gpu_memory_observed": bool(metrics.get("torch_peak_reserved_bytes", 0)),
        }
    return {"run_id": summary["run_id"], "status": summary["status"],
            "config_fingerprint": manifest["config_fingerprint"],
            "configuration": manifest["config"], "stages": stages,
            "elapsed_seconds": summary["elapsed_seconds"],
            "scope": "Completed calls in this invocation; includes cold starts. Stage times overlap. GPU memory is PyTorch allocator memory, not all device allocations. RSS is a process-lifetime high-water mark.",
            "automatic_resource_plan": False}
