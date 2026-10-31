"""Summarize measured runs without launching models or changing configuration."""
import json
from pathlib import Path


def profile_run(directory, invocation=None):
    root = Path(directory)
    if invocation is not None:
        if not invocation or Path(invocation).name != invocation or invocation in (".", ".."):
            raise ValueError("Invalid invocation ID")
        root = root / "invocations" / invocation
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
            "invocation_id": summary.get("invocation_id"),
            "config_fingerprint": manifest["config_fingerprint"],
            "configuration": manifest["config"], "stages": stages,
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "scope": "Completed calls in this invocation; includes cold starts. Stage times overlap. GPU memory is PyTorch allocator memory, not all device allocations. RSS is a process-lifetime high-water mark.",
            "automatic_resource_plan": False}


def list_invocations(directory):
    records = []
    for path in sorted((Path(directory) / 'invocations').glob('*/run_summary.json')):
        data = json.loads(path.read_text())
        records.append({'invocation_id': path.parent.name, 'status': data['status'],
                        'started_at': data.get('started_at'),
                        'elapsed_seconds': data.get('elapsed_seconds'),
                        'completed_clips': data.get('completed_clips', 0),
                        'skipped_clips': data.get('skipped_clips', 0),
                        'legacy': (path.parent / 'legacy.json').exists()})
    return {'invocations': records}
