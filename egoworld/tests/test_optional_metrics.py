"""The base package must not require the optional Prometheus client."""
import subprocess
import sys


def test_metrics_import_and_updates_without_prometheus():
    program = '''
import sys
sys.modules['prometheus_client'] = None
from egoworld.observability.metrics import DEFAULT_METRICS as metrics
metrics.throughput.inc()
metrics.queue_length.set(3)
metrics.stage_latency.observe(.5)
metrics.failure_count.inc()
'''
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
