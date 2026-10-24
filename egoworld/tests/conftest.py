"""A real asynchronous CPU backend for scheduler tests; inference is injected per test."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
import pytest
from egoworld.pipeline.stages import PrepareActor, ModelActor
from egoworld.pipeline.artifacts import WriterActor
from egoworld.contracts import GPU_STAGES


class ThreadBackend:
    def __init__(self, config):
        self.config = config
        self.pools = {"prepare": ThreadPoolExecutor(config["execution"]["cpu_workers"]),
                      "gpu": ThreadPoolExecutor(max(1, config["num_gpus"])),
                      "cpu": ThreadPoolExecutor(config["execution"]["cpu_operator_workers"]),
                      "write": ThreadPoolExecutor(config["execution"]["write_workers"])}
        self.limits = {"prepare": config["execution"]["cpu_workers"], "gpu": max(1, config["num_gpus"]),
                       "cpu": config["execution"]["cpu_operator_workers"], "write": config["execution"]["write_workers"]}
        self.pending = {}
        self.preparer = PrepareActor(config)
        self.models = ModelActor(config)
        self.writer = WriterActor(config)
        self.closed = False

    def pool(self, kind, stage):
        return "write" if kind in ("write", "finalize") else ("gpu" if stage in GPU_STAGES else "cpu") if kind == "compute" else kind

    def available(self, kind, stage=None):
        pool = self.pool(kind, stage)
        return sum(task["pool"] == pool for task in self.pending.values()) < self.limits[pool]

    def submit(self, kind, job, prepared=None, inputs=None, result=None):
        pool = self.pool(kind, job.get("stage"))
        def work():
            if kind == "prepare": return self.preparer.prepare(job)
            if kind == "compute": return self.models.process(job, prepared, dict(inputs))
            if kind == "write": return self.writer.write_stage(job, result.result())
            return self.writer.finalize(job, dict(inputs), dict(result))
        future = self.pools[pool].submit(work)
        self.pending[future] = {"kind": kind, "pool": pool, "job": dict(job), "result": result, "started": time.monotonic()}
        return future

    def poll(self, timeout):
        if not self.pending: return []
        done, _ = wait(self.pending, timeout=timeout, return_when=FIRST_COMPLETED)
        events = []
        for future in done:
            task = self.pending.pop(future)
            error = value = None
            if task["kind"] == "compute":
                value = future
            else:
                try: value = future.result()
                except Exception as exc:
                    error = exc; value = task["result"]
                    if task["kind"] == "write":
                        try: task["result"].result()
                        except Exception: task["job"]["_retry_compute"] = True
            events.append((task["kind"], task["job"], value, error, time.monotonic() - task["started"]))
        return events

    def count(self, kind):
        return sum(task["kind"] == kind for task in self.pending.values())

    def close(self):
        self.closed = True
        for pool in self.pools.values(): pool.shutdown(wait=True)


@pytest.fixture
def thread_backend():
    return ThreadBackend
