"""Ray resource pools. The driver fetches receipts, never full model outputs."""
from __future__ import annotations
import time
from egoworld.contracts import GPU_STAGES
from egoworld.pipeline.stages import PrepareActor, ModelActor
from egoworld.pipeline.artifacts import WriterActor


class RayBackend:
    def __init__(self, config):
        import ray
        self.ray, self.config = ray, config
        self.slots, self.pending = [], {}
        groups = config["execution"].get("gpu_stage_groups") or [list(GPU_STAGES)] * config["num_gpus"]
        definitions = [("prepare", PrepareActor, {"num_cpus": 1}, None)] * config["execution"]["cpu_workers"]
        definitions += [("write", WriterActor, {"num_cpus": .5}, None)] * config["execution"]["write_workers"]
        enabled_gpu = any(config["operators"][s]["enabled"] for s in GPU_STAGES)
        if enabled_gpu:
            definitions += [("gpu", ModelActor, {"num_cpus": 1, "num_gpus": 1}, group) for group in groups]
        if config["operators"]["dex_retarget"]["enabled"]:
            definitions += [("cpu", ModelActor, {"num_cpus": 1}, ["dex_retarget"])] * config["execution"]["cpu_operator_workers"]
        required_cpus = sum(resources["num_cpus"] for _, _, resources, _ in definitions)
        if ray.cluster_resources().get("CPU", 0) < required_cpus:
            raise ValueError(f"Worker pools require {required_cpus} Ray CPUs; reduce execution worker counts")
        for kind, cls, resources, stages in definitions:
            self.slots.append({"kind": kind, "class": cls, "resources": resources, "stages": stages,
                               "actor": self._actor(cls, resources), "busy": False, "last_stage": None})

    def _actor(self, cls, resources):
        return self.ray.remote(cls).options(**resources, max_restarts=0, max_task_retries=0).remote(self.config)

    def available(self, kind, stage=None):
        pool = "write" if kind in ("write", "finalize") else ("gpu" if stage in GPU_STAGES else "cpu") if kind == "compute" else kind
        return [i for i, slot in enumerate(self.slots) if slot["kind"] == pool and not slot["busy"]
                and (stage is None or stage in slot["stages"])]

    def stage_priority(self, stage):
        """Prefer work that reuses a model already warm on an idle worker."""
        return 0 if any(self.slots[i]["last_stage"] == stage for i in self.available("compute", stage)) else 1

    def submit(self, kind, job, prepared=None, inputs=None, result=None):
        candidates = self.available(kind, job.get("stage") if kind == "compute" else None)
        if not candidates:
            raise RuntimeError("No idle worker")
        index = min(candidates, key=lambda i: self.slots[i]["last_stage"] != job.get("stage"))
        slot = self.slots[index]; slot["busy"] = True
        if kind == "prepare":
            ref = slot["actor"].prepare.remote(job)
        elif kind == "compute":
            ref, result = slot["actor"].compute.options(num_returns=2).remote(job, prepared, inputs)
        elif kind == "write":
            ref = slot["actor"].write_stage.remote(job, result)
        else:
            ref = slot["actor"].finalize.remote(job, inputs, result)
        self.pending[ref] = {"kind": kind, "job": job, "slot": index, "started": time.monotonic(), "result": result}
        return ref

    def poll(self, timeout):
        if not self.pending:
            return []
        done, _ = self.ray.wait(list(self.pending), num_returns=1, timeout=timeout)
        now = time.monotonic()
        expired = [ref for ref, task in self.pending.items() if now - task["started"] > self.config["execution"]["stage_timeout_s"]]
        events = []
        for ref in set(done) | set(expired):
            task = self.pending.pop(ref); slot = self.slots[task["slot"]]
            slot["busy"] = False; slot["last_stage"] = task["job"].get("stage")
            error, value = None, None
            if ref in expired:
                error = TimeoutError(f"{task['kind']} exceeded stage_timeout_s")
                if task["kind"] == "write":
                    value = task["result"]
                self.ray.kill(slot["actor"], no_restart=True)
                slot["actor"] = self._actor(slot["class"], slot["resources"])
            elif task["kind"] == "compute":
                try:
                    self.ray.get(ref)  # small status return; never the model payload
                    value = task["result"]
                except Exception as exc:
                    error = exc
                    if self._unhealthy(exc):
                        self.ray.kill(slot["actor"], no_restart=True)
                        slot["actor"] = self._actor(slot["class"], slot["resources"])
            else:
                try:
                    value = self.ray.get(ref)
                except Exception as exc:
                    error = exc
                    if task["kind"] == "write":
                        value = task["result"]
                        try:
                            self.ray.get(task["result"])  # error path only; hot path never fetches model data
                        except Exception as source_error:
                            task["job"]["_retry_compute"] = True
                            error = source_error
                    # An error dependency must not kill a healthy writer; replace only dead actors.
                    if not isinstance(exc, self.ray.exceptions.RayTaskError) and any(name in type(exc).__name__ for name in ("ActorDied", "ActorUnavailable")):
                        slot["actor"] = self._actor(slot["class"], slot["resources"])
            events.append((task["kind"], task["job"], value, error, now - task["started"]))
        return events

    @staticmethod
    def _unhealthy(error):
        return (any(name in type(error).__name__ for name in ("ActorDied", "ActorUnavailable"))
                or any(text in str(error).lower() for text in ("actor died", "cuda", "out of memory")))

    def count(self, kind):
        return sum(task["kind"] == kind for task in self.pending.values())

    def close(self):
        for slot in self.slots:
            try:
                self.ray.kill(slot["actor"], no_restart=True)
            except Exception:
                pass
