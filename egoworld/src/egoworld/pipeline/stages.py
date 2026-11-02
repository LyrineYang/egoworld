"""Model-independent stage registry and worker. Large results go directly to Writer."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import gc
import json
import shutil
import subprocess
import os
import signal
import sys
import time

from egoworld.contracts import STAGE_ARTIFACTS, validate_rows
from egoworld.operators.sam2_op import Sam2Operator, _extract_clip
from egoworld.operators.hamer_op import HamerOperator
from egoworld.operators.foundationpose_op import FoundationPoseOperator
from egoworld.operators.dex_retarget_op import DexRetargetOperator
from egoworld.operators.fast3r_op import Fast3ROperator
from egoworld.utils.video import get_video_info
from egoworld.utils.performance import packet_bytes, process_peak_rss_bytes, cuda_metrics, model_load_seconds

OPERATORS = {"sam2": Sam2Operator, "hamer": HamerOperator, "foundationpose": FoundationPoseOperator,
             "dex_retarget": DexRetargetOperator, "fast3r": Fast3ROperator}


class PrepareActor:
    def __init__(self, config):
        self.config = config

    def prepare(self, job):
        from egoworld.utils.timing import profile_session, phase
        identity = {"invocation_id": job.get("invocation_id"), "clip_id": job["clip"]["clip_id"], "operation": "video_prepare"}
        with profile_session(self.config.get("execution", {}).get("phase_profiling", False), identity) as profile:
            with phase("video_prepare"):
                result = self._prepare(job)
            if profile:
                result["profile"] = profile.finish()
                result["stats"] = dict(profile.stats)
            return result

    def _prepare(self, job):
        clip = job["clip"]
        target = Path(job["work_dir"]) / "clip.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            from egoworld.utils.timing import phase
            with phase("video_extract"):
                extracted = _extract_clip(clip["video_path"], clip["start_s"], clip["end_s"])
            try:
                shutil.move(extracted, str(target) + ".part")
                Path(str(target) + ".part").replace(target)
            except Exception:
                Path(extracted).unlink(missing_ok=True)
                raise
        info = get_video_info(str(target))
        if info.frame_count != clip["frame_end"] - clip["frame_start"] or abs(info.fps - clip["fps"]) > .01:
            target.unlink(missing_ok=True)
            raise ValueError("Prepared clip frame count/FPS differs from manifest; normalize to constant FPS")
        result = {"video_path": str(target), "frames": info.frame_count, "prepared_bytes": target.stat().st_size}
        from egoworld.pipeline.input_cache import prepare_cache
        spec = prepare_cache(self.config, job, target)
        if spec:
            result["input_cache"] = spec
        return result


class ModelActor:
    def __init__(self, config):
        self.config = config
        self.cache = OrderedDict()
        self.session = None
        import cv2
        import pyarrow as pa
        threads = config["execution"].get("worker_threads", 1)
        cv2.setNumThreads(threads)
        pa.set_cpu_count(threads)
        pa.set_io_thread_count(threads)

    def compute(self, job, prepared, inputs, result=None):
        """Ray exposes status and payload as separate returns; the driver gets status only."""
        if result is not None:
            prepared = dict(prepared, _phase_input=result)
        packet = self.process(job, prepared, inputs)
        return {"result_bytes": packet["stats"]["result_bytes"], "work_units": packet["stats"].get("work_units")}, packet

    def close(self):
        if self.session:
            self.session.close()
            self.session = None
        self._release_models()

    def _release_models(self, keep=0):
        if len(self.cache) <= keep:
            return
        while len(self.cache) > keep:
            self.cache.popitem(last=False)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def process(self, job, prepared, inputs):
        from egoworld.product.gates import branch_admission
        filtered, inputs = branch_admission(self.config, job, prepared, inputs)
        if filtered is not None:
            return filtered
        previous = Path.cwd()
        working_dir = self.config["operators"][job["stage"]]["params"].get("working_dir")
        try:
            if working_dir:
                os.chdir(working_dir)
            from egoworld.pipeline.input_cache import cache_session
            with cache_session(prepared.get("input_cache")) as cache:
                from egoworld.utils.timing import profile_session, phase
                identity = {"invocation_id": job.get("invocation_id"), "clip_id": job["clip"].get("clip_id"),
                            "stage": job["stage"], "operation": job.get("_operation", "stage"),
                            "trial": job.get("trial"), "attempt": job.get("attempt", 0)}
                with profile_session(self.config.get("execution", {}).get("phase_profiling", False), identity) as profile:
                    try:
                        with phase("worker_request", cuda=job.get("_operation") not in ("input", "post")):
                            packet = self._process(job, prepared, inputs)
                    except BaseException:
                        if profile:
                            from egoworld.io.writers import write_json
                            from uuid import uuid4
                            file = Path(job["work_dir"]) / "phase-failures" / f"{uuid4().hex}.json"
                            try:
                                write_json(str(file), profile.finish())
                            except Exception:
                                pass  # profiling must never hide the original model error
                        raise
                    if profile:
                        observation = profile.finish()
                        packet.setdefault("profiles", []).append(observation)
                        for key, value in observation["stats"].items():
                            packet["stats"][key] = packet["stats"].get(key, 0) + value
                if cache is not None:
                    for key, value in cache.stats.items():
                        packet["stats"].setdefault(key, value)
                packet["stats"]["result_bytes"] = max(packet["stats"].get("result_bytes", 0), packet_bytes(packet))
                return packet
        finally:
            if working_dir:
                os.chdir(previous)

    def _process(self, job, prepared, inputs):
        stage, clip = job["stage"], job["clip"]
        params = dict(self.config["operators"][stage]["params"])
        command = params.pop("runner_command", None)
        params.pop("working_dir", None)
        started = time.perf_counter()
        cpu_start = time.process_time()
        measure_gpu = self.config["execution"].get("measure_gpu_memory", False) and job.get("_operation") not in ("input", "post")
        if measure_gpu:
            cuda_metrics(reset=True)
        if command:
            # Native weights must not overlap an external runner on the same GPU.
            self._release_models()
            packet = self._external(command, job, prepared, inputs)
            packet.setdefault("stats", {})["runner_roundtrip_seconds"] = packet.get("stats", {}).get("runner_roundtrip_seconds", 0.) + time.perf_counter() - started
        else:
            if self.session:
                self.session.close()
                self.session = None
            cache_hit = stage in self.cache
            if stage not in self.cache:
                self._release_models(self.config["execution"]["max_cached_models"] - 1)
                self.cache[stage] = OPERATORS[stage](**params)
            self.cache.move_to_end(stage)
            operator = self.cache[stage]
            load_before = model_load_seconds(operator)
            if job.get("_operation"):
                from egoworld.pipeline.partitioned import run_phase
                packet = run_phase(operator, job, prepared, inputs, self.config["execution"])
            elif stage == "sam2":
                from egoworld.pipeline.partitioned import sam_packet
                packet = sam_packet(clip, operator.run(prepared["video_path"], 0, 0))
            else:
                packet = operator.run(clip, prepared["video_path"], inputs)
            stats = packet.setdefault("stats", {})
            if job.get("_operation") not in ("input", "post"):
                stats["model_cache_hit"] = int(cache_hit)
                stats["model_load_seconds"] = model_load_seconds(operator) - load_before
        if not packet.get("_handoff") and set(packet.get("tables", {})) != set(STAGE_ARTIFACTS[stage]):
            raise ValueError(f"{stage} returned an incomplete/unexpected output contract")
        for name, data in packet.get("tables", {}).items():
            validate_rows(name, data, clip)
        stats = packet.setdefault("stats", {})
        wall = time.perf_counter() - started
        if job.get("_operation") and not command:
            stats["inference_seconds"] = stats.get("inference_seconds", 0.) + wall
        elif not (command and job.get("_operation")):
            stats["inference_seconds"] = wall  # legacy worker wall-time name
        stats["stage_wall_seconds"] = stats["inference_seconds"]
        if not command:
            stats["process_cpu_seconds"] = stats.get("process_cpu_seconds", 0.) + time.process_time() - cpu_start
        stats["process_peak_rss_bytes"] = max(stats.get("process_peak_rss_bytes", 0), process_peak_rss_bytes())
        stats["result_bytes"] = max(stats.get("result_bytes", 0), packet_bytes(packet))
        if measure_gpu and not command:
            stats.update(cuda_metrics())
        return packet

    def _external(self, command, job, prepared, inputs):
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise ValueError("runner_command must be a nonempty argv list; shell execution is not supported")
        folder = Path(job["work_dir"]) / job["stage"]
        if job.get("_operation"):
            folder /= job["_operation"]
        folder.mkdir(parents=True, exist_ok=True)
        request, response = folder / "request.json", folder / "response.json"
        response.unlink(missing_ok=True)
        request.write_text(json.dumps({"config": self.config, "job": job, "prepared": prepared, "inputs": inputs}))
        mode = self.config["execution"].get("external_runner_mode", "auto")
        builtin = any(command[i:i + 2] == ["-m", "egoworld.worker"] for i in range(len(command) - 1))
        if mode == "persistent" or (mode == "auto" and builtin):
            from egoworld.pipeline.external import RunnerSession
            key = (tuple(command), job["stage"])
            if self.session and self.session.key != key:
                self.session.close()
                self.session = None
            if not self.session:
                self.session = RunnerSession(command, job["stage"], Path(job["work_dir"]).parent / ".runners",
                                             self.config["execution"].get("worker_threads", 1),
                                             environment={"CUDA_VISIBLE_DEVICES": ""} if job.get("_operation") in ("input", "post") else None)
            (folder / "runner.log").write_text(f"Persistent runner log: {self.session.log_path}\n")
            try:
                reply = self.session.run(request.resolve(), response.resolve(), self.config["execution"]["stage_timeout_s"] * .9)
                if "error" in reply:
                    self._runner_error(reply, folder)
                return reply
            except BaseException:
                self.session.close()
                self.session = None
                raise
        if self.session:
            self.session.close()
            self.session = None
        from egoworld.pipeline.external import runner_environment
        with (folder / "runner.log").open("wb") as log:
            # The guard watches this pipe. Actor death closes it even on SIGKILL,
            # so cleanup does not depend on a finally block in the dead actor.
            process = subprocess.Popen([sys.executable, "-m", "egoworld.utils.process_guard", "--"] +
                                       command + ["--request", str(request), "--response", str(response)],
                                       stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env=runner_environment(self.config["execution"].get("worker_threads", 1)))
            try:
                # Leave time for group cleanup before the actor watchdog expires.
                process.wait(timeout=self.config["execution"]["stage_timeout_s"] * .9)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"External {job['stage']} runner timed out; process group killed") from exc
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            reply = json.loads(response.read_text()) if response.is_file() else {}
            self._runner_error(reply, folder)
            if process.returncode:
                raise RuntimeError(f"External {job['stage']} runner exited {process.returncode}; see {folder / 'runner.log'}")
        if not reply:
            raise ValueError(f"External runner wrote no response; see {folder / 'runner.log'}")
        return reply

    @staticmethod
    def _runner_error(reply, folder):
        if "error" in reply:
            from egoworld.utils.errors import RetryableError
            detail = reply["error"]
            error_class = RetryableError if detail.get("retryable") else ValueError
            raise error_class(f"External {detail['type']}: {detail['message']}; see {folder / 'runner.log'}")
