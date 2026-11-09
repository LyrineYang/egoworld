"""Pilot bounded local pipeline using production stage actors and durable writers."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
from threading import Lock
import time

DEVICES = tuple(os.getenv("EGOWORLD_STUDY_DEVICES", "2,3").split(","))


def stage_worker(config_path, stage, fd, verify):
    from egoworld.operators.sam2_op import _load_prompt_config
    from egoworld.pipeline.artifacts import WriterActor
    from egoworld.pipeline.stages import ModelActor, OPERATORS

    if verify:
        from contextlib import redirect_stdout
        from egoworld.preflight import runtime_report
        with redirect_stdout(sys.stderr):
            report = runtime_report(config_path, gpu=True, stages=[stage], external=False, coordinator=False)
        with os.fdopen(os.dup(fd), "w", buffering=1) as pipe:
            pipe.write(json.dumps(report) + "\n")
        if not report["ok"]:
            raise RuntimeError(f"Native worker readiness failed for {stage}")
    config = json.loads(Path(config_path).read_text())
    params = config["operators"][stage]["params"]
    params.pop("runner_command")
    actor = ModelActor(config)
    writer = WriterActor(config)
    with os.fdopen(fd, "w", buffering=1) as result:
        started = time.perf_counter()
        try:
            if stage in ("sam2", "hamer"):
                operator = OPERATORS[stage](**{k: v for k, v in params.items() if k != "working_dir"})
                if stage == "hamer":
                    operator._load()
                else:
                    operator._ensure_predictor()
                    prompts = _load_prompt_config(params["prompting"])
                    if prompts.source == "groundingdino":
                        operator._ensure_groundingdino(prompts)._ensure_model()
                actor.cache[stage] = operator
            warm_seconds = time.perf_counter() - started
            for line in sys.stdin:
                job, prepared, receipts = json.loads(line)
                started = time.perf_counter()
                packet = actor.process(job, prepared, receipts)
                receipt = writer.write_stage(job, packet)
                result.write(json.dumps({"receipt": receipt, "service_seconds": time.perf_counter() - started,
                                         "preload_seconds": warm_seconds}) + "\n")
        finally:
            actor.close()


class Slot:
    def __init__(self, stage, config_path, params, device, log, verify, decoder_threads=None):
        command = params["runner_command"]
        if command[1:] != ["-m", "egoworld.worker"]:
            raise ValueError("Pilot only supports the standard EgoWorld native stage interpreter")
        read_fd, write_fd = os.pipe()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
        if stage == "sam2" and decoder_threads is not None:
            env["OPENCV_FFMPEG_THREADS"] = str(decoder_threads)
        try:
            with log.open("w") as stream:
                self.process = subprocess.Popen(
                    [command[0], "-B", str(Path(__file__).resolve()), str(config_path), stage, str(write_fd), str(int(verify))],
                    stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, text=True,
                    pass_fds=(write_fd,), cwd=params.get("working_dir"),
                    env=env)
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        self.log = log
        self.device = device
        self.reader = os.fdopen(read_fd)
        self.lock = Lock()
        self.preload_seconds = None

    def read(self, timeout):
        if not select.select([self.reader], [], [], timeout)[0]:
            raise TimeoutError(f"Native stage timed out; see {self.log}")
        response = self.reader.readline()
        if not response:
            raise RuntimeError(f"Native stage exited before committing; see {self.log}")
        return json.loads(response)

    def request(self, job, prepared, receipts, timeout):
        queued = time.perf_counter()
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"Native stage exited; see {self.log}")
            self.process.stdin.write(json.dumps([job, prepared, receipts]) + "\n")
            self.process.stdin.flush()
            report = self.read(timeout)
            self.preload_seconds = report["preload_seconds"]
        return report["receipt"], {"service_seconds": report["service_seconds"],
                                    "queue_inclusive_seconds": time.perf_counter() - queued}

    def close(self):
        try:
            self.process.stdin.close()
            try:
                status = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                status = self.process.wait()
            if status:
                raise RuntimeError(f"Native stage failed; see {self.log}")
        finally:
            self.reader.close()


def pipeline(config_path, videos, clips, *, run_id, layout, backend_factory=None, sam2_decoder_threads=None):
    from egoworld.config import load_config
    from egoworld.contracts import OUTPUT_SCHEMA_VERSION, enabled_stages
    from egoworld.pipeline.artifacts import WriterActor
    from egoworld.pipeline.driver import _build_clip_tasks, _clip_to_dict, load_manifest
    from egoworld.pipeline.input_cache import release_prepared_cache
    from egoworld.pipeline.stages import PrepareActor
    from egoworld.utils.provenance import provenance
    from product_comparison import write

    started = time.perf_counter()
    config = load_config(config_path).resolved()
    graph = list(enabled_stages(config))
    fused_checks = layout.startswith("stream-checked-")
    if (fused_checks != bool(backend_factory) or config.num_gpus != 2 or len(set(DEVICES)) != 2
            or len(DEVICES) != 2 or graph != ["sam2", "hamer", "dex_retarget"]):
        raise ValueError("Pilot supports two local GPUs and the frozen hand DAG")
    payload = asdict(config)
    root = Path(config.paths.output_root) / f"run_id={run_id}"
    root.mkdir(exist_ok=True)
    work = root / "local-stream"
    work.mkdir()
    video_index = {video["video_id"]: video for video in load_manifest(videos)}
    tasks = [_clip_to_dict(task) for task in _build_clip_tasks(load_manifest(clips), video_index)]
    for clip in tasks:
        clip["fps"] = float(video_index[clip["video_id"]]["fps"])
        clip["video_sha256"] = video_index[clip["video_id"]]["checksum"]
    identity = provenance(config, videos, clips)
    write(root / "run_manifest.json", {**config.to_run_manifest(), "config_fingerprint": identity["fingerprint"],
          "provenance": identity, "config": payload, "stages": graph,
          "output_schema_version": OUTPUT_SCHEMA_VERSION, "executor": layout})
    if layout in ("stream-pinned", "stream-checked-pinned"):
        placements = {"sam2": (DEVICES[0],), "hamer": (DEVICES[1],), "dex_retarget": ("",)}
    elif layout in ("stream-dual", "stream-checked-dual"):
        placements = {"sam2": DEVICES, "hamer": DEVICES, "dex_retarget": ("",)}
    else:
        raise ValueError(f"Unknown streaming layout {layout}")
    receipts = []
    with ExitStack() as stack:
        slots = {}
        for stage, devices in placements.items():
            slots[stage] = []
            for index, device in enumerate(devices):
                slot = Slot(stage, config_path, getattr(config.operators, stage).params, device,
                            work / f"{stage}-{index}.log", fused_checks, sam2_decoder_threads)
                stack.callback(slot.close)
                slots[stage].append(slot)
        if fused_checks:
            from egoworld.preflight import runtime_report
            static = runtime_report(config_path, assets=True)
            runtime = [{"stage": stage, "device": slot.device, **slot.read(90)}
                       for stage, group in slots.items() for slot in group]
            ready = static["ok"] and all(row["ok"] for row in runtime)
            write(Path(config_path).parent / "readiness.json",
                  {"ok": ready, "scope": "Static coordinator and actual pinned model worker interpreters",
                   "coordinator": static, "workers": runtime})
            if not ready:
                raise RuntimeError("Static assets or native pinned workers failed readiness")
        prepare = PrepareActor(payload)
        writer = WriterActor(payload)

        def execute(item):
            number, clip = item
            job = {"clip": clip, "work_dir": str(root / ".work" / clip["clip_id"]),
                   "directory": str(root / f"video_id={clip['video_id']}" / f"clip_id={clip['clip_id']}"),
                   "fingerprint": identity["fingerprint"]}
            prepared = prepare.prepare(job)
            receipts, observations = {}, {}
            try:
                for stage in graph:
                    group = slots[stage]
                    receipts[stage], observations[stage] = group[number % len(group)].request(
                        {**job, "stage": stage}, prepared, receipts, config.execution.stage_timeout_s)
                writer.finalize(job, receipts, {})
                return {"clip_id": clip["clip_id"], "committed_seconds": time.perf_counter() - started,
                        "stage_observations": observations}
            finally:
                release_prepared_cache(prepared)
                if not config.execution.keep_prepared:
                    shutil.rmtree(job["work_dir"], ignore_errors=True)

        active = min(config.execution.max_active_clips, len(tasks))
        with ThreadPoolExecutor(max_workers=active) as pool:
            remaining = iter(enumerate(tasks))
            pending = {pool.submit(execute, next(remaining)) for _ in range(active)}
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    receipts.append(future.result())
                    item = next(remaining, None)
                    if item is not None:
                        pending.add(pool.submit(execute, item))
    preload = {stage: [{"device": slot.device, "seconds": slot.preload_seconds} for slot in group]
               for stage, group in slots.items()}
    write(work / "execution.json", {"wall_seconds": time.perf_counter() - started,
          "sam2_decoder_threads": sam2_decoder_threads,
          "model_preload_seconds": preload, "clip_commits": sorted(receipts, key=lambda r: r["committed_seconds"])})
    return run_id


if __name__ == "__main__":
    _, config, stage, fd, verify = sys.argv
    stage_worker(config, stage, int(fd), bool(int(verify)))
