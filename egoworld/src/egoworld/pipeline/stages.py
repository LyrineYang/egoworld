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

OPERATORS = {"sam2": Sam2Operator, "hamer": HamerOperator, "foundationpose": FoundationPoseOperator,
             "dex_retarget": DexRetargetOperator, "fast3r": Fast3ROperator}


class PrepareActor:
    def __init__(self, config):
        self.config = config

    def prepare(self, job):
        clip = job["clip"]
        target = Path(job["work_dir"]) / "clip.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
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
        return {"video_path": str(target), "frames": info.frame_count}


class ModelActor:
    def __init__(self, config):
        self.config = config
        self.cache = OrderedDict()

    def compute(self, job, prepared, inputs):
        """Ray exposes status and payload as separate returns; the driver gets status only."""
        return None, self.process(job, prepared, inputs)

    def process(self, job, prepared, inputs):
        previous = Path.cwd()
        working_dir = self.config["operators"][job["stage"]]["params"].get("working_dir")
        try:
            if working_dir:
                os.chdir(working_dir)
            return self._process(job, prepared, inputs)
        finally:
            if working_dir:
                os.chdir(previous)

    def _process(self, job, prepared, inputs):
        stage, clip = job["stage"], job["clip"]
        params = dict(self.config["operators"][stage]["params"])
        command = params.pop("runner_command", None)
        params.pop("working_dir", None)
        started = time.perf_counter()
        if command:
            packet = self._external(command, job, prepared, inputs)
        else:
            if stage not in self.cache:
                while len(self.cache) >= self.config["execution"]["max_cached_models"]:
                    self.cache.popitem(last=False)
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                self.cache[stage] = OPERATORS[stage](**params)
            self.cache.move_to_end(stage)
            operator = self.cache[stage]
            if stage == "sam2":
                result = operator.run(prepared["video_path"], 0, 0)
                frames, instances = result.get("frames", []), result.get("instances", [])
                for row in frames + instances:
                    row["frame_index"] += clip["frame_start"]
                    row["timestamp_s"] = row["frame_index"] / clip["fps"]
                tracks = [{k: row[k] for k in ("frame_index", "timestamp_s", "object_id", "label", "area_px")} |
                          {"center_xy": [(row["bbox_xyxy"][0] + row["bbox_xyxy"][2]) / 2,
                                         (row["bbox_xyxy"][1] + row["bbox_xyxy"][3]) / 2] if row["bbox_xyxy"] else None}
                          for row in instances]
                packet = {"tables": {"masks": frames, "instances": instances, "tracks": tracks},
                          "stats": {"empty_mask_rate": result.get("empty_mask_rate", 1.),
                                    "processed_frames": result.get("processed_frames", len(frames))}}
            else:
                packet = operator.run(clip, prepared["video_path"], inputs)
        if set(packet.get("tables", {})) != set(STAGE_ARTIFACTS[stage]):
            raise ValueError(f"{stage} returned an incomplete/unexpected output contract")
        for name, data in packet["tables"].items():
            validate_rows(name, data, clip)
        packet.setdefault("stats", {})["inference_seconds"] = time.perf_counter() - started
        return packet

    def _external(self, command, job, prepared, inputs):
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise ValueError("runner_command must be a nonempty argv list; shell execution is not supported")
        folder = Path(job["work_dir"]) / job["stage"]
        folder.mkdir(parents=True, exist_ok=True)
        request, response = folder / "request.json", folder / "response.json"
        response.unlink(missing_ok=True)
        request.write_text(json.dumps({"config": self.config, "job": job, "prepared": prepared, "inputs": inputs}))
        with (folder / "runner.log").open("wb") as log:
            # The guard watches this pipe. Actor death closes it even on SIGKILL,
            # so cleanup does not depend on a finally block in the dead actor.
            process = subprocess.Popen([sys.executable, "-m", "egoworld.utils.process_guard", "--"] +
                                       command + ["--request", str(request), "--response", str(response)],
                                       stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
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
            if "error" in reply:
                from egoworld.utils.errors import RetryableError
                detail = reply["error"]
                error_class = RetryableError if detail.get("retryable") else ValueError
                raise error_class(f"External {job['stage']} {detail['type']}: {detail['message']}; see {folder / 'runner.log'}")
            if process.returncode:
                raise RuntimeError(f"External {job['stage']} runner exited {process.returncode}; see {folder / 'runner.log'}")
        if not reply:
            raise ValueError(f"External runner wrote no response; see {folder / 'runner.log'}")
        return reply
