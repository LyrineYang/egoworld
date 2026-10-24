"""Read-only checks usable before installing any GPU dependencies."""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import sys
import zipfile
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
import xml.etree.ElementTree as ET

from egoworld.config import load_config


def checkpoint_error(path: str) -> str | None:
    file = Path(path)
    if not file.is_file():
        return f"Missing checkpoint: {file}"
    with file.open("rb") as handle:
        header = handle.read(4)
    if header != b"PK\x03\x04" or not zipfile.is_zipfile(file):
        return f"Checkpoint is not a complete PyTorch ZIP archive; download again: {file}"
    # Check the central directory only: do not load pickle or allocate model tensors.
    with zipfile.ZipFile(file) as archive:
        if not any(name.endswith("data.pkl") for name in archive.namelist()):
            return f"Checkpoint has no PyTorch tensor metadata: {file}"
    return None


def asset_report(config, *, videos=None, clips=None, stages=None) -> dict:
    """Static startup checks: no tensor loading, CUDA or model imports."""
    import numpy as np
    from egoworld.contracts import enabled_stages
    checks = []
    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    def file(name, value):
        path = Path(value)
        ok = path.is_file() and path.stat().st_size > 0 and os.access(path, os.R_OK)
        check(name, ok, path)
        return ok
    for name in ("output_root", "manifest_path", "state_db_path"):
        path = Path(getattr(config.paths, name))
        if name == "state_db_path":
            path = path.parent
        while not path.exists():
            path = path.parent
        check(f"Writable {name} parent", path.is_dir() and os.access(path, os.W_OK), path)
    for stage in enabled_stages(config):
        if stages is not None and stage not in stages:
            continue
        params = getattr(config.operators, stage).params
        if params.get("working_dir"):
            check(f"{stage} working directory", Path(params["working_dir"]).is_dir(), params["working_dir"])
        if stage == "sam2":
            prompts = params.get("prompting", {})
            for key, value in (("SAM2", params.get("checkpoint", "")), ("GroundingDINO", prompts.get("gd_checkpoint", ""))):
                if file(f"{key} checkpoint", value):
                    error = checkpoint_error(value)
                    check(f"{key} archive", error is None, error or value)
            file("SAM2 config", params.get("config", ""))
            file("GroundingDINO config", prompts.get("gd_config", ""))
        elif stage == "hamer":
            checkpoint = Path(params["checkpoint"])
            file("HaMeR checkpoint", checkpoint)
            file("HaMeR model config", checkpoint.parent.parent / "model_config.yaml")
            file("MANO right hand", Path(params.get("working_dir", Path.cwd())) / "_DATA/data/mano/MANO_RIGHT.pkl")
        elif stage == "foundationpose":
            check("FoundationPose configured objects", bool(params["objects"]), "at least one label/mesh is required")
            repo = Path(params.get("repo_path", ""))
            file("FoundationPose source", repo / "estimater.py")
            for version in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
                for name in ("model_best.pth", "config.yml"):
                    file(f"FoundationPose {version}/{name}", repo / "weights" / version / name)
            for label, spec in params["objects"].items():
                file(f"{label} mesh", spec["mesh_path"])
                scale = float(spec.get("mesh_scale_to_meters", 0))
                check(f"{label} mesh scale", np.isfinite(scale) and scale > 0, scale)
            if file("RGB-D assets manifest", params["assets_manifest"]):
                try:
                    from egoworld.operators.common import rgbd_assets
                    assets = rgbd_assets(params["assets_manifest"])
                    targets = {v["video_id"]: v for v in videos} if videos is not None else {key: {} for key in assets}
                    if not targets:
                        raise ValueError("RGB-D assets manifest is empty")
                    for video_id, video in targets.items():
                        item = assets[video_id]
                        matrix = np.asarray(item["intrinsics"], dtype=float).reshape(3, 3)
                        scale = float(item["depth_scale_to_meters"])
                        if not (np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0 and np.allclose(matrix[2], [0, 0, 1]) and np.isfinite(scale) and scale > 0):
                            raise ValueError(f"Invalid calibration/scale: {video_id}")
                        first = min((c["frame_start"] for c in clips or [] if c["video_id"] == video_id), default=0)
                        depth = item["depth_pattern"].format(frame_index=first)
                        if not file(f"{video_id} first requested depth", depth):
                            continue
                        import cv2
                        sample = np.load(depth, allow_pickle=False, mmap_mode="r") if depth.endswith(".npy") else cv2.imread(depth, cv2.IMREAD_UNCHANGED)
                        expected = (video.get("height"), video.get("width"))
                        valid = sample is not None and sample.ndim == 2 and np.isfinite(sample).all() and np.all(sample >= 0) and np.any(sample > 0)
                        check(f"{video_id} depth sample", valid and (not all(expected) or sample.shape == expected), depth)
                    check("RGB-D manifest calibration", True, f"{len(targets)} videos; first requested depth sampled")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    check("RGB-D manifest calibration", False, exc)
        elif stage == "dex_retarget":
            try:
                rotation = np.asarray(params["human_to_robot_rotation"], dtype=float).reshape(3, 3)
                check("Robot axes", np.allclose(rotation.T @ rotation, np.eye(3)) and np.linalg.det(rotation) > .99, "proper 3x3 rotation")
                units = params["joint_units"]
                check("Robot side/units", params["handedness"] in ("left", "right") and bool(units) and not (set([units] if isinstance(units, str) else units) - {"radians", "meters"}))
                if file("Robot retarget YAML", params["config"]):
                    import yaml
                    data = yaml.safe_load(Path(params["config"]).read_text())["retargeting"]
                    urdf = Path(data["urdf_path"])
                    if not urdf.is_absolute():
                        urdf = Path(params["urdf_dir"]) / urdf
                    if file("Robot URDF", urdf):
                        check("Robot URDF XML", ET.parse(urdf).getroot().tag == "robot", urdf)
            except (OSError, ValueError, KeyError, TypeError, ImportError, ET.ParseError) as exc:
                check("Robot assets", False, exc)
        elif stage == "fast3r":
            root = Path(params["model_name_or_path"])
            file("Fast3R local config", root / "config.json")
            weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
            check("Fast3R local weights", bool(weights) and all(p.stat().st_size > 0 for p in weights), root)
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def runtime_report(config_path: str, *, gpu: bool = False, stages: list[str] | None = None,
                   assets: bool = False, external: bool = True, coordinator: bool = True) -> dict:
    config = load_config(config_path).resolved()
    from egoworld.contracts import enabled_stages, GPU_STAGES
    selected = set(stages) if stages else set(enabled_stages(config))
    if selected - set(enabled_stages(config)):
        raise ValueError("Doctor stages must be enabled in the config")
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for binary in ("ffmpeg", "ffprobe"):
        check(binary, shutil.which(binary) is not None, shutil.which(binary) or "not on PATH")
    modules = ["numpy", "pyarrow", "cv2"]
    if gpu:
        if coordinator:
            modules += ["ray"]
        if selected & GPU_STAGES:
            modules += ["torch"]
    imported = {}
    for name in modules:
        try:
            imported[name] = importlib.import_module(name)
            check(name, True, getattr(imported[name], "__version__", "importable"))
        except Exception as exc:
            check(name, False, str(exc))
    if gpu:
        torch = imported.get("torch")
        if torch and "sam2" in selected and (not external or not config.operators.sam2.params.get("runner_command")):
            version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:3])
            check("Torch >= 2.5.1", version >= (2, 5, 1), torch.__version__)
        count = torch.cuda.device_count() if torch else 0
        if selected & GPU_STAGES:
            requested = config.num_gpus if coordinator else 1
            check("CUDA devices", count >= requested, f"requested={requested}, visible={count}")
        if torch and count:
            check("GPU", True, torch.cuda.get_device_name(0))
            precision = config.operators.sam2.params.get("precision", "fp16") if "sam2" in selected else "fp32"
            if precision == "bf16":
                check("bf16", torch.cuda.is_bf16_supported(), "use precision=fp16 if unavailable")
        native_modules = {
            "sam2": ["sam2.build_sam", "groundingdino.util.inference", "decord"],
            "hamer": ["hamer.models", "hamer.datasets.vitdet_dataset"],
            "foundationpose": ["estimater", "nvdiffrast.torch", "trimesh"],
            "dex_retarget": ["dex_retargeting.retargeting_config"],
            "fast3r": ["fast3r.models.fast3r", "fast3r.dust3r.inference_multiview", "fast3r.dust3r.cloud_opt.init_im_poses"],
        }
        for stage in sorted(selected):
            params = getattr(config.operators, stage).params
            if external and (command := params.get("runner_command")):
                valid = isinstance(command, list) and bool(command) and all(isinstance(arg, str) for arg in command)
                try:
                    if not valid or shutil.which(command[0]) is None:
                        raise ValueError("runner_command must start with an available Python executable")
                    with tempfile.TemporaryDirectory(prefix="egoworld-doctor-") as folder:
                        snapshot = Path(folder) / "config.json"
                        snapshot.write_text(json.dumps(asdict(config)))
                        result = subprocess.run(command + ["--doctor", "--config", str(snapshot), "--stage", stage],
                            cwd=params.get("working_dir"), capture_output=True, text=True,
                            timeout=min(60, config.execution.stage_timeout_s))
                    report = json.loads(result.stdout)
                    check(f"{stage} external environment", result.returncode == 0 and report["ok"],
                          result.stderr[-2000:] or "native imports checked inside runner environment")
                    checks[-1]["report"] = report
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                    check(f"{stage} external environment", False, str(exc))
                continue
            if stage == "foundationpose" and params.get("repo_path"):
                sys.path.insert(0, str(Path(params["repo_path"]).resolve()))
            for name in native_modules[stage]:
                try:
                    importlib.import_module(name)
                    check(name, True, "native API importable")
                except Exception as exc:
                    check(name, False, str(exc))
            files = []
            if stage == "sam2":
                prompts = params.get("prompting", {})
                for name, path in (("SAM2 checkpoint", params.get("checkpoint", "")),
                                   ("GroundingDINO checkpoint", prompts.get("gd_checkpoint", ""))):
                    error = checkpoint_error(path)
                    check(name, error is None, error or path)
                files.append(("GroundingDINO config", prompts.get("gd_config", "")))
                from egoworld.operators.sam2_op import _resolve_model_cfg
                try:
                    path = _resolve_model_cfg(params.get("config"))
                    check("SAM2 config", bool(path and Path(path).is_file()), path or "missing")
                except Exception as exc:
                    check("SAM2 config", False, str(exc))
            elif stage == "hamer":
                files.append(("HaMeR checkpoint", params["checkpoint"]))
            elif stage == "foundationpose":
                files.append(("RGB-D asset manifest", params["assets_manifest"]))
                for label, spec in params["objects"].items():
                    files.append((f"{label} mesh", spec.get("mesh_path", "")))
            elif stage == "dex_retarget":
                files.append(("Robot retarget config", params["config"]))
            elif stage == "fast3r":
                path = Path(params["model_name_or_path"])
                local = path.is_dir() and (path / "config.json").is_file() and (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")))
                check("Fast3R pinned local snapshot", local, str(path))
            for name, path in files:
                check(name, Path(path).is_file() and Path(path).stat().st_size > 0, path)
    if assets:
        checks.extend(asset_report(config, stages=selected)["checks"])
    return {"mode": "gpu" if gpu else "cpu", "stages": sorted(selected),
            "ok": all(item["ok"] for item in checks), "checks": checks}
