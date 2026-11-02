"""Verified stage cache, restored through the existing stage receipt protocol.

No hardlinks, guessed compatibility or model-name-only keys. Runtime/package
and source identities are intentionally conservative. A code change can miss
the cache even when a more detailed dependency analysis might allow reuse.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from egoworld.contracts import OUTPUT_SCHEMA_VERSION
from egoworld.io.writers import write_json
from egoworld.pipeline.artifacts import load_stage
from egoworld.utils.hashing import sha256_file
from egoworld.product.identity import source_identity, semantic_params
from egoworld.product.store import initialize, locked


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_identity(command):
    executable = command[0]
    script = """import sys,json,hashlib,importlib.metadata as m,importlib.util as u
from pathlib import Path
sources={}
for name in ('sam2','groundingdino','hamer','fast3r','dex_retargeting'):
    spec=u.find_spec(name)
    if spec and spec.origin:
        root=Path(spec.origin).parent
        h=hashlib.sha256()
        for p in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json')):
            h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
        sources[name]=h.hexdigest()
print(json.dumps({'python':sys.version,'packages':sorted((d.metadata.get('Name',''),d.version) for d in m.distributions()),'model_sources':sources}))
"""
    result = subprocess.run([executable, "-c", script], text=True, capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def directory_identity(path):
    root = Path(path)
    return {str(p.relative_to(root)): sha256_file(str(p)) for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}


def stage_identity(config, provenance, stage, clip, upstream, runtime):
    params = asdict(getattr(config.operators, stage))["params"]
    assets = {key: value for key, value in provenance["local_assets"].items() if key.startswith(stage + ".")}
    if stage == "fast3r":
        model_root = Path(params["model_name_or_path"])
        if model_root.is_dir():
            assets.update({key: value for key, value in provenance["local_assets"].items()
                           if Path(key).is_relative_to(model_root)})
    extras = {}
    if stage == "dex_retarget" and params.get("urdf_dir"):
        extras["robot_assets"] = directory_identity(params["urdf_dir"])
    if stage == "foundationpose":
        from egoworld.operators.common import rgbd_assets
        item = rgbd_assets(params["assets_manifest"])[clip["video_id"]]
        extras["depth_frames"] = {str(i): sha256_file(item["depth_pattern"].format(frame_index=i))
            for i in range(clip["frame_start"], clip["frame_end"])}
    for key in ("working_dir", "repo_path"):
        if params.get(key):
            repo = Path(params[key])
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
                changes = subprocess.check_output(["git", "diff", "HEAD", "--", "*.py", "*.yaml", "*.yml", "*.cpp", "*.cu", "*.h", "*.hpp", "*.json"], cwd=repo)
                extras[key] = {"commit": commit, "tracked_changes_sha256": hashlib.sha256(changes).hexdigest()}
                if stage == "foundationpose":
                    extras["native_libraries"] = {str(p.relative_to(repo)): sha256_file(str(p)) for p in repo.rglob("*.so") if ".git" not in p.parts}
            except subprocess.CalledProcessError:
                extras[key] = directory_identity(repo)
    paths = {str(Path(v["path"]).resolve()): v["sha256"] for v in assets.values() if v.get("path")}
    normalized_assets = {key if key.startswith(stage + ".") else Path(key).name:
                         {k: v for k, v in item.items() if k != "path"} for key, item in assets.items()}
    runtime = dict(runtime)
    packages = {"sam2": ("sam2", "groundingdino"), "hamer": ("hamer",),
                "fast3r": ("fast3r",), "dex_retarget": ("dex_retargeting",)}
    if "model_sources" in runtime:
        runtime["model_sources"] = {k: v for k, v in runtime["model_sources"].items() if k in packages.get(stage, ())}
    return {"version": 2, "schema_version": OUTPUT_SCHEMA_VERSION, "stage": stage,
            "split_execution": getattr(config.execution, "split_execution", False) and stage in getattr(config.execution, "split_stages", []),
            "sources": source_identity(stage), "models": {stage: config.model_versions.get(stage)},
            "params": semantic_params(params, paths), "assets": normalized_assets, "extra_assets": extras, "runtime": runtime,
            "clip": {k: clip[k] for k in ("frame_start", "frame_end", "fps", "video_sha256")},
            "upstream": upstream, "coordinates": asdict(config.coordinates),
            "branch_cleaning": config.extra.get("branch_cleaning") if stage in ("hamer", "foundationpose", "dex_retarget") else None}


def file_signature(receipt):
    return {name: info["sha256"] for name, info in receipt["files"].items()}


class StageCache:
    def __init__(self, root):
        self.root = Path(root)
        initialize(root)

    @locked
    def lookup(self, identity):
        key = digest(identity)
        folder = self.root / identity["stage"] / key
        try:
            entry = json.loads((folder / "entry.json").read_text())
            if entry["identity"] != identity or digest(entry["identity"]) != key:
                return None
            return load_stage(folder, identity["stage"], key, verify_hashes=True)
        except (OSError, ValueError, KeyError):
            return None

    @locked
    def restore(self, identity, destination, fingerprint):
        receipt = self.lookup(identity)
        if receipt is None:
            return None
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in receipt["files"]:
            if Path(name).name != name:
                raise ValueError("Invalid cache artifact filename")
            temporary = destination / f".{name}.{uuid4().hex}.tmp"
            shutil.copyfile(Path(receipt["directory"]) / name, temporary)
            os.replace(temporary, destination / name)
        receipt = {**receipt, "directory": str(destination), "fingerprint": fingerprint,
                   "cache_origin": {"key": digest(identity), "directory": receipt["directory"]}}
        write_json(str(destination / f"stage_{identity['stage']}.json"), receipt)
        restored = load_stage(destination, identity["stage"], fingerprint, True)
        if restored is None:
            raise ValueError("Restored cache output failed verification")
        return restored

    @locked
    def publish(self, identity, source, fingerprint):
        if self.lookup(identity):
            return digest(identity)
        stage, key = identity["stage"], digest(identity)
        receipt = load_stage(source, stage, fingerprint, True)
        if receipt is None:
            raise ValueError("Only complete, hash-verified stages can enter the cache")
        parent = self.root / stage
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{key}.{uuid4().hex}.tmp"
        temporary.mkdir()
        target = parent / key
        try:
            for name in receipt["files"]:
                shutil.copyfile(Path(source) / name, temporary / name)
            cached = {**receipt, "fingerprint": key, "directory": str(target)}
            write_json(str(temporary / f"stage_{stage}.json"), cached)
            write_json(str(temporary / "entry.json"), {"identity": identity, "origin": str(source)})
            if target.exists():
                # A damaged entry is quarantined, never silently trusted.
                if self.lookup(identity):
                    return key
                target.rename(parent / f".{key}.invalid.{uuid4().hex}")
            try:
                temporary.rename(target)
            except OSError:
                if not self.lookup(identity):
                    raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return key

    @locked
    def explain(self, identity):
        key = digest(identity)
        folder = self.root / identity["stage"] / key
        if folder.exists():
            return {"key": key, "reason": "hit" if self.lookup(identity) else "corrupt_entry"}
        parent = folder.parent
        candidates = sorted(parent.glob("*/entry.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:32]
        for candidate in candidates:
            try:
                previous = json.loads(candidate.read_text())["identity"]
                if previous.get("clip", {}).get("video_sha256") == identity["clip"]["video_sha256"]:
                    changed = [k for k in identity if identity[k] != previous.get(k)]
                    return {"key": key, "reason": "identity_changed", "changed": changed}
            except (ValueError, KeyError, OSError):
                continue
        return {"key": key, "reason": "not_found"}
