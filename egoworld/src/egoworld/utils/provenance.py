"""Automatic identities for code, input manifests and configured local model assets."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
from egoworld.contracts import OUTPUT_SCHEMA_VERSION
from egoworld.utils.hashing import sha256_file


def provenance(config, videos, clips):
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        digest.update(str(file.relative_to(package)).encode()); digest.update(file.read_bytes())
    assets = {}
    for stage, operator in asdict(config.operators).items():
        if not operator["enabled"]:
            continue
        def visit(value, key=""):
            if isinstance(value, dict):
                for name, nested in value.items():
                    visit(nested, f"{key}.{name}")
            elif isinstance(value, list):
                for i, nested in enumerate(value):
                    visit(nested, f"{key}[{i}]")
            elif isinstance(value, str) and Path(value).is_file():
                assets[f"{stage}{key}"] = {"path": str(Path(value).resolve()), "sha256": sha256_file(value)}
            elif isinstance(value, str) and Path(value).is_dir() and key.endswith("model_name_or_path"):
                for file in sorted(Path(value).rglob("*")):
                    if file.is_file() and file.suffix in (".safetensors", ".bin", ".json"):
                        assets[str(file.resolve())] = {"sha256": sha256_file(str(file))}
        visit(operator["params"])
    try:
        code_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        code_hash = "unavailable"
    identity = {"source_tree_hash": digest.hexdigest(), "git_commit": code_hash, "local_assets": assets,
                "input_manifests": {"videos": sha256_file(videos), "clips": sha256_file(clips)}}
    behavior = {"operators": asdict(config.operators), "coordinates": asdict(config.coordinates),
                "metrics": asdict(config.metrics), "dataset_hash": config.dataset_hash,
                "model_versions": config.model_versions, "schema": OUTPUT_SCHEMA_VERSION, **identity}
    identity["fingerprint"] = hashlib.sha256(json.dumps(behavior, sort_keys=True).encode()).hexdigest()
    return identity
