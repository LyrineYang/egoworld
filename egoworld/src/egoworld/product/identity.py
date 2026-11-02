"""Content identities for the code actually imported by a model adapter."""
import ast
import hashlib
from pathlib import Path


def source_identity(stage):
    root = Path(__file__).resolve().parents[1]
    pending = [f"operators/{stage}_op.py", "contracts.py", "io/writers.py"]
    if stage in ("hamer", "foundationpose", "dex_retarget"):
        pending.append("product/gates.py")
    seen = {}
    while pending:
        relative = pending.pop()
        path = root / relative
        if relative in seen or not path.is_file():
            continue
        content = path.read_bytes()
        seen[relative] = hashlib.sha256(content).hexdigest()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    parent = list(Path(relative).parent.parts)
                    prefix = parent[:len(parent) - node.level + 1]
                    modules = ["egoworld." + ".".join(prefix + node.module.split("."))]
                else:
                    modules = [node.module]
            for module in modules:
                if module.startswith("egoworld."):
                    candidate = module[len("egoworld."):].replace(".", "/")
                    pending.extend([candidate + ".py", candidate + "/__init__.py"])
    # Shared execution adapters also derive frame fields/tracks and can affect
    # output bytes. Include their content without pulling every model imported
    # by the registry into every other model's dependency closure.
    for name in ("pipeline/stages.py", "pipeline/partitioned.py", "worker.py"):
        path = root / name
        if path.is_file():
            seen[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return seen


def semantic_params(value, asset_paths):
    if isinstance(value, dict):
        return {k: semantic_params(v, asset_paths) for k, v in value.items() if k not in ("working_dir", "repo_path")}
    if isinstance(value, (tuple, list)):
        return [semantic_params(v, asset_paths) for v in value]
    if isinstance(value, str):
        try:
            path = str(Path(value).resolve())
        except (OSError, ValueError):
            return value
        if path in asset_paths:
            return {"asset_sha256": asset_paths[path]}
    return value
