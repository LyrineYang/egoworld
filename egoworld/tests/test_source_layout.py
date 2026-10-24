from pathlib import Path
import subprocess


def test_manifest_source_is_visible_to_git():
    root = Path(__file__).resolve().parents[2]
    files = ["egoworld/src/egoworld/manifests/" + name for name in ("__init__.py", "build_manifest.py", "schema.py")]
    assert all((root / name).is_file() for name in files)
    result = subprocess.run(["git", "check-ignore", "--no-index", *files], cwd=root, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
