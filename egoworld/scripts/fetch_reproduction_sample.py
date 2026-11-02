"""Fetch a fixed official RGB preview and extract one continuous egocentric view."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request


REVISION = "fce55699745b9abcee20f7b3366ff039c82660a6"
SOURCE = (f"https://raw.githubusercontent.com/epic-kitchens/VISOR/{REVISION}/"
          "landing_static/media/videos/epic_kitchens_videowall.mp4")
SOURCE_SHA256 = "8103fd021d134ed4db88b0e0b233d9e4c08d79dc3e41070b364179d92e6fe932"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "data/samples")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "epic-kitchens-videowall.mp4"
    if not source.is_file() or sha256(source) != SOURCE_SHA256:
        temporary = source.with_suffix(".part")
        try:
            with urllib.request.urlopen(SOURCE, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError("Official sample checksum differs from the pinned version")
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    ffmpeg = shutil.which("ffmpeg") or str(root / ".venv/tools/ffmpeg")
    sample = args.output_dir / "epic-dough-3s.mp4"
    command = [ffmpeg, "-v", "error", "-y", "-ss", "0", "-i", str(source), "-t", "3",
               "-vf", "crop=426:206:426:480,fps=10", "-an", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", str(sample)]
    subprocess.run(command, check=True)
    metadata = {"source_url": SOURCE, "source_sha256": SOURCE_SHA256,
        "project_url": "https://epic-kitchens.github.io/VISOR/",
        "purpose": "Local inference reproduction; no training or ground-truth evaluation",
        "source_kind": "Official EPIC-KITCHENS RGB preview; cropped bottom-center dough view",
        "terms": "Original EPIC-KITCHENS/VISOR data terms apply; preview is not an annotated benchmark",
        "transform": {"start_seconds": 0, "duration_seconds": 3, "crop_xywh": [426, 480, 426, 206], "fps": 10},
        "command": command, "sample_path": str(sample.resolve()), "sample_sha256": sha256(sample)}
    (args.output_dir / "sample-source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
