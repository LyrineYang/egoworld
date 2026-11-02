"""Build a portable static website using only the Python standard library."""

from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
DEST = HERE / "dist"
ASSETS = HERE.parent / "docs" / "assets"


def main():
    DEST.mkdir(exist_ok=True)
    (DEST / "assets").mkdir(exist_ok=True)
    for name in ("index.html", "styles.css"):
        shutil.copyfile(HERE / name, DEST / name)
    for name in ("mark.svg", "observation.svg", "pipeline-simple.svg"):
        shutil.copyfile(ASSETS / name, DEST / "assets" / name)
    (DEST / ".nojekyll").touch()
    print(f"Built static website: {DEST}")


if __name__ == "__main__":
    main()
