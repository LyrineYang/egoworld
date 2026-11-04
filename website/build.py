"""Build a portable static website using only the Python standard library."""

from html import escape
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
DEST = HERE / "dist"
DOCS = HERE.parent / "docs"
ASSETS = DOCS / "assets"
SUMMARY = "efficiency-convergence-2026-11-04.json"
DIAGNOSTICS = "efficiency-iterations-2026-11-04.json"
FIGURES = (
    "scheduling-iterations.svg", "pipeline-outcomes.svg", "paired-runs.svg",
    "workload-size.svg", "native-quality-cost.svg", "slow-round.svg",
)
LABELS = {
    "egoworld": "EgoWorld",
    "shared-process": "DJ process map",
    "shared-inprocess": "DJ in-process map",
    "native-cpu": "DJ native CPU",
    "native-cuda": "DJ native CUDA",
    "native-cuda-minimal": "DJ CUDA minimal",
}


def trial_rows(report):
    if report["status"] != "completed_audited" or len(report["trial_rows"]) != report["executions"]:
        raise ValueError("Expected a completed benchmark with all audited trials")

    previous_group = None
    for trial in report["trial_rows"]:
        score = trial["quality"]["exact_frames"]
        error = score["mpjpe_root_mm_detected_only"]
        group = trial["group"]
        cells = (
            group, LABELS[trial["name"]], trial["source_images"],
            f"{trial['wall_seconds']:.2f}", f"{score['common_output_ready_seconds']:.2f}",
            f"{score['matched_hands']}/{score['gt_hands']}",
            f"{score['quality_hands']['20']}/{score['gt_hands']}",
            "—" if error is None else f"{error:.2f}",
        )
        css_class = ' class="group-start"' if group != previous_group else ""
        yield f"<tr{css_class}>" + "".join(f"<td>{escape(str(value))}</td>" for value in cells) + "</tr>"
        previous_group = group


def main():
    DEST.mkdir(exist_ok=True)
    (DEST / "assets").mkdir(exist_ok=True)
    for name in ("index.html", "styles.css"):
        shutil.copyfile(HERE / name, DEST / name)
    for name in ("mark.svg", "observation.svg", "pipeline-simple.svg"):
        shutil.copyfile(ASSETS / name, DEST / "assets" / name)
    benchmark_assets = DEST / "assets" / "benchmarks"
    benchmark_assets.mkdir(exist_ok=True)
    for name in (*FIGURES, "trials.csv"):
        shutil.copyfile(ASSETS / "benchmarks" / name, benchmark_assets / name)
    for name in (SUMMARY, DIAGNOSTICS):
        shutil.copyfile(DOCS / "benchmarks" / name, benchmark_assets / name)
    report = json.loads((DOCS / "benchmarks" / SUMMARY).read_text(encoding="utf-8"))
    page = (HERE / "benchmarks.html").read_text(encoding="utf-8")
    marker = "<!-- TRIAL_ROWS -->"
    if page.count(marker) != 1:
        raise ValueError("Expected one trial table insertion point")
    (DEST / "benchmarks.html").write_text(
        page.replace(marker, "\n                ".join(trial_rows(report))), encoding="utf-8"
    )
    (DEST / ".nojekyll").touch()
    print(f"Built static website: {DEST}")


if __name__ == "__main__":
    main()
