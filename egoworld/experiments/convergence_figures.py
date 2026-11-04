"""Render the published convergence figures and trial CSV from checked-in summaries."""

import csv
from io import StringIO
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator


DOCS = Path(__file__).resolve().parents[2] / "docs"
SUMMARY = "efficiency-convergence-2026-11-04.json"
ITERATIONS = "efficiency-iterations-2026-11-04.json"
PAPER = "#f6f3eb"
INK = "#282820"
MUTED = "#655e52"
RULE = "#d2cbbd"
ACCENT = "#a04b30"
GREEN = "#426152"
BLUE = "#495d82"
COLORS = {
    "egoworld": ACCENT,
    "shared-process": GREEN,
    "shared-inprocess": BLUE,
    "native-cpu": GREEN,
    "native-cuda": BLUE,
    "native-cuda-minimal": "#73547c",
}
LABELS = {
    "egoworld": "EgoWorld",
    "shared-process": "DJ process map",
    "shared-inprocess": "DJ in-process map",
    "native-cpu": "DJ native CPU",
    "native-cuda": "DJ native CUDA",
    "native-cuda-minimal": "DJ CUDA minimal",
}
FIXED = ("egoworld", "shared-process", "shared-inprocess")
NATIVE = ("egoworld", "native-cpu", "native-cuda", "native-cuda-minimal")


def figure(title, subtitle, columns, *, height=4.1):
    fig, axes = plt.subplots(1, columns, figsize=(10.4, height), squeeze=False)
    fig.subplots_adjust(left=.1, right=.96, top=.72, bottom=.23, wspace=.35)
    fig.text(.055, .91, title, fontfamily="serif", fontsize=20)
    fig.text(.055, .82, subtitle, color=MUTED, fontsize=10)
    return fig, axes[0]


def save(fig, output, name):
    with StringIO() as svg:
        fig.savefig(svg, format="svg", metadata={"Date": None})
        content = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    (output / f"{name}.svg").write_text(content, encoding="utf-8")
    plt.close(fig)


def schedule_figure(data, output):
    fig, axes = figure(
        "The scheduling iterations",
        "Single diagnostic run per setting · 12 clips · 54 original hand export rows match exactly",
        3,
    )
    labels = ("1 active\nclip", "Resident\n+ split", "Resident\n+ whole")
    rows = data["schedule"]
    values = (
        [row["wall_seconds"] for row in rows],
        [row["runner_logs"] for row in rows],
        [sum(stage["model_load_seconds"] for stage in row["stages"].values()) for row in rows],
    )
    for ax, heading, metric in zip(
        axes, ("Whole run (s)", "Runner logs (count)", "Model loading (s, summed)"), values
    ):
        ax.bar(range(3), metric, color=["#a49b89", "#768875", ACCENT], width=.64)
        ax.set_title(heading, loc="left", pad=12)
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, max(metric) * 1.19)
        for index, value in enumerate(metric):
            ax.annotate(f"{value:.0f}" if isinstance(value, int) else f"{value:.1f}",
                        (index, value), xytext=(0, 5), textcoords="offset points", ha="center")
    fig.text(.055, .045, "The load metric sums incremental SAM2 and HaMeR load spans. The bars represent separate metrics.",
             color=MUTED, fontsize=9)
    save(fig, output, "scheduling-iterations")


def pipeline_figure(data, output):
    fig, axes = figure(
        "Where the pipeline changed",
        "Logical data flow · measured outcomes from the 12-image development fixture",
        1, height=5.2,
    )
    ax = axes[0]
    fig.subplots_adjust(left=.055, right=.945, top=.76, bottom=.09)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = (
        ("Input + scheduler", "Admit more clips\nand keep models resident"),
        ("SAM2 + DINO", "Filter nested hand boxes\nbefore reconstruction"),
        ("HaMeR", "Reconstruct 21\nhand joints"),
        ("Dex + export", "Retarget and verify\nstructured outputs"),
    )
    for index, (name, detail) in enumerate(stages):
        left = index * .25
        box = FancyBboxPatch((left + .006, .62), .224, .29, boxstyle="round,pad=.006",
                             facecolor="#eeeade", edgecolor=RULE)
        ax.add_patch(box)
        ax.text(left + .021, .83, name, fontfamily="serif", fontsize=13, va="top")
        ax.text(left + .021, .74, detail, fontsize=10, va="top", color=MUTED, linespacing=1.4)
        if index < len(stages) - 1:
            ax.annotate("", xy=(left + .253, .765), xytext=(left + .236, .765),
                        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.1})

    first, _, last = data["schedule"]
    before, after = (data["quality"][key] for key in ("baseline", "area_policy"))
    rows = data["quality"]["raw_prediction_rows"]
    metrics = (
        (0, "Schedule tuning", f"{first['wall_seconds']:.2f} → {last['wall_seconds']:.2f} s",
         f"{data['same_export_rows']} exported hand rows\nmatch the original run"),
        (.34, "Box filtering", f"{rows['baseline']} → {rows['area_policy']} raw rows",
         f"{before['false_positives']} → {after['false_positives']} extra predictions\non the scored frames"),
        (.68, "Matched hand MPJPE", f"{before['mpjpe_root_mm_detected_only']:.2f} → "
         f"{after['mpjpe_root_mm_detected_only']:.2f} mm",
         f"Matched hands stay at {after['matched_hands']}/{after['gt_hands']}\n"
         f"{after['quality_hands']['20']}/{after['gt_hands']} within 20 mm"),
    )
    for left, title, number, detail in metrics:
        ax.plot((left, left + .3), (.48, .48), color=RULE)
        ax.text(left, .415, title, color=MUTED, va="top")
        ax.text(left, .30, number, fontfamily="serif", fontsize=16, color=ACCENT, va="top")
        ax.text(left, .17, detail, color=MUTED, va="top", linespacing=1.4)
    save(fig, output, "pipeline-outcomes")


def paired_figure(report, output):
    fig, (wall, ratio) = figure(
        "Five paired runs, with the slow run retained",
        "12 clips · SAM2 → HaMeR → Dex · same weights, inputs, GPU and audited outputs",
        2, height=4.7,
    )
    groups = range(1, 6)
    rows = report["trial_rows"]
    fifth_egoworld = None
    for name in FIXED:
        values = [
            next(row["wall_seconds"] for row in rows
                 if row["group"] == f"fixed3-r{index - 1}" and row["name"] == name)
            for index in groups
        ]
        wall.plot(groups, values, "o-", color=COLORS[name], label=LABELS[name], lw=1.8)
        if name == "egoworld":
            fifth_egoworld = values[-1]
        if name in report["eligible_fixed_model_ratios"]:
            values = report["eligible_fixed_model_ratios"][name]["values"]
            ratio.plot(groups, values, "o-", color=COLORS[name], lw=1.8)
    for ax in (wall, ratio):
        ax.axvline(3.5, color=RULE, linestyle="--", zorder=0)
        ax.set_xticks(list(groups))
        ax.set_xlabel("Paired run")
    wall.set_ylim(70, 155)
    wall.set_title("Whole-run seconds", loc="left", pad=12)
    wall.annotate(f"{fifth_egoworld:.2f} s", (5, fifth_egoworld), xytext=(-10, 8),
                  textcoords="offset points", ha="right")
    wall.legend(loc="upper left", frameon=False)
    ratio.set_ylim(.7, 1.4)
    ratio.axhline(1, color=MUTED, linestyle=":")
    ratio.set_title("Paired ratio · DJ seconds / EgoWorld seconds", loc="left", pad=12)
    fig.text(.055, .045, "Dashed divider separates the first three runs from the two runs after restart. Ratios above 1 favor EgoWorld.",
             color=MUTED, fontsize=9)
    save(fig, output, "paired-runs")


def volume_figure(report, output):
    fig, (ax,) = figure(
        "Latency at three input sizes",
        "Fixed models · nested prefixes at 1 and 4 clips · five repeated runs at 12 clips",
        1,
    )
    rows = report["trial_rows"]
    for index, name in enumerate(FIXED):
        color = COLORS[name]
        positions = [pos + (index - 1) * .15 for pos in range(3)]
        for position, clips in zip(positions, (1, 4, 12)):
            values = [row["wall_seconds"] for row in rows if row["track"] == "fixed_models"
                      and row["source_images"] == clips and row["name"] == name]
            center = median(values)
            ax.errorbar(position, center, yerr=[[center - min(values)], [max(values) - center]],
                        fmt="o", color=color, capsize=4, label=LABELS[name] if clips == 1 else None)
    ax.set_xticks(range(3), ("1 clip · n=1", "4 clips · n=1", "12 clips · n=5"))
    ax.set_ylabel("Whole-run seconds")
    ax.set_ylim(50, 155)
    ax.legend(loc="upper left", frameon=False, ncols=3)
    fig.text(.055, .045, "The 12-clip whiskers span the observed minimum and maximum across the five repetitions.",
             color=MUTED, fontsize=9)
    save(fig, output, "workload-size")


def native_figure(report, output):
    fig, (time_ax, quality_ax, error_ax) = figure(
        "Native hand products trade time for coverage and error",
        "Same wrist-relative OpenPose21 contract · exact source frames · separate native models and exports",
        3, height=4.6,
    )
    fig.subplots_adjust(left=.19, wspace=.36)
    rows = [row for row in report["trial_rows"] if row["track"] == "native_product"
            and row["source_images"] == 12]
    for index, name in enumerate(NATIVE):
        dev = [row["quality"]["exact_frames"] for row in rows
               if row["name"] == name and row["role"] == "development_repeats"]
        holdout = next(row["quality"]["exact_frames"] for row in rows
                       if row["name"] == name and row["role"] == "frozen_disjoint_image_holdout")
        color = COLORS[name]
        for ax, metric in (
            (time_ax, lambda score: score["common_output_ready_seconds"]),
            (quality_ax, lambda score: score["quality_hands"]["20"]),
            (error_ax, lambda score: score["mpjpe_root_mm_detected_only"]),
        ):
            ax.scatter([metric(score) for score in dev], [index - .14] * len(dev),
                       color=color, alpha=.7, s=25)
            ax.scatter(metric(holdout), index + .14, color=color, marker="D", s=35)
    for ax in (time_ax, quality_ax, error_ax):
        ax.set_yticks(range(4), [LABELS[name] for name in NATIVE] if ax is time_ax else [])
        ax.set_ylim(3.55, -.55)
        ax.yaxis.grid(True, color=RULE, alpha=.45)
        ax.xaxis.grid(False)
    time_ax.set_title("Ready time (s)", loc="left", pad=12)
    quality_ax.set_title("Hands within 20 mm", loc="left", pad=12)
    quality_ax.set_xlim(0, 12.7)
    quality_ax.set_xticks([0, 4, 8, 12])
    error_ax.set_title("Matched MPJPE (mm)", loc="left", pad=12)
    error_ax.set_xlim(0, 15)
    error_ax.xaxis.set_major_locator(MultipleLocator(5))
    fig.text(.055, .08, "● Development, three runs on the same 12 images     ◆ Holdout, one run on 12 disjoint images",
             color=MUTED, fontsize=9)
    fig.text(.055, .035, "Qualified counts use all 12 targets. MPJPE averages only matched hands.",
             color=MUTED, fontsize=9)
    save(fig, output, "native-quality-cost")


def outlier_figure(data, output):
    fig, (ax,) = figure(
        "The slow round spent more time in several measured phases",
        "EgoWorld, fixed models · fourth and fifth paired runs on the same workload",
        1,
    )
    rounds = {row["group"]: row for row in data["timing_components"]}
    keys = ("pipeline_seconds", "provenance_seconds", "publish_export_seconds")
    for index, (key, label, color) in enumerate((
        ("fixed3-r3", "Run 4", GREEN), ("fixed3-r4", "Run 5", ACCENT),
    )):
        values = [rounds[key][phase] for phase in keys]
        positions = [phase + (index - .5) * .34 for phase in range(3)]
        ax.bar(positions, values, width=.3, color=color, label=label)
        ax.bar_label(ax.containers[-1], fmt="%.1f", padding=3)
    ax.set_xticks(range(3), ("Pipeline", "Asset verification", "Export"))
    ax.set_ylabel("Measured seconds")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper right", frameon=False)
    fig.text(.055, .045, "Each stage retained one runner with 11 cache hits in both rounds.",
             color=MUTED, fontsize=9)
    save(fig, output, "slow-round")


def write_csv(report, output):
    common = ("group", "track", "role", "name", "kind", "status", "environment_block",
              "source_images", "full_parity", "wall_seconds", "process_tree_peak_rss_bytes")
    fields = (*common, "gpu_whole_device_peak_used_bytes", "scoring_mode", "gt_hands",
              "matched_hands", "missed_hands", "false_positives", "mpjpe_root_mm_detected_only",
              "pa_mpjpe_mm_detected_only", "reprojection_mean_px_detected_only",
              "common_output_rows", "normalization_seconds", "common_output_ready_seconds",
              "pck_20", "pck_50", "qualified_20", "qualified_50", "qualified_20_per_ready_second",
              "qualified_50_per_ready_second", "qualified_20_per_raw_second", "qualified_50_per_raw_second")
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for trial in report["trial_rows"]:
            for mode, score in trial["quality"].items():
                row = {key: trial[key] for key in common}
                row["scoring_mode"] = mode
                row["gpu_whole_device_peak_used_bytes"] = json.dumps(trial["gpu_whole_device_peak_used_bytes"])
                row.update({key: score[key] for key in fields if key in score})
                for threshold in ("20", "50"):
                    row[f"pck_{threshold}"] = score["pck_all_gt"][threshold]
                    row[f"qualified_{threshold}"] = score["quality_hands"][threshold]
                    row[f"qualified_{threshold}_per_ready_second"] = score["quality_hands_per_second_common"][threshold]
                    row[f"qualified_{threshold}_per_raw_second"] = score["quality_hands_per_second_raw"][threshold]
                writer.writerow(row)


def main():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": RULE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "axes.grid.axis": "y", "grid.color": RULE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "svg.hashsalt": "egoworld-convergence-2026-11-04",
    })
    report = json.loads((DOCS / "benchmarks" / SUMMARY).read_text())
    diagnostics = json.loads((DOCS / "benchmarks" / ITERATIONS).read_text())
    if report["status"] != "completed_audited" or len(report["trial_rows"]) != report["executions"]:
        raise ValueError("Expected the audited benchmark summary")
    output = DOCS / "assets" / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    schedule_figure(diagnostics, output)
    pipeline_figure(diagnostics, output)
    paired_figure(report, output)
    volume_figure(report, output)
    native_figure(report, output)
    outlier_figure(diagnostics, output)
    write_csv(report, output)


if __name__ == "__main__":
    main()
