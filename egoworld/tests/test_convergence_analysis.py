"""Final comparison reports must retain all eligible pairs and reject partial evidence."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from convergence_analysis import analyse, paired_ratios


def fixed_pairs():
    return [
        {"track": "fixed_models", "role": "development_repeats",
         "full_parity": True, "status": "passed", "group": f"r{i}",
         "name": name, "wall_seconds": seconds}
        for i in range(5)
        for name, seconds in (("egoworld", 20 if i == 4 else 10), ("shared-process", 12))
    ]


def test_paired_summary_keeps_a_slower_reference_trial():
    report = paired_ratios(fixed_pairs(), "egoworld", "shared-process", 5)
    assert report["n"] == 5
    assert report["reference_faster_pairs"] == 4
    assert report["values"] == [1.2, 1.2, 1.2, 1.2, .6]


@pytest.mark.parametrize("remove_trial", [False, True])
def test_unverified_pair_cannot_generate_ratio(remove_trial):
    rows = fixed_pairs()
    if remove_trial:
        rows.pop()
    else:
        rows[-1]["full_parity"] = False
    with pytest.raises(ValueError):
        paired_ratios(rows, "egoworld", "shared-process", 5)


def test_running_study_is_not_a_final_report(tmp_path):
    (tmp_path / "study-state.json").write_text(json.dumps({"status": "running"}))
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="not terminal-completed"):
        analyse(tmp_path)
