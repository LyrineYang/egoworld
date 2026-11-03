"""Metrics must not reward missed hands, oracle matching or reflected poses."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from output_quality import canonical, datajuicer_predictions, evaluate, similarity_align


@pytest.fixture
def case():
    xyz = np.random.default_rng(7).normal(size=(21, 3)) * .03
    xyz -= xyz[0].copy()
    pred = canonical("s", 1, "p", "right", xyz, [0, 0, 1], 200, [224, 224])
    truth = {"sample_id": "s", "frame_index": 1, "hand_id": "0", "handedness": "right",
        "joints_camera_m": (xyz + [0, 0, 1]).tolist(), "joints_image_px": copy.deepcopy(pred["joints_image_px"])}
    samples = {"s": {"image_size": [224, 224], "score_frames": [1]}}
    policy = {"wrist_match_diagonal_fraction": .25, "mpjpe_thresholds_mm": [20, 50]}
    return pred, truth, samples, policy


def test_perfect_prediction_and_missing_do_not_share_denominator(case):
    p, g, samples, policy = case
    score = evaluate([p], [g], samples, policy)
    assert score["recall"] == 1 and score["quality_yield_all_gt"]["20"] == 1
    assert score["mpjpe_root_mm_detected_only"] < 1e-10
    missing = evaluate([], [g], samples, policy)
    assert missing["recall"] == 0 and missing["quality_yield_all_gt"]["20"] == 0
    assert missing["mpjpe_root_mm_detected_only"] is None
    assert missing["pck_all_gt"]["20"] == 0


def test_extra_hand_fp_and_wrong_side_are_visible(case):
    p, g, samples, policy = case
    wrong = dict(p, prediction_id="extra", handedness="left")
    score = evaluate([wrong, p], [g], samples, policy)
    assert score["scored_predictions"] == 2 and score["false_positives"] == 1
    assert score["precision"] == .5
    assert score["handedness_accuracy_matched"] == 0
    assert score["quality_yield_all_gt"]["20"] == 0


def test_matching_does_not_choose_best_3d_pose(case):
    p, g, samples, policy = case
    poor = copy.deepcopy(p)
    poor["joints_root_m"][2][0] += 2.
    best = copy.deepcopy(p)
    best["prediction_id"] = "oracle"
    best["joints_image_px"][0][0] += 1
    score = evaluate([poor, best], [g], samples, policy)
    assert score["details"][0]["prediction_id"] == "p"
    assert score["quality_yield_all_gt"]["20"] == 0


def test_pa_recovers_similarity_without_hiding_root_scale_error(case):
    p, g, samples, policy = case
    target = np.asarray(p["joints_root_m"])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    changed = target @ rotation * 2. + [3, 4, 5]
    np.testing.assert_allclose(similarity_align(changed, target), target, atol=1e-12)
    p["joints_root_m"] = changed.tolist()
    score = evaluate([p], [g], samples, policy)
    assert score["mpjpe_root_mm_detected_only"] > 20
    assert score["pa_mpjpe_mm_detected_only"] < 1e-9


def test_pa_cannot_flip_reflections(case):
    p, *_ = case
    target = np.asarray(p["joints_root_m"])
    reflected = target * [-1, 1, 1]
    assert np.linalg.norm(similarity_align(reflected, target) - target) > .01
    with pytest.raises(ValueError, match="Degenerate"):
        similarity_align(np.zeros((21, 3)), target)


def test_duplicate_id_and_unlabelled_corpus_rejected(case):
    p, g, samples, policy = case
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate([p, p], [g], samples, policy)
    with pytest.raises(ValueError, match="ground truth"):
        evaluate([p], [], samples, policy)


def test_native_mano_origin_normalization_and_identity(tmp_path, case):
    p, *_ = case
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"sample_id": "s"}) + "\n")
    xyz = np.asarray(p["joints_root_m"]) + [.01, .02, .03]
    tags = {"joints": [[xyz.tolist()]], "if_right_hand": [[0]], "camera_translation": [[[0, 0, 1]]]}
    sidecar = tmp_path / "out_stats.jsonl"
    meta = {"input_id": "s", "hand_reconstruction_tags": tags}
    sidecar.write_text(json.dumps({"__dj__meta__": meta}) + "\n")
    samples = {"s": {"image_size": [224, 224], "native_frame_indices": [1]}}
    rows, _ = datajuicer_predictions(path, samples, 200 / 224)
    np.testing.assert_allclose(rows[0]["joints_root_m"], p["joints_root_m"], atol=1e-12)
    assert rows[0]["handedness"] == "left"  # already mirrored upstream, do not flip X twice
    meta["input_id"] = "unknown"
    sidecar.write_text(json.dumps({"__dj__meta__": meta}) + "\n")
    with pytest.raises(ValueError, match="identity"):
        datajuicer_predictions(path, samples, 1.)


def test_off_image_prediction_is_miss_and_fp(case):
    p, g, samples, policy = case
    p["joints_image_px"][0] = [10000., 10000.]
    score = evaluate([p], [g], samples, policy)
    assert score["false_positives"] == score["missed_hands"] == 1
    assert score["mpjpe_root_mm_detected_only"] is None


def test_camera_projection_is_same_after_rebasing_mano_origin(case):
    p, *_ = case
    root = np.asarray(p["joints_root_m"])
    wrist_offset = np.array([.03, -.02, .01])
    raw = canonical("s", 1, "wilor", "right", root + wrist_offset, [0, 0, 1], 200, [224, 224])
    rebased = canonical("s", 1, "hamer", "right", root, np.array([0, 0, 1]) + wrist_offset, 200, [224, 224])
    np.testing.assert_allclose(raw["joints_root_m"], rebased["joints_root_m"], atol=1e-12)
    np.testing.assert_allclose(raw["joints_image_px"], rebased["joints_image_px"], atol=1e-12)


def test_unknown_ground_truth_units_are_not_silently_scored(tmp_path):
    from output_quality import run
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"ground_truth_unit": "millimeters", "ground_truth_joint_order": "openpose21"}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"corpus": str(corpus)}))
    with pytest.raises(ValueError, match="unit/joint order"):
        run(spec, tmp_path / "scores")


def test_double_digit_native_frame_identity_uses_actual_lexical_order(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from bind_native_quality_frames import frame_indices
    for i in range(12):
        (tmp_path / f"frame_{i}.jpg").touch()
    indices, _ = frame_indices(list(range(12)), tmp_path.iterdir())
    assert indices == [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]


def test_missing_native_frame_file_does_not_invent_coverage(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from bind_native_quality_frames import frame_indices
    (tmp_path / 'frame_0.jpg').touch()
    with pytest.raises(ValueError, match="missing"):
        frame_indices([0, 1], tmp_path.iterdir())


def test_exact_frame_miss_and_identical_image_scoring_are_distinct(tmp_path, monkeypatch):
    import cv2
    from bind_native_quality_frames import bind
    video = tmp_path / 'clip.mp4'
    pixels = np.full((16, 16, 3), 100, dtype=np.uint8)
    folder = tmp_path / 'frames/clip'
    folder.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(folder / f'frame_{i}.jpg'), pixels)
    class Capture:
        def __init__(self, path): pass
        def set(self, key, value): pass
        def read(self): return True, pixels.copy()
        def release(self): pass
    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    corpus = tmp_path / 'corpus.json'
    value = {'samples': {'s': {'video_path': str(video), 'native_frame_indices': [0, 1, 2], 'score_frames': [1]}},
        'source_images': {'image': {'montage_frames': [0, 1, 2]}}, 'ground_truth': ['unchanged-labels']}
    corpus.write_text(json.dumps(value))
    times = tmp_path / 'times.json'
    times.write_text(json.dumps({'video': str(video), 'frames': [{'source_frame': i} for i in [0, 2, 2]]}))
    export = tmp_path / 'raw.jsonl'
    export.write_text(json.dumps({'sample_id': 's'}) + '\n')
    sidecar = tmp_path / 'raw_stats.jsonl'
    meta = {'input_id': 's', 'hand_reconstruction_tags': {'joints': [[i] for i in range(3)]}}
    sidecar.write_text(json.dumps({'__dj__meta__': meta}) + '\n')
    before = sidecar.read_bytes()
    exact = bind(corpus, tmp_path / 'frames', times, export, tmp_path / 'exact.json')
    equivalent = bind(corpus, tmp_path / 'frames', times, export, tmp_path / 'equivalent.json', 'identical-image')
    assert exact['samples']['s']['native_frame_indices'] == []
    assert equivalent['samples']['s']['native_frame_indices'] == [1]
    assert exact['native_frame_binding']['records']['s']['strict_missing_score_frames'] == [1]
    assert exact['ground_truth'] == equivalent['ground_truth'] == value['ground_truth']
    assert sidecar.read_bytes() == before
