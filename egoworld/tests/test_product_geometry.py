import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egoworld.contracts import schemas
from egoworld.product.geometry import fit_similarity, robust_similarity, transform_points, fast3r_source_pixels
from egoworld.product.world import fuse_world
from egoworld.utils.hashing import sha256_file


def test_similarity_recovers_metric_rotation_scale_with_outliers():
    rng = np.random.default_rng(9)
    points = rng.normal(size=(200, 3))
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    matrix[:3, 3] = [3, -2, 1]
    target = transform_points(points, matrix, 2.5)
    target[::7] += 10
    result = robust_similarity(points, target, .001)
    assert result["scale"] == pytest.approx(2.5)
    np.testing.assert_allclose(result["transform"], matrix.ravel(), atol=1e-12)
    assert result["fit"]["heldout_inlier_fraction"] < 1


def test_alignment_rejects_degenerate_geometry_and_bad_heldout_pairs():
    line = np.arange(15).reshape(5, 3)
    with pytest.raises(ValueError, match="Collinear"):
        fit_similarity(line, line)
    points = np.random.default_rng(1).normal(size=(100, 3))
    target = points.copy()
    target[::5] += 100
    with pytest.raises(ValueError, match="held-out"):
        robust_similarity(points, target, .001)


def test_pixel_correspondence_inverts_center_crop_with_pixel_centers():
    points = fast3r_source_pixels([[0, 0], [223, 223]], (640, 480), 224, (224, 224))
    expected = (np.array([[0, 0], [223, 223]]) + [37, 0] + .5) * [640 / 299, 480 / 224] - .5
    np.testing.assert_allclose(points, expected)
    with pytest.raises(ValueError, match="shape changed"):
        fast3r_source_pixels([[0, 0]], (640, 480), 224, (512, 512))


@pytest.fixture
def world_case(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta.json").write_text(json.dumps({"status": "Done", "clip": {
        "video_id": "v", "clip_id": "c", "frame_start": 0, "frame_end": 3, "fps": 10}}))
    reconstruction = [{"frame_index": 0, "timestamp_s": 0., "view_index": 0,
        "camera_to_world": np.eye(4).ravel().tolist(), "intrinsics": np.eye(3).ravel().tolist(),
        "image_width": 1, "image_height": 1, "pointcloud_path": "reconstruction-pointcloud.npz",
        "coord_frame": "first_view_world", "length_unit": "relative", "scale_status": "unscaled"}]
    pq.write_table(pa.Table.from_pylist(reconstruction, schema=schemas()["reconstruction"]), source / "reconstruction.parquet")
    np.savez(source / "reconstruction-pointcloud.npz", points=[[1., 2., 3.]], confidence=[1.], view_index=[0])
    obj = np.eye(4)
    obj[:3, 3] = [1, 0, 1]
    objects = [{"frame_index": f, "timestamp_s": f / 10, "object_id": 1, "label": "box",
        "transform": obj.ravel().tolist(), "coord_frame": "object_to_camera", "length_unit": "meters", "mesh_path": "box.obj"} for f in (0, 1)]
    pq.write_table(pa.Table.from_pylist(objects, schema=schemas()["object_pose"]), source / "object_pose.parquet")
    alignment = np.eye(4)
    alignment[:3, 3] = [10, 0, 0]
    calibration = {"version": 1, "video_id": "v", "clip_id": "c", "world_id": "metric-test", "length_unit": "meters",
        "scene_alignment": {"scale": 2., "transform": alignment.ravel().tolist()},
        "bound_files": {name: sha256_file(str(source / name)) for name in ("reconstruction.parquet", "reconstruction-pointcloud.npz")},
        "evidence": {"method": "Synthetic test correspondences; not model validation"}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    return source, path, tmp_path / "world"


def test_metric_object_translation_is_not_scaled_again_and_frames_are_exact(world_case):
    source, calibration, output = world_case
    result = fuse_world(source, calibration, ["world_object", "world_scene", "world_camera"], output)
    assert result["status"] == "partial" and result["skipped_frames"]["world_object"] == [1]
    obj = pq.ParquetFile(output / "world_object.parquet").read().to_pylist()[0]
    np.testing.assert_allclose(np.array(obj["transform"]).reshape(4, 4)[:3, 3], [11, 0, 1])
    with np.load(output / "world_scene.npz") as content:
        np.testing.assert_allclose(content["points"], [[12, 4, 6]])


def test_world_alignment_cannot_be_reused_for_different_clip_or_changed_points(world_case):
    source, calibration, output = world_case
    original = json.loads(calibration.read_text())
    calibration.write_text(json.dumps({**original, "clip_id": "other"}))
    with pytest.raises(ValueError, match="exact clip"):
        fuse_world(source, calibration, ["world_scene"], output)
    calibration.write_text(json.dumps(original))
    (source / "reconstruction-pointcloud.npz").write_bytes(b"different")
    with pytest.raises(ValueError, match="identity changed"):
        fuse_world(source, calibration, ["world_scene"], output)


def test_prior_hand_is_not_silently_promoted_to_calibrated_meters(world_case):
    from test_pipeline_logic import full_packet
    source, calibration, output = world_case
    hand = full_packet("hamer", {"frame_start": 0, "fps": 10})["tables"]["hand_pose"]
    pq.write_table(pa.Table.from_pylist(hand, schema=schemas()["hand_pose"]), source / "hand_pose.parquet")
    with pytest.raises(ValueError, match="model_prior is insufficient"):
        fuse_world(source, calibration, ["world_hand"], output)
    assert not (output / "world_manifest.json").exists()


def test_measured_hand_alignment_is_bound_to_predictions_and_applied_before_world_pose(world_case):
    from test_pipeline_logic import full_packet
    from egoworld.product.world import calibrate_hand
    source, calibration, output = world_case
    hand = full_packet("hamer", {"frame_start": 0, "fps": 10})["tables"]["hand_pose"]
    joints = np.random.default_rng(12).normal(size=(21, 3)) * .03
    joints[0] = 0
    hand[0]["joints_3d"] = joints.ravel().tolist()
    pq.write_table(pa.Table.from_pylist(hand, schema=schemas()["hand_pose"]), source / "hand_pose.parquet")
    target = (joints + [0, 0, 1]) * 1.5 + [0, 0, -.7]
    measured = {"version": 1, "video_id": "v", "clip_id": "c", "joint_order": "openpose21", "length_unit": "meters",
        "samples": [{"frame_index": 0, "object_id": 1, "joints_camera_meters": target.ravel().tolist(),
                     "evidence": "Known synthetic transform for a unit test, not observed hand measurements"}]}
    measurement_path = output.parent / "measurements.json"
    measurement_path.write_text(json.dumps(measured))
    calibrated = output.parent / "hand-calibration.json"
    calibrate_hand(source, calibration, measurement_path, calibrated, threshold_m=1e-6)
    result = fuse_world(source, calibrated, ["world_hand"], output)
    assert result["status"] == "passed"
    row = pq.ParquetFile(output / "world_hand.parquet").read().to_pylist()[0]
    np.testing.assert_allclose(np.array(row["joints_3d"]).reshape(21, 3), target + [10, 0, 0], atol=1e-12)
    assert row["scale_status"] == "calibrated"
    # Measurement alignment must not apply to a newly changed prediction.
    hand[0]["camera_translation"][2] += .1
    pq.write_table(pa.Table.from_pylist(hand, schema=schemas()["hand_pose"]), source / "hand_pose.parquet")
    with pytest.raises(ValueError, match="measured camera alignment"):
        fuse_world(source, calibrated, ["world_hand"], output)
