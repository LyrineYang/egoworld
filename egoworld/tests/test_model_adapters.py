"""R04–R07: execute native adapter control flow against small official-API doubles."""
from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from egoworld.contracts import validate_rows

CLIP = {"clip_id": "c", "video_id": "v", "frame_start": 10, "frame_end": 20, "fps": 10.}


def module(monkeypatch, name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules: monkeypatch.setitem(sys.modules, key, ModuleType(key))
    value = ModuleType(name)
    for key, obj in attrs.items(): setattr(value, key, obj)
    monkeypatch.setitem(sys.modules, name, value)


def test_hamer_batches_and_preserves_handedness_and_mano(monkeypatch):
    from egoworld.operators import hamer_op as target
    detections = [{"frame_index": 10, "object_id": i + 1, "label": side + " hand", "area_px": 64, "bbox_xyxy": [0, 0, 8, 8]} for i, side in enumerate(("left", "right"))]
    batch = {"right": np.array([0, 1]), "box_center": np.array([[4., 4.], [4., 4.]]),
             "box_size": np.array([8., 8.]), "img_size": np.array([[32., 32.], [32., 32.]])}
    torch = SimpleNamespace(inference_mode=nullcontext, utils=SimpleNamespace(data=SimpleNamespace(DataLoader=lambda *args, **kwargs: [batch])))
    monkeypatch.setitem(sys.modules, "torch", torch)
    module(monkeypatch, "hamer.datasets.vitdet_dataset", ViTDetDataset=lambda *args, **kwargs: object())
    module(monkeypatch, "hamer.utils", recursive_to=lambda batch, device: batch)
    monkeypatch.setattr(target, "rows", lambda *args: detections)
    monkeypatch.setattr(target, "frame_reader", lambda *args: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(target, "read_frame", lambda *args: np.zeros((32, 32, 3), dtype=np.uint8))
    operator = target.HamerOperator(checkpoint="not-read", frame_stride=1)
    monkeypatch.setattr(operator, "_load", lambda: None)
    operator.model_cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000), MODEL=SimpleNamespace(IMAGE_SIZE=256))
    operator.model = lambda data: {"pred_keypoints_3d": np.tile(np.arange(63).reshape(21, 3), (2, 1, 1)).astype(float),
        "pred_cam": np.ones((2, 3)), "pred_mano_params": {"global_orient": np.tile(np.eye(3), (2, 1, 1)),
        "hand_pose": np.tile(np.eye(3), (2, 15, 1, 1)), "betas": np.zeros((2, 10))}}
    packet = operator.run(CLIP, "unused", {})
    saved = packet["tables"]["hand_pose"]; validate_rows("hand_pose", saved, CLIP)
    assert [r["handedness"] for r in saved] == ["left", "right"]
    assert saved[0]["joints_3d"][3] == -saved[1]["joints_3d"][3]
    assert saved[0]["scale_status"] == "model_prior" and len(saved[0]["mano_pose"]) == 135
    detections[0]["label"] = "hand"
    with pytest.raises(ValueError, match="refusing to guess"): operator.run(CLIP, "unused", {})


def test_dex_sequences_reset_per_identity_and_apply_axes(monkeypatch):
    from egoworld.operators import dex_retarget_op as target
    xyz = np.zeros((21, 3)); xyz[1] = [1, 0, 0]
    hands = [{"frame_index": frame, "timestamp_s": frame / 10, "object_id": oid, "handedness": "right",
              "joints_3d": xyz.ravel().tolist(), "scale_status": "model_prior"} for oid, frame in [(1, 10), (1, 11), (2, 10)]]
    monkeypatch.setattr(target, "rows", lambda *args: hands)
    calls, built = [], []
    def build():
        state = []
        def retarget(reference):
            calls.append(reference.copy()); state.append(1); return [.1 * len(state)]
        built.append(state)
        return SimpleNamespace(joint_names=["joint"], retarget=retarget,
                               optimizer=SimpleNamespace(target_link_human_indices=np.array([[0], [1]]), retargeting_type="VECTOR"))
    operator = target.DexRetargetOperator(config="not-read", robot_id="test", handedness="right", joint_units="radians",
        human_to_robot_rotation=[0, -1, 0, 1, 0, 0, 0, 0, 1])
    monkeypatch.setattr(operator, "_build", build)
    output = operator.run(CLIP, "unused", {})["tables"]["mapping"]
    validate_rows("mapping", output, CLIP)
    assert len(built) == 2
    np.testing.assert_allclose(calls[0], [[0, 1, 0]])
    assert [row["joint_positions"][0] for row in output] == [.1, .2, .1]


def test_foundationpose_uses_metric_depth_register_then_track(monkeypatch, tmp_path):
    from egoworld.operators import foundationpose_op as target
    for frame in (10, 11): np.save(tmp_path / f"{frame}.npy", np.full((2, 2), 1000, dtype=np.uint16))
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"v": {"intrinsics": [100, 0, 1, 0, 100, 1, 0, 0, 1], "depth_pattern": str(tmp_path / "{frame_index}.npy"), "depth_scale_to_meters": .001}}))
    instances = [{"frame_index": f, "timestamp_s": f / 10, "object_id": 1, "label": "cup", "area_px": 4,
                  "mask_rle": '{"size":[2,2],"counts":[0,4]}'} for f in (10, 11)]
    monkeypatch.setattr(target, "rows", lambda *args: instances)
    monkeypatch.setattr(target, "frame_reader", lambda *args: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(target, "read_frame", lambda *args: np.zeros((2, 2, 3), dtype=np.uint8))
    mesh = SimpleNamespace(vertices=np.zeros((3, 3)), vertex_normals=np.zeros((3, 3)), apply_scale=lambda scale: None)
    module(monkeypatch, "trimesh", load=lambda *args, **kwargs: mesh)
    log = []
    class Tracker:
        def __init__(self, **kwargs): pass
        def register(self, **kwargs):
            log.append("register"); assert np.all(kwargs["depth"] == 1); assert kwargs["ob_mask"].all(); return np.eye(4)
        def track_one(self, **kwargs): log.append("track"); return np.eye(4)
    operator = target.FoundationPoseOperator(assets_manifest=str(assets), objects={"cup": {"mesh_path": "not-read", "mesh_scale_to_meters": .001}})
    monkeypatch.setattr(operator, "_load", lambda: Tracker)
    output = operator.run(CLIP, "unused", {})["tables"]["object_pose"]
    validate_rows("object_pose", output, CLIP)
    assert log == ["register", "track"] and output[0]["length_unit"] == "meters"


def test_fast3r_relative_reconstruction_and_pnp_failure(monkeypatch):
    from egoworld.operators import fast3r_op as target
    torch = SimpleNamespace(inference_mode=nullcontext, float32="fp32", float16="fp16", bfloat16="bf16", from_numpy=lambda a: a)
    monkeypatch.setitem(sys.modules, "torch", torch)
    class Model:
        @classmethod
        def from_pretrained(cls, name): return cls()
        def to(self, device): return self
        def eval(self): return self
    pred = {"pts3d_in_other_view": np.ones((1, 3, 3, 3)), "conf": np.full((1, 3, 3), 2.)}
    module(monkeypatch, "fast3r.models.fast3r", Fast3R=Model)
    module(monkeypatch, "fast3r.dust3r.inference_multiview", inference=lambda images, *args, **kwargs: ({"preds": [pred for _ in images]}, {}))
    module(monkeypatch, "fast3r.dust3r.utils.image", load_images=lambda files, **kwargs: files)
    module(monkeypatch, "fast3r.dust3r.cloud_opt.init_im_poses", fast_pnp=lambda *args, **kwargs: (100., np.eye(4)))
    monkeypatch.setattr(target, "frame_reader", lambda *args: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(target, "read_frame", lambda *args: np.zeros((3, 3, 3), dtype=np.uint8))
    operator = target.Fast3ROperator(model_name_or_path="not-read", max_views=2)
    packet = operator.run(CLIP, "unused", {})
    validate_rows("reconstruction", packet["tables"]["reconstruction"], CLIP)
    assert len(packet["tables"]["reconstruction"]) == 2
    assert packet["tables"]["reconstruction"][0]["length_unit"] == "relative"
    assert len(packet["arrays"]["reconstruction-pointcloud.npz"]["points"]) <= 100000
    module(monkeypatch, "fast3r.dust3r.cloud_opt.init_im_poses", fast_pnp=lambda *args, **kwargs: (None, None))
    with pytest.raises(ValueError, match="PnP failed"): operator.run(CLIP, "unused", {})
