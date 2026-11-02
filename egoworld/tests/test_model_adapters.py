"""R04–R07: execute native adapter control flow against small official-API doubles."""
from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from egoworld.contracts import validate_rows

CLIP = {"clip_id": "c", "video_id": "v", "frame_start": 10, "frame_end": 20, "fps": 10.}


def test_bfloat16_outputs_are_converted_before_numpy():
    from egoworld.operators.common import array
    class Tensor:
        dtype = "torch.bfloat16"
        def detach(self): return self
        def cpu(self): return self
        def float(self):
            self.dtype = "torch.float32"
            return self
        def numpy(self):
            assert self.dtype == "torch.float32"
            return np.ones((2, 3), dtype=np.float32)
    assert array(Tensor()).dtype == np.float32


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
    raw_joints = np.arange(63).reshape(21, 3).astype(float) + [.05, 0, 0]
    operator.model = lambda data: {"pred_keypoints_3d": np.tile(raw_joints, (2, 1, 1)),
        "pred_cam": np.ones((2, 3)), "pred_mano_params": {"global_orient": np.tile(np.eye(3), (2, 1, 1)),
        "hand_pose": np.tile(np.eye(3), (2, 15, 1, 1)), "betas": np.zeros((2, 10))}}
    packet = operator.run(CLIP, "unused", {})
    saved = packet["tables"]["hand_pose"]; validate_rows("hand_pose", saved, CLIP)
    assert [r["handedness"] for r in saved] == ["left", "right"]
    assert saved[0]["joints_3d"][3] == -saved[1]["joints_3d"][3]
    assert saved[0]["scale_status"] == "model_prior" and len(saved[0]["mano_pose"]) == 135
    # Independently calculated full-image cameras: f=625, bs=8,
    # center offset=(-12,-12), original crop tx/ty=(1,1).
    for row, sign, camera in zip(saved, [-1, 1], [[-4, -2, 156.25], [-2, -2, 156.25]]):
        local = np.asarray(row["joints_3d"]).reshape(21, 3)
        np.testing.assert_allclose(local[0], 0)
        np.testing.assert_allclose(local + row["camera_translation"],
                                   raw_joints * [sign, 1, 1] + camera, atol=1e-7)
    detections[0]["label"] = "hand"
    with pytest.raises(ValueError, match="refusing to guess"): operator.run(CLIP, "unused", {})
    # Two valid hands plus a false hand-like sleeve must not reach inference
    # when the user explicitly selected the two tracked hand identities.
    detections[0]["label"] = "left hand"
    detections.append(dict(detections[0], object_id=3, label="hand"))
    operator.params["object_ids"] = [1, 2]
    assert len(operator.run(CLIP, "unused", {})["tables"]["hand_pose"]) == 2
    operator.params["object_ids"] = [99]
    with pytest.raises(ValueError, match="No nonempty"):
        operator.run(CLIP, "unused", {})


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
        return SimpleNamespace(joint_names=["joint"], retarget=retarget, has_joint_limits=True,
                               optimizer=SimpleNamespace(robot=SimpleNamespace(dof_joint_names=["joint"], joint_limits=np.array([[-1., 1.]])),
                               target_link_human_indices=np.array([[0], [1]]), retargeting_type="VECTOR"))
    operator = target.DexRetargetOperator(config="not-read", robot_id="test", handedness="right", joint_units="radians",
        human_to_robot_rotation=[0, -1, 0, 1, 0, 0, 0, 0, 1])
    monkeypatch.setattr(operator, "_build", build)
    output = operator.run(CLIP, "unused", {})["tables"]["mapping"]
    validate_rows("mapping", output, CLIP)
    assert len(built) == 2
    np.testing.assert_allclose(calls[0], [[0, 1, 0]])
    assert [row["joint_positions"][0] for row in output] == [.1, .2, .1]


@pytest.mark.parametrize("enabled", [True, False])
def test_dex_build_does_not_expand_configured_joint_limits(monkeypatch, enabled):
    from egoworld.operators.dex_retarget_op import DexRetargetOperator
    calls = []
    limits = np.array([[-.47, .47]])
    sequence = SimpleNamespace(has_joint_limits=enabled, joint_limits=limits,
        optimizer=SimpleNamespace(set_joint_limit=lambda bounds, epsilon: calls.append((bounds.copy(), epsilon))))
    module(monkeypatch, "dex_retargeting.retargeting_config", RetargetingConfig=SimpleNamespace(
        load_from_file=lambda _: SimpleNamespace(build=lambda: sequence)))
    assert DexRetargetOperator(config="fixture")._build() is sequence
    assert len(calls) == int(enabled)
    if enabled:
        np.testing.assert_array_equal(calls[0][0], limits)
        assert calls[0][1] == 0.0


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("overshoot", [2e-8, 1e-3])
def test_dex_corrects_roundoff_but_rejects_real_limit_violation(monkeypatch, side, overshoot):
    from egoworld.operators import dex_retarget_op as target
    hand = {"frame_index": 10, "timestamp_s": 1., "object_id": 1, "handedness": side,
            "joints_3d": np.zeros(63).tolist(), "scale_status": "model_prior"}
    monkeypatch.setattr(target, "rows", lambda *args: [hand])
    # Deliberately reverse the robot's named order: limits must follow names.
    robot = SimpleNamespace(dof_joint_names=["b", "a"], joint_limits=np.array([[-1., 1.], [-.47, .47]]))
    sequence = SimpleNamespace(has_joint_limits=True, joint_names=["a", "b"],
        retarget=lambda _: [.47 + overshoot, 0.], optimizer=SimpleNamespace(robot=robot,
        target_link_human_indices=np.array([[0], [1]]), retargeting_type="VECTOR"))
    operator = target.DexRetargetOperator(config="fixture", robot_id="fixture", handedness=side,
        joint_units="radians", human_to_robot_rotation=np.eye(3).ravel().tolist())
    monkeypatch.setattr(operator, "_build", lambda: sequence)
    if overshoot > 1e-6:
        with pytest.raises(ValueError, match="strict URDF"):
            operator.run(CLIP, "unused", {})
    else:
        packet = operator.run(CLIP, "unused", {})
        assert packet["tables"]["mapping"][0]["joint_positions"] == [.47, 0.]
        assert packet["stats"]["joint_limit_adjustments"] == 1


def test_foundationpose_uses_metric_depth_register_then_track(monkeypatch, tmp_path):
    from egoworld.operators import foundationpose_op as target
    for frame in (10, 11): np.save(tmp_path / f"{frame}.npy", np.full((2, 2), 1000, dtype=np.uint16))
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"v": {"intrinsics": [100, 0, 1, 0, 100, 1, 0, 0, 1], "depth_pattern": str(tmp_path / "{frame_index}.npy"), "depth_scale_to_meters": .001}}))
    instances = [{"frame_index": f, "timestamp_s": f / 10, "object_id": oid, "label": "cup", "area_px": 4,
                  "mask_rle": '{"size":[2,2],"counts":[0,4]}'} for f in (10, 11) for oid in (1, 2)]
    monkeypatch.setattr(target, "rows", lambda *args: instances)
    monkeypatch.setattr(target, "frame_reader", lambda *args: SimpleNamespace(release=lambda: None))
    decoded = []
    def decode(*args):
        decoded.append(args[1]); return np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(target, "read_frame", decode)
    mesh = SimpleNamespace(vertices=np.zeros((3, 3)), vertex_normals=np.zeros((3, 3)), apply_scale=lambda scale: None)
    import copy
    mesh.copy = lambda: copy.copy(mesh)
    mesh_loads = []
    def load_mesh(*args, **kwargs):
        mesh_loads.append(args[0]); return mesh
    module(monkeypatch, "trimesh", load=load_mesh)
    log = []
    class Tracker:
        def __init__(self, **kwargs): pass
        def register(self, **kwargs):
            log.append("register"); assert np.all(kwargs["depth"] == 1); assert kwargs["ob_mask"].all()
            kwargs["depth"][:] = 5  # one tracker must not corrupt the next object
            return np.eye(4)
        def track_one(self, **kwargs):
            log.append("track"); assert np.all(kwargs["depth"] == 1); kwargs["depth"][:] = 5
            return np.eye(4)
    operator = target.FoundationPoseOperator(assets_manifest=str(assets), objects={"cup": {"mesh_path": "not-read", "mesh_scale_to_meters": .001}})
    monkeypatch.setattr(operator, "_load", lambda: Tracker)
    output = operator.run(CLIP, "unused", {})["tables"]["object_pose"]
    validate_rows("object_pose", output, CLIP)
    assert log == ["register", "register", "track", "track"] and output[0]["length_unit"] == "meters"
    assert decoded == [0, 1] and len(mesh_loads) == 1


@pytest.mark.parametrize("return_tuple", [False, True])
@pytest.mark.parametrize("precision,upstream_precision", [("fp32", "32"), ("fp16", "16-mixed"), ("bf16", "bf16-mixed")])
def test_fast3r_relative_reconstruction_and_pnp_failure(monkeypatch, return_tuple, precision, upstream_precision):
    from egoworld.operators import fast3r_op as target
    torch = SimpleNamespace(inference_mode=nullcontext, device=lambda name: SimpleNamespace(type=name), from_numpy=lambda a: a)
    monkeypatch.setitem(sys.modules, "torch", torch)
    class Model:
        @classmethod
        def from_pretrained(cls, name): return cls()
        def to(self, device): return self
        def eval(self): return self
        def modules(self):
            return [attention]
    attention = SimpleNamespace(attn_implementation="flash_attention")
    pred = {"pts3d_in_other_view": np.ones((1, 3, 3, 3)), "conf": np.full((1, 3, 3), 2.)}
    module(monkeypatch, "fast3r.models.fast3r", Fast3R=Model)
    def inference(images, model, device, **kwargs):
        assert device.type == "cuda" and kwargs["dtype"] == upstream_precision
        result = {"views": images, "preds": [pred for _ in images], "loss": None}
        return (result, {}) if return_tuple else result
    module(monkeypatch, "fast3r.dust3r.inference_multiview", inference=inference)
    module(monkeypatch, "fast3r.dust3r.utils.image", load_images=lambda files, **kwargs: files)
    module(monkeypatch, "fast3r.dust3r.cloud_opt.init_im_poses", fast_pnp=lambda *args, **kwargs: (100., np.eye(4)))
    monkeypatch.setattr(target, "frame_reader", lambda *args: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(target, "read_frame", lambda *args: np.zeros((3, 3, 3), dtype=np.uint8))
    operator = target.Fast3ROperator(model_name_or_path="not-read", max_views=2,
                                   dtype=precision, attn_implementation="pytorch_auto")
    packet = operator.run(CLIP, "unused", {})
    validate_rows("reconstruction", packet["tables"]["reconstruction"], CLIP)
    assert len(packet["tables"]["reconstruction"]) == 2
    assert attention.attn_implementation == "pytorch_auto"
    assert packet["tables"]["reconstruction"][0]["length_unit"] == "relative"
    assert len(packet["arrays"]["reconstruction-pointcloud.npz"]["points"]) <= 100000
    module(monkeypatch, "fast3r.dust3r.cloud_opt.init_im_poses", fast_pnp=lambda *args, **kwargs: (None, None))
    with pytest.raises(ValueError, match="PnP failed"): operator.run(CLIP, "unused", {})
