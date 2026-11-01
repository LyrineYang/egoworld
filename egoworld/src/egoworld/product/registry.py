"""Typed output catalog. One model kernel can provide several selected outputs."""
from dataclasses import dataclass

from egoworld.contracts import STAGE_DEPENDENCIES, GPU_STAGES, schemas


@dataclass(frozen=True)
class DataType:
    kernel: str
    artifact: str
    fields: tuple[str, ...]
    description: str


DATA_TYPES = {
    "masks": DataType("sam2", "instances", ("mask_rle", "label", "area_px"), "Per-instance segmentation masks"),
    "boxes": DataType("sam2", "instances", ("bbox_xyxy", "label", "detection_score"), "Per-instance 2D boxes"),
    "tracks": DataType("sam2", "tracks", ("center_xy", "area_px", "label"), "Clip-local 2D trajectories"),
    "hand_joints": DataType("hamer", "hand_pose", ("handedness", "joints_3d", "camera_translation", "coord_frame", "length_unit", "scale_status", "joint_order"), "21 wrist-relative hand joints and estimated camera translation"),
    "hand_mano": DataType("hamer", "hand_pose", ("handedness", "global_orient", "mano_pose", "betas"), "Canonical MANO rotation and shape parameters"),
    "object_poses": DataType("foundationpose", "object_pose", ("label", "transform", "coord_frame", "length_unit", "mesh_path"), "Metric object-to-camera transforms; requires RGB-D and meshes"),
    "robot_joints": DataType("dex_retarget", "mapping", ("handedness", "robot_id", "joint_names", "joint_positions", "joint_units", "source_scale_status"), "Named robot joint values for the configured hand"),
    "camera_poses": DataType("fast3r", "reconstruction", ("view_index", "camera_to_world", "intrinsics", "image_width", "image_height", "coord_frame", "length_unit", "scale_status"), "Sampled camera poses; relative unless calibrated"),
    "scene_points": DataType("fast3r", "reconstruction-pointcloud.npz", ("points", "confidence", "view_index"), "Bounded reconstructed point cloud"),
    "world_camera": DataType("world", "world_camera", (), "Metric camera poses in an explicitly anchored world"),
    "world_scene": DataType("world", "world_scene.npz", (), "Metric scene points; requires measured scene alignment"),
    "world_object": DataType("world", "world_object", (), "Metric object-to-world poses at matching camera frames"),
    "world_hand": DataType("world", "world_hand", (), "Metric world hand joints; additionally requires measured hand alignment"),
}
WORLD_DEPENDENCIES = {"world_camera": ("fast3r",), "world_scene": ("fast3r",),
                      "world_object": ("fast3r", "foundationpose"), "world_hand": ("fast3r", "hamer")}


def resolve_outputs(outputs):
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Recipe outputs must be a nonempty list; use data catalog")
    selections = {}
    for item in outputs:
        if isinstance(item, str):
            name, fields = item, None
        elif isinstance(item, dict) and set(item) <= {"type", "fields"} and "type" in item:
            name, fields = item["type"], item.get("fields")
        else:
            raise ValueError("Each output is a data type name or {type, fields}")
        if name not in DATA_TYPES:
            raise ValueError(f"Unknown data type {name!r}; use data catalog")
        if name in selections:
            raise ValueError(f"Duplicate output {name}")
        definition = DATA_TYPES[name]
        allowed = definition.fields
        if fields is not None and (not isinstance(fields, list) or not fields or
                                   any(not isinstance(f, str) or f not in allowed for f in fields) or len(fields) != len(set(fields))):
            raise ValueError(f"Invalid fields for {name}; allowed: {list(allowed)}")
        selections[name] = tuple(fields) if fields is not None else allowed
    kernels = set()
    def add(kernel):
        for dep in STAGE_DEPENDENCIES[kernel]:
            add(dep)
        kernels.add(kernel)
    for name in selections:
        definition = DATA_TYPES[name]
        for kernel in WORLD_DEPENDENCIES.get(name, (definition.kernel,)):
            add(kernel)
    graph = {name: deps for name, deps in STAGE_DEPENDENCIES.items() if name in kernels}
    return selections, graph


def catalog():
    return {name: {"kernel": spec.kernel, "fields": list(spec.fields), "description": spec.description,
                   "requires": list(WORLD_DEPENDENCIES.get(name, STAGE_DEPENDENCIES.get(spec.kernel, ())))}
            for name, spec in DATA_TYPES.items()}
