"""Sequential, per-hand robot retargeting through dex-retargeting's optimizer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from egoworld.operators.common import rows


class DexRetargetOperator:
    name = "dex_retarget"

    def __init__(self, **params):
        self.params = params

    def _build(self):
        from dex_retargeting.retargeting_config import RetargetingConfig
        if self.params.get("urdf_dir"):
            RetargetingConfig.set_default_urdf_dir(Path(self.params["urdf_dir"]))
        return RetargetingConfig.load_from_file(self.params["config"]).build()

    def run(self, clip, video_path, inputs):
        groups = {}
        for row in rows(inputs, "hamer", "hand_pose"):
            groups.setdefault((row["object_id"], row["handedness"]), []).append(row)
        rotation = np.asarray(self.params["human_to_robot_rotation"], dtype=float).reshape(3, 3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < .99:
            raise ValueError("human_to_robot_rotation must be a proper rotation")
        output = []
        for (_, side), hand_rows in groups.items():
            configured_side = self.params.get("handedness")
            if configured_side not in ("left", "right") or side not in ("left", "right"):
                raise ValueError("Retargeting requires a configured left/right robot hand and known source side")
            if side != configured_side:
                continue
            optimizer = self._build()  # reset optimizer/filter state between clips and hand identities
            indices = np.asarray(optimizer.optimizer.target_link_human_indices)
            joint_names = list(optimizer.joint_names)
            units = self.params["joint_units"]
            if isinstance(units, str):
                units = [units] * len(joint_names)
            if len(units) != len(joint_names):
                raise ValueError("joint_units length must match robot joints")
            for row in sorted(hand_rows, key=lambda r: r["frame_index"]):
                xyz = np.asarray(row["joints_3d"], dtype=np.float32).reshape(21, 3)
                xyz = xyz @ rotation.T * float(self.params.get("scale_factor", 1.0))
                kind = optimizer.optimizer.retargeting_type.upper()
                if kind == "POSITION":
                    reference = xyz[indices]
                elif kind in ("VECTOR", "DEXPILOT"):
                    reference = xyz[indices[1]] - xyz[indices[0]]
                else:
                    raise ValueError(f"Unsupported retargeting type: {kind}")
                qpos = np.asarray(optimizer.retarget(reference), dtype=float)
                if not np.isfinite(qpos).all():
                    raise ValueError("Retargeting returned nonfinite joints")
                output.append({"frame_index": row["frame_index"], "timestamp_s": row["timestamp_s"],
                    "object_id": row["object_id"], "handedness": side, "robot_id": self.params["robot_id"],
                    "joint_names": joint_names, "joint_positions": qpos.tolist(), "joint_units": units,
                    "source_scale_status": row["scale_status"]})
        return {"tables": {"mapping": output}, "stats": {"hands": len(groups)}}
