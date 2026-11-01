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
        sequence = RetargetingConfig.load_from_file(self.params["config"]).build()
        if sequence.has_joint_limits:
            # Upstream defaults to expanding every bound by 1e-3. Use its
            # public API to keep the configured URDF constraints exact.
            sequence.optimizer.set_joint_limit(sequence.joint_limits, epsilon=0.0)
        return sequence

    def run(self, clip, video_path, inputs):
        groups = {}
        for row in rows(inputs, "hamer", "hand_pose", ["frame_index", "timestamp_s", "object_id", "handedness", "joints_3d", "scale_status"]):
            groups.setdefault((row["object_id"], row["handedness"]), []).append(row)
        rotation = np.asarray(self.params["human_to_robot_rotation"], dtype=float).reshape(3, 3)
        input_transform = self.params.get("input_transform", "configured_rotation")
        if input_transform not in ("configured_rotation", "dex_wrist"):
            raise ValueError("input_transform must be configured_rotation or dex_wrist")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < .99:
            raise ValueError("human_to_robot_rotation must be a proper rotation")
        output = []
        limited_hands = adjustments = 0
        max_adjustment = 0.0
        for (_, side), hand_rows in groups.items():
            configured_side = self.params.get("handedness")
            if configured_side not in ("left", "right") or side not in ("left", "right"):
                raise ValueError("Retargeting requires a configured left/right robot hand and known source side")
            if side != configured_side:
                continue
            optimizer = self._build()  # reset optimizer/filter state between clips and hand identities
            indices = np.asarray(optimizer.optimizer.target_link_human_indices)
            joint_names = list(optimizer.joint_names)
            limits = None
            if optimizer.has_joint_limits:
                robot = optimizer.optimizer.robot
                names = list(robot.dof_joint_names)
                raw_limits = np.asarray(robot.joint_limits, dtype=float)
                if raw_limits.shape != (len(names), 2) or np.isnan(raw_limits).any():
                    raise ValueError("Robot joint limits must match its named DOFs")
                limits = raw_limits[[names.index(name) for name in joint_names]]
                if np.any(limits[:, 0] > limits[:, 1]):
                    raise ValueError("Robot joint lower limit exceeds upper limit")
                limited_hands += 1
            units = self.params["joint_units"]
            if isinstance(units, str):
                units = [units] * len(joint_names)
            if len(units) != len(joint_names):
                raise ValueError("joint_units length must match robot joints")
            for row in sorted(hand_rows, key=lambda r: r["frame_index"]):
                xyz = np.asarray(row["joints_3d"], dtype=np.float32).reshape(21, 3)
                if input_transform == "dex_wrist":
                    from egoworld.hand_coordinates import dex_wrist_coordinates
                    xyz = dex_wrist_coordinates(xyz, side)
                xyz = xyz @ rotation.T * float(self.params.get("scale_factor", 1.0))
                kind = optimizer.optimizer.retargeting_type.upper()
                if kind == "POSITION":
                    reference = xyz[indices]
                elif kind in ("VECTOR", "DEXPILOT"):
                    reference = xyz[indices[1]] - xyz[indices[0]]
                else:
                    raise ValueError(f"Unsupported retargeting type: {kind}")
                qpos = np.asarray(optimizer.retarget(reference), dtype=float)
                if qpos.shape != (len(joint_names),) or not np.isfinite(qpos).all():
                    raise ValueError("Retargeting returned invalid joint positions")
                if limits is not None:
                    bounded = np.clip(qpos, limits[:, 0], limits[:, 1])
                    correction = np.abs(bounded - qpos)
                    largest = float(correction.max(initial=0.0))
                    if largest > 1e-6:
                        raise ValueError("Retargeting exceeded strict URDF joint limits")
                    # Float32 conversion/filtering can round a valid solution
                    # just outside a bound. Correct only this numerical residue.
                    adjustments += int(np.count_nonzero(correction))
                    max_adjustment = max(max_adjustment, largest)
                    qpos = bounded
                output.append({"frame_index": row["frame_index"], "timestamp_s": row["timestamp_s"],
                    "object_id": row["object_id"], "handedness": side, "robot_id": self.params["robot_id"],
                    "joint_names": joint_names, "joint_positions": qpos.tolist(), "joint_units": units,
                    "source_scale_status": row["scale_status"]})
        return {"tables": {"mapping": output}, "stats": {"hands": len(groups),
            "input_transform": input_transform,
            "joint_limit_hands": limited_hands, "joint_limit_adjustments": adjustments,
            "max_joint_limit_adjustment": max_adjustment}}
