"""Read retargeting output independently and check URDF limits and fingertip FK.

Run in the isolated Dex environment. This checks kinematic validity, not pose
accuracy, collision freedom or correspondence with a source video.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq


def verify(mapping, urdf):
    rows = pq.ParquetFile(mapping).read().to_pylist()
    if not rows:
        raise ValueError("Mapping is empty")
    joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf).getroot().findall("joint")
              if joint.attrib["type"] != "fixed"}
    # This verifier targets the fixed-base Allegro sample: one angle per joint.
    if any(joint.attrib["type"] != "revolute" for joint in joints.values()):
        raise ValueError("Verifier requires revolute joints and a fixed base")
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    tips = sorted(frame.name for frame in model.frames if frame.name.endswith("_tip")
                  and frame.type == pin.BODY)
    if not tips:
        raise ValueError("No fingertip frames in URDF")
    minimum_margin = float("inf")
    violations = []
    positions = []
    for row in rows:
        names, values = row["joint_names"], row["joint_positions"]
        if len(names) != len(set(names)) or set(names) != set(joints):
            raise ValueError("Output joint names do not match the URDF")
        if len(values) != len(names) or row["joint_units"] != ["radians"] * len(names):
            raise ValueError("Joint values or units do not match")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite joint positions")
        qpos = pin.neutral(model)
        for name, value in zip(names, values):
            limit = joints[name].find("limit")
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            margin = min(value - lower, upper - value)
            minimum_margin = min(minimum_margin, margin)
            if margin < -1e-6:
                violations.append({"frame_index": row["frame_index"], "joint": name,
                                   "position": value, "lower": lower, "upper": upper,
                                   "excess_radians": -margin})
            joint = model.joints[model.getJointId(name)]
            if joint.nq != 1:
                raise ValueError("Expected one angle per joint")
            qpos[joint.idx_q] = value
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        sample = []
        for name in tips:
            pose = data.oMf[model.getFrameId(name, pin.BODY)].homogeneous
            rotation = pose[:3, :3]
            if (not np.isfinite(pose).all() or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
                    not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
                raise ValueError("Invalid fingertip forward kinematics")
            sample.append(pose[:3, 3].tolist())
        positions.append(sample)
    return {"status": "review" if violations else "passed", "rows": len(rows), "joints": len(joints),
            "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "minimum_joint_limit_margin_radians": minimum_margin,
            "strict_limit_violations": violations, "forward_kinematics_valid": True,
            "fingertip_names": tips, "fingertip_positions_meters": positions,
            "scope": "Independent Parquet/URDF read and Pinocchio FK; no optimizer calls. "
                     "Not pose accuracy, collision checking or a video-to-robot validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.mapping, args.urdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "fingertip_positions_meters"}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
