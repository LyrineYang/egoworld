"""Wrist coordinates from dex-retargeting's reference hand detector.

The input has OpenPose/MediaPipe's wrist, index MCP and middle MCP at 0, 5, 9.
This is orientation normalization, not learned hand tracking or calibration.
"""
import numpy as np


def dex_wrist_coordinates(joints, side):
    points = np.asarray(joints, dtype=float).reshape(21, 3)
    if side not in ("left", "right") or not np.isfinite(points).all():
        raise ValueError("Wrist normalization requires finite joints and a known side")
    points = points - points[0]
    palm = points[[0, 5, 9]]
    x_vector = palm[0] - palm[2]
    centered = palm - palm.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered)
    if singular[0] < 1e-8 or singular[1] / singular[0] < 1e-4:
        raise ValueError("Degenerate palm cannot define a wrist frame")
    normal = basis[2]
    x = x_vector - np.dot(x_vector, normal) * normal
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("Degenerate palm longitudinal axis")
    x /= np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.dot(z, centered[1] - centered[2]) < 0:
        normal = -normal
        z = -z
    frame = np.stack([x, normal, z], axis=1)
    operator_to_mano = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]] if side == "right"
                                else [[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    return points @ frame @ operator_to_mano
