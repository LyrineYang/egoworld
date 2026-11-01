"""Explicit row-vector geometry; matrices use T_destination_from_source."""
import numpy as np


def rigid(value):
    matrix = np.asarray(value, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("Expected a finite proper rigid transform")
    return matrix


def similarity(value):
    scale = float(value["scale"])
    transform = rigid(value["transform"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity scale must be positive")
    return scale, transform


def transform_points(points, transform, scale=1.):
    return np.asarray(points) @ transform[:3, :3].T * scale + transform[:3, 3]


def fit_similarity(source, target):
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("Similarity fit needs at least three finite paired 3D points")
    xc, yc = x - x.mean(0), y - y.mean(0)
    for centered in (xc, yc):
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[0] < 1e-10 or singular[1] / singular[0] < 1e-5:
            raise ValueError("Collinear or coincident alignment points")
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float(np.dot(singular, sign) / np.mean(np.sum(xc * xc, axis=1)))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = y.mean(0) - scale * rotation @ x.mean(0)
    similarity({"scale": scale, "transform": transform})
    return scale, transform


def robust_similarity(source, target, threshold_m, min_inlier_fraction=.5, trials=200):
    x, y = np.asarray(source, float), np.asarray(target, float)
    if len(x) < 12 or x.shape != y.shape or not np.isfinite([x, y]).all():
        raise ValueError("Robust alignment needs at least twelve finite paired points")
    if not 0 < threshold_m or not np.isfinite(threshold_m) or not 0 < min_inlier_fraction <= 1:
        raise ValueError("Invalid alignment acceptance thresholds")
    # Held-out points do not participate in fitting or model selection.
    holdout = np.arange(len(x)) % 5 == 0
    train = np.flatnonzero(~holdout)
    rng = np.random.default_rng(0)
    best = np.zeros(len(train), bool)
    for _ in range(trials):
        sample = rng.choice(train, 3, replace=False)
        try:
            scale, transform = fit_similarity(x[sample], y[sample])
        except ValueError:
            continue
        good = np.linalg.norm(transform_points(x[train], transform, scale) - y[train], axis=1) <= threshold_m
        if good.sum() > best.sum():
            best = good
    if best.sum() < max(6, min_inlier_fraction * len(train)):
        raise ValueError("No alignment satisfies the minimum training inlier fraction")
    scale, transform = fit_similarity(x[train[best]], y[train[best]])
    residual = np.linalg.norm(transform_points(x, transform, scale) - y, axis=1)
    test_good = residual[holdout] <= threshold_m
    if test_good.mean() < min_inlier_fraction:
        raise ValueError("Alignment fails held-out metric correspondence checks")
    return {"scale": scale, "transform": transform.reshape(-1).tolist(),
            "fit": {"pairs": len(x), "threshold_m": threshold_m,
                    "train_inlier_fraction": float((residual[~holdout] <= threshold_m).mean()),
                    "heldout_inlier_fraction": float(test_good.mean()),
                    "heldout_inlier_rmse_m": float(np.sqrt(np.mean(residual[holdout][test_good] ** 2)))}}


def fast3r_source_pixels(pixels, source_size, model_size, output_size):
    """Invert pinned upstream resize + center crop, including pixel centers."""
    width, height = source_size
    long_edge = round(model_size * max(width / height, height / width)) if model_size == 224 else model_size
    resized = np.array([round(width * long_edge / max(width, height)),
                        round(height * long_edge / max(width, height))])
    cx, cy = resized // 2
    if model_size == 224:
        halfw = halfh = min(cx, cy)
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if resized[0] == resized[1]:
            halfh = 3 * halfw / 4
    if tuple(map(int, [2 * halfw, 2 * halfh])) != tuple(output_size):
        raise ValueError("Fast3R preprocessing shape changed; cannot infer pixel correspondence")
    return (np.asarray(pixels) + [cx - halfw, cy - halfh] + .5) * np.array(source_size) / resized - .5
