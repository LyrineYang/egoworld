"""Compare saved hand points with an independent, same-input upstream forward.

Run in the native HaMeR environment. This checks numerical reproduction for
every saved row, including left-hand reflection; it does not measure accuracy.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch


def verify(acceptances, output, preview_dir=None):
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.models import load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full

    original_cwd = Path.cwd()
    model = cfg = model_key = None
    cases = []
    try:
        for acceptance in acceptances:
            acceptance = Path(acceptance).resolve()
            report = json.loads(acceptance.read_text())
            if report["status"] != "passed":
                raise ValueError(f"Acceptance did not pass: {acceptance}")
            run = Path(report["outputs"]["run_dir"])
            requests = list(run.glob(".work/*/hamer/request.json"))
            if len(requests) != 1:
                raise ValueError("Verifier requires one retained HaMeR request per acceptance")
            request = json.loads(requests[0].read_text())
            params = request["config"]["operators"]["hamer"]["params"]
            key = (params["working_dir"], params["checkpoint"], params.get("device", "cuda"))
            os.chdir(params["working_dir"])
            if key != model_key:
                model, cfg = load_hamer(params["checkpoint"])
                model = model.to(key[2]).eval()
                model_key = key
            folder = next(run.glob("video_id=*/clip_id=*"))
            rows = pq.ParquetFile(folder / "hand_pose.parquet").read().to_pylist()
            if not rows:
                raise ValueError("Hand output is empty")
            instances = pq.ParquetFile(Path(request["inputs"]["sam2"]["directory"]) /
                                       "instances.parquet").read().to_pylist()
            detections = {(r["frame_index"], r["object_id"]): r for r in instances}
            cap = cv2.VideoCapture(request["prepared"]["video_path"])
            comparisons, tiles = [], []
            try:
                for row in rows:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            row["frame_index"] - request["job"]["clip"]["frame_start"])
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError("Cannot decode saved input frame")
                    detection = detections[(row["frame_index"], row["object_id"])]
                    right = int(row["handedness"] == "right")
                    if row["handedness"] not in ("left", "right"):
                        raise ValueError("Unknown handedness")
                    sign = 2 * right - 1
                    dataset = ViTDetDataset(cfg, image, np.array([detection["bbox_xyxy"]]),
                                           np.array([right]),
                                           rescale_factor=params.get("rescale_factor", 2.0))
                    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1,
                                                                  shuffle=False, num_workers=0)))
                    batch = recursive_to(batch, key[2])
                    with torch.inference_mode():
                        prediction = model(batch)
                    camera = prediction["pred_cam"].clone()
                    camera[:, 1] *= sign
                    focal = float(params.get("focal_length_px") or
                                  cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * batch["img_size"].max())
                    full = cam_crop_to_full(camera, batch["box_center"], batch["box_size"],
                                            batch["img_size"], focal)
                    raw = prediction["pred_keypoints_3d"].detach().cpu().numpy()[0].copy()
                    raw[:, 0] *= sign
                    expected = raw + full.detach().cpu().numpy()[0]
                    local = np.asarray(row["joints_3d"]).reshape(21, 3)
                    saved = local + row["camera_translation"]
                    if (not np.isfinite(saved).all() or np.any(saved[:, 2] <= 0) or
                            not np.allclose(local[0], 0, atol=1e-7)):
                        raise ValueError("Invalid wrist-relative or camera-space output")
                    def project(points):
                        return points[:, :2] / points[:, 2:] * focal + np.array(
                            [image.shape[1] / 2, image.shape[0] / 2])
                    error = float(np.abs(saved - expected).max())
                    pixel_error = float(np.linalg.norm(project(saved) - project(expected), axis=1).max())
                    if error >= 1e-4 or pixel_error >= .1:
                        raise ValueError(f"Upstream mismatch: {error} m, {pixel_error} px")
                    comparisons.append({"frame_index": row["frame_index"], "object_id": row["object_id"],
                        "handedness": row["handedness"], "max_camera_difference_meters": error,
                        "max_projection_difference_pixels": pixel_error})
                    if preview_dir:
                        xy = np.rint(project(saved)).astype(int)
                        for finger, color in enumerate([(0, 80, 255), (0, 220, 255), (80, 220, 0),
                                                        (255, 200, 0), (220, 0, 220)]):
                            chain = [0] + list(range(1 + finger * 4, 5 + finger * 4))
                            cv2.polylines(image, [xy[chain]], False, color, 1, cv2.LINE_AA)
                        for point in xy:
                            cv2.circle(image, tuple(point), 2, (255, 255, 255), -1)
                        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (100, 255, 100), 1)
                        image = cv2.resize(image, (426, 240))
                        cv2.putText(image, f"frame {row['frame_index']} / id {row['object_id']} / {row['handedness']}",
                                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
                        tiles.append(image)
            finally:
                cap.release()
            if preview_dir:
                preview_dir.mkdir(parents=True, exist_ok=True)
                columns = 3
                tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % columns)
                sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
                cv2.imwrite(str(preview_dir / f"{acceptance.parents[1].name}.png"), sheet)
            cases.append({"acceptance": str(acceptance), "rows": len(rows), "comparisons": comparisons})
    finally:
        os.chdir(original_cwd)
    result = {"status": "passed", "cases": cases,
              "scope": "Independent upstream forward and cam_crop_to_full on the same RGB, boxes and supplied "
                       "handedness; numerical reproduction, not ground-truth accuracy or handedness classification."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    result = verify([path.resolve() for path in args.acceptance], args.output.resolve(),
                    args.preview_dir.resolve() if args.preview_dir else None)
    print(json.dumps({"status": result["status"], "rows_per_case": [c["rows"] for c in result["cases"]]}))


if __name__ == "__main__":
    main()
