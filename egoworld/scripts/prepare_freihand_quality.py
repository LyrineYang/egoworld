#!/usr/bin/env python3
"""Freeze a seeded FreiHAND evaluation subset, downloading only ZIP members.

This is an image-pose pilot: repeat each image as three identical video frames
for the native video interfaces, and score only frame 1. No temporal claims.
Original dataset terms apply; do not redistribute assets in the Git repository.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
import zipfile

URL = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip"
TERMS = "https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class RangeFile(io.RawIOBase):
    """Seekable verified HTTP ranges for zipfile, using the host curl TLS stack."""
    def __init__(self, url, directory):
        self.url, self.directory, self.position, self.requests = url, Path(directory), 0, []
        header = subprocess.check_output(["curl", "-fsSIL", "--retry", "2", "--max-time", "60", url], text=True)
        lengths = re.findall(r"(?im)^content-length:\s*(\d+)", header)
        tags = re.findall(r"(?im)^etag:\s*(.+)", header)
        if not lengths or not tags:
            raise ValueError("Archive requires stable Content-Length and ETag")
        self.size, self.etag = int(lengths[-1]), tags[-1].strip()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid remote ZIP seek")
        return self.position

    def read(self, length=-1):
        length = min(self.size - self.position, length if length >= 0 else self.size)
        if length == 0:
            return b""
        if length > 16 * 1024**2:
            raise ValueError("Refusing large range; only annotations/selected images are required")
        start, end = self.position, self.position + length - 1
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            header, body = Path(tmp) / "headers", Path(tmp) / "body"
            subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "90",
                "--max-filesize", str(length), "--range", f"{start}-{end}",
                "--header", "If-Range: " + self.etag, "--dump-header", str(header),
                "--output", str(body), self.url], check=True)
            headers = header.read_text()
            expected = f"bytes {start}-{end}/{self.size}"
            ranges = re.findall(r"(?im)^content-range:\s*(.+)", headers)
            tags = re.findall(r"(?im)^etag:\s*(.+)", headers)
            if not ranges or ranges[-1].strip() != expected or not tags or tags[-1].strip() != self.etag:
                raise ValueError("Server ignored range or archive identity changed")
            data = body.read_bytes()
        if len(data) != length:
            raise ValueError("Incomplete archive range")
        self.requests.append({"start": start, "end": end, "bytes": length, "sha256": sha(data)})
        self.position += length
        return data


def prepare(args):
    import cv2
    import numpy as np
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Freeze selection before model inference; never select on model results.
    rng = random.Random(args.seed)
    remote = None
    if args.archive:
        archive = zipfile.ZipFile(args.archive)
        archive_identity = {"path": str(Path(args.archive).resolve()), "sha256": file_sha(args.archive)}
    else:
        remote = RangeFile(URL, output)
        archive = zipfile.ZipFile(remote)
        archive_identity = {"url": URL, "size": remote.size, "etag": remote.etag}
    records = []
    def member(name):
        candidates = [p for p in archive.namelist() if p == name or p.endswith("/" + name)]
        if len(candidates) != 1:
            raise ValueError(f"Missing/ambiguous archive member {name}")
        name = candidates[0]
        data = archive.read(name)  # zipfile also verifies member CRC32.
        records.append({"member": name, "bytes": len(data), "sha256": sha(data)})
        return data
    xyz_data = member("evaluation_xyz.json")
    camera_data = member("evaluation_K.json")
    xyz, cameras = json.loads(xyz_data), json.loads(camera_data)
    if not len(xyz) == len(cameras) == 3960 or not 1 <= args.samples <= len(xyz):
        raise ValueError("Unexpected FreiHAND evaluation cardinality or sample count")
    selected = sorted(rng.sample(range(len(xyz)), args.samples))
    (output / "selection.json").write_text(json.dumps({"seed": args.seed, "indices": selected,
        "rule": "Python Random.sample without replacement over all 3960 evaluation images, before inference"}, indent=2) + "\n")
    samples, truth = {}, []
    for index in selected:
        name = f"freihand-eval-{index:08d}"
        image_bytes = member(f"evaluation/rgb/{index:08d}.jpg")
        image_path = output / (name + ".jpg")
        image_path.write_bytes(image_bytes)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (224, 224, 3):
            raise ValueError("Unexpected official image shape")
        video = output / (name + ".mp4")
        subprocess.run([args.ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", "224x224", "-framerate", "1", "-i", "pipe:0", "-frames:v", "3",
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-threads", "1", str(video)],
            input=image.tobytes() * 3, check=True)
        points = np.asarray(xyz[index], dtype=float)
        intrinsic = np.asarray(cameras[index], dtype=float)
        if points.shape != (21, 3) or intrinsic.shape != (3, 3) or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            raise ValueError("Invalid official annotations")
        projection = points @ intrinsic.T
        projection = projection[:, :2] / projection[:, 2:]
        samples[name] = {"official_index": index, "image_path": str(image_path), "image_sha256": sha(image_bytes),
            "video_path": str(video), "video_sha256": sha(video.read_bytes()), "image_size": [224, 224],
            "native_frame_indices": [0, 1, 2], "score_frames": [1], "intrinsics": intrinsic.tolist()}
        truth.append({"sample_id": name, "frame_index": 1, "hand_id": "0", "handedness": "right",
            "joints_camera_m": points.tolist(), "joints_image_px": projection.tolist()})
        print(f"Prepared {name}", flush=True)
    archive.close()
    corpus = {"version": 1, "dataset": "FreiHAND official evaluation v2", "terms_url": TERMS,
        "terms": "Research use only; commercial use prohibited; cite Zimmermann et al. ICCV2019",
        "selection_seed": args.seed, "selection_indices": selected, "archive": archive_identity,
        "archive_members": records, "range_requests": remote.requests if remote else [],
        "ground_truth_joint_order": "openpose21", "ground_truth_unit": "meters",
        "scope": "Small static image-pose pilot through real video interfaces; frame1 scored once per source image",
        "limits": ["Not a temporal or egocentric generalization benchmark.",
            "Evaluation split; pretrained model training membership not independently audited.",
            "Images can be correlated; no claim of independent subjects/captures or confidence intervals."],
        "samples": samples, "ground_truth": truth}
    (output / "corpus.json").write_text(json.dumps(corpus, indent=2, allow_nan=False) + "\n")
    print(f"Frozen {len(samples)} images; source assets remain outside the Git repository", flush=True)


def make_montage(corpus_path, output_path, ffmpeg):
    """One video avoids replicating CPU models across a many-row dataset.

    This is an explicit data layout, not a modification to native resource
    policy. Each source image gets a separate EgoWorld three-frame clip; native
    WiLoR is configured to sample every frame of the common 36-frame video.
    """
    import cv2
    path, output = Path(corpus_path), Path(output_path).resolve()
    corpus = json.loads(path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    chunks, truth, sources = [], [], {}
    name = f"freihand-pilot-{len(corpus['samples'])}"
    for i, (source, sample) in enumerate(corpus["samples"].items()):
        image_bytes = Path(sample["image_path"]).read_bytes()
        if sha(image_bytes) != sample["image_sha256"]:
            raise ValueError("Source image changed")
        image = cv2.imread(sample["image_path"])
        chunks.extend([image.tobytes()] * 3)
        gt = next(r for r in corpus["ground_truth"] if r["sample_id"] == source)
        truth.append({**gt, "sample_id": name, "source_image_id": source, "frame_index": 3 * i + 1})
        sources[source] = {**sample, "montage_frames": [3 * i, 3 * i + 1, 3 * i + 2]}
    video = output / (name + ".mp4")
    subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "224x224", "-framerate", "1", "-i", "pipe:0", "-frames:v", str(len(chunks)),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-threads", "1", str(video)],
        input=b"".join(chunks), check=True)
    corpus.update(source_corpus=str(path.resolve()), source_corpus_sha256=sha(path.read_bytes()),
        source_images=sources, ground_truth=truth,
        samples={name: {"video_path": str(video), "video_sha256": sha(video.read_bytes()),
            "image_size": [224, 224], "native_frame_indices": list(range(len(chunks))),
            "score_frames": [r["frame_index"] for r in truth]}},
        layout="One 1fps montage, three identical frames per original image, no temporal quality claim")
    (output / "corpus.json").write_text(json.dumps(corpus, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--archive", help="Optional already-downloaded official evaluation ZIP")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--from-corpus", help="Build one montage from a previously frozen per-image corpus")
    args = parser.parse_args()
    if args.from_corpus:
        make_montage(args.from_corpus, args.output, args.ffmpeg)
    else:
        prepare(args)
