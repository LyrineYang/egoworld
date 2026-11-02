"""CPU-only cache-format experiment on real reproduction artifacts.

This measures codecs and serialization, not Data-Juicer or GPU throughput.
The input is never tiled: ratios reflect only the specified short samples.
Writes are buffered (no fsync); reads have warm OS caches. Outputs may be
placed on a mounted drive without changing the production pipeline.
"""
import argparse
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path
import resource
import statistics
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha(data):
    return hashlib.sha256(data).hexdigest()


def timed(call, repeats):
    observations = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = call()
        observations.append(time.perf_counter() - start)
    return value, {"median_ms": 1000 * statistics.median(observations),
                   "samples_ms": [1000 * t for t in observations]}


def codec_pair(name, size):
    if name == "none":
        return lambda x: x, lambda x: x
    if name == "gzip":
        return lambda x: gzip.compress(x, compresslevel=9, mtime=0), gzip.decompress
    codec = pa.Codec(name)
    return (lambda x: codec.compress(x).to_pybytes(),
            lambda x: codec.decompress(x, decompressed_size=size).to_pybytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Frozen product reproduction root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repeats")
    import cv2
    args.output.mkdir(parents=True, exist_ok=True)
    raw_root = args.root / "data-runs"
    hand_root = raw_root / "run_id=release_chicken-hands" / "model-output"
    scene_root = raw_root / "run_id=verified_rgbd-world" / "model-output"
    def unique(root, name):
        paths = [p for p in root.rglob(name) if any(part.startswith("clip_id=") for part in p.parts)]
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one {name}: {paths}")
        return paths[0]
    paths = {"instances": unique(hand_root, "instances.parquet"),
             "hand_pose": unique(hand_root, "hand_pose.parquet"),
             "pointcloud_npz": unique(scene_root, "reconstruction-pointcloud.npz")}
    video = args.root / "samples/epic-chicken-3s.mp4"
    reader = cv2.VideoCapture(str(video))
    frames = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        reader.release()
    if not frames:
        raise ValueError("No video frames decoded")
    buffer = io.BytesIO()
    np.save(buffer, np.stack(frames), allow_pickle=False)
    payloads = {"decoded_frames_npy": buffer.getvalue()}
    with np.load(paths["pointcloud_npz"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payloads["pointcloud_uncompressed_npz"] = buffer.getvalue()
    for name, path in paths.items():
        payloads[name + "_existing_file"] = path.read_bytes()
    report = {"version": 1, "scope": "Codec microbenchmark, NOT a Data-Juicer comparison; warm page cache, buffered I/O, no fsync; tiny real samples, no tiling",
              "platform": platform.platform(), "numpy": np.__version__,
              "pyarrow": pa.__version__, "opencv": cv2.__version__,
              "repeats": args.repeats,
              "inputs": {k: {"path": str(p), "bytes": p.stat().st_size,
                              "sha256": sha(p.read_bytes())} for k, p in paths.items()},
              "video": {"path": str(video), "sha256": sha(video.read_bytes()),
                        "decoded_shape": list(np.stack(frames).shape)},
              "blob_results": [], "parquet_results": []}
    for name, raw in payloads.items():
        for codec in ("none", "gzip", "zstd", "lz4_frame"):
            encode, decode = codec_pair(codec, len(raw))
            encoded, enc_time = timed(lambda: encode(raw), args.repeats)
            decoded, dec_time = timed(lambda: decode(encoded), args.repeats)
            assert decoded == raw, (name, codec)
            output = args.output / f"{name}.{codec}"
            _, write_time = timed(lambda: output.write_bytes(encoded), args.repeats)
            disk_bytes, read_time = timed(output.read_bytes, args.repeats)
            assert disk_bytes == encoded
            report["blob_results"].append({"sample": name, "codec": codec,
                "input_bytes": len(raw), "stored_bytes": len(encoded),
                "stored_fraction": len(encoded) / len(raw), "input_sha256": sha(raw),
                "encode": enc_time, "decode": dec_time,
                "buffered_write": write_time, "warm_read": read_time,
                "lossless_roundtrip": True})
    for name in ("instances", "hand_pose"):
        table = pq.ParquetFile(paths[name]).read()
        for codec in ("none", "zstd", "snappy", "lz4"):
            def encode_table():
                target = io.BytesIO()
                pq.write_table(table, target, compression=codec,
                               row_group_size=65536, data_page_size=1048576)
                return target.getvalue()
            encoded, write_time = timed(encode_table, args.repeats)
            decoded, read_time = timed(lambda: pq.ParquetFile(io.BytesIO(encoded)).read(), args.repeats)
            assert table.equals(decoded), (name, codec)
            (args.output / f"{name}.reencoded.{codec}.parquet").write_bytes(encoded)
            report["parquet_results"].append({"sample": name, "codec": codec,
                "rows": table.num_rows, "stored_bytes": len(encoded),
                "encode": write_time, "decode": read_time,
                "lossless_roundtrip": True})
    report["whole_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["memory_scope"] = "Whole experiment process high-water mark; NOT isolated codec peak or pipeline memory"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output / 'report.json'),
                      "blob_cases": len(report['blob_results']),
                      "parquet_cases": len(report['parquet_results']),
                      "all_roundtrips_passed": True}))


if __name__ == "__main__":
    main()
