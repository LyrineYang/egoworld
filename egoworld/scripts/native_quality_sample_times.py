#!/usr/bin/env python3
"""Run in the native DJ environment: observe stock uniform sampler PTS only."""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    import av
    from data_juicer.utils.mm_utils import extract_video_frames_uniformly
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame-num", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    with av.open(args.video) as container:
        rate = float(container.streams.video[0].average_rate)
        frames = extract_video_frames_uniformly(container, args.frame_num)
        records = [{"ordinal": i, "pts": f.pts, "time_base": str(f.time_base),
            "time_seconds": float(f.pts * f.time_base),
            "source_frame": round(float(f.pts * f.time_base) * rate)} for i, f in enumerate(frames)]
    path.write_text(json.dumps({"video": str(Path(args.video).resolve()), "rate": rate,
        "method": "Unmodified native DJ uniform sampler PTS; no model call", "frames": records}, indent=2) + "\n")
