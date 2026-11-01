"""Internal CPU input -> GPU -> CPU output steps of one logical stage.

Reuses ModelActor and the built-in runner in its original model environment.
The handoff contains descriptors only; Writer still owns the sole stage commit.
"""
from pathlib import Path
from egoworld.pipeline.spool import Spool, spool_path


def sam_packet(clip, result):
    frames, instances = result.get("frames", []), result.get("instances", [])
    for row in frames + instances:
        row["frame_index"] += clip["frame_start"]
        row["timestamp_s"] = row["frame_index"] / clip["fps"]
    tracks = [{k: row[k] for k in ("frame_index", "timestamp_s", "object_id", "label", "area_px")} |
              {"center_xy": [(row["bbox_xyxy"][0] + row["bbox_xyxy"][2]) / 2,
                             (row["bbox_xyxy"][1] + row["bbox_xyxy"][3]) / 2] if row["bbox_xyxy"] else None}
              for row in instances]
    return {"tables": {"masks": frames, "instances": instances, "tracks": tracks},
            "stats": {"empty_mask_rate": result.get("empty_mask_rate", 1.),
                      "processed_frames": result.get("processed_frames", len(frames))}}


def run_phase(operator, job, prepared, inputs, execution):
    stage, operation, clip = job["stage"], job["_operation"], job["clip"]
    spool = Spool(spool_path(execution, job["work_dir"], stage),
                  execution["split_stage_bytes"], execution["split_block_bytes"])
    previous = prepared.get("_phase_input", {})
    before = spool.used
    def consume():
        for descriptor in previous["blocks"]:
            data = spool.get(descriptor)
            # No failed step is retried against partially consumed scratch:
            # Engine restarts from CPU input, retaining verified upstream commits.
            spool.discard(descriptor)  # get() already closed its mapping and owns a copy
            yield data
    blocks, packet, units, frame_source = [], None, 0, None
    if operation == "input":
        spool.clear(); spool.root.mkdir(parents=True, exist_ok=True)
        if stage == "hamer":
            batches = operator.prepare_batches(clip, prepared["video_path"], inputs, cross_frame=True)
        elif stage == "foundationpose":
            batches = operator.prepare_frames(clip, prepared["video_path"], inputs)
        elif stage == "fast3r":
            batches = [operator.prepare(clip, prepared["video_path"])]
        elif stage == "sam2":
            from egoworld.operators.sam2_op import _load_prompt_config
            from egoworld.utils.video import iter_frames, get_video_info
            from itertools import islice
            cfg = _load_prompt_config(operator.params.get("prompting", {}))
            fps = get_video_info(prepared["video_path"]).fps
            frame_source = iter_frames(prepared["video_path"], 0., 1e9, max(1, int(round(cfg.prompt_interval_s * fps))))
            batches = islice(frame_source, cfg.max_prompts_per_clip)
        else:
            raise ValueError(f"No split execution for {stage}")
        try:
            for data in batches:
                blocks.append(spool.put(data))
                units += (len(data["metadata"]) if stage == "hamer" else len(data["rows"]) if stage == "foundationpose"
                          else len(data["indices"]) if stage == "fast3r" else 0)
        finally:
            if hasattr(batches, "close"):
                batches.close()
            if frame_source is not None:
                frame_source.close()
    elif operation == "infer":
        if stage == "hamer":
            for data in consume():
                blocks.append(spool.put(operator.infer_batch(data)))
        elif stage == "fast3r":
            for data in consume():
                blocks.append(spool.put(operator.infer(data)))
        elif stage == "foundationpose":
            blocks.append(spool.put(operator.infer_frames(consume())))
        else:
            from egoworld.utils.performance import packet_bytes
            pending, pending_bytes = [], 0
            max_frames = int(operator.params.get("mask_block_frames", 16))
            max_bytes = min(int(operator.params.get("mask_block_bytes", 8 * 1024**2)), spool.block_limit // 2)
            if max_frames < 1 or max_bytes < 1:
                raise ValueError("SAM2 mask block limits must be positive")
            def flush():
                nonlocal pending, pending_bytes
                if pending:
                    blocks.append(spool.put({"frames": pending}))
                    pending, pending_bytes = [], 0
            def receive(data):
                nonlocal pending_bytes
                size = packet_bytes(data)
                if size > max_bytes:
                    raise ValueError("One SAM2 mask packet exceeds mask_block_bytes or split_block_bytes")
                if pending and (len(pending) >= max_frames or pending_bytes + size > max_bytes):
                    flush()
                pending.append(data); pending_bytes += size
            result = operator.run(prepared["video_path"], 0., 0.,
                prepared_prompts=consume() if previous["blocks"] else [], mask_sink=receive)
            flush()
            previous = dict(previous, sam_stats={k: result[k] for k in ("empty_mask_rate", "processed_frames")})
    elif operation == "post":
        if stage == "hamer":
            output = []
            for data in consume():
                output.extend(operator.post_batch(clip, data))
            packet = {"tables": {"hand_pose": output},
                      "stats": {"sampled_frames": len({r["frame_index"] for r in output})}}
        elif stage == "fast3r":
            packet = operator.postprocess(clip, next(consume()))
        elif stage == "foundationpose":
            packet = next(consume())
        else:
            from egoworld.operators.sam2_op import encode_masks
            frames, instances = [], []
            for block in consume():
                for data in block["frames"]:
                    a, b = encode_masks(data); frames.extend(a); instances.extend(b)
            packet = sam_packet(clip, dict(previous["sam_stats"], frames=frames, instances=instances))
    else:
        raise ValueError(f"Unknown internal operation: {operation}")
    if packet is None:
        packet = {"blocks": blocks, "_handoff": True}
        if "sam_stats" in previous:
            packet["sam_stats"] = previous["sam_stats"]
    current = packet.setdefault("stats", {})
    if operation == "input":
        current["work_units"] = units if stage != "sam2" else clip["frame_end"] - clip["frame_start"]
    for key, value in previous.get("stats", {}).items():
        current.setdefault(key, value)
    current["spool_peak_bytes"] = max(current.get("spool_peak_bytes", 0), before, spool.used, spool.peak)
    packet["profiles"] = list(previous.get("profiles", []))
    return packet
