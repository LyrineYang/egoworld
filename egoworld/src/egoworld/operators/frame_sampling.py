"""Read nearby increasing indices without re-seeking an inter-frame codec."""


def read_index(cap, index, max_forward_gap=32, phase_factory=None):
    import cv2
    from contextlib import nullcontext
    phase = phase_factory or (lambda name: nullcontext())
    if not isinstance(index, int) or index < 0:
        raise ValueError("Frame index must be a nonnegative integer")
    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    gap = index - current
    if 0 < gap <= max_forward_gap:
        with phase("video_decode_skip"):
            for _ in range(gap):
                if not cap.grab():
                    raise ValueError(f"Cannot advance to frame {index}")
    elif gap:
        with phase("video_seek"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    with phase("video_decode"):
        ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}")
    return frame
