import os
import tempfile

from egoworld.pipeline.state_store import (
    bulk_insert_pending,
    get_clip_state,
    init_db,
    mark_dead_letter,
    upsert_clip_status,
)


def test_state_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "state.db")
        init_db(db_path)
        clips = [
            {"clip_id": "c1", "video_id": "v1"},
            {"clip_id": "c2", "video_id": "v1"},
        ]
        bulk_insert_pending(db_path, clips)
        upsert_clip_status(db_path, "c1", "v1", "Running", "", 0)
        state = get_clip_state(db_path, "c1")
        assert state is not None
        assert state.status == "Running"
        mark_dead_letter(db_path, "c2", "v1", "bad")
