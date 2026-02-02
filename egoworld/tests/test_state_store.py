import os
import tempfile

from egoworld.pipeline.state_store import (
    bulk_insert_pending,
    get_clip_state,
    get_resumable_clips,
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


def test_get_resumable_clips_includes_running_and_writing():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "state.db")
        init_db(db_path)
        clips = [
            {"clip_id": "c1", "video_id": "v1"},
            {"clip_id": "c2", "video_id": "v1"},
            {"clip_id": "c3", "video_id": "v1"},
            {"clip_id": "c4", "video_id": "v1"},
            {"clip_id": "c5", "video_id": "v1"},
        ]
        bulk_insert_pending(db_path, clips)
        upsert_clip_status(db_path, "c2", "v1", "Done", "", 0)
        upsert_clip_status(db_path, "c3", "v1", "Running", "", 0)
        upsert_clip_status(db_path, "c4", "v1", "Writing", "", 0)
        upsert_clip_status(db_path, "c5", "v1", "Failed", "oops", 1)

        resumable = set(get_resumable_clips(db_path))
        assert "c1" in resumable
        assert "c3" in resumable
        assert "c4" in resumable
        assert "c5" in resumable
        assert "c2" not in resumable

        failed_only = set(get_resumable_clips(db_path, statuses=("Failed",)))
        assert failed_only == {"c5"}
