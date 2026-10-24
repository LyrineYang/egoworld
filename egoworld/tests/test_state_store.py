import os
import tempfile
import sqlite3

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


def test_runs_keep_independent_clip_states_and_dead_letters(tmp_path):
    db = str(tmp_path / "state.db")
    init_db(db)
    clips = [{"clip_id": "c1", "video_id": "v1"}]
    for run_id in ("A", "B"):
        bulk_insert_pending(db, clips, run_id=run_id)
    upsert_clip_status(db, "c1", "v1", "Done", run_id="A")
    upsert_clip_status(db, "c1", "v1", "Failed", "bad", 2, run_id="B")
    mark_dead_letter(db, "c1", "v1", "bad", run_id="B")
    assert get_resumable_clips(db, run_id="A") == []
    assert get_resumable_clips(db, run_id="B") == ["c1"]
    assert get_clip_state(db, "c1", run_id="A").status == "Done"
    assert get_clip_state(db, "c1", run_id="B").retry_count == 2
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT run_id, error FROM dead_letter").fetchall() == [("B", "bad")]


def test_legacy_database_migration_preserves_records(tmp_path):
    db = str(tmp_path / "legacy.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE clip_status (clip_id TEXT PRIMARY KEY, video_id TEXT, status TEXT, last_error TEXT, retry_count INTEGER, updated_at REAL)")
        conn.execute("INSERT INTO clip_status VALUES ('c1', 'v1', 'Done', '', 2, 123.0)")
        conn.execute("CREATE TABLE dead_letter (clip_id TEXT, video_id TEXT, error TEXT, updated_at REAL)")
        conn.execute("INSERT INTO dead_letter VALUES ('c2', 'v1', 'bad', 123.0)")
    init_db(db)
    init_db(db)
    legacy = get_clip_state(db, "c1")
    assert (legacy.status, legacy.retry_count, legacy.updated_at) == ("Done", 2, 123.0)
    bulk_insert_pending(db, [{"clip_id": "c1", "video_id": "v1"}], run_id="new")
    assert get_resumable_clips(db, run_id="new") == ["c1"]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT run_id, error FROM dead_letter").fetchall() == [("", "bad")]
