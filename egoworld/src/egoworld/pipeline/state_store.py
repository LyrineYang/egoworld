"""SQLite-backed state store for clip processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import sqlite3
import time


@dataclass
class ClipState:
    clip_id: str
    video_id: str
    status: str
    last_error: str
    retry_count: int
    updated_at: float
    run_id: str = ""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(clip_status)")}
        legacy = bool(columns) and "run_id" not in columns
        if legacy:
            conn.execute("ALTER TABLE clip_status RENAME TO clip_status_legacy")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_status (
                run_id TEXT NOT NULL DEFAULT '',
                clip_id TEXT NOT NULL,
                video_id TEXT,
                status TEXT,
                last_error TEXT,
                retry_count INTEGER,
                updated_at REAL,
                PRIMARY KEY (run_id, clip_id)
            )
            """
        )
        if legacy:
            conn.execute(
                """
                INSERT INTO clip_status
                (run_id, clip_id, video_id, status, last_error, retry_count, updated_at)
                SELECT '', clip_id, video_id, status, last_error, retry_count, updated_at
                FROM clip_status_legacy
                """
            )
            conn.execute("DROP TABLE clip_status_legacy")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_letter (
                run_id TEXT NOT NULL DEFAULT '',
                clip_id TEXT,
                video_id TEXT,
                error TEXT,
                updated_at REAL
            )
            """
        )
        dead_letter_columns = {row[1] for row in conn.execute("PRAGMA table_info(dead_letter)")}
        if "run_id" not in dead_letter_columns:
            conn.execute("ALTER TABLE dead_letter ADD COLUMN run_id TEXT NOT NULL DEFAULT ''")
        conn.commit()


def upsert_clip_status(
    path: str,
    clip_id: str,
    video_id: str,
    status: str,
    last_error: str = "",
    retry_count: int = 0,
    *,
    run_id: str = "",
) -> None:
    now = time.time()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO clip_status (run_id, clip_id, video_id, status, last_error, retry_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, clip_id) DO UPDATE SET
                video_id=excluded.video_id,
                status=excluded.status,
                last_error=excluded.last_error,
                retry_count=excluded.retry_count,
                updated_at=excluded.updated_at
            """,
            (run_id, clip_id, video_id, status, last_error, retry_count, now),
        )
        conn.commit()


def mark_dead_letter(path: str, clip_id: str, video_id: str, error: str, *, run_id: str = "") -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO dead_letter (run_id, clip_id, video_id, error, updated_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, clip_id, video_id, error, time.time()),
        )
        conn.commit()


def get_resumable_clips(
    path: str, statuses: Optional[Sequence[str]] = None, *, run_id: str = "",
) -> List[str]:
    statuses = statuses or ("Pending", "Failed", "Writing", "Running")
    placeholders = ",".join(["?"] * len(statuses))
    query = f"SELECT clip_id FROM clip_status WHERE run_id=? AND status IN ({placeholders})"
    with sqlite3.connect(path) as conn:
        rows = conn.execute(query, (run_id, *statuses)).fetchall()
    return [row[0] for row in rows]


def get_clip_state(path: str, clip_id: str, *, run_id: str = "") -> Optional[ClipState]:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT clip_id, video_id, status, last_error, retry_count, updated_at FROM clip_status WHERE run_id=? AND clip_id=?",
            (run_id, clip_id),
        ).fetchone()
    if not row:
        return None
    return ClipState(*row, run_id=run_id)


def bulk_insert_pending(path: str, clips: Iterable[dict], *, run_id: str = "") -> None:
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO clip_status
            (run_id, clip_id, video_id, status, last_error, retry_count, updated_at)
            VALUES (?, ?, ?, 'Pending', '', 0, ?)
            """,
            [(run_id, clip["clip_id"], clip["video_id"], time.time()) for clip in clips],
        )
        conn.commit()


def init_stage_db(path: str) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS stage_status (
            run_id TEXT, clip_id TEXT, stage TEXT, status TEXT, attempt INTEGER,
            error TEXT, updated_at REAL, PRIMARY KEY(run_id, clip_id, stage))""")


def set_stage(path: str, run_id: str, clip_id: str, stage: str, status: str,
              attempt: int = 0, error: str = "") -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""INSERT INTO stage_status VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, clip_id, stage) DO UPDATE SET
            status=excluded.status, attempt=excluded.attempt, error=excluded.error, updated_at=excluded.updated_at""",
            (run_id, clip_id, stage, status, attempt, error, time.time()))


def run_counts(path: str, run_id: str) -> dict:
    with sqlite3.connect(path) as conn:
        return dict(conn.execute("SELECT status, COUNT(*) FROM clip_status WHERE run_id=? GROUP BY status", (run_id,)))
