"""SQLite-backed state store for clip processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
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


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_status (
                clip_id TEXT PRIMARY KEY,
                video_id TEXT,
                status TEXT,
                last_error TEXT,
                retry_count INTEGER,
                updated_at REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_letter (
                clip_id TEXT,
                video_id TEXT,
                error TEXT,
                updated_at REAL
            )
            """
        )
        conn.commit()


def upsert_clip_status(
    path: str,
    clip_id: str,
    video_id: str,
    status: str,
    last_error: str = "",
    retry_count: int = 0,
) -> None:
    now = time.time()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO clip_status (clip_id, video_id, status, last_error, retry_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                status=excluded.status,
                last_error=excluded.last_error,
                retry_count=excluded.retry_count,
                updated_at=excluded.updated_at
            """,
            (clip_id, video_id, status, last_error, retry_count, now),
        )
        conn.commit()


def mark_dead_letter(path: str, clip_id: str, video_id: str, error: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO dead_letter (clip_id, video_id, error, updated_at) VALUES (?, ?, ?, ?)",
            (clip_id, video_id, error, time.time()),
        )
        conn.commit()


def get_pending_clips(path: str, limit: int = 1000) -> List[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT clip_id FROM clip_status WHERE status IN ('Pending','Failed') LIMIT ?",
            (limit,),
        ).fetchall()
    return [row[0] for row in rows]


def get_clip_state(path: str, clip_id: str) -> Optional[ClipState]:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT clip_id, video_id, status, last_error, retry_count, updated_at FROM clip_status WHERE clip_id=?",
            (clip_id,),
        ).fetchone()
    if not row:
        return None
    return ClipState(*row)


def bulk_insert_pending(path: str, clips: Iterable[dict]) -> None:
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO clip_status
            (clip_id, video_id, status, last_error, retry_count, updated_at)
            VALUES (?, ?, 'Pending', '', 0, ?)
            """,
            [(clip["clip_id"], clip["video_id"], time.time()) for clip in clips],
        )
        conn.commit()
