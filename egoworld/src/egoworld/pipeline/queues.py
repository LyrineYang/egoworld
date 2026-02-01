"""Backpressure helpers for Ray and asyncio."""

from __future__ import annotations

import asyncio
from typing import Iterable, List, Tuple


def enforce_in_flight(
    pending: List[object],
    max_in_flight: int,
) -> Tuple[List[object], List[object]]:
    """Block until in-flight count is below max.

    Returns (done_refs, remaining_refs).
    """
    if max_in_flight <= 0:
        return [], pending
    if len(pending) < max_in_flight:
        return [], pending

    try:
        import ray  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ray is required for in-flight enforcement") from exc

    done, remaining = ray.wait(pending, num_returns=1)
    return list(done), list(remaining)


class BoundedAsyncQueue:
    def __init__(self, max_size: int):
        self._queue: asyncio.Queue = asyncio.Queue(max_size)

    async def put(self, item: object) -> None:
        await self._queue.put(item)

    async def get(self) -> object:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()

    def full(self) -> bool:
        return self._queue.full()

    def empty(self) -> bool:
        return self._queue.empty()


async def drain_queue(queue: BoundedAsyncQueue) -> Iterable[object]:
    while not queue.empty():
        yield await queue.get()
