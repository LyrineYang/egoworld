"""Output writers with fixed Parquet parameters and atomic writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import os

from egoworld.config import ParquetConfig


def _pa():  # pragma: no cover - optional dependency
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def _empty_table(schema):
    pa, _ = _pa()
    data = {field.name: pa.array([], type=field.type) for field in schema}
    return pa.table(data, schema=schema)


def write_parquet_table(
    path: str,
    rows: List[Dict[str, Any]],
    schema=None,
    parquet: Optional[ParquetConfig] = None,
) -> None:
    pa, pq = _pa()
    parquet = parquet or ParquetConfig()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"

    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        if schema is None:
            schema = pa.schema([])
        table = _empty_table(schema)

    pq.write_table(
        table,
        tmp_path,
        compression=parquet.compression,
        row_group_size=parquet.row_group_size,
        data_page_size=parquet.data_page_size,
    )
    os.replace(tmp_path, path)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    os.replace(tmp_path, path)


def write_run_manifest(path: str, manifest: Dict[str, Any]) -> None:
    write_json(path, manifest)


def write_json_lines(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(tmp_path, path)
