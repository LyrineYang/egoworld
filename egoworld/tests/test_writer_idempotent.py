import os
import tempfile

import pytest

from egoworld.io.writers import write_json


def test_write_json_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "meta.json")
        payload = {"a": 1}
        write_json(path, payload)
        assert os.path.exists(path)
        write_json(path, payload)
        assert os.path.exists(path)


def test_parquet_write_if_available():
    pyarrow = pytest.importorskip("pyarrow")
    from egoworld.io.writers import write_parquet_table

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.parquet")
        write_parquet_table(path, [{"x": 1}], schema=pyarrow.schema([("x", pyarrow.int64())]))
        assert os.path.exists(path)
