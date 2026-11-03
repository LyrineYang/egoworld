"""Full-chain equality must cover upstream artifacts and prohibit result reuse."""
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from matched_audit import audit_recipe_pair
from test_product_recipes import recipe_case


def test_complete_recipe_audit_checks_masks_not_only_final_hands(recipe_case):
    data, run, _, root = recipe_case
    data['cache'].update(read=False, write=False)
    first, second = run('a'), run('b')
    manifests = [Path(r['run_dir']) / 'data_manifest.json' for r in (first, second)]
    assert audit_recipe_pair(*manifests)['passed']
    folder = next((Path(second['run_dir']) / 'model-output').glob('run_id=*/video_id=*/clip_id=*'))
    table_path = folder / 'instances.parquet'
    table = pq.ParquetFile(table_path).read()
    rows = table.to_pylist(); rows[0]['label'] = 'different semantic output'
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), table_path)
    stage_path = folder / 'stage_sam2.json'
    receipt = json.loads(stage_path.read_text())
    receipt['files']['instances.parquet'].update(bytes=table_path.stat().st_size,
        sha256=hashlib.sha256(table_path.read_bytes()).hexdigest())
    stage_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='stage/instances'):
        audit_recipe_pair(*manifests)


def test_complete_recipe_audit_rejects_prediction_cache_substitution(recipe_case):
    _, run, _, _ = recipe_case
    first, cached = run('first'), run('cached')
    with pytest.raises(ValueError, match='result caches'):
        audit_recipe_pair(Path(first['run_dir']) / 'data_manifest.json', Path(cached['run_dir']) / 'data_manifest.json')
