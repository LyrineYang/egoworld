"""Per-invocation records; the root summary remains a backwards-compatible latest view."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

from egoworld.io.writers import write_json


def new_record(root):
    root = Path(root)
    identifier = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid4().hex[:8]
    directory = root / identifier
    directory.mkdir(parents=True, exist_ok=False)
    return identifier, directory


def preserve_legacy_summary(root):
    """Preserve the last pre-history invocation before the first new-format run."""
    root = Path(root)
    summary = root / 'run_summary.json'
    if not summary.is_file():
        return
    data = json.loads(summary.read_text())
    if data.get('invocation_id'):
        return
    _, destination = new_record(root / 'invocations')
    shutil.copyfile(summary, destination / 'run_summary.json')
    if (root / 'run_manifest.json').is_file():
        shutil.copyfile(root / 'run_manifest.json', destination / 'run_manifest.json')
    write_json(destination / 'legacy.json', {'scope': 'Last summary before invocation history; earlier overwritten calls cannot be recovered.'})
