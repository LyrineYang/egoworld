#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"

REPO_ID="${REPO_ID:-Frywind/AwesomeDataset}"
TARGET_DIR="${TARGET_DIR:-$DATA_DIR/AwesomeDataset}"
REVISION="${REVISION:-main}"
MAX_WORKERS="${MAX_WORKERS:-8}"
TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--token TOKEN] [--repo REPO_ID] [--target DIR] [--revision REV] [--max-workers N]

Defaults:
  REPO_ID     Frywind/AwesomeDataset
  TARGET_DIR  $TARGET_DIR
  REVISION    main
  MAX_WORKERS 8

Environment:
  HF_TOKEN or HUGGINGFACE_TOKEN  Hugging Face access token (required).
  DATA_DIR                      Base data directory (default: $DATA_DIR).
  REPO_ID, TARGET_DIR, REVISION, MAX_WORKERS can also be set via env.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)
      TOKEN="${2:-}"
      shift 2
      ;;
    --repo)
      REPO_ID="${2:-}"
      shift 2
      ;;
    --target)
      TARGET_DIR="${2:-}"
      shift 2
      ;;
    --revision)
      REVISION="${2:-}"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: HF_TOKEN is required. Set HF_TOKEN/HUGGINGFACE_TOKEN or pass --token."
  exit 1
fi

mkdir -p "$TARGET_DIR"

ensure_pip_module() {
  local module="$1"
  if python - <<PY
try:
    import ${module}  # noqa: F401
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
  then
    return 0
  else
    echo "[install] python module ${module}"
    python -m pip install -q "${module}"
  fi
}

ensure_pip_module huggingface_hub

export HF_TOKEN="$TOKEN"
export REPO_ID TARGET_DIR REVISION MAX_WORKERS

python - <<'PY'
import inspect
import os

from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]
repo_id = os.environ["REPO_ID"]
target_dir = os.environ["TARGET_DIR"]
revision = os.environ.get("REVISION") or None
max_workers = int(os.environ.get("MAX_WORKERS", "8"))

kwargs = dict(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
)

sig = inspect.signature(snapshot_download)
if "revision" in sig.parameters and revision:
    kwargs["revision"] = revision
if "local_dir_use_symlinks" in sig.parameters:
    kwargs["local_dir_use_symlinks"] = False
if "max_workers" in sig.parameters:
    kwargs["max_workers"] = max_workers
if "resume_download" in sig.parameters:
    kwargs["resume_download"] = True
if "token" in sig.parameters:
    kwargs["token"] = token
elif "use_auth_token" in sig.parameters:
    kwargs["use_auth_token"] = token

snapshot_download(**kwargs)
PY

echo "Download completed: $TARGET_DIR"
