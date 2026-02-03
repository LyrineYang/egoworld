#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${ENV_NAME:-egoworld-base}"
ENV_DIR="$REPO_ROOT/egoworld/env"
BASE_YML="$ENV_DIR/base.yml"
LOCK_FILE="$ENV_DIR/locks/linux-64/base.lock"

USE_LOCK=1
WITH_WEIGHTS=0
RUN_SMOKE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--no-lock] [--weights] [--smoke] [--env-name NAME]

  --no-lock     Skip conda-lock install even if lock file exists.
  --weights     Download model weights (runs scripts/download_models.sh).
  --smoke       Run SAM2 smoke test after install.
  --env-name    Override conda env name (default: egoworld-base).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-lock)
      USE_LOCK=0
      shift
      ;;
    --weights)
      WITH_WEIGHTS=1
      shift
      ;;
    --smoke)
      RUN_SMOKE=1
      shift
      ;;
    --env-name)
      ENV_NAME="${2:-}"
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

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but not found on PATH."
  exit 1
fi

cd "$REPO_ROOT"

if [[ "$USE_LOCK" -eq 1 && -f "$LOCK_FILE" ]]; then
  if ! command -v conda-lock >/dev/null 2>&1; then
    echo "ERROR: conda-lock not found. Install with: pip install conda-lock"
    exit 1
  fi
  conda-lock install --name "$ENV_NAME" "$LOCK_FILE"
else
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$BASE_YML"
  else
    conda env create -n "$ENV_NAME" -f "$BASE_YML"
  fi
fi

if [[ "$WITH_WEIGHTS" -eq 1 ]]; then
  bash "$REPO_ROOT/egoworld/scripts/download_models.sh"
fi

if [[ "$RUN_SMOKE" -eq 1 ]]; then
  PYTHONPATH="$REPO_ROOT/egoworld/src" EGOWORLD_SAM2_SMOKE=1 conda run -n "$ENV_NAME" pytest -q egoworld/tests/test_sam2_integration.py
fi
