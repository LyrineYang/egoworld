#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"

cd "$ROOT_DIR"
mkdir -p "$MODELS_DIR"

if command -v wget >/dev/null 2>&1; then
  DL_CMD=(wget -q -O)
elif command -v curl >/dev/null 2>&1; then
  DL_CMD=(curl --fail --location --output)
else
  echo "ERROR: wget or curl is required to download weights."
  exit 1
fi

valid_download() {
  local file="$1" min_bytes="$2"
  [[ -f "$file" && $(wc -c <"$file") -ge "$min_bytes" ]] || return 1
  case "$file" in
    *.pt|*.pth|*.pt.part|*.pth.part)
      python - "$file" <<'CHECKPOINT'
import sys
import zipfile
path = sys.argv[1]
if not zipfile.is_zipfile(path):
    raise SystemExit(1)
with zipfile.ZipFile(path) as archive:
    if not any(name.endswith("data.pkl") for name in archive.namelist()):
        raise SystemExit(1)
CHECKPOINT
      ;;
  esac
}

download() {
  local url="$1" out="$2" min_bytes="${3:-0}"
  if valid_download "$out" "$min_bytes"; then
    echo "[skip] $out verified"
    return 0
  fi
  echo "[download] $url -> $out"
  if ! "${DL_CMD[@]}" "$out.part" "$url"; then
    return 1
  fi
  if ! valid_download "$out.part" "$min_bytes"; then
    echo "ERROR: incomplete or invalid download: $out.part" >&2
    return 1
  fi
  mv "$out.part" "$out"
}

download_fallback() {
  local out="$1"
  local min_bytes="$2"
  shift 2
  for url in "$@"; do
    if download "$url" "$out" "$min_bytes"; then
      return 0
    fi
  done
  return 1
}

# SAM2.1 (small)
SAM2_DIR="$MODELS_DIR/sam2"
mkdir -p "$SAM2_DIR"
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
download "$SAM2_BASE_URL/sam2.1_hiera_small.pt" "$SAM2_DIR/sam2.1_hiera_small.pt" 100000000
# Config file from official repo
SAM2_CFG_URL1="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CFG_URL2="https://github.com/facebookresearch/sam2/raw/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
download_fallback "$SAM2_DIR/sam2.1_hiera_s.yaml" 1024 "$SAM2_CFG_URL1" "$SAM2_CFG_URL2" || \
  echo "[warn] SAM2 config download failed; will use config from installed sam2 package if available."

# GroundingDINO
GD_DIR="$MODELS_DIR/groundingdino"
mkdir -p "$GD_DIR"
GD_WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GD_CFG_URL="https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/config/GroundingDINO_SwinT_OGC.py"
download "$GD_WEIGHTS_URL" "$GD_DIR/groundingdino_swint_ogc.pth" 100000000
download "$GD_CFG_URL" "$GD_DIR/GroundingDINO_SwinT_OGC.py" 512

echo "SAM2 + GroundingDINO downloads verified. Additional model assets require separate setup; see egoworld/env/README.md."
