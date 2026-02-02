#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"

mkdir -p "$MODELS_DIR"

if command -v wget >/dev/null 2>&1; then
  DL_CMD="wget -q -c -O"
elif command -v curl >/dev/null 2>&1; then
  DL_CMD="curl -L -C - -o"
else
  echo "ERROR: wget or curl is required to download weights."
  exit 1
fi

download() {
  local url="$1"
  local out="$2"
  local min_bytes="${3:-0}"
  if [ -s "$out" ]; then
    local size
    size=$(wc -c <"$out")
    if [ "$size" -ge "$min_bytes" ]; then
      echo "[skip] $out exists"
      return
    fi
    echo "[redo] $out exists but too small ($size bytes)"
  fi
  echo "[download] $url -> $out"
  $DL_CMD "$out" "$url"
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

# SAM2.1 (small)
SAM2_DIR="$MODELS_DIR/sam2"
mkdir -p "$SAM2_DIR"
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
download "$SAM2_BASE_URL/sam2.1_hiera_small.pt" "$SAM2_DIR/sam2.1_hiera_small.pt" 100000000
# Config file from official repo
SAM2_CFG_URL1="https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CFG_URL2="https://github.com/facebookresearch/sam2/raw/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
download_fallback "$SAM2_DIR/sam2.1_hiera_s.yaml" 1024 "$SAM2_CFG_URL1" "$SAM2_CFG_URL2" || \
  echo "[warn] SAM2 config download failed; will use config from installed sam2 package if available."

# GroundingDINO
GD_DIR="$MODELS_DIR/groundingdino"
mkdir -p "$GD_DIR"
GD_WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GD_CFG_URL="https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
download "$GD_WEIGHTS_URL" "$GD_DIR/groundingdino_swint_ogc.pth"
download "$GD_CFG_URL" "$GD_DIR/GroundingDINO_SwinT_OGC.py"

# Fast3R (Hugging Face)
FAST3R_DIR="$MODELS_DIR/fast3r"
mkdir -p "$FAST3R_DIR"
ensure_pip_module huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jedyang97/Fast3R_ViT_Large_512",
    local_dir="./models/fast3r",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.safetensors.index.json"],
)
PY

# FoundationPose weights (Google Drive folder)
FP_DIR="$MODELS_DIR/foundationpose"
mkdir -p "$FP_DIR/weights"
ensure_pip_module gdown
python - <<'PY'
import gdown
folder_url = "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
gdown.download_folder(folder_url, output="./models/foundationpose/weights", quiet=False, use_cookies=False)
PY

# HaMeR demo weights (official script)
HAMER_DIR="$MODELS_DIR/hamer"
HAMER_SRC="$HAMER_DIR/_src"
if [ ! -d "$HAMER_SRC" ]; then
  echo "[clone] HaMeR repo"
  git clone --recursive --depth 1 https://github.com/geopavlakos/hamer.git "$HAMER_SRC"
fi
(
  cd "$HAMER_SRC"
  bash fetch_demo_data.sh
)

# DexRetargeting has no model weights; installed via pip when needed.

echo "All requested model downloads completed."
