#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Fresh-machine install for the LAMP pipeline.
#
# This script treats uv as a fast venv/pip replacement:
#   1. Create `.venv` with Python 3.12.
#   2. Activate it and install `requirements.txt` with `uv pip`.
#   3. Optionally install TensorRT in the staged order required by its wheel.
#   4. Run smoke tests with the activated `python`.
#
# Assumes a Linux box with an NVIDIA GPU and CUDA 12.x driver for the default
# TensorRT path. If TensorRT install fails, the script keeps a usable eager
# RF-DETR environment and prints the `--bbox-backend rfdetr` fallback command.
#
# Usage:
#   bash scripts/install.sh             # env + TensorRT
#   bash scripts/install.sh --minimal   # env without TensorRT; use --bbox-backend rfdetr
#
# Re-running is safe; uv pip detects already-satisfied packages.

set -euo pipefail
export UV_PYTHON_DOWNLOADS="auto"

INSTALL_TRT=1
TRT_AVAILABLE=0
for arg in "$@"; do
    case "$arg" in
        --minimal) INSTALL_TRT=0 ;;
        --help|-h)
            sed -n '2,/^set -euo pipefail/{s/^# \{0,1\}//p}' "$0"
            exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
VENV_DIR="${LAMP_VENV_DIR:-.venv}"
echo "[install] repo root: $REPO_ROOT"
echo "[install] venv: $VENV_DIR"

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[install] uv not found on PATH. Install it first:
    curl -LsSf https://astral.sh/uv/install.sh | sh
then re-run this script.
EOF
    exit 1
fi
echo "[install] uv: $(uv --version)"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[install] WARNING: nvidia-smi not found or failed to execute. The pipeline
runs on CPU as a fallback but is much slower than the CUDA path.
EOF
    read -rp "Continue without GPU? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
else
    echo "[install] GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
fi

echo "[install] step 1: create/refresh venv"
uv venv "$VENV_DIR" --python 3.12
source "$VENV_DIR/bin/activate"
python -m pip --version | sed 's/^/[install] /'

echo "[install] step 2: install Python requirements"
uv pip install -r requirements.txt

if [[ $INSTALL_TRT -eq 1 ]]; then
    echo "[install] step 3: install TensorRT 10.x"
    # wheel_stub must be installed first because tensorrt-cu12 lacks complete
    # build-system metadata and needs --no-build-isolation.
    uv pip install wheel_stub
    if uv pip install --no-build-isolation 'tensorrt-cu12==10.16.1.11'; then
        TRT_AVAILABLE=1
    else
        TRT_AVAILABLE=0
        cat >&2 <<'EOF'
[install] WARNING: TensorRT install failed. Continuing with the eager RF-DETR
backend. The default `rfdetr-trt` path will not work in this environment; run
with `--bbox-backend rfdetr` instead.
EOF
    fi
else
    echo "[install] step 3: TensorRT skipped (--minimal)."
fi

echo "[install] step 4: enforce headless OpenCV"
# rfdetr/roboflow may pull opencv-python; remove it to avoid GL library import
# errors on headless machines.
uv pip uninstall opencv-python || true
uv pip install --force-reinstall opencv-python-headless==4.13.0.92

echo "[install] step 5: smoke tests"
python scripts/smoke_test.py

cat <<EOF

[install] DONE.

Activate this environment in new shells with:
     source $VENV_DIR/bin/activate

Fetch the public LAMP checkpoint and sample recording:
     bash scripts/fetch_artifacts.sh

Download the SMPL neutral model from https://smplify.is.tue.mpg.de/index.html
and save it as:
     ./data/SMPL_NEUTRAL.pkl

Run the demo from the repo root:
     python -m lamp.app.cli run \
         --recording ./data/test-library \
         --checkpoint ./ckpts/lamp_smpl_aria_gen2.pt \
         --smpl-model-path ./data/SMPL_NEUTRAL.pkl
EOF

if [[ $TRT_AVAILABLE -eq 0 ]]; then
    cat <<'EOF'

TensorRT is not available in this environment. Use the eager detector backend:
     python -m lamp.app.cli run \
         --recording ./data/test-library \
         --checkpoint ./ckpts/lamp_smpl_aria_gen2.pt \
         --smpl-model-path ./data/SMPL_NEUTRAL.pkl \
         --bbox-backend rfdetr
EOF
fi

cat <<'EOF'

RF-DETR weights auto-download/cache to ~/.cache/lamp on first run (set
LAMP_CACHE_DIR to relocate). If TensorRT fails to load due to a driver
mismatch, fall back to eager BF16 by passing --bbox-backend rfdetr.
EOF
