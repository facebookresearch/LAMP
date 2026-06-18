#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${1:-"$ROOT_DIR/ckpts"}"
DATA_DIR="${2:-"$ROOT_DIR/data"}"
RECORDING_DIR="$DATA_DIR/test-library"
SMPL_PATH="$ROOT_DIR/data/SMPL_NEUTRAL.pkl"

if ! command -v hf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[fetch] `hf` command not found. Activate the LAMP venv and install the base
requirements first:
  source .venv/bin/activate
  uv pip install -r requirements.txt
EOF
    exit 1
fi

mkdir -p "$CKPT_DIR" "$DATA_DIR"

hf download facebook/LAMP lamp_smpl_aria_gen2.pt --local-dir "$CKPT_DIR"
# The dataset repo keeps each recording in its own folder (e.g. test-library/);
# downloading into $DATA_DIR lands the recording at $DATA_DIR/test-library/.
hf download facebook/LAMP --repo-type dataset --include "test-library/*" --local-dir "$DATA_DIR"

cat <<EOF
Downloaded LAMP artifacts:
  checkpoint: $CKPT_DIR/lamp_smpl_aria_gen2.pt
  recording:  $RECORDING_DIR

SMPL is not redistributed with LAMP. Download SMPL_NEUTRAL.pkl from
https://smpl.is.tue.mpg.de, save it as:
  $SMPL_PATH

Then run:
  python -m lamp.app.cli run \\
    --recording $RECORDING_DIR \\
    --checkpoint $CKPT_DIR/lamp_smpl_aria_gen2.pt \\
    --smpl-model-path $SMPL_PATH
EOF
