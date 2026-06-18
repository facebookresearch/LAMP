# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared RF-DETR backend helpers."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import os
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


_RFDETR_PERSON_CLASS_ID: int = 1

# Where we cache the auto-downloaded RF-DETR-Nano checkpoint when the caller
# doesn't supply one. Override with the LAMP_CACHE_DIR env var.
_LAMP_CACHE_DIR: Path = Path(
    os.environ.get("LAMP_CACHE_DIR", str(Path.home() / ".cache" / "lamp"))
)
# The `rfdetr` package writes its auto-download under this exact filename.
_RFDETR_NANO_CACHE_NAME: str = "rf-detr-nano.pth"


def _resolve_rfdetr_weights(weights_path: Path | None) -> Path:
    """Return a path to RF-DETR-Nano weights, auto-downloading if needed."""
    if weights_path is not None:
        return Path(weights_path)
    _LAMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _LAMP_CACHE_DIR / _RFDETR_NANO_CACHE_NAME
    if not cached.exists():
        from rfdetr import RFDETRNano  # pyright: ignore[reportMissingImports]

        logger.info(
            "No RF-DETR weights provided; downloading the stock RF-DETR-Nano "
            "checkpoint into %s (one-time, ~349 MB).",
            _LAMP_CACHE_DIR,
        )
        cwd = os.getcwd()
        try:
            os.chdir(_LAMP_CACHE_DIR)
            RFDETRNano()  # triggers the download of rf-detr-nano.pth into cwd
        finally:
            os.chdir(cwd)
        if not cached.exists():
            raise RuntimeError(
                f"RF-DETR auto-download did not produce {cached}. Check network "
                "access, or set LAMP_CACHE_DIR to a writable cache directory."
            )
    return cached
