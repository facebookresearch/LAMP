#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    try:
        import torch
    except ImportError as exc:
        print(f"[smoke] torch import failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[smoke] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        print(f"[smoke]    device 0: {torch.cuda.get_device_name(0)}")

    try:
        from lamp.app.pipeline import LampPipeline  # noqa: F401
    except ImportError as exc:
        print(f"[smoke] lamp import failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("[smoke] lamp package importable")

    try:
        import transformers
        from transformers import VitPoseForPoseEstimation  # noqa: F401

        print(f"[smoke] transformers {transformers.__version__} (VitPose available)")
    except ImportError as exc:
        print(f"[smoke] transformers/VitPose import failed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        import rfdetr
        from rfdetr import RFDETRNano  # noqa: F401

        print(
            f"[smoke] rfdetr {getattr(rfdetr, '__version__', '?')} (RFDETRNano available)"
        )
    except ImportError:
        print("[smoke] rfdetr not installed")

    import cv2

    print(f"[smoke] cv2 {cv2.__version__} (headless build expected)")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print(f"[smoke] onnxruntime providers: {providers}")
        if "TensorrtExecutionProvider" not in providers:
            print(
                "[smoke] note: TensorrtExecutionProvider not listed; the rfdetr-trt "
                "backend is unavailable — run with --bbox-backend rfdetr"
            )
    except ImportError:
        print("[smoke] onnxruntime not installed")


if __name__ == "__main__":
    main()
