# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Public detector contracts and settings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from lamp.core.types import CameraOrientation

DEFAULT_KP_HF_MODEL_ID = "usyd-community/vitpose-plus-base"


@dataclass(slots=True)
class DetectorStats:
    """Per-call wall-clock breakdown for `Detector2D.detect`."""

    bbox_ms: float = 0.0
    kp_ms: float = 0.0
    post_ms: float = 0.0


@dataclass(slots=True)
class PeopleDetector2dSettings:
    """Backend choices and post-processing thresholds for the 2D detector."""

    bbox_backend: Literal["rfdetr", "rfdetr-trt"] = "rfdetr-trt"
    kp_hf_model_id: str = DEFAULT_KP_HF_MODEL_ID
    min_box_conf: float = 0.5
    min_kp_conf: float = 0.7
    min_reliable_kp_num: int = 8
    min_box_size_ratio: float = 0.05
    max_num_box_per_image: int = -1


@dataclass(slots=True)
class KeypointDetectionResult:
    """Per-camera keypoint-detector output in original image coordinates."""

    kps_ij: np.ndarray
    kp_scores: np.ndarray
    bboxes_xyxy: np.ndarray
    bbox_scores: np.ndarray


class PeopleKeypointBackend(ABC):
    """Unified people+keypoint backend used by `Detector2D`."""

    @abstractmethod
    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation],
    ) -> tuple[dict[int, KeypointDetectionResult], DetectorStats]:
        """Return keypoint detections plus backend timing stats."""


class BBoxDetector(ABC):
    """Per-frame people-box backend for the two-stage detector path."""

    @abstractmethod
    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation] | None = None,
    ) -> dict[int, np.ndarray]:
        """Return `{cam_idx: (N, 5) [x1, y1, x2, y2, score]}`."""

    @classmethod
    def from_rfdetr_trt(
        cls,
        weights_path: Path | None = None,
        device: str = "cuda:0",
        score_thresh: float = 0.85,
        min_box_size_ratio: float = 0.0,
        onnx_path: Path | None = None,
        engine_cache_dir: Path | None = None,
        fp16: bool = False,
    ) -> BBoxDetector:
        from lamp.detection.rfdetr_trt import _RfDetrTrtBboxDetector

        return _RfDetrTrtBboxDetector(
            rfdetr_weights=weights_path,
            onnx_path=onnx_path,
            engine_cache_dir=engine_cache_dir,
            device=device,
            score_thresh=score_thresh,
            min_box_size_ratio=min_box_size_ratio,
            fp16=fp16,
        )

    @classmethod
    def from_rfdetr(
        cls,
        weights_path: Path | None = None,
        device: str = "cuda:0",
        score_thresh: float = 0.85,
        min_box_size_ratio: float = 0.0,
        optimize_for_inference: bool = False,
        optimize_batch_size: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ) -> BBoxDetector:
        from lamp.detection.rfdetr_torch import _RfDetrBboxDetector

        return _RfDetrBboxDetector(
            weights_path=weights_path,
            device=device,
            score_thresh=score_thresh,
            min_box_size_ratio=min_box_size_ratio,
            optimize_for_inference=optimize_for_inference,
            optimize_batch_size=optimize_batch_size,
            dtype=dtype,
        )

    @classmethod
    def with_dummy_backend(cls) -> BBoxDetector:
        from lamp.detection.dummy import DummyBBoxDetector

        return DummyBBoxDetector()


class KeypointDetector(ABC):
    """ViTPose-style keypoint backend, batched across cameras."""

    @abstractmethod
    def detect(
        self,
        images: dict[int, np.ndarray],
        bboxes_per_cam: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation] | None = None,
    ) -> dict[int, KeypointDetectionResult]:
        """Run keypoints for the given `(N, 7)` boxes per camera."""

    @classmethod
    def from_huggingface(
        cls,
        model_id: str = DEFAULT_KP_HF_MODEL_ID,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        dataset_index: int = 0,
    ) -> KeypointDetector:
        from lamp.detection.vitpose import _HuggingFaceViTPoseKeypointDetector

        return _HuggingFaceViTPoseKeypointDetector(
            model_id=model_id,
            device=device,
            dtype=dtype,
            dataset_index=dataset_index,
        )
