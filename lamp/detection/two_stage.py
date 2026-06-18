# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Two-stage person-box plus keypoint detector backend."""

from __future__ import annotations

import time

import numpy as np
from lamp.core.types import CameraOrientation
from lamp.detection.geometry import expand_boxes_5_to_7
from lamp.detection.types import (
    BBoxDetector,
    DetectorStats,
    KeypointDetectionResult,
    KeypointDetector,
    PeopleKeypointBackend,
)


class TwoStagePeopleKeypointBackend(PeopleKeypointBackend):
    def __init__(self, bbox: BBoxDetector, keypoint: KeypointDetector) -> None:
        self._bbox = bbox
        self._keypoint = keypoint

    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation],
    ) -> tuple[dict[int, KeypointDetectionResult], DetectorStats]:
        stats = DetectorStats()

        t0 = time.perf_counter()
        bboxes_5 = self._bbox.detect(images, camera_orientations)
        stats.bbox_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bboxes_7 = {cam: expand_boxes_5_to_7(b) for cam, b in bboxes_5.items()}
        stats.post_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        kp_results = self._keypoint.detect(images, bboxes_7, camera_orientations)
        stats.kp_ms = (time.perf_counter() - t0) * 1000.0
        return kp_results, stats
