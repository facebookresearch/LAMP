# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Public detector facade and post-processing."""

from __future__ import annotations

import time

import numpy as np
from lamp.core.types import CameraOrientation, Detection2D
from lamp.detection.two_stage import TwoStagePeopleKeypointBackend
from lamp.detection.types import (
    BBoxDetector,
    DetectorStats,
    KeypointDetectionResult,
    KeypointDetector,
    PeopleDetector2dSettings,
    PeopleKeypointBackend,
)

__all__ = [
    "BBoxDetector",
    "CameraOrientation",
    "Detection2D",
    "Detector2D",
    "DetectorStats",
    "KeypointDetectionResult",
    "KeypointDetector",
    "PeopleDetector2dSettings",
    "PeopleKeypointBackend",
]


class Detector2D:
    """Detector facade plus common post-processing into `Detection2D` records."""

    def __init__(
        self,
        bbox: BBoxDetector | None = None,
        keypoint: KeypointDetector | None = None,
        settings: PeopleDetector2dSettings | None = None,
        backend: PeopleKeypointBackend | None = None,
    ) -> None:
        self._settings = (
            settings if settings is not None else PeopleDetector2dSettings()
        )
        if backend is None:
            if bbox is None or keypoint is None:
                raise ValueError(
                    "Detector2D requires either a backend or bbox+keypoint detectors."
                )
            backend = TwoStagePeopleKeypointBackend(bbox, keypoint)
        self._backend = backend
        # Keep the latest per-stage detector timing for pipeline stats.
        self.last_detect_stats: DetectorStats = DetectorStats()

    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation],
        timestamp_ns: int,
    ) -> dict[int, list[Detection2D]]:
        kp_results, stats = self._backend.detect(images, camera_orientations)

        t0 = time.perf_counter()
        out: dict[int, list[Detection2D]] = {}
        for cam_idx in images:
            kp_res = kp_results.get(cam_idx)
            if kp_res is None or kp_res.bboxes_xyxy.shape[0] == 0:
                out[cam_idx] = []
                continue
            out[cam_idx] = self._build_detections(
                kp_res, cam_idx, timestamp_ns, images[cam_idx]
            )
        stats.post_ms += (time.perf_counter() - t0) * 1000.0
        self.last_detect_stats = stats
        return out

    def _build_detections(
        self,
        kp_res: KeypointDetectionResult,
        cam_idx: int,
        timestamp_ns: int,
        image: np.ndarray,
    ) -> list[Detection2D]:
        s = self._settings
        h, w = image.shape[:2]
        min_side = min(h, w)
        min_size = s.min_box_size_ratio * float(min_side)

        # Suppress per-keypoint values below the conf threshold.
        reliable_kp = kp_res.kp_scores >= s.min_kp_conf
        thresholded_kps = kp_res.kps_ij.copy()
        thresholded_scores = kp_res.kp_scores.copy()
        thresholded_kps[~reliable_kp] = 0.0
        thresholded_scores[~reliable_kp] = 0.0

        reliable_count = reliable_kp.sum(axis=1)

        # Min size filter on the bbox.
        widths = kp_res.bboxes_xyxy[:, 2] - kp_res.bboxes_xyxy[:, 0]
        heights = kp_res.bboxes_xyxy[:, 3] - kp_res.bboxes_xyxy[:, 1]
        size_mask = (widths >= min_size) & (heights >= min_size)
        reliable_mask = reliable_count >= s.min_reliable_kp_num
        keep_mask = size_mask & reliable_mask

        # Top-N by bbox score if requested.
        keep_idx = np.where(keep_mask)[0]
        if s.max_num_box_per_image > 0 and keep_idx.size > s.max_num_box_per_image:
            scores_for_sort = kp_res.bbox_scores[keep_idx]
            top = np.argsort(-scores_for_sort)[: s.max_num_box_per_image]
            keep_idx = keep_idx[top]

        detections: list[Detection2D] = []
        for i in keep_idx:
            kps_xy = thresholded_kps[i]  # (17, 2)
            kp_sc = thresholded_scores[i]  # (17,)
            kps_xys = np.empty((17, 3), dtype=np.float32)
            kps_xys[:, :2] = kps_xy
            kps_xys[:, 2] = kp_sc
            detections.append(
                Detection2D(
                    box_xyxy=kp_res.bboxes_xyxy[i].astype(np.float32, copy=True),
                    box_score=float(kp_res.bbox_scores[i]),
                    keypoints=kps_xys,
                    cam_idx=cam_idx,
                    timestamp_ns=timestamp_ns,
                    has_keypoints=True,
                    det_crop=None,
                    det_crop_raw=None,
                )
            )
        return detections
