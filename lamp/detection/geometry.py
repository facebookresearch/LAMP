# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image normalization and coordinate transforms for detector backends."""

from __future__ import annotations

import numpy as np
import torch
from lamp.core.types import CameraOrientation
from lamp.detection.types import KeypointDetectionResult


def _resolve_detector_device(device: str) -> str:
    """Return a torch-compatible device string with CPU fallback."""

    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _ensure_hwc3_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize public detector images to HWC 3-channel uint8 arrays."""

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        raise ValueError(
            f"expected HxW, HxWx1, or HxWx3 image; got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _orient_image_for_detector(
    image: np.ndarray,
    orientation: CameraOrientation,
) -> np.ndarray:
    """Apply the per-camera orientation before a detector sees the image."""

    if orientation == CameraOrientation.ROTATE_90_CW:
        return np.ascontiguousarray(np.rot90(image, k=3))
    return np.ascontiguousarray(image)


def _points_oriented_to_original(
    points_xy: np.ndarray,
    orig_h: int,
    orientation: CameraOrientation,
) -> np.ndarray:
    """Map points from an oriented detector image back to original image coords."""

    if points_xy.shape[0] == 0:
        return points_xy.astype(np.float32, copy=True)
    out = points_xy.astype(np.float32, copy=True)
    if orientation == CameraOrientation.ROTATE_90_CW:
        x_r = out[..., 0].copy()
        y_r = out[..., 1].copy()
        out[..., 0] = y_r
        out[..., 1] = float(orig_h - 1) - x_r
    return out


def _boxes_oriented_to_original(
    boxes_xyxy: np.ndarray,
    orig_h: int,
    orientation: CameraOrientation,
) -> np.ndarray:
    """Map xyxy boxes from an oriented detector image to original image coords."""

    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if orientation == CameraOrientation.UPRIGHT:
        return boxes_xyxy.astype(np.float32, copy=True)

    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    corners = np.stack(
        (
            np.stack((x1, y1), axis=1),
            np.stack((x2, y1), axis=1),
            np.stack((x1, y2), axis=1),
            np.stack((x2, y2), axis=1),
        ),
        axis=1,
    )
    orig = _points_oriented_to_original(corners, orig_h, orientation)
    return np.stack(
        (
            orig[:, :, 0].min(axis=1),
            orig[:, :, 1].min(axis=1),
            orig[:, :, 0].max(axis=1),
            orig[:, :, 1].max(axis=1),
        ),
        axis=1,
    ).astype(np.float32)


def _points_original_to_oriented(
    points_xy: np.ndarray,
    orig_h: int,
    orientation: CameraOrientation,
) -> np.ndarray:
    """Inverse of `_points_oriented_to_original`."""
    if points_xy.shape[0] == 0:
        return points_xy.astype(np.float32, copy=True)
    out = points_xy.astype(np.float32, copy=True)
    if orientation == CameraOrientation.ROTATE_90_CW:
        # Inverting `_points_oriented_to_original`:
        #   x_orig = y_oriented  =>  y_oriented = x_orig
        #   y_orig = orig_h - 1 - x_oriented  =>  x_oriented = orig_h - 1 - y_orig
        x_orig = out[..., 0].copy()
        y_orig = out[..., 1].copy()
        out[..., 0] = float(orig_h - 1) - y_orig
        out[..., 1] = x_orig
    return out


def _boxes_original_to_oriented(
    boxes_xyxy: np.ndarray,
    orig_h: int,
    orientation: CameraOrientation,
) -> np.ndarray:
    """Inverse of `_boxes_oriented_to_original`."""
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if orientation == CameraOrientation.UPRIGHT:
        return boxes_xyxy.astype(np.float32, copy=True)
    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    corners = np.stack(
        (
            np.stack((x1, y1), axis=1),
            np.stack((x2, y1), axis=1),
            np.stack((x1, y2), axis=1),
            np.stack((x2, y2), axis=1),
        ),
        axis=1,
    )
    rotated = _points_original_to_oriented(corners, orig_h, orientation)
    return np.stack(
        (
            rotated[:, :, 0].min(axis=1),
            rotated[:, :, 1].min(axis=1),
            rotated[:, :, 0].max(axis=1),
            rotated[:, :, 1].max(axis=1),
        ),
        axis=1,
    ).astype(np.float32)


def empty_keypoint_result() -> KeypointDetectionResult:
    return KeypointDetectionResult(
        kps_ij=np.zeros((0, 17, 2), dtype=np.float32),
        kp_scores=np.zeros((0, 17), dtype=np.float32),
        bboxes_xyxy=np.zeros((0, 4), dtype=np.float32),
        bbox_scores=np.zeros((0,), dtype=np.float32),
    )


def expand_boxes_5_to_7(boxes_5: np.ndarray) -> np.ndarray:
    """Convert `(N, 5) [xyxy, score]` boxes to the keypoint backend format."""
    if boxes_5.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)
    out = np.zeros((boxes_5.shape[0], 7), dtype=np.float32)
    out[:, :4] = boxes_5[:, :4]
    out[:, 4] = boxes_5[:, 4]
    out[:, 5] = 1.0
    out[:, 6] = 0.0
    return out
