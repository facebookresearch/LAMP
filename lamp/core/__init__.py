# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Core geometry and data contracts."""

from lamp.core.se3 import (
    as_4x4_f32,
    compose,
    invert,
    slerp_se3_batched,
    slerp_so3_batched,
)
from lamp.core.types import (
    box_iou_xyxy,
    CameraOrientation,
    color_from_id,
    Detection2D,
    Frameset,
    LampResult,
    Person,
    PersonState,
    Skeleton,
    SMPL_SKELETON_EDGES,
)

__all__ = [
    "SMPL_SKELETON_EDGES",
    "CameraOrientation",
    "Detection2D",
    "Frameset",
    "LampResult",
    "Person",
    "PersonState",
    "Skeleton",
    "as_4x4_f32",
    "box_iou_xyxy",
    "color_from_id",
    "compose",
    "invert",
    "slerp_se3_batched",
    "slerp_so3_batched",
]
