# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""2D person detection public API."""

from lamp.detection.detector import Detector2D
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
    "Detector2D",
    "DetectorStats",
    "KeypointDetectionResult",
    "KeypointDetector",
    "PeopleDetector2dSettings",
    "PeopleKeypointBackend",
]
