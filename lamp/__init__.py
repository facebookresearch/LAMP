# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Public import surface for LAMP."""

from __future__ import annotations

from lamp.app.pipeline import LampPipeline, LampPipelineSettings, LampPipelineStats
from lamp.core.types import (
    Detection2D,
    Frameset,
    LampResult,
    Person,
    PersonState,
    Skeleton,
)
from lamp.detection.detector import (
    BBoxDetector,
    CameraOrientation,
    Detector2D,
    DetectorStats,
    KeypointDetectionResult,
    KeypointDetector,
    PeopleDetector2dSettings,
    PeopleKeypointBackend,
)
from lamp.io.persistence import load_results, save_results
from lamp.io.sensor_io import CameraCalibration, MpsLoader, VrsLoader
from lamp.models.lifter import is_outlier_pose, Lifter
from lamp.tracking.tracker import LampTracker, LampTrackerSettings
from lamp.visualization.visualizer import Visualizer

# Short aliases for common public names.
Detection2DType = Detection2D
Tracker = LampTracker

__all__ = [
    "BBoxDetector",
    "CameraCalibration",
    "CameraOrientation",
    "Detection2D",
    "Detection2DType",
    "Detector2D",
    "DetectorStats",
    "Frameset",
    "KeypointDetectionResult",
    "KeypointDetector",
    "LampPipeline",
    "LampPipelineSettings",
    "LampPipelineStats",
    "LampResult",
    "LampTracker",
    "LampTrackerSettings",
    "Lifter",
    "MpsLoader",
    "PeopleDetector2dSettings",
    "PeopleKeypointBackend",
    "Person",
    "PersonState",
    "Skeleton",
    "Tracker",
    "Visualizer",
    "VrsLoader",
    "is_outlier_pose",
    "load_results",
    "save_results",
]
