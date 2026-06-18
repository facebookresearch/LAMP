# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""No-op detector backends used by tests and dry runs."""

from __future__ import annotations

import numpy as np
from lamp.core.types import CameraOrientation
from lamp.detection.types import BBoxDetector


class DummyBBoxDetector(BBoxDetector):
    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation] | None = None,
    ) -> dict[int, np.ndarray]:
        del camera_orientations
        return {cam_idx: np.zeros((0, 5), dtype=np.float32) for cam_idx in images}
