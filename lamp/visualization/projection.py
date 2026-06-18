# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Projection helpers for viewer overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
from lamp.core.se3 import invert
from lamp.core.types import color_from_id, Person
from lamp.visualization.timeline import lookup_delayed_state

ProjectedSkeletons = dict[
    int, list[tuple[int, np.ndarray, tuple[float, float, float, float]]]
]


def project_skeletons_for_overlay(
    people: dict[int, Person],
    T_world_cams: dict[int, np.ndarray],
    cam_models: dict[int, Any],
    current_ts_ns: int,
    delay: int,
    *,
    joints_by_person: dict[int, np.ndarray] | None = None,
) -> ProjectedSkeletons:
    out: ProjectedSkeletons = {}
    for person_id, person in people.items():
        if not person.active or person.last_lifted_ts == -1:
            continue
        state = lookup_delayed_state(person, current_ts_ns, delay)
        if state is None or state.skeleton is None:
            continue

        kp_world = (
            joints_by_person.get(person_id)
            if joints_by_person is not None
            else state.skeleton.kp_world
        )
        if kp_world is None or kp_world.size == 0:
            continue
        color_rgba = person.color or color_from_id(person_id)
        for cam_idx, T_world_cam in T_world_cams.items():
            cam = cam_models.get(cam_idx)
            if cam is None or cam.project is None:
                continue
            kp_xy = _project_points(T_world_cam, cam, kp_world)
            if not np.isnan(kp_xy[:, 0]).all():
                out.setdefault(cam_idx, []).append((int(person_id), kp_xy, color_rgba))
    return out


def _project_points(
    T_world_cam: np.ndarray, cam: Any, points_world: np.ndarray
) -> np.ndarray:
    T_cam_world = invert(T_world_cam)
    points_cam = (T_cam_world[:3, :3] @ points_world.T).T + T_cam_world[:3, 3]
    points_xy = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    for idx, point_cam in enumerate(points_cam):
        if point_cam[2] <= 0.0:
            continue
        projected = cam.project(point_cam)
        if projected is not None:
            points_xy[idx] = projected
    return points_xy
