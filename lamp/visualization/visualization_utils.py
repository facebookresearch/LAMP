# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualization helpers shared by the Viser frontend."""

from __future__ import annotations

import socket
from collections import deque
from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import numpy as np
from lamp.core.types import color_from_id, Detection2D, SMPL_SKELETON_EDGES

_COCO_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
_UNTRACKED_RGB: tuple[int, int, int] = (200, 200, 200)
_KP_DRAW_SCORE_THRES: float = 0.02
COMPOSITE_HANDLE_KEY: int = -1
TRAJ_WINDOW_NS: int = 5_000_000_000
DEVICE_TRAJ_MAXLEN: int = 200
PERSON_TRAJ_MAXLEN: int = 5000
INITIAL_VIEW_BEHIND_M: float = 3.0
INITIAL_VIEW_UP_M: float = 2.0
INITIAL_VIEW_LOOK_M: float = 2.0


class SmplParams(NamedTuple):
    """SMPL params used for batched mesh forwarding."""

    betas: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray


@dataclass(slots=True)
class RenderFrame:
    """Per-timestamp state used to align delayed poses with images."""

    timestamp_ns: int
    T_world_cams: dict[int, np.ndarray]
    images: dict[int, np.ndarray]
    detections: dict[int, list[Detection2D]] | None
    intrinsics: dict[int, np.ndarray]
    cam_models: dict[int, Any] | None
    T_world_device: np.ndarray | None


def skeleton_edges_for(n_joints: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (a, b) for a, b in SMPL_SKELETON_EDGES if a < n_joints and b < n_joints
    )


def smpl_params_from_skeleton(skel: Any) -> SmplParams | None:
    joints_rot_mat = skel.joints_rot_mat
    shape = skel.shape
    T_world_pelvis = skel.T_world_pelvis
    if joints_rot_mat.shape != (24, 3, 3) or int(shape.size) != 10:
        return None
    if T_world_pelvis.shape != (4, 4):
        return None
    return SmplParams(
        betas=np.asarray(shape, dtype=np.float32).reshape(10),
        global_orient=np.asarray(joints_rot_mat[0:1], dtype=np.float32),
        body_pose=np.asarray(joints_rot_mat[1:24], dtype=np.float32),
        transl=np.asarray(T_world_pelvis[:3, 3], dtype=np.float32).reshape(3),
    )


def compose_camera_grid(images: dict[int, np.ndarray], columns: int = 2) -> np.ndarray:
    """Tile sorted camera images into a fixed-column grid."""
    cam_indices = sorted(images.keys())
    sample = images[cam_indices[0]]
    h, w = sample.shape[:2]
    cells: list[np.ndarray] = []
    for idx in cam_indices:
        img = images[idx]
        if img.shape[:2] != (h, w):
            ys = (np.arange(h) * (img.shape[0] / h)).astype(np.int32)
            xs = (np.arange(w) * (img.shape[1] / w)).astype(np.int32)
            img = img[ys][:, xs]
        cells.append(img)
    rows = (len(cells) + columns - 1) // columns
    blank = np.zeros_like(sample)
    while len(cells) < rows * columns:
        cells.append(blank)
    row_arrays = [
        np.hstack(cells[r * columns : (r + 1) * columns]) for r in range(rows)
    ]
    return np.vstack(row_arrays)


def _track_color_rgb(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return _UNTRACKED_RGB
    r, g, b, _ = color_from_id(track_id)
    return (round(r * 255), round(g * 255), round(b * 255))


def resolve_port(host: str, requested: int) -> int:
    """Resolve `0` to an available TCP port."""
    if requested != 0:
        return requested
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def p50(buf: deque[float]) -> float:
    if not buf:
        return 0.0
    s = sorted(buf)
    return s[len(s) // 2]


def trim_traj(buf: deque[tuple[int, np.ndarray]], current_ts_ns: int) -> None:
    cutoff = current_ts_ns - TRAJ_WINDOW_NS
    while buf and buf[0][0] < cutoff:
        buf.popleft()


def draw_overlays(
    img: np.ndarray,
    dets: list[Detection2D],
    projected_skeletons: list[tuple[int, np.ndarray, tuple[float, float, float, float]]]
    | None = None,
) -> np.ndarray:
    """Draw detections and projected skeletons on a copy of one image."""
    canvas = img.copy()
    for det in dets:
        color = _track_color_rgb(det.track_id)
        x1, y1, x2, y2 = (float(v) for v in det.box_xyxy[:4])
        cv2.rectangle(
            canvas,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            color,
            thickness=2,
        )
        if det.track_id is not None:
            cv2.putText(
                canvas,
                f"#{det.track_id}",
                (round(x1), max(round(y1) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        if det.has_keypoints and det.keypoints.shape[0] >= 17:
            _draw_keypoints(canvas, det.keypoints, color)

    if projected_skeletons:
        h, w = canvas.shape[:2]
        for person_id, kp_xy, color_rgba in projected_skeletons:
            color = (
                round(color_rgba[0] * 255),
                round(color_rgba[1] * 255),
                round(color_rgba[2] * 255),
            )
            _draw_projected_skeleton(canvas, kp_xy, color, w, h, person_id)
    return canvas


def _draw_projected_skeleton(
    canvas: np.ndarray,
    kp_xy: np.ndarray,
    color: tuple[int, int, int],
    img_w: int,
    img_h: int,
    person_id: int,
) -> None:
    visible = np.zeros(kp_xy.shape[0], dtype=bool)
    for j in range(kp_xy.shape[0]):
        x = float(kp_xy[j, 0])
        y = float(kp_xy[j, 1])
        if not np.isnan(x) and not np.isnan(y) and 0 <= x < img_w and 0 <= y < img_h:
            visible[j] = True

    n_joints = int(kp_xy.shape[0])
    for a, b in skeleton_edges_for(n_joints):
        if visible[a] and visible[b]:
            cv2.line(
                canvas,
                (round(float(kp_xy[a, 0])), round(float(kp_xy[a, 1]))),
                (round(float(kp_xy[b, 0])), round(float(kp_xy[b, 1]))),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    for j in range(kp_xy.shape[0]):
        if visible[j]:
            cv2.circle(
                canvas,
                (round(float(kp_xy[j, 0])), round(float(kp_xy[j, 1]))),
                radius=3,
                color=color,
                thickness=-1,
            )

    head_idx = 15
    label_idx = head_idx if n_joints > head_idx and visible[head_idx] else -1
    if label_idx < 0 and n_joints > 0 and visible[0]:
        label_idx = 0
    if label_idx >= 0:
        x = round(float(kp_xy[label_idx, 0]))
        y = round(float(kp_xy[label_idx, 1])) - 8
        cv2.putText(
            canvas,
            f"ID:{person_id}",
            (x, max(y, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def _draw_keypoints(
    canvas: np.ndarray, kps: np.ndarray, color: tuple[int, int, int]
) -> None:
    visible = [bool(float(kps[k, 2]) >= _KP_DRAW_SCORE_THRES) for k in range(17)]
    for a, b in _COCO_SKELETON_EDGES:
        if not (visible[a] and visible[b]):
            continue
        ax, ay = float(kps[a, 0]), float(kps[a, 1])
        bx, by = float(kps[b, 0]), float(kps[b, 1])
        cv2.line(canvas, (round(ax), round(ay)), (round(bx), round(by)), color, 1)
    for k in range(17):
        if visible[k]:
            x, y = float(kps[k, 0]), float(kps[k, 1])
            cv2.circle(canvas, (round(x), round(y)), 3, color, -1)


def se3_to_viser(
    T: np.ndarray,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z)), (
        float(t[0]),
        float(t[1]),
        float(t[2]),
    )
