# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""JSON serialization for `LampResult`.

The result schema is intentionally stable across releases. Detection keypoints
are stored in COCO-17 order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from lamp.core.types import Detection2D, LampResult, Person, PersonState, Skeleton


def save_results(
    path: Path,
    result: LampResult,
    cam_index_to_name: dict[int, str],
    vrs_uri: str,
    *,
    all_timestamps: list[int] | None = None,
) -> None:
    """Write a `LampResult` to JSON."""
    payload: dict[str, Any] = {
        "vrs": vrs_uri,
        "all_timestamps": list(all_timestamps) if all_timestamps is not None else [],
        "people": [],
        "num_tracks": 0,
    }
    num_tracks = 0
    for person_id in sorted(result.people.keys()):
        person = result.people[person_id]
        if person.last_lifted_ts == -1:
            continue
        num_tracks += 1
        person_json = _person_to_json(person, cam_index_to_name)
        if person_json["tsToStates"]:
            payload["people"].append(person_json)
    payload["num_tracks"] = num_tracks

    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _person_to_json(
    person: Person, cam_index_to_name: dict[int, str]
) -> dict[str, Any]:
    """Serialize one `Person`."""
    color = person.color if person.color is not None else (0.0, 0.0, 0.0, 1.0)
    person_json: dict[str, Any] = {
        "personId": int(person.id),
        "active": bool(person.active),
        "inactiveTs": int(person.inactive_ts),
        "lastLiftedTs": int(person.last_lifted_ts),
        "lastObsTs": int(person.last_obs_ts),
        "uncertainty": float(person.uncertainty),
        "color": [float(color[0]), float(color[1]), float(color[2])],
        "tsToStates": [],
    }
    for ts in sorted(person.ts_to_states.keys()):
        state = person.ts_to_states[ts]
        if state.skeleton is None:
            continue
        person_json["tsToStates"].append(_state_to_json(ts, state, cam_index_to_name))
    return person_json


def _state_to_json(
    ts: int, state: PersonState, cam_index_to_name: dict[int, str]
) -> dict[str, Any]:
    """Serialize one `PersonState`."""
    skeleton = state.skeleton
    assert skeleton is not None  # caller filtered; helps the type checker
    state_json: dict[str, Any] = {
        "ts": int(ts),
        "numFuses": int(state.num_fuses),
        "skeleton": _skeleton_to_json(skeleton, state.detection2ds, cam_index_to_name),
        "detection2ds": [_detection2d_to_json(d) for d in state.detection2ds],
    }
    return state_json


def _skeleton_to_json(
    skeleton: Skeleton,
    detection2ds: list[Detection2D],
    cam_index_to_name: dict[int, str],
) -> dict[str, Any]:
    """Serialize a skeleton and its observed 2D boxes by camera name."""
    obs2d: dict[str, list[float]] = {}
    for det in detection2ds:
        cam_name = cam_index_to_name.get(det.cam_idx, f"cam_{det.cam_idx}")
        box = det.box_xyxy
        obs2d[cam_name] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

    return {
        "kpWorld": skeleton.kp_world.astype(np.float64).tolist(),
        "obs2d_bbx_xyxy": obs2d,
        "shape": skeleton.shape.astype(np.float64).tolist(),
        "joints_rot_mat": skeleton.joints_rot_mat.astype(np.float64).tolist(),
        "T_w_pelvis": skeleton.T_world_pelvis.astype(np.float64).tolist(),
    }


def _detection2d_to_json(det: Detection2D) -> dict[str, Any]:
    """Serialize one `Detection2D` with COCO-17 keypoints."""
    box = det.box_xyxy
    kpts_json: list[dict[str, float]] = []
    if det.keypoints.size > 0:
        for kp in det.keypoints[:17]:
            kpts_json.append(
                {"score": float(kp[2]), "x": float(kp[0]), "y": float(kp[1])}
            )
    return {
        "camIdx": int(det.cam_idx),
        "timestampNs": int(det.timestamp_ns),
        "hasKpts": bool(det.has_keypoints),
        "boxScore": float(det.box_score),
        "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        "kpts": kpts_json,
    }


def load_results(path: Path) -> LampResult:
    """Reconstruct a `LampResult` from a JSON file written by `save_results`."""
    raw = json.loads(path.read_text())
    people: dict[int, Person] = {}
    latest_ts = 0
    for person_json in raw["people"]:
        person = _person_from_json(person_json)
        people[person.id] = person
        if person.last_lifted_ts > latest_ts:
            latest_ts = person.last_lifted_ts
    return LampResult(timestamp_ns=latest_ts, people=people)


def _person_from_json(person_json: dict[str, Any]) -> Person:
    """Rebuild one `Person`."""
    color_arr = person_json["color"]
    color: tuple[float, float, float, float] = (
        float(color_arr[0]),
        float(color_arr[1]),
        float(color_arr[2]),
        1.0,
    )
    person = Person(
        id=int(person_json["personId"]),
        color=color,
        last_obs_ts=int(person_json.get("lastObsTs", -1)),
        last_lifted_ts=int(person_json["lastLiftedTs"]),
        active=bool(person_json.get("active", True)),
        inactive_ts=int(person_json["inactiveTs"]),
        uncertainty=float(person_json.get("uncertainty", 0.0)),
    )
    for state_json in person_json["tsToStates"]:
        ts = int(state_json["ts"])
        person.ts_to_states[ts] = _state_from_json(state_json)
    return person


def _state_from_json(state_json: dict[str, Any]) -> PersonState:
    """Rebuild one `PersonState`."""
    num_fuses = int(state_json.get("numFuses", 1))
    skeleton: Skeleton | None = None
    sk_json = state_json.get("skeleton")
    if sk_json:
        skeleton = _skeleton_from_json(sk_json)
    detection2ds: list[Detection2D] = []
    for det_json in state_json.get("detection2ds", []):
        detection2ds.append(_detection2d_from_json(det_json))
    return PersonState(
        detection2ds=detection2ds, skeleton=skeleton, num_fuses=num_fuses
    )


def _skeleton_from_json(sk_json: dict[str, Any]) -> Skeleton:
    """Rebuild a `Skeleton`."""
    # nlohmann::json loads numbers as `double`; the Skeleton dataclass declares
    # float32 — the down-cast is intentional and matches the rest of the codebase.
    kp_world = np.asarray(sk_json["kpWorld"], dtype=np.float32)
    shape = np.asarray(sk_json.get("shape", []), dtype=np.float32)
    joints_rot_mat = np.asarray(sk_json.get("joints_rot_mat", []), dtype=np.float32)
    if joints_rot_mat.ndim != 3:
        # Empty list deserializes to shape (0,); reshape to (0, 3, 3) to
        # keep the contract.
        joints_rot_mat = np.zeros((0, 3, 3), dtype=np.float32)
    T_world_pelvis = np.asarray(sk_json["T_w_pelvis"], dtype=np.float32)
    # `kp_score` is not persisted. Reload as ones so downstream consumers do
    # not see all-zero confidences.
    kp_score = np.ones(kp_world.shape[0], dtype=np.float32)
    return Skeleton(
        kp_world=kp_world,
        kp_score=kp_score,
        T_world_pelvis=T_world_pelvis,
        shape=shape,
        joints_rot_mat=joints_rot_mat,
    )


def _detection2d_from_json(det_json: dict[str, Any]) -> Detection2D:
    """Rebuild a `Detection2D`."""
    box_arr = det_json["box"]
    box_xyxy = np.asarray(
        [float(box_arr[0]), float(box_arr[1]), float(box_arr[2]), float(box_arr[3])],
        dtype=np.float32,
    )
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for kp_idx, kp in enumerate(det_json.get("kpts", [])):
        if kp_idx >= 17:
            break
        keypoints[kp_idx, 0] = float(kp["x"])
        keypoints[kp_idx, 1] = float(kp["y"])
        keypoints[kp_idx, 2] = float(kp["score"])
    return Detection2D(
        box_xyxy=box_xyxy,
        box_score=float(det_json["boxScore"]),
        keypoints=keypoints,
        cam_idx=int(det_json["camIdx"]),
        timestamp_ns=int(det_json["timestampNs"]),
        has_keypoints=bool(det_json["hasKpts"]),
    )
