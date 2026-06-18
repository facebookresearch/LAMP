# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Viser-based 3D viewer and per-camera image grid."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import viser
from lamp.core.types import color_from_id, Detection2D, Person
from lamp.visualization.projection import (
    project_skeletons_for_overlay,
    ProjectedSkeletons,
)
from lamp.visualization.timeline import lookup_delayed_state
from lamp.visualization.visualization_utils import (
    compose_camera_grid,
    COMPOSITE_HANDLE_KEY,
    DEVICE_TRAJ_MAXLEN,
    draw_overlays,
    INITIAL_VIEW_BEHIND_M,
    INITIAL_VIEW_LOOK_M,
    INITIAL_VIEW_UP_M,
    p50,
    PERSON_TRAJ_MAXLEN,
    RenderFrame,
    resolve_port,
    se3_to_viser,
    skeleton_edges_for,
    smpl_params_from_skeleton,
    SmplParams,
    trim_traj,
)

logger: logging.Logger = logging.getLogger(__name__)

# Opacity for inactive people when "Show inactive" is enabled.
_INACTIVE_ALPHA = 0.35


class Visualizer:
    """Owns the Viser server, the 3D scene, and the per-camera image grid."""

    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        smpl_faces: np.ndarray | None = None,
        frustum_cam_indices: set[int] | None = None,
        smpl_forward_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
        temporal_window: int = 20,
    ) -> None:
        self._host = host
        self._port = resolve_port(host, port)
        self._stopped = False
        self._server: viser.ViserServer = viser.ViserServer(
            host=host, port=self._port, verbose=False
        )

        self._smpl_faces: np.ndarray | None = (
            np.asarray(smpl_faces, dtype=np.uint32) if smpl_faces is not None else None
        )

        self._smpl_forward_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = (
            smpl_forward_fn
        )
        # Set before `_build_gui`; the slider range depends on it.
        self._temporal_window: int = int(temporal_window)
        self._render_frame_history: deque[RenderFrame] = deque(
            maxlen=max(2 * self._temporal_window, self._temporal_window + 1, 30)
        )
        self._render_frame_history_lock: threading.Lock = threading.Lock()

        self._floor_anchor_xy: tuple[float, float] = (0.0, 0.0)
        self._initial_render_cam: (
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
            | None
        ) = None

        self._frustum_cam_indices: set[int] | None = frustum_cam_indices

        self._server.gui.configure_theme(control_width="large", dark_mode=True)

        self._gui_handles: dict[str, Any] = {}
        # Spacebar hotkey command (created inside `_build_gui`); held here so
        # the CommandHandle isn't garbage-collected and the binding stays live.
        self._toggle_autoplay_cmd: Any | None = None
        self._build_gui()

        self._image_handles: dict[int, Any] = {}

        # 3D scene: keep frustum handles per camera so we can reposition them
        # in `set_camera_frustums` without recreating.
        self._frustum_handles: dict[int, Any] = {}

        self._skeleton_handles: dict[int, Any] = {}
        # Per-person SMPL mesh handles. Populated lazily on first frame
        # that emits non-empty `verts_w`. `.vertices` is updated in place
        # on subsequent frames; faces are constant.
        self._mesh_handles: dict[int, Any] = {}

        self._points_handle: Any | None = None
        self._points_size: float = 0.005
        self._points_xyz: np.ndarray | None = None
        self._points_color: tuple[int, int, int] = (160, 160, 160)
        # Wire GUI callbacks: visibility toggle, Z-clip slider, and point-size
        # slider all operate on `_points_handle` without touching scene topology.
        self._gui_handles["show_points"].on_update(self._on_show_points_toggle)
        self._gui_handles["points_z_max"].on_update(self._on_points_zclip)
        self._gui_handles["points_size"].on_update(self._on_points_size)

        self._floor_selected: bool = False
        self._floor_finalized: bool = False
        self._selected_floor_z: float | None = None
        self._floor_plane_handle: Any | None = None
        self._floor_xy_half_extent: float = 10.0
        self._floor_grid_signature: tuple[float, bool] | None = None
        self._gui_handles["floor_z"].on_update(self._on_floor_z_update)
        self._gui_handles["select_floor"].on_update(self._on_select_floor)
        self._gui_handles["show_floor"].on_update(self._on_show_floor_toggle)

        self._device_traj: deque[tuple[int, np.ndarray]] = deque(
            maxlen=DEVICE_TRAJ_MAXLEN
        )
        self._person_traj: dict[int, deque[tuple[int, np.ndarray]]] = {}
        self._device_traj_handle: Any | None = None
        self._person_traj_handles: dict[int, Any] = {}

        # Cosy axis handle, gated by the `show_world_cosy` GUI checkbox.
        self._world_cosy_handle: Any | None = None

        for _vis_toggle in (
            "show_skeleton",
            "show_mesh",
            "show_camera_frustums",
            "show_world_cosy",
            "show_device_trajectory",
            "show_person_trajectories",
        ):
            self._gui_handles[_vis_toggle].on_update(self._on_visibility_toggle)

        self._runtime_gap_ms: deque[float] = deque(maxlen=30)
        self._runtime_compute_ms: deque[float] = deque(maxlen=30)
        self._runtime_viz_ms: deque[float] = deque(maxlen=30)

        self._server.on_client_connect(self._on_client_connect)

    def _on_client_connect(self, client: Any) -> None:
        """Frame a newly-connected client on the stored startup anchor."""
        if self._initial_render_cam is None:
            return
        position, look_at, up = self._initial_render_cam
        client.camera.position = position
        client.camera.look_at = look_at
        client.camera.up_direction = up

    # GUI construction

    def _build_gui(self) -> None:
        gui = self._server.gui

        self._image_folder: viser.GuiFolderHandle = gui.add_folder("Playback")
        with self._image_folder:
            self._gui_handles["auto_play"] = gui.add_checkbox(
                "Auto play", initial_value=False
            )
            self._gui_handles["next"] = gui.add_button("Next frame")
            # Fixed FPS on caps the live view at --fps; off runs as fast as
            # compute allows.
            self._gui_handles["fixed_fps"] = gui.add_checkbox(
                "Fixed FPS", initial_value=True
            )

        self._toggle_autoplay_cmd = gui.add_command("Toggle auto play", hotkey="space")

        @self._toggle_autoplay_cmd.on_trigger
        def _(_event: Any) -> None:
            handle = self._gui_handles["auto_play"]
            handle.value = not handle.value

        with gui.add_folder("Floor"):
            self._gui_handles["floor_z"] = gui.add_slider(
                "Floor Z (m)",
                min=-3.0,
                max=3.0,
                step=0.01,
                initial_value=0.0,
                disabled=True,
                hint=(
                    "Pick the floor height, then click `Select floor` BEFORE "
                    "starting the run. Fed to the model as a ground plane."
                ),
            )
            self._gui_handles["select_floor"] = gui.add_checkbox(
                "Select floor",
                initial_value=False,
                disabled=True,
                hint=(
                    "Lock the current Floor Z as the floor for this run. "
                    "Untoggle to adjust Floor Z again with live preview."
                ),
            )

            self._gui_handles["show_floor"] = gui.add_checkbox(
                "Show floor plane",
                initial_value=True,
                disabled=True,
                hint="Hide the selected floor plane to declutter the scene.",
            )
        with gui.add_folder("Scene"):
            self._gui_handles["show_world_cosy"] = gui.add_checkbox(
                "Show world cosy", initial_value=False
            )
            self._gui_handles["show_camera_frustums"] = gui.add_checkbox(
                "Show camera frustums", initial_value=True
            )

            self._gui_handles["show_points"] = gui.add_checkbox(
                "Show MPS points", initial_value=True
            )

            self._gui_handles["points_z_max"] = gui.add_slider(
                "MPS points: max Z",
                min=-10.0,
                max=10.0,
                step=0.05,
                initial_value=10.0,
            )

            self._gui_handles["points_size"] = gui.add_slider(
                "MPS point size (m)",
                min=0.002,
                max=0.02,
                step=0.001,
                initial_value=0.005,
            )
            self._gui_handles["show_device_trajectory"] = gui.add_checkbox(
                "Show device trajectory (5s)", initial_value=True
            )
            self._gui_handles["show_person_trajectories"] = gui.add_checkbox(
                "Show person trajectories (full)", initial_value=True
            )
        with gui.add_folder("People"):
            self._gui_handles["show_skeleton"] = gui.add_checkbox(
                "Show skeleton", initial_value=False
            )
            self._gui_handles["show_mesh"] = gui.add_checkbox(
                "Show SMPL mesh", initial_value=True
            )
            self._gui_handles["show_inactive"] = gui.add_checkbox(
                "Show inactive (faded)", initial_value=False
            )
            self._gui_handles["show_id"] = gui.add_checkbox(
                "Show id label", initial_value=True
            )
            self._gui_handles["show_projected_skeleton"] = gui.add_checkbox(
                "Show projected skeleton", initial_value=True
            )
            self._gui_handles["smoothing_delay"] = gui.add_slider(
                "Smoothing Delay",
                min=0,
                max=max(self._temporal_window - 1, 0),
                step=1,
                initial_value=max(self._temporal_window - 1, 0),
            )
        with gui.add_folder("Runtime"):
            self._gui_handles["runtime_vrs_ms"] = gui.add_text(
                "VRS Processing",
                initial_value="—",
                disabled=True,
                hint="Inter-call gap — VRS H.265 decode + Frameset construction.",
            )
            self._gui_handles["runtime_compute_ms"] = gui.add_text(
                "Frameset Processing",
                initial_value="—",
                disabled=True,
                hint="Inside _process_frameset minus viz: detect + track + lift + smoothing.",
            )
            self._gui_handles["runtime_viz_ms"] = gui.add_text(
                "Visualization",
                initial_value="—",
                disabled=True,
                hint="Sub-ms; the scene push runs on a worker thread.",
            )
            self._gui_handles["runtime_fps"] = gui.add_text(
                "FPS (p50)",
                initial_value="—",
                disabled=True,
                hint="1000 / (VRS + Frameset + Visualization) p50.",
            )

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def gui_handles(self) -> dict[str, Any]:
        """Viser handle objects keyed by GUI control names."""
        return self._gui_handles

    def set_camera_frustums(
        self,
        T_world_cams: dict[int, np.ndarray],
        intrinsics: dict[int, np.ndarray],  # 3x3 K per cam
        image_size: dict[int, tuple[int, int]],  # (W, H) per cam
    ) -> None:
        """Add or update one frustum per camera in the 3D scene."""
        visible = bool(self._gui_handles["show_camera_frustums"].value)
        for cam_idx, T_world_cam in T_world_cams.items():
            # Skip RGB / other non-track-creating cams per the constructor's
            # `frustum_cam_indices` allow-set. None = render all (e.g. tests).
            if (
                self._frustum_cam_indices is not None
                and cam_idx not in self._frustum_cam_indices
            ):
                continue
            K = intrinsics.get(cam_idx)
            size = image_size.get(cam_idx)
            if K is None or size is None:
                continue
            width, height = size
            fy = float(K[1, 1])
            fov = 2.0 * float(np.arctan2(0.5 * float(height), max(fy, 1e-6)))
            aspect = float(width) / float(max(height, 1))
            wxyz, position = se3_to_viser(T_world_cam)

            handle = self._frustum_handles.get(cam_idx)
            if handle is None:
                self._frustum_handles[cam_idx] = self._server.scene.add_camera_frustum(
                    name=f"/cameras/cam_{cam_idx}",
                    fov=fov,
                    aspect=aspect,
                    scale=0.1,
                    color=(255, 255, 255),
                    wxyz=wxyz,
                    position=position,
                    visible=visible,
                )
            else:
                handle.wxyz = wxyz
                handle.position = position
                handle.fov = fov
                handle.aspect = aspect
                handle.visible = visible

    def set_image_grid(self, images: dict[int, np.ndarray]) -> None:
        """Render all cameras as a single 2-column composite image."""
        if not images:
            return
        composite = compose_camera_grid(images, columns=2)
        handle = self._image_handles.get(COMPOSITE_HANDLE_KEY)
        if handle is None:
            with self._image_folder:
                self._image_handles[COMPOSITE_HANDLE_KEY] = self._server.gui.add_image(
                    composite, label="cams (2x2)", format="jpeg", jpeg_quality=60
                )
        else:
            handle.image = composite

    def set_2d_overlays(
        self,
        images: dict[int, np.ndarray],
        detections: dict[int, list[Detection2D]],
        projected_skeletons: dict[
            int, list[tuple[int, np.ndarray, tuple[float, float, float, float]]]
        ]
        | None = None,
    ) -> dict[int, np.ndarray]:
        """Draw bbox + keypoints + projected 3D skeleton onto each camera image."""
        out: dict[int, np.ndarray] = {}
        for cam_idx, img in images.items():
            cam_dets = detections.get(cam_idx, [])
            cam_proj = (
                projected_skeletons.get(cam_idx, [])
                if projected_skeletons is not None
                else []
            )
            out[cam_idx] = draw_overlays(img, cam_dets, cam_proj)
        return out

    def cache_update_frame(self, *args: Any, **kwargs: Any) -> None:
        """Cache frame data from an `update(...)` call without rendering it."""
        timestamp_ns = args[0] if len(args) >= 1 else kwargs.get("timestamp_ns")
        T_world_cams = args[1] if len(args) >= 2 else kwargs.get("T_world_cams")
        images = args[2] if len(args) >= 3 else kwargs.get("images")
        if timestamp_ns is None or T_world_cams is None or images is None:
            return
        self._cache_render_frame(
            timestamp_ns=int(timestamp_ns),
            T_world_cams=T_world_cams,
            images=images,
            detections=kwargs.get("detections"),
            intrinsics=kwargs.get("intrinsics"),
            cam_models=kwargs.get("cam_models"),
            T_world_device=kwargs.get("T_world_device"),
        )

    def _cache_render_frame(
        self,
        *,
        timestamp_ns: int,
        T_world_cams: dict[int, np.ndarray],
        images: dict[int, np.ndarray],
        detections: dict[int, list[Detection2D]] | None,
        intrinsics: dict[int, np.ndarray] | None,
        cam_models: dict[int, Any] | None,
        T_world_device: np.ndarray | None,
    ) -> RenderFrame:
        if intrinsics is None:
            intrinsics = {cam: np.eye(3, dtype=np.float32) for cam in T_world_cams}
        frame = RenderFrame(
            timestamp_ns=int(timestamp_ns),
            T_world_cams=dict(T_world_cams),
            images=dict(images),
            detections=(
                {cam: list(dets) for cam, dets in detections.items()}
                if detections is not None
                else None
            ),
            intrinsics=dict(intrinsics),
            cam_models=dict(cam_models) if cam_models is not None else None,
            T_world_device=T_world_device,
        )
        with self._render_frame_history_lock:
            for idx in range(len(self._render_frame_history) - 1, -1, -1):
                if self._render_frame_history[idx].timestamp_ns == frame.timestamp_ns:
                    self._render_frame_history[idx] = frame
                    break
            else:
                self._render_frame_history.append(frame)
        return frame

    def _select_delayed_render_frame(
        self, current_frame: RenderFrame, delay: int
    ) -> RenderFrame:
        if delay <= 0:
            return current_frame
        with self._render_frame_history_lock:
            frames = list(self._render_frame_history)
        if not frames:
            return current_frame
        return frames[max(0, len(frames) - 1 - int(delay))]

    def update(
        self,
        timestamp_ns: int,
        T_world_cams: dict[int, np.ndarray],
        images: dict[int, np.ndarray],
        people: dict[int, Person],
        detections: dict[int, list[Detection2D]] | None = None,
        intrinsics: dict[int, np.ndarray] | None = None,
        cam_models: dict[int, Any] | None = None,
        T_world_device: np.ndarray | None = None,
    ) -> None:
        """Push one frame's worth of state into the viewer."""
        current_ts_ns = int(timestamp_ns)
        current_frame = self._cache_render_frame(
            timestamp_ns=current_ts_ns,
            T_world_cams=T_world_cams,
            images=images,
            detections=detections,
            intrinsics=intrinsics,
            cam_models=cam_models,
            T_world_device=T_world_device,
        )
        smoothing_delay = int(self._gui_handles["smoothing_delay"].value)
        render_frame = self._select_delayed_render_frame(current_frame, smoothing_delay)
        render_ts_ns = render_frame.timestamp_ns

        # Delayed poses are rendered with the matching delayed image/camera frame.
        image_sizes: dict[int, tuple[int, int]] = {}
        for cam, img in render_frame.images.items():
            if img.ndim >= 2:
                image_sizes[cam] = (int(img.shape[1]), int(img.shape[0]))
        # Fall back to a 1:1 aspect square if we don't have an image for a cam.
        for cam in render_frame.T_world_cams:
            image_sizes.setdefault(cam, (640, 640))
        self.set_camera_frustums(
            render_frame.T_world_cams, render_frame.intrinsics, image_sizes
        )

        self._update_world_cosy()

        joints_by_person = self._update_skeletons(people, render_ts_ns)

        self._update_trajectories(
            render_ts_ns,
            render_frame.T_world_device,
            people,
            joints_by_person=joints_by_person,
        )

        show_projected_skeleton = bool(
            self._gui_handles["show_projected_skeleton"].value
        )
        projected_skeletons = None
        if show_projected_skeleton and render_frame.cam_models is not None and people:
            projected_skeletons = self._project_skeletons_for_overlay(
                people,
                render_frame.T_world_cams,
                render_frame.cam_models,
                render_ts_ns,
                joints_by_person=joints_by_person,
            )

        if render_frame.detections is not None or projected_skeletons is not None:
            grid_images = self.set_2d_overlays(
                render_frame.images,
                render_frame.detections or {},
                projected_skeletons=projected_skeletons,
            )
        else:
            grid_images = render_frame.images
        self.set_image_grid(grid_images)

    # 3D scene update helpers

    def _update_world_cosy(self) -> None:
        """Add or hide the world-frame cosy axes (origin, length 1m)."""
        visible = bool(self._gui_handles["show_world_cosy"].value)
        if not visible:
            if self._world_cosy_handle is not None:
                self._world_cosy_handle.visible = False
            return
        if self._world_cosy_handle is None:
            self._world_cosy_handle = self._server.scene.add_frame(
                "/cosy/world",
                axes_length=1.0,
                axes_radius=0.01,
            )
        else:
            self._world_cosy_handle.visible = True

    def _update_skeletons(
        self,
        people: dict[int, Person],
        current_ts_ns: int,
    ) -> dict[int, np.ndarray]:
        """Render 3D skeletons and meshes for every active person."""
        show_skel = bool(self._gui_handles["show_skeleton"].value)
        show_mesh = bool(self._gui_handles["show_mesh"].value)
        show_inactive = bool(self._gui_handles["show_inactive"].value)
        delay = int(self._gui_handles["smoothing_delay"].value)
        drawn_skel: set[int] = set()
        drawn_mesh: set[int] = set()
        joints_by_person: dict[int, np.ndarray] = {}
        batched: list[tuple[int, tuple[float, float, float, float], SmplParams]] = []
        for person_id, person in people.items():
            if not person.active and not show_inactive:
                continue
            if person.last_lifted_ts == -1:
                continue
            target_state = lookup_delayed_state(person, current_ts_ns, delay)
            if target_state is None or target_state.skeleton is None:
                continue
            skel = target_state.skeleton
            base = person.color or color_from_id(person_id)
            alpha = 1.0 if person.active else _INACTIVE_ALPHA
            color_rgba = (base[0], base[1], base[2], alpha)
            params = smpl_params_from_skeleton(skel)
            if self._smpl_forward_fn is not None and params is not None:
                batched.append((person_id, color_rgba, params))
                continue

            if skel.kp_world.size > 0:
                joints = np.asarray(skel.kp_world, dtype=np.float32)
                joints_by_person[person_id] = joints
                self._render_skeleton(person_id, joints, color_rgba, visible=show_skel)
                drawn_skel.add(person_id)
            if self._smpl_faces is not None and skel.verts_w.size > 0:
                self._render_mesh(
                    person_id, skel.verts_w, color_rgba, visible=show_mesh
                )
                drawn_mesh.add(person_id)

        if self._smpl_forward_fn is not None and batched:
            betas = np.stack([p.betas for _, _, p in batched], axis=0)
            global_orient = np.stack([p.global_orient for _, _, p in batched], axis=0)
            body_pose = np.stack([p.body_pose for _, _, p in batched], axis=0)
            transl = np.stack([p.transl for _, _, p in batched], axis=0)
            joints_batch, verts_batch = self._smpl_forward_fn(
                betas=betas,
                global_orient_rotmat=global_orient,
                body_pose_rotmat=body_pose,
                transl=transl,
            )
            for i, (person_id, color_rgba, _params) in enumerate(batched):
                joints = np.ascontiguousarray(joints_batch[i], dtype=np.float32)
                joints_by_person[person_id] = joints
                self._render_skeleton(person_id, joints, color_rgba, visible=show_skel)
                drawn_skel.add(person_id)
                if self._smpl_faces is not None:
                    self._render_mesh(
                        person_id, verts_batch[i], color_rgba, visible=show_mesh
                    )
                    drawn_mesh.add(person_id)

        for stale_id in list(self._skeleton_handles.keys()):
            if stale_id not in drawn_skel:
                handle = self._skeleton_handles.pop(stale_id, None)
                if handle is not None:
                    handle.remove()
        for stale_id in list(self._mesh_handles.keys()):
            if stale_id not in drawn_mesh:
                handle = self._mesh_handles.pop(stale_id, None)
                if handle is not None:
                    handle.remove()
        return joints_by_person

    def _project_skeletons_for_overlay(
        self,
        people: dict[int, Person],
        T_world_cams: dict[int, np.ndarray],
        cam_models: dict[int, Any],
        current_ts_ns: int,
        *,
        joints_by_person: dict[int, np.ndarray] | None = None,
    ) -> ProjectedSkeletons:
        return project_skeletons_for_overlay(
            people,
            T_world_cams,
            cam_models,
            current_ts_ns,
            int(self._gui_handles["smoothing_delay"].value),
            joints_by_person=joints_by_person,
        )

    def _render_skeleton(
        self,
        person_id: int,
        kp_world: np.ndarray,
        color_rgba: tuple[float, float, float, float],
        visible: bool,
    ) -> None:
        """Render one person's SMPL skeleton as line segments."""
        if kp_world.size == 0:
            return
        edges = skeleton_edges_for(int(kp_world.shape[0]))
        segments = np.array(
            [[kp_world[a], kp_world[b]] for a, b in edges],
            dtype=np.float32,
        )
        # Line segments have no opacity in viser, so fade an inactive person by
        # blending the color toward white in proportion to its alpha.
        alpha = float(color_rgba[3])
        rgb_uint = np.array(
            [round((color_rgba[i] * alpha + (1.0 - alpha)) * 255) for i in range(3)],
            dtype=np.uint8,
        )

        # Viser's `add_line_segments` colors take `(3,)` global, `(N, 3)`
        # per-segment, or `(N, 2, 3)` per-endpoint. Use the global form.
        handle = self._skeleton_handles.get(person_id)
        # If the segment count changes, recreate the handle because viser does
        # not reliably resize line-segment geometry in place.
        if handle is not None and handle.points.shape != segments.shape:
            handle.remove()
            handle = None
        if handle is None:
            self._skeleton_handles[person_id] = self._server.scene.add_line_segments(
                name=f"/people/person_{person_id}/skeleton",
                points=segments,
                colors=rgb_uint,
                line_width=2.0,
                visible=visible,
            )
        else:
            handle.points = segments
            handle.colors = rgb_uint
            handle.visible = visible

    def _render_mesh(
        self,
        person_id: int,
        verts_w: np.ndarray,
        color_rgba: tuple[float, float, float, float],
        visible: bool,
    ) -> None:
        """Render or update one person's SMPL mesh."""
        assert self._smpl_faces is not None
        color = (
            round(color_rgba[0] * 255),
            round(color_rgba[1] * 255),
            round(color_rgba[2] * 255),
        )
        opacity = float(color_rgba[3])
        verts = np.ascontiguousarray(verts_w, dtype=np.float32)
        handle = self._mesh_handles.get(person_id)
        if handle is None:
            self._mesh_handles[person_id] = self._server.scene.add_mesh_simple(
                name=f"/people/person_{person_id}/mesh",
                vertices=verts,
                faces=self._smpl_faces,
                color=color,
                opacity=opacity,
                visible=visible,
            )
        else:
            handle.vertices = verts
            handle.color = color
            handle.opacity = opacity
            handle.visible = visible

    # Rolling trajectory trails (device + per person, 5s window)

    def _update_trajectories(
        self,
        current_ts_ns: int,
        T_world_device: np.ndarray | None,
        people: dict[int, Person],
        *,
        joints_by_person: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Append the current device + per-person pelvis positions to the trails."""
        show_dev = bool(self._gui_handles["show_device_trajectory"].value)
        show_ppl = bool(self._gui_handles["show_person_trajectories"].value)

        # Device trail (5 s window).
        if T_world_device is not None:
            pos = T_world_device[:3, 3].astype(np.float32, copy=True)
            self._device_traj.append((current_ts_ns, pos))
            trim_traj(self._device_traj, current_ts_ns)
        self._device_traj_handle = self._render_trajectory_line_handle(
            handle=self._device_traj_handle,
            scene_name="/scene/device_trajectory",
            buf=self._device_traj,
            color=np.array([255, 255, 255], dtype=np.uint8),  # white, matches frustums
            visible=show_dev,
        )

        drawn_pids: set[int] = set()
        for pid, person in people.items():
            if not person.active:
                continue
            if person.last_lifted_ts == -1:
                continue
            target = lookup_delayed_state(
                person,
                current_ts_ns,
                int(self._gui_handles["smoothing_delay"].value),
            )
            if target is None or target.skeleton is None:
                continue
            joints = (
                joints_by_person.get(pid)
                if joints_by_person is not None
                else target.skeleton.kp_world
            )
            if joints is not None and joints.size > 0:
                pelvis_pos = joints[0].astype(np.float32, copy=True)
            else:
                pelvis_pos = target.skeleton.T_world_pelvis[:3, 3].astype(
                    np.float32, copy=True
                )
            buf = self._person_traj.setdefault(pid, deque(maxlen=PERSON_TRAJ_MAXLEN))
            buf.append((current_ts_ns, pelvis_pos))
            color_rgba = person.color or color_from_id(pid)
            rgb = np.array(
                [
                    round(color_rgba[0] * 255),
                    round(color_rgba[1] * 255),
                    round(color_rgba[2] * 255),
                ],
                dtype=np.uint8,
            )
            handle = self._person_traj_handles.get(pid)
            self._person_traj_handles[pid] = self._render_trajectory_line_handle(
                handle=handle,
                scene_name=f"/people/person_{pid}/trajectory",
                buf=buf,
                color=rgb,
                visible=show_ppl,
            )
            drawn_pids.add(pid)
        # Drop trails for persons that disappeared.
        for stale_pid in list(self._person_traj_handles.keys()):
            if stale_pid not in drawn_pids:
                handle = self._person_traj_handles.pop(stale_pid, None)
                if handle is not None:
                    handle.remove()
                self._person_traj.pop(stale_pid, None)

    def _render_trajectory_line_handle(
        self,
        handle: Any | None,
        scene_name: str,
        buf: deque[tuple[int, np.ndarray]],
        color: np.ndarray,
        visible: bool,
    ) -> Any | None:
        """Build / update one LineSegments handle from a trajectory buffer."""
        if buf.maxlen is None:
            raise ValueError(
                "Trajectory deque must have a bounded maxlen for fixed-shape "
                "rendering; pre-allocate via deque(maxlen=...)."
            )
        n_real_pts = len(buf)
        if n_real_pts < 2:
            if handle is not None:
                handle.visible = False
            return handle
        n_seg_total = buf.maxlen - 1
        n_seg_real = n_real_pts - 1
        pts = np.stack([p for _, p in buf], axis=0).astype(np.float32, copy=False)
        last_pt = pts[-1]
        segments = np.empty((n_seg_total, 2, 3), dtype=np.float32)
        segments[:n_seg_real, 0] = pts[:-1]
        segments[:n_seg_real, 1] = pts[1:]
        # Pad: last_pt → last_pt is a zero-length segment, invisible.
        segments[n_seg_real:, 0] = last_pt
        segments[n_seg_real:, 1] = last_pt
        if handle is not None:
            handle.points = segments
            handle.colors = color
            handle.visible = visible
            return handle
        return self._server.scene.add_line_segments(
            name=scene_name,
            points=segments,
            colors=color,
            line_width=2.0,
            visible=visible,
        )

    # Static MPS semidense point cloud (one-shot upload at startup)

    def set_semidense_points(
        self,
        points: np.ndarray,
        color: tuple[int, int, int] = (200, 200, 200),
        point_size: float = 0.005,
    ) -> None:
        """Upload the MPS semidense point cloud as a single static scene node."""
        self._points_xyz = np.ascontiguousarray(points, dtype=np.float32)
        self._points_color = color
        self._points_size = float(point_size)

        z = self._points_xyz[:, 2]

        z_lo, z_hi = float(np.percentile(z, 5.0)), float(z.max())
        slider = self._gui_handles["points_z_max"]
        slider.min = z_lo
        slider.max = z_hi
        slider.value = z_hi  # show all points; user drags down to clip ceiling

        if not self._floor_selected and not self._floor_finalized:
            floor_slider = self._gui_handles["floor_z"]
            floor_slider.min = z_lo
            floor_slider.max = z_hi
            floor_slider.value = z_lo
            floor_slider.disabled = False
            self._gui_handles["select_floor"].disabled = False
        # Size the floor preview grid to the cloud's XY extent so it visually
        # spans the scene (a bit of padding past the raw min/max). Robust
        # percentiles avoid a single stray point blowing the grid up.
        x_span = float(np.percentile(self._points_xyz[:, 0], 95.0)) - float(
            np.percentile(self._points_xyz[:, 0], 5.0)
        )
        y_span = float(np.percentile(self._points_xyz[:, 1], 95.0)) - float(
            np.percentile(self._points_xyz[:, 1], 5.0)
        )
        self._floor_xy_half_extent = max(0.5 * max(x_span, y_span), 1.0)
        self._render_floor_plane()

        if self._points_handle is not None:
            self._points_handle.remove()
            self._points_handle = None
        self._publish_points(point_size=point_size)

    def _publish_points(self, point_size: float | None = None) -> None:
        """Create or update the point-cloud handle from the cached `_points_xyz`."""
        if self._points_xyz is None:
            return
        if point_size is not None:
            self._points_size = float(point_size)
        z_max = float(self._gui_handles["points_z_max"].value)
        mask = self._points_xyz[:, 2] <= z_max
        filtered = self._points_xyz[mask]
        if filtered.shape[0] == 0:
            # Viser's add_point_cloud rejects empty arrays; just hide.
            if self._points_handle is not None:
                self._points_handle.visible = False
            return
        visible = bool(self._gui_handles["show_points"].value)
        colors = np.tile(
            np.asarray(self._points_color, dtype=np.uint8), (filtered.shape[0], 1)
        )
        if self._points_handle is None:
            # float32 precision so the wide (tens-of-meters) world coords don't
            # quantize under viser's default float16 buffers.
            self._points_handle = self._server.scene.add_point_cloud(
                name="/scene/mps_points",
                points=filtered,
                colors=colors,
                point_size=self._points_size,
                point_shape="circle",
                precision="float32",
                point_shading="flat",
                visible=visible,
            )
        else:
            self._points_handle.points = filtered
            self._points_handle.colors = colors
            self._points_handle.visible = visible

    def _on_show_points_toggle(self, _evt: Any) -> None:
        """GUI callback: flip the static point-cloud handle's visibility."""
        if self._points_handle is None:
            return
        self._points_handle.visible = bool(self._gui_handles["show_points"].value)

    def _on_visibility_toggle(self, _evt: Any = None) -> None:
        """Apply every scene-visibility toggle to its cached handles right now."""
        g = self._gui_handles
        show_skel = bool(g["show_skeleton"].value)
        for handle in self._skeleton_handles.values():
            handle.visible = show_skel
        show_mesh = bool(g["show_mesh"].value)
        for handle in self._mesh_handles.values():
            handle.visible = show_mesh
        show_frustums = bool(g["show_camera_frustums"].value)
        for handle in self._frustum_handles.values():
            handle.visible = show_frustums
        if self._world_cosy_handle is not None:
            self._world_cosy_handle.visible = bool(g["show_world_cosy"].value)
        if self._device_traj_handle is not None:
            self._device_traj_handle.visible = bool(g["show_device_trajectory"].value)
        show_ppl = bool(g["show_person_trajectories"].value)
        for handle in self._person_traj_handles.values():
            handle.visible = show_ppl

    def _on_points_zclip(self, _evt: Any) -> None:
        """GUI callback: re-filter the cached points by world-Z and re-push."""
        self._publish_points()

    def _on_points_size(self, _evt: Any) -> None:
        """GUI callback: update the MPS point-cloud render size in place."""
        self._points_size = float(self._gui_handles["points_size"].value)
        if self._points_handle is not None:
            self._points_handle.point_size = self._points_size

    # First camera pose startup anchor (pre-run setup)

    def set_initial_view(
        self,
        T_world_device: np.ndarray,
        T_world_cams: dict[int, np.ndarray],
    ) -> None:
        """Anchor the viewer to the FIRST valid camera pose, before the run."""
        del T_world_device  # framing derives from the anchor cam; kept for API
        if not T_world_cams:
            return
        anchor = min(T_world_cams)
        cam = T_world_cams[anchor]
        cam_pos = cam[:3, 3]
        # The SLAM cam's +z is its viewing direction (forward).
        cam_fwd = cam[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=cam.dtype)
        world_up = np.array([0.0, 0.0, 1.0], dtype=cam.dtype)  # LAMP world is Z-up

        # (#3) Floor anchor: center the floor grid under the anchor cam's XY.
        self._floor_anchor_xy = (float(cam_pos[0]), float(cam_pos[1]))
        self._render_floor_plane()

        # (#2) Render camera: up-and-behind the anchor cam, looking in front.
        position_arr = (
            cam_pos - cam_fwd * INITIAL_VIEW_BEHIND_M + world_up * INITIAL_VIEW_UP_M
        )
        look_at_arr = cam_pos + cam_fwd * INITIAL_VIEW_LOOK_M
        position = (
            float(position_arr[0]),
            float(position_arr[1]),
            float(position_arr[2]),
        )
        look_at = (float(look_at_arr[0]), float(look_at_arr[1]), float(look_at_arr[2]))
        up = (float(world_up[0]), float(world_up[1]), float(world_up[2]))
        self._initial_render_cam = (position, look_at, up)
        # Apply to clients already connected; `_on_client_connect` handles the
        # ones that connect after this point.
        for client in self._server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up

    # Fixed-floor selection (pre-run setup)

    def _render_floor_plane(self) -> None:
        """Draw / move the floor preview grid at the current slider Z."""
        z = float(self._gui_handles["floor_z"].value)
        d = self._floor_xy_half_extent
        # Center the grid under the startup anchor's XY (set by
        # `set_initial_view`); defaults to the world origin (0, 0) until then.
        ax, ay = self._floor_anchor_xy
        position = (ax, ay, z)
        if self._floor_selected:
            cell_color = (90, 210, 120)  # locked = green
            section_color = (60, 170, 90)
        else:
            cell_color = (120, 170, 255)  # preview = blue-ish
            section_color = (90, 140, 230)

        signature = (round(d, 4), self._floor_selected)
        handle = self._floor_plane_handle
        if handle is None or signature != self._floor_grid_signature:
            # Recreate (size/color changed). Preserve the live visibility so a
            # `Show floor plane` hide choice survives the lock-time recolor.
            visible = True if handle is None else bool(handle.visible)
            if handle is not None:
                handle.remove()
            self._floor_plane_handle = self._server.scene.add_grid(
                name="/scene/floor_plane",
                width=2.0 * d,
                height=2.0 * d,
                plane="xy",  # horizontal in the Z-up LAMP world
                cell_color=cell_color,
                cell_size=1.0,  # 1 m cells
                section_color=section_color,
                section_thickness=2.0,
                section_size=5.0,  # heavier line every 5 m
                position=position,
                visible=visible,
            )
            self._floor_grid_signature = signature
        else:
            # Common case: just slide it to the new Z / anchor.
            handle.position = position

    def _on_floor_z_update(self, _evt: Any) -> None:
        """GUI callback: re-render the preview grid as the slider moves."""
        if self._floor_selected:
            return
        self._render_floor_plane()

    def _on_show_floor_toggle(self, _evt: Any) -> None:
        """GUI callback: flip the floor plane's visibility.

        Only meaningful once the plane exists (drawn by `set_semidense_points`)
        and the toggle is enabled (after `Select floor`). A no-op before then.
        """
        if self._floor_plane_handle is not None:
            self._floor_plane_handle.visible = bool(
                self._gui_handles["show_floor"].value
            )

    def _on_select_floor(self, _evt: Any) -> None:
        """GUI callback: lock / unlock the floor plane at the current slider Z.

        Checked locks the current Floor Z as the chosen floor and enables the
        `Show floor plane` toggle. Unchecking releases the lock so Floor Z can
        be dragged again with live preview.
        """
        self._floor_selected = bool(self._gui_handles["select_floor"].value)
        if self._floor_selected:
            z = float(self._gui_handles["floor_z"].value)
            self._selected_floor_z = z
            self._gui_handles["show_floor"].disabled = False
            logger.info("Floor selected at z=%.3f m", z)
        else:
            self._selected_floor_z = None
            logger.info("Floor selection cleared; Floor Z is adjustable again.")
        self._render_floor_plane()

    def finalize_floor(self) -> float | None:
        """Disable the floor controls and return the selected floor height."""
        self._floor_finalized = True
        self._gui_handles["floor_z"].disabled = True
        self._gui_handles["select_floor"].disabled = True
        # Never selected → the preview grid (if any was drawn) is dead weight;
        # hide it so the scene isn't littered with an unused plane.
        if not self._floor_selected and self._floor_plane_handle is not None:
            self._floor_plane_handle.visible = False
        return self._selected_floor_z if self._floor_selected else None

    # Runtime stats (live pipeline → UI breakdown)

    def set_runtime_stats(self, stats: Any) -> None:
        """Push the latest `LampPipelineStats` into the Runtime GUI panel."""
        self._runtime_gap_ms.append(float(stats.gap_ms))
        compute_ms = float(stats.proc_total_ms) - float(stats.viz_ms)
        self._runtime_compute_ms.append(max(compute_ms, 0.0))
        self._runtime_viz_ms.append(float(stats.viz_ms))

        gapp50 = p50(self._runtime_gap_ms)
        computep50 = p50(self._runtime_compute_ms)
        vizp50 = p50(self._runtime_viz_ms)
        totalp50 = gapp50 + computep50 + vizp50

        self._gui_handles["runtime_vrs_ms"].value = f"{gapp50:6.1f} ms"
        self._gui_handles["runtime_compute_ms"].value = f"{computep50:6.1f} ms"
        self._gui_handles["runtime_viz_ms"].value = f"{vizp50:6.1f} ms"
        self._gui_handles["runtime_fps"].value = (
            f"{1000.0 / totalp50:5.2f} fps" if totalp50 > 0 else "—"
        )

    # Lifecycle

    def shutdown(self) -> None:
        """Stop the Viser server. Safe to call more than once."""
        if not self._stopped:
            self._server.stop()
            self._stopped = True
