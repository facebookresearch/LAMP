# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end orchestrator: VRS + MPS -> detector -> tracker -> lifter -> sinks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from lamp.core.types import Frameset, LampResult
from lamp.detection.detector import (
    BBoxDetector,
    Detector2D,
    KeypointDetector,
    PeopleDetector2dSettings,
)
from lamp.io.parallel_vrs import ParallelVrsLoader
from lamp.io.persistence import save_results
from lamp.io.sensor_io import (
    MpsLoader,
    PerCameraCalibration,
    resolve_recording_paths,
    VrsLoader,
)
from lamp.models.lifter import Lifter, LifterSettings
from lamp.tracking.tracker import LampTracker, LampTrackerSettings
from lamp.visualization.threaded import ThreadedVisualizer
from lamp.visualization.visualizer import Visualizer

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class LampPipelineSettings:
    """User-facing configuration for an end-to-end pipeline run."""

    detector: PeopleDetector2dSettings = field(default_factory=PeopleDetector2dSettings)
    tracker: LampTrackerSettings = field(default_factory=LampTrackerSettings)
    lifter: LifterSettings = field(default_factory=LifterSettings)

    smpl_model_path: Path | None = None

    cuda_graphs: bool = True
    # Static CUDA Graph replay batch; larger batches run eager.
    capture_batch_size: int = 8
    device: str = "cuda:0"
    enable_visualizer: bool = True
    visualizer_port: int = 8080
    # When set, `run()` writes JSON results to this path. None skips persistence.
    save_path: Path | None = None
    # Subset of camera indices to drive the pipeline with. None uses all cameras.
    cam_index_subset: list[int] | None = None
    # Process-level parallel SLAM decode. Project Aria's HEVC path is
    # single-threaded per stream, so Gen2 benefits from one worker per SLAM cam.
    vrs_decode_workers: int = 4
    vrs_decode_prefetch: int = 4
    # Cameras that are shown in the viewer but skipped by person detection and
    # cannot seed new tracks. Defaults to RGB cameras reported by `VrsLoader`.
    non_track_creating_cam_indices: list[int] | None = None

    fps: float | None = 10.0

    start_ts_s: float = 0.0


@dataclass
class LampPipelineStats:
    """Per-frameset wall-clock budget surfaced after each `_process_frameset`."""

    detect_ms: float = 0.0
    # Sub-detector breakdown populated from `Detector2D.last_detect_stats`.
    bbox_ms: float = 0.0
    kp_ms: float = 0.0
    detect_post_ms: float = 0.0
    track_ms: float = 0.0
    lift_ms: float = 0.0
    smoothing_ms: float = 0.0
    viz_ms: float = 0.0
    # Wall time between consecutive `_process_frameset` calls (VRS decode +
    # frameset build); 0.0 on the first call.
    gap_ms: float = 0.0
    # Wall time of the current `_process_frameset` call (sum of the stage times
    # plus MPS-query and Python overhead).
    proc_total_ms: float = 0.0
    n_lifts: int = 0
    n_merges: int = 0


def _resolve_non_track_creating_cams(
    settings: LampPipelineSettings, vrs: VrsLoader | ParallelVrsLoader
) -> set[int]:
    """Cameras excluded from detection, defaulting to the RGB cameras."""
    if settings.non_track_creating_cam_indices is None:
        return set(vrs.rgb_camera_indices)
    return set(settings.non_track_creating_cam_indices)


def _pacing_interval(fixed_fps: bool, fps: float | None) -> float:
    """Minimum seconds between consecutive live frames.

    Fixed-FPS mode caps the live view at the ``--fps`` target (``1/fps``); with
    the toggle off (or no/zero ``fps``) it returns 0 — no pacing, i.e. run as
    fast as compute allows.
    """
    if fixed_fps and fps and fps > 0:
        return 1.0 / fps
    return 0.0


class _LivePacer:
    """Spaces consecutive live frames at least one interval apart.

    The interval is anchored to each frame's START, so the rate is capped at
    `1/interval` and is NOT inflated by per-frame compute time — the period is
    `max(interval, work)`, dropping below the target only when a frame takes
    longer than the interval. `wait()` is pure (returns the sleep duration); the
    caller does the sleeping so this stays testable with a fake clock.
    """

    def __init__(self, now: float) -> None:
        self._next_start = now

    def wait(self, interval_s: float, now: float) -> float:
        """Return seconds to sleep before the current frame, advancing the anchor."""
        sleep_s = max(0.0, self._next_start - now)
        # The frame starts after sleeping; the next is due one interval later.
        self._next_start = now + sleep_s + interval_s
        return sleep_s


class LampPipeline:
    """Orchestrates one full LAMP pipeline run over a recording folder."""

    def __init__(
        self,
        *,
        settings: LampPipelineSettings,
        vrs: VrsLoader | ParallelVrsLoader,
        mps: MpsLoader,
        detector: Detector2D,
        tracker: LampTracker,
        lifter: Lifter,
        visualizer: Visualizer | ThreadedVisualizer | None = None,
    ) -> None:
        self._settings: LampPipelineSettings = settings
        self._vrs: VrsLoader | ParallelVrsLoader = vrs
        self._mps: MpsLoader = mps
        self._detector: Detector2D = detector
        self._tracker: LampTracker = tracker
        self._lifter: Lifter = lifter
        # Either the raw Visualizer or its ThreadedVisualizer wrapper — both
        # expose the same public surface (gui_handles, host, port, update,
        # shutdown), so the per-frame loop here is unaware of which it has.
        self._visualizer: Visualizer | ThreadedVisualizer | None = visualizer

        # cam_idx -> PerCameraCalibration cache, snapshotted at construction
        # so the per-frame loop doesn't re-derive it. Honors `cam_index_subset`
        # when set; falls back to every camera the VRS exposes otherwise.
        active_cams = (
            list(settings.cam_index_subset)
            if settings.cam_index_subset is not None
            else list(vrs.camera_indices)
        )
        self._cam_models: dict[int, PerCameraCalibration | None] = {
            idx: vrs.calibration.camera(idx) for idx in active_cams
        }

        # Exclude non-track-creating cameras from detection. The viewer still
        # receives those images and frustums.
        non_track_creating = _resolve_non_track_creating_cams(settings, vrs)
        self._detector_cam_indices: set[int] = {
            idx for idx in active_cams if idx not in non_track_creating
        }

        # FPS throttling lives in the VRS streaming layer, before H.265 decode.

        # Bookkeeping for `save_results`.
        self._all_timestamps: list[int] = []
        # Per-frame profiling populated after each `_process_frameset`.
        self._last_run_stats: LampPipelineStats = LampPipelineStats()

        self._last_proc_end_perf: float | None = None

        self.frame_gate: Callable[[], None] | None = None

    # Construction from on-disk artifacts

    @classmethod
    def from_paths(
        cls,
        recording: Path,
        lamp_ckpt: Path,
        device: str = "cuda:0",
        settings: LampPipelineSettings | None = None,
    ) -> LampPipeline:
        """Build the full pipeline from on-disk artifact paths."""
        if settings is None:
            settings = LampPipelineSettings(device=device)
        else:
            # Caller supplied a settings struct; honor its `device` rather
            # than the bare `device=` arg if the two disagree.
            device = settings.device

        # Fail-fast on missing SMPL model — the checkpoint lifter is the only path
        # and it needs the SMPL neutral .pkl. Do this before any expensive
        # component construction (detector init alone can take seconds).
        if settings.smpl_model_path is None:
            raise ValueError(
                "smpl_model_path is required for the checkpoint lifter. Download the "
                "SMPL neutral .pkl from https://smpl.is.tue.mpg.de and set "
                "settings.smpl_model_path (or pass --smpl-model-path on the CLI)."
            )

        recording_paths = resolve_recording_paths(recording)
        vrs_metadata = VrsLoader(
            recording_paths.vrs,
            decode_camera_indices=settings.cam_index_subset,
        )
        mps_loader = MpsLoader(recording_paths.root)

        non_track_creating = _resolve_non_track_creating_cams(settings, vrs_metadata)

        ds = settings.detector
        # BBox stage. RF-DETR-Nano weights auto-download and cache; the TRT
        # path derives its ONNX + engine-cache paths from the weights.
        if ds.bbox_backend == "rfdetr":
            bbox = BBoxDetector.from_rfdetr(
                device=device,
                score_thresh=ds.min_box_conf,
                min_box_size_ratio=ds.min_box_size_ratio,
            )
        elif ds.bbox_backend == "rfdetr-trt":
            bbox = BBoxDetector.from_rfdetr_trt(
                device=device,
                score_thresh=ds.min_box_conf,
                min_box_size_ratio=ds.min_box_size_ratio,
            )
        else:
            raise ValueError(f"unsupported bbox_backend: {ds.bbox_backend!r}")
        # Keypoint stage — HuggingFace ViTPose is the only path.
        kp = KeypointDetector.from_huggingface(
            model_id=ds.kp_hf_model_id, device=device
        )
        det = Detector2D(bbox, kp, ds)

        # `smpl_model_path is not None` is guaranteed by the fail-fast
        # check at the top of this method.
        assert settings.smpl_model_path is not None
        lifter = Lifter.from_checkpoint(
            lamp_ckpt,
            smpl_model_path=settings.smpl_model_path,
            device=device,
            settings=settings.lifter,
            capture_cuda_graph=settings.cuda_graphs,
            capture_batch_size=settings.capture_batch_size,
        )

        tracker = LampTracker(
            num_cameras=len(vrs_metadata.camera_indices),
            settings=settings.tracker,
            non_track_creating_cam_indices=non_track_creating,
        )

        visualizer: Visualizer | ThreadedVisualizer | None = None
        if settings.enable_visualizer:
            active_cams = (
                set(settings.cam_index_subset)
                if settings.cam_index_subset is not None
                else set(vrs_metadata.camera_indices)
            )
            frustum_cams = active_cams - non_track_creating
            inner_viz = Visualizer(
                port=settings.visualizer_port,
                smpl_faces=lifter.smpl_faces,
                frustum_cam_indices=frustum_cams,
                smpl_forward_fn=lifter.forward_smpl_geometry,
                # Sizes the "Smoothing Delay" slider range to the model's
                # temporal window.
                temporal_window=lifter.snippet_length,
            )
            logger.info("Visualizer at http://localhost:%d", inner_viz.port)
            # MPS semidense points are static in the world frame — upload
            # once at startup. Filtered + subsampled in `semidense_points`;
            # returns None when the recording folder has no point cloud.
            points = mps_loader.semidense_points()
            if points is not None:
                logger.info("Loaded %d MPS semidense points", points.shape[0])
                inner_viz.set_semidense_points(points)

            # Visualizer updates run on a worker thread with single-slot
            # dropping so the ~22 ms scene push stays off the compute budget.
            visualizer = ThreadedVisualizer(inner_viz)

        vrs_loader: VrsLoader | ParallelVrsLoader
        if settings.vrs_decode_workers > 1:
            vrs_loader = ParallelVrsLoader(
                recording_paths.vrs,
                base_loader=vrs_metadata,
                max_workers=settings.vrs_decode_workers,
                prefetch_frames=settings.vrs_decode_prefetch,
            )
        else:
            vrs_loader = vrs_metadata

        return cls(
            settings=settings,
            vrs=vrs_loader,
            mps=mps_loader,
            detector=det,
            tracker=tracker,
            lifter=lifter,
            visualizer=visualizer,
        )

    # Properties (handy for tests + downstream callers)

    @property
    def settings(self) -> LampPipelineSettings:
        return self._settings

    @property
    def vrs(self) -> VrsLoader | ParallelVrsLoader:
        return self._vrs

    @property
    def mps(self) -> MpsLoader:
        return self._mps

    @property
    def tracker(self) -> LampTracker:
        return self._tracker

    @property
    def lifter(self) -> Lifter:
        return self._lifter

    @property
    def visualizer(self) -> Visualizer | ThreadedVisualizer | None:
        return self._visualizer

    @property
    def last_run_stats(self) -> LampPipelineStats:
        """Per-stage wall-clock budget for the most recently processed frame."""
        return self._last_run_stats

    # Startup anchor (pre-run setup)

    def _first_localized_frame(self) -> tuple[int, int, np.ndarray] | None:
        """Find the first frameset with a valid MPS pose, without decoding."""
        ts_list = self._vrs.frameset_timestamps_ns()
        # Honor `--start-ts`: don't anchor on a frame in the skipped lead-in.
        start_offset_ns = max(0, int(self._settings.start_ts_s * 1e9))
        first_ts = ts_list[0] if ts_list else 0
        for idx, ts in enumerate(ts_list):
            if start_offset_ns > 0 and (ts - first_ts) < start_offset_ns:
                continue
            T_world_device = self._mps.query_T_world_device(ts)
            if T_world_device is not None:
                return idx, ts, T_world_device
        return None

    def first_valid_camera_pose(
        self,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]] | None:
        """Compute the first frameset's camera poses UPFRONT, without decoding."""
        found = self._first_localized_frame()
        if found is None:
            return None
        _idx, _ts, T_world_device = found
        T_world_cams: dict[int, np.ndarray] = {
            idx: T_world_device @ cam.T_device_camera
            for idx, cam in self._cam_models.items()
            if cam is not None
        }
        return T_world_device, T_world_cams

    def seed_initial_view(self) -> bool:
        """Render the first localized frameset at startup, before the run gate."""
        if self._visualizer is None:
            return False
        found = self._first_localized_frame()
        if found is None:
            return False
        idx, _ts, T_world_device = found
        fs = self._vrs.frameset_at_index(idx)
        if fs is None:
            return False

        # Same `cam_index_subset` image filter as `_process_frameset`.
        images = (
            {i: img for i, img in fs.images.items() if i in self._cam_models}
            if self._settings.cam_index_subset is not None
            else fs.images
        )
        # `T_world_cams` + `intrinsics` over every active, non-None cam (incl.
        # the RGB cam) — `set_camera_frustums` applies its own frustum allow-set.
        T_world_cams: dict[int, np.ndarray] = {}
        intrinsics: dict[int, np.ndarray] = {}
        for cam_idx, cam in self._cam_models.items():
            if cam is None:
                continue
            T_world_cams[cam_idx] = T_world_device @ cam.T_device_camera
            intrinsics[cam_idx] = cam.intrinsic_matrix
        # `image_sizes` from the decoded images, with `update()`'s 640x640
        # fallback for cams that have a pose but no image (e.g. the RGB cam).
        image_sizes: dict[int, tuple[int, int]] = {
            cam_idx: (int(img.shape[1]), int(img.shape[0]))
            for cam_idx, img in images.items()
            if img.ndim >= 2
        }
        for cam_idx in T_world_cams:
            image_sizes.setdefault(cam_idx, (640, 640))

        self._visualizer.set_initial_view(T_world_device, T_world_cams)
        self._visualizer.set_camera_frustums(T_world_cams, intrinsics, image_sizes)
        self._visualizer.set_image_grid(images)
        return True

    # Floor plane (pre-run setup)

    def set_floor_z(self, z: float | None) -> None:
        """Set (or clear) the fixed floor-plane height fed to the lifter."""
        self._lifter.set_floor_plane(z)

    def run(self, max_framesets: int | None = None) -> LampResult:
        """Process all framesets (or up to `max_framesets`) and return the result."""
        last_ts: int = -1

        min_frame_gap_ns = (
            int(1e9 / (self._settings.fps + 0.1))
            if self._settings.fps is not None
            else 0
        )
        # Skip the opening `start_ts_s` seconds — dropped before decode by the
        # streamer, composes with the fps throttle + max_framesets.
        start_offset_ns = max(0, int(self._settings.start_ts_s * 1e9))
        pacer = _LivePacer(time.perf_counter())
        try:
            with torch.no_grad():
                for fs in self._vrs.iter_framesets(
                    max_framesets=max_framesets,
                    min_frame_gap_ns=min_frame_gap_ns,
                    start_offset_ns=start_offset_ns,
                ):
                    if self.frame_gate is not None:
                        self.frame_gate()
                    sleep_s = pacer.wait(self._live_interval(), time.perf_counter())
                    if sleep_s > 0.0:
                        t0 = time.perf_counter()
                        time.sleep(sleep_s)
                        # Exclude the actual pacing wait (incl. sleep overshoot)
                        # from the gap_ms stat.
                        if self._last_proc_end_perf is not None:
                            self._last_proc_end_perf += time.perf_counter() - t0
                    self._process_frameset(fs)
                    last_ts = fs.timestamp_ns
        finally:
            # Visualizer + persistence both live in `finally` so a
            # KeyboardInterrupt mid-loop still flushes whatever we have.
            self._maybe_persist()

        return LampResult(timestamp_ns=last_ts, people=self._tracker.people)

    # Internal helpers

    def _live_interval(self) -> float:
        """Target gap before the next live frame, per the viewer's pacing toggle.

        Returns 0 (no pacing) without a visualizer or when auto-play is off — the
        start/step gate handles stepping. Otherwise Fixed FPS caps at the --fps
        target; with it off, 0 (run as fast as compute allows).
        """
        viz = self._visualizer
        if viz is None or not bool(viz.gui_handles["auto_play"].value):
            return 0.0
        return _pacing_interval(
            bool(viz.gui_handles["fixed_fps"].value), self._settings.fps
        )

    def _process_frameset(self, fs: Frameset) -> None:
        """Run one frame: MPS-pose -> detect -> track -> lift -> merge -> visualize."""
        ts = fs.timestamp_ns
        proc_start_perf = time.perf_counter()
        stats = LampPipelineStats()
        self._last_run_stats = stats
        if self._last_proc_end_perf is not None:
            stats.gap_ms = (proc_start_perf - self._last_proc_end_perf) * 1000.0

        images = (
            {idx: img for idx, img in fs.images.items() if idx in self._cam_models}
            if self._settings.cam_index_subset is not None
            else fs.images
        )

        # 1. Resolve T_world_device before detection so unlocalized frames do not
        # spend detector time.
        T_world_device = self._mps.query_T_world_device(ts)
        if T_world_device is None:
            # No `intrinsics=` / `detections=` — empty `T_world_cams` means no
            # frustums are drawn (so identity-K fallback wouldn't matter), and
            # the detector hasn't run yet so there are no overlays to push.
            if self._visualizer is not None:
                t0 = time.perf_counter()
                self._visualizer.update(ts, {}, images, self._tracker.people)
                stats.viz_ms = (time.perf_counter() - t0) * 1000.0
            proc_end_perf = time.perf_counter()
            stats.proc_total_ms = (proc_end_perf - proc_start_perf) * 1000.0
            self._last_proc_end_perf = proc_end_perf
            if self._visualizer is not None:
                self._visualizer.set_runtime_stats(stats)
            return

        # 2. Per-cam world transforms (T_world_cam = T_world_device @ T_device_cam).
        T_world_cams: dict[int, np.ndarray] = {}
        for cam_idx, cam in self._cam_models.items():
            if cam is None:
                continue
            T_world_cams[cam_idx] = T_world_device @ cam.T_device_camera

        # 3. Detect 2D in one batched call over detector-enabled cameras.
        detector_images = {
            idx: img for idx, img in images.items() if idx in self._detector_cam_indices
        }
        t0 = time.perf_counter()
        detections = self._detector.detect(
            detector_images, self._vrs.camera_orientations, ts
        )
        stats.detect_ms = (time.perf_counter() - t0) * 1000.0
        sub = self._detector.last_detect_stats
        stats.bbox_ms = sub.bbox_ms
        stats.kp_ms = sub.kp_ms
        stats.detect_post_ms = sub.post_ms

        # Persist only timestamps that reached inference.
        self._all_timestamps.append(int(ts))

        t0 = time.perf_counter()
        self._tracker.track_frameset(detections, T_world_cams, self._cam_models, ts)
        stats.track_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Lift eligible tracks and fuse every snippet timestep. Snapshot
        # `last_lifted_ts` before lifting so the merge step can distinguish a
        # first lift from a re-lifted established track.
        prev_last_lifted_ts = {
            pid: p.last_lifted_ts for pid, p in self._tracker.people.items()
        }
        self._run_lifter(T_world_cams, stats)

        # 6. Post-lift spatial merge. Read the threshold from settings so
        # injected test lifters can still override behavior.
        stats.n_merges = self._tracker.merge_lifted_tracks(
            threshold_m=self._settings.lifter.merge_threshold_m,
            current_ts_ns=ts,
            prev_last_lifted_ts=prev_last_lifted_ts,
        )

        # 7. Visualizer (optional). Pass the FULL image dict (including the
        # RGB cam) so the per-cam image grid still renders the RGB feed even
        # though we skipped it from the detector.
        if self._visualizer is not None:
            intrinsics: dict[int, np.ndarray] = {}
            for idx in T_world_cams:
                cam = self._cam_models.get(idx)
                if cam is not None:
                    intrinsics[idx] = cam.intrinsic_matrix
            t0 = time.perf_counter()
            self._visualizer.update(
                ts,
                T_world_cams,
                images,
                self._tracker.people,
                detections=detections,
                intrinsics=intrinsics,
                cam_models=self._cam_models,
                T_world_device=T_world_device,
            )
            stats.viz_ms = (time.perf_counter() - t0) * 1000.0

        proc_end_perf = time.perf_counter()
        stats.proc_total_ms = (proc_end_perf - proc_start_perf) * 1000.0
        self._last_proc_end_perf = proc_end_perf
        if self._visualizer is not None:
            self._visualizer.set_runtime_stats(stats)

    def _run_lifter(
        self,
        T_world_cams: dict[int, np.ndarray],
        stats: LampPipelineStats,
    ) -> None:
        """Build snippets, run the model, fuse outputs, and attach skeletons."""
        ls = self._settings.lifter
        snippets = self._tracker.get_snippets_for_lifting(
            snippet_length=self._lifter.snippet_length,
            T_gravityWorld_world=self._mps.T_gravityWorld_world,
            kp_thres=ls.kp_thres_for_binary,
            num_views=self._lifter.expected_num_views,
        )
        if not snippets:
            return

        t0 = time.perf_counter()
        batched = self._lifter.lift_all_steps_batched(snippets)
        stats.lift_ms = (time.perf_counter() - t0) * 1000.0
        for person_id, skeletons_with_ts in batched.items():
            latest_ts_in_snippet = snippets[person_id].snippet_timestamps_ns[-1]
            t1 = time.perf_counter()
            self._tracker.attach_skeletons(
                person_id,
                skeletons_with_ts,
                T_world_cams=T_world_cams,
                T_gravityWorld_world=self._mps.T_gravityWorld_world,
                min_pose_depth=ls.min_pose_depth,
                max_pose_depth=ls.max_pose_depth,
            )
            stats.smoothing_ms += (time.perf_counter() - t1) * 1000.0
            if self._tracker.people[person_id].last_lifted_ts == latest_ts_in_snippet:
                stats.n_lifts += 1

    def _maybe_persist(self) -> None:
        """Write JSON results if a save path is configured."""
        if self._settings.save_path is None:
            return
        save_path = Path(self._settings.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Store the input VRS path string in the result metadata.
        vrs_uri = str(self._vrs_path_for_uri())
        cam_index_to_name = {
            idx: cam.label for idx, cam in self._cam_models.items() if cam is not None
        }
        result = LampResult(
            timestamp_ns=self._all_timestamps[-1] if self._all_timestamps else -1,
            people=self._tracker.people,
        )
        t0 = time.perf_counter()
        save_results(
            save_path,
            result,
            cam_index_to_name=cam_index_to_name,
            vrs_uri=vrs_uri,
            all_timestamps=list(self._all_timestamps),
        )
        logger.info(
            "Saved results to %s (%d people, %d timestamps) in %.2fs",
            save_path,
            len(result.people),
            len(self._all_timestamps),
            time.perf_counter() - t0,
        )

    def _vrs_path_for_uri(self) -> str:
        """The on-disk VRS path, used as the JSON `vrs` key."""
        return str(self._vrs.path)
