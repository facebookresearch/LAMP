# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process-parallel VRS image decoding."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from lamp.core.types import CameraOrientation, Frameset
from lamp.io.sensor_io import (
    _to_uint8_hwc3,
    CameraCalibration,
    VrsFrameIndex,
    VrsLoader,
)

logger: logging.Logger = logging.getLogger(__name__)

type _DecodeResult = tuple[int, int, int, np.ndarray | None, str | None]


def _decode_worker_main(
    vrs_path: str,
    cam_idx: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        loader = VrsLoader(Path(vrs_path), decode_camera_indices=[cam_idx])
        while True:
            frame_idx = task_queue.get()
            if frame_idx is None:
                return
            ts_ns, image = loader.decode_camera_image(int(frame_idx), cam_idx)
            result_queue.put((int(frame_idx), cam_idx, ts_ns, image, None))
    except BaseException as exc:  # pragma: no cover - exercised by integration failures
        result_queue.put((-1, cam_idx, 0, None, f"{type(exc).__name__}: {exc}"))


class _DecodeWorker:
    def __init__(
        self,
        *,
        ctx: Any,
        vrs_path: Path,
        cam_idx: int,
        result_queue: Any,
        queue_size: int,
    ) -> None:
        self.cam_idx = cam_idx
        self._tasks = ctx.Queue(maxsize=queue_size)
        self._process = ctx.Process(
            target=_decode_worker_main,
            args=(str(vrs_path), cam_idx, self._tasks, result_queue),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def submit(self, frame_idx: int) -> None:
        self._tasks.put(frame_idx)

    def request_stop(self) -> None:
        with suppress(queue.Full):
            self._tasks.put_nowait(None)

    def close(self, *, abort: bool) -> None:
        if not abort:
            self.request_stop()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)


class _ParallelDecodeSession:
    def __init__(
        self,
        *,
        vrs_path: Path,
        camera_indices: Sequence[int],
        prefetch_frames: int,
    ) -> None:
        self._vrs_path = vrs_path
        self._camera_indices = list(camera_indices)
        self._ctx = mp.get_context("spawn")
        queue_size = max(1, prefetch_frames)
        self._results = self._ctx.Queue(
            maxsize=max(1, len(self._camera_indices) * (prefetch_frames + 1))
        )
        self._workers = [
            _DecodeWorker(
                ctx=self._ctx,
                vrs_path=self._vrs_path,
                cam_idx=cam_idx,
                result_queue=self._results,
                queue_size=queue_size,
            )
            for cam_idx in self._camera_indices
        ]
        self._pending: dict[int, dict[int, tuple[int, np.ndarray]]] = {}

    def __enter__(self) -> _ParallelDecodeSession:
        for worker in self._workers:
            worker.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(abort=exc_type is not None)

    def submit(self, frame_idx: int) -> None:
        for worker in self._workers:
            worker.submit(frame_idx)

    def collect(self, frame_idx: int) -> dict[int, tuple[int, np.ndarray]]:
        ready = self._pending.pop(frame_idx, {})
        while len(ready) < len(self._camera_indices):
            try:
                result = self._results.get(timeout=30.0)
            except queue.Empty as exc:
                raise RuntimeError(
                    f"timed out waiting for VRS decode workers at frame {frame_idx}"
                ) from exc
            got_idx, cam_idx, ts_ns, image, error = self._parse_result(result)
            if error is not None:
                raise RuntimeError(f"VRS decode worker {cam_idx} failed: {error}")
            if image is None:
                raise RuntimeError(f"VRS decode worker {cam_idx} returned no image")

            target = (
                ready if got_idx == frame_idx else self._pending.setdefault(got_idx, {})
            )
            target[cam_idx] = (ts_ns, image)
        return ready

    def close(self, *, abort: bool = False) -> None:
        for worker in self._workers:
            worker.close(abort=abort)

    @staticmethod
    def _parse_result(result: Any) -> _DecodeResult:
        frame_idx, cam_idx, ts_ns, image, error = result
        return int(frame_idx), int(cam_idx), int(ts_ns), image, error


class ParallelVrsLoader:
    """`VrsLoader` wrapper that decodes each SLAM camera in its own process."""

    def __init__(
        self,
        vrs_path: Path,
        *,
        base_loader: VrsLoader | None = None,
        decode_camera_indices: Sequence[int] | None = None,
        max_workers: int = 4,
        prefetch_frames: int = 4,
    ) -> None:
        self._base = base_loader or VrsLoader(
            vrs_path, decode_camera_indices=decode_camera_indices
        )
        self._max_workers = max(1, int(max_workers))
        self._prefetch_frames = max(1, int(prefetch_frames))

    @property
    def path(self) -> Path:
        return self._base.path

    @property
    def device_family(self) -> str:
        return self._base.device_family

    @property
    def camera_indices(self) -> list[int]:
        return self._base.camera_indices

    @property
    def rgb_camera_indices(self) -> list[int]:
        return self._base.rgb_camera_indices

    @property
    def slam_camera_indices(self) -> list[int]:
        return self._base.slam_camera_indices

    @property
    def num_framesets(self) -> int:
        return self._base.num_framesets

    @property
    def calibration(self) -> CameraCalibration:
        return self._base.calibration

    @property
    def camera_orientations(self) -> dict[int, CameraOrientation]:
        return self._base.camera_orientations

    @property
    def labels(self) -> dict[int, str]:
        return self._base.labels

    def frameset_timestamps_ns(self) -> list[int]:
        return self._base.frameset_timestamps_ns()

    def iter_frame_indices(
        self,
        max_framesets: int | None = None,
        min_frame_gap_ns: int = 0,
        start_offset_ns: int = 0,
    ) -> Iterator[VrsFrameIndex]:
        yield from self._base.iter_frame_indices(
            max_framesets=max_framesets,
            min_frame_gap_ns=min_frame_gap_ns,
            start_offset_ns=start_offset_ns,
        )

    def iter_framesets(
        self,
        max_framesets: int | None = None,
        min_frame_gap_ns: int = 0,
        start_offset_ns: int = 0,
    ) -> Iterator[Frameset]:
        slam_indices = self._base.slam_camera_indices
        if len(slam_indices) <= 1 or self._max_workers <= 1:
            yield from self._base.iter_framesets(
                max_framesets=max_framesets,
                min_frame_gap_ns=min_frame_gap_ns,
                start_offset_ns=start_offset_ns,
            )
            return

        if self._max_workers < len(slam_indices):
            logger.warning(
                "VRS parallel decode needs one worker per SLAM camera; "
                "got %d worker(s) for %d cameras, falling back to sequential decode.",
                self._max_workers,
                len(slam_indices),
            )
            yield from self._base.iter_framesets(
                max_framesets=max_framesets,
                min_frame_gap_ns=min_frame_gap_ns,
                start_offset_ns=start_offset_ns,
            )
            return

        frames = list(
            self._base.iter_frame_indices(
                max_framesets=max_framesets,
                min_frame_gap_ns=min_frame_gap_ns,
                start_offset_ns=start_offset_ns,
            )
        )
        if not frames:
            return

        in_flight: deque[VrsFrameIndex] = deque()
        next_submit = 0

        def submit_until_full(session: _ParallelDecodeSession) -> None:
            nonlocal next_submit
            while next_submit < len(frames) and len(in_flight) < self._prefetch_frames:
                frame = frames[next_submit]
                session.submit(frame.frame_idx)
                in_flight.append(frame)
                next_submit += 1

        with _ParallelDecodeSession(
            vrs_path=self._base.path,
            camera_indices=slam_indices,
            prefetch_frames=self._prefetch_frames,
        ) as session:
            submit_until_full(session)
            while in_flight:
                frame = in_flight.popleft()
                decoded = session.collect(frame.frame_idx)
                submit_until_full(session)
                yield self._form_frameset(slam_indices, frame, decoded)

    def frameset_at_index(self, idx: int) -> Frameset | None:
        return self._base.frameset_at_index(idx)

    @staticmethod
    def _form_frameset(
        slam_indices: Sequence[int],
        frame: VrsFrameIndex,
        decoded: dict[int, tuple[int, np.ndarray]],
    ) -> Frameset:
        images: dict[int, np.ndarray] = {}
        canonical_ts: int | None = None
        for cam_idx in slam_indices:
            ts_ns, raw = decoded[cam_idx]
            if canonical_ts is None:
                canonical_ts = ts_ns
            images[cam_idx] = _to_uint8_hwc3(raw)

        assert canonical_ts is not None
        if canonical_ts != frame.timestamp_ns:
            raise RuntimeError(
                "VRS worker timestamp diverged from anchor timestamp "
                f"at frame {frame.frame_idx}: {canonical_ts} != {frame.timestamp_ns}"
            )
        return Frameset(timestamp_ns=canonical_ts, images=images)
