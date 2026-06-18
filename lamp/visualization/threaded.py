# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Threaded wrapper for the Viser visualizer."""

from __future__ import annotations

import logging
import threading
from typing import Any, cast, Protocol

import numpy as np


class VisualizerLike(Protocol):
    @property
    def host(self) -> str: ...

    @property
    def port(self) -> int: ...

    @property
    def gui_handles(self) -> dict[str, Any]: ...

    def set_camera_frustums(self, *args: Any, **kwargs: Any) -> None: ...
    def set_image_grid(self, *args: Any, **kwargs: Any) -> None: ...
    def set_2d_overlays(self, *args: Any, **kwargs: Any) -> dict[int, np.ndarray]: ...
    def set_runtime_stats(self, *args: Any, **kwargs: Any) -> None: ...
    def set_semidense_points(self, *args: Any, **kwargs: Any) -> None: ...
    def finalize_floor(self) -> float | None: ...
    def set_initial_view(self, *args: Any, **kwargs: Any) -> None: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...
    def shutdown(self) -> None: ...


logger: logging.Logger = logging.getLogger(__name__)


class ThreadedVisualizer:
    """Off-thread wrapper around `Visualizer.update(...)`."""

    _SHUTDOWN: object = object()

    def __init__(
        self,
        inner: VisualizerLike,
        *,
        join_timeout_s: float = 5.0,
    ) -> None:
        self._inner: VisualizerLike = inner
        self._join_timeout_s: float = join_timeout_s
        self._pending: tuple[tuple[Any, ...], dict[str, Any]] | object | None = None
        self._lock: threading.Lock = threading.Lock()
        self._wakeup: threading.Event = threading.Event()
        self._stopped: bool = False
        self._dropped_count: int = 0
        self._worker: threading.Thread = threading.Thread(
            target=self._run_worker,
            name="ThreadedVisualizer-worker",
            daemon=True,
        )
        self._worker.start()

    @property
    def host(self) -> str:
        return self._inner.host

    @property
    def port(self) -> int:
        return self._inner.port

    @property
    def gui_handles(self) -> dict[str, Any]:
        """Direct passthrough — safe to read AND write `.value` from any thread."""
        return self._inner.gui_handles

    @property
    def inner(self) -> VisualizerLike:
        """Underlying `Visualizer`. Exposed for tests + advanced callers."""
        return self._inner

    @property
    def dropped_count(self) -> int:
        """Number of `update(...)` calls dropped because the slot was occupied."""
        with self._lock:
            return self._dropped_count

    def set_camera_frustums(self, *args: Any, **kwargs: Any) -> None:
        """Inline proxy to `Visualizer.set_camera_frustums`."""
        self._inner.set_camera_frustums(*args, **kwargs)

    def set_image_grid(self, *args: Any, **kwargs: Any) -> None:
        """Inline proxy to `Visualizer.set_image_grid`. See `set_camera_frustums`."""
        self._inner.set_image_grid(*args, **kwargs)

    def set_2d_overlays(self, *args: Any, **kwargs: Any) -> dict[int, np.ndarray]:
        """Inline proxy to `Visualizer.set_2d_overlays`. See `set_camera_frustums`."""
        return self._inner.set_2d_overlays(*args, **kwargs)

    def set_runtime_stats(self, *args: Any, **kwargs: Any) -> None:
        """Inline proxy to `Visualizer.set_runtime_stats`. See `set_camera_frustums`."""
        self._inner.set_runtime_stats(*args, **kwargs)

    def set_semidense_points(self, *args: Any, **kwargs: Any) -> None:
        """Inline proxy to `Visualizer.set_semidense_points`. One-shot at startup."""
        self._inner.set_semidense_points(*args, **kwargs)

    def finalize_floor(self) -> float | None:
        """Inline proxy to `Visualizer.finalize_floor`. Called once at run start."""
        return self._inner.finalize_floor()

    def set_initial_view(self, *args: Any, **kwargs: Any) -> None:
        """Inline proxy to `Visualizer.set_initial_view`. Called once at startup."""
        self._inner.set_initial_view(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Enqueue an `update` call and return immediately."""
        with self._lock:
            if self._stopped:
                return
        cache_update_frame = getattr(self._inner, "cache_update_frame", None)
        if cache_update_frame is not None:
            cache_update_frame(*args, **kwargs)
        with self._lock:
            if self._stopped:
                # Silently drop; shutdown can race with the final update.
                return
            had_pending = self._pending is not None
            self._pending = (args, kwargs)
            if had_pending:
                self._dropped_count += 1
        # `set()` is thread-safe and idempotent — the worker may consume
        # multiple `set` calls' worth of work in one wakeup if it was busy.
        self._wakeup.set()

    def _run_worker(self) -> None:
        """Daemon loop: wait for args, call `inner.update`, repeat."""
        while True:
            self._wakeup.wait()
            with self._lock:
                payload = self._pending
                self._pending = None

                self._wakeup.clear()
            if payload is None:
                # Spurious wakeup (e.g. `shutdown` cleared the slot before we
                # got here). Loop and wait again.
                continue
            if payload is self._SHUTDOWN:
                return
            payload_t = cast(tuple[tuple[Any, ...], dict[str, Any]], payload)
            args, kwargs = payload_t
            try:
                self._inner.update(*args, **kwargs)
            except Exception:
                ts_hint: object = args[0] if args else "<no-args>"
                logger.exception(
                    "ThreadedVisualizer.update failed at ts=%s; continuing.",
                    ts_hint,
                )

    def shutdown(self) -> None:
        """Stop the worker, drain pending work, and shut down the inner viz."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            # Replace any pending args with the sentinel — the worker will
            # see `_SHUTDOWN` after its current work finishes (or immediately
            # if it was idle) and return.
            self._pending = self._SHUTDOWN
        self._wakeup.set()
        self._worker.join(timeout=self._join_timeout_s)
        if self._worker.is_alive():
            logger.warning(
                "ThreadedVisualizer worker did not exit within %.1fs; "
                "shutting down inner anyway.",
                self._join_timeout_s,
            )
        self._inner.shutdown()
