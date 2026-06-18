# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line entry point for running the LAMP pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from lamp.app.cli_parser import _build_parser
from lamp.app.pipeline import LampPipeline, LampPipelineSettings
from lamp.detection.detector import PeopleDetector2dSettings
from lamp.io.sensor_io import resolve_recording_paths
from lamp.models.lifter import LifterSettings
from lamp.tracking.tracker import LampTrackerSettings

logger: logging.Logger = logging.getLogger(__name__)

# Exit codes — kept as constants so calling code (and tests) can refer to them.
_EXIT_OK = 0
_EXIT_USAGE = 2
_EXIT_RUNTIME = 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `lamp` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=getattr(args, "verbose", False))

    if args.subcommand == "run":
        return _cmd_run(args)
    # argparse already prints a usage error for missing subcommands, but the
    # caller can also reach this path if no subcommand was given on Python
    # versions where `required=True` falls through; mirror argparse exit code.
    parser.print_help(sys.stderr)
    return _EXIT_USAGE


# Subcommand handlers


def _cmd_run(args: argparse.Namespace) -> int:
    """Implements `lamp run`."""
    try:
        recording_paths = resolve_recording_paths(args.recording)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return _EXIT_RUNTIME
    if not args.checkpoint.exists():
        logger.error("--checkpoint path does not exist: %s", args.checkpoint)
        return _EXIT_RUNTIME
    if not args.smpl_model_path.exists():
        logger.error("--smpl-model-path does not exist: %s", args.smpl_model_path)
        return _EXIT_RUNTIME

    save_path = _resolve_save_path(args, recording_paths.vrs)
    # `--fps 0` (or any value <= 0) disables the throttle (settings.fps=None);
    # otherwise the value (default 10.0) caps the VRS stream rate. The argparse
    # default is 10.0 so omitting the flag inherits the 10 Hz throttle.
    fps_setting = args.fps if args.fps is not None and args.fps > 0 else None
    settings = LampPipelineSettings(
        detector=PeopleDetector2dSettings(
            bbox_backend=args.bbox_backend,
            kp_hf_model_id=args.kp_hf_model_id,
            min_box_conf=args.min_box_conf,
            min_kp_conf=args.min_kp_conf,
            min_reliable_kp_num=args.min_reliable_kp_num,
            min_box_size_ratio=args.min_box_size_ratio,
        ),
        tracker=LampTrackerSettings(
            min_track_frame_ratio=args.min_track_frame_ratio,
        ),
        lifter=LifterSettings(snippet_length=args.snippet_length),
        smpl_model_path=args.smpl_model_path,
        cuda_graphs=args.cuda_graphs,
        capture_batch_size=args.capture_batch_size,
        device=args.device,
        enable_visualizer=not args.no_viser,
        visualizer_port=args.port,
        save_path=save_path,
        cam_index_subset=args.cameras,
        vrs_decode_workers=args.vrs_decode_workers,
        fps=fps_setting,
        start_ts_s=args.start_ts,
    )

    logger.info("Building pipeline (device=%s)...", settings.device)
    pipeline = LampPipeline.from_paths(
        recording=args.recording,
        lamp_ckpt=args.checkpoint,
        device=settings.device,
        settings=settings,
    )

    if pipeline.visualizer is not None:
        pipeline.seed_initial_view()

    if pipeline.visualizer is not None and not args.auto_start:
        auto_play = pipeline.visualizer.gui_handles["auto_play"]
        next_handle = pipeline.visualizer.gui_handles["next"]
        logger.info(
            "Pipeline ready. Open http://localhost:%d and toggle `Auto play` "
            "in the Playback folder to start inference (untoggle to pause "
            "mid-stream, or pass --auto-start to skip this gate).",
            pipeline.visualizer.port,
        )

        def _wait_for_auto_play() -> None:
            while not (auto_play.value or next_handle.value):
                time.sleep(0.05)
            if next_handle.value:
                next_handle.value = False

        # Open the start gate without consuming a queued "Next" click, so the
        # first per-frame `frame_gate` still advances frame 1 on a single click.
        try:
            while not (auto_play.value or next_handle.value):
                time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("Interrupted before start; exiting.")
            return _EXIT_OK
        pipeline.frame_gate = _wait_for_auto_play

    if pipeline.visualizer is not None:
        floor_z = pipeline.visualizer.finalize_floor()
        pipeline.set_floor_z(floor_z)
        if floor_z is not None:
            logger.info("Using selected floor plane at z=%.3f m.", floor_z)

    logger.info("Running pipeline (max_framesets=%s)...", args.max_framesets)
    pipeline.run(max_framesets=args.max_framesets)
    return _EXIT_OK


# Helpers


def _configure_logging(*, verbose: bool) -> None:
    """Set the root logger to INFO (or DEBUG when `-v`)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_save_path(args: argparse.Namespace, vrs: Path) -> Path | None:
    """Compute the JSON results path."""
    if args.no_save:
        return None
    if args.out is not None:
        return args.out
    return vrs.parent / f"{vrs.stem}_lamp.json"


if __name__ == "__main__":
    raise SystemExit(main())
