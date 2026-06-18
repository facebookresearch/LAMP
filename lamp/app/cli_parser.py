# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Argument parser construction for the LAMP CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from lamp.app.pipeline import LampPipelineSettings
from lamp.detection.detector import PeopleDetector2dSettings
from lamp.models.lifter import LifterSettings
from lamp.tracking.tracker import LampTrackerSettings


def _path_arg(s: str) -> Path:
    """argparse `type=` for path flags; expands `~`."""
    return Path(s).expanduser()


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level `lamp` parser with the `run` subcommand."""
    parser = argparse.ArgumentParser(
        prog="lamp",
        description=(
            "LAMP - Multi-view 3D human pose estimation for egocentric AR/VR "
            "recordings. The `run` subcommand executes the full pipeline on a "
            "recording folder."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run the full LAMP pipeline on a recording folder.",
        description=(
            "Run the full LAMP pipeline. Loads a flat folder containing "
            "video.vrs, closed_loop_trajectory.csv, online_calibration.jsonl, "
            "and semidense_points.csv.gz, then loads the LAMP checkpoint and 2D "
            "detector backend, then streams framesets through detector -> tracker -> 3D lifter and "
            "writes a JSON result."
        ),
    )

    # I/O paths
    io = run.add_argument_group("input / output")
    io.add_argument(
        "--recording",
        type=_path_arg,
        required=True,
        help=(
            "Flat recording folder containing video.vrs, closed_loop_trajectory.csv, "
            "online_calibration.jsonl, and semidense_points.csv.gz."
        ),
    )
    io.add_argument(
        "--checkpoint",
        type=_path_arg,
        required=True,
        help=("Path to the LAMPNet checkpoint (.pt state_dict)."),
    )
    io.add_argument(
        "--out",
        type=_path_arg,
        default=None,
        help=(
            "Path to the JSON results file. "
            "Defaults to `<video_stem>_lamp.json` next to video.vrs."
        ),
    )
    io.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the JSON result.",
    )

    # Runtime / device
    runtime = run.add_argument_group("runtime")
    runtime.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=(
            "Torch device for model inference (`cpu`, `cuda:0`, `cuda:1`, ...). "
            "Falls back to CPU via `Lifter.resolve_device` if CUDA isn't "
            "available. Multi-GPU hosts can use any `cuda:N` device."
        ),
    )
    runtime.add_argument(
        "--max-framesets",
        type=int,
        default=None,
        help="Process at most this many framesets (None = all).",
    )
    runtime.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="Subset of camera indices to drive the pipeline with (default: all).",
    )
    runtime.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help=(
            "Throttle the VRS stream to ~FPS framesets per second of capture "
            "time (default 10, the model's native rate). Frames whose capture "
            "timestamp is less than 1/(FPS+0.1)s after the previous YIELDED "
            "frame are skipped BEFORE H.265 decode — so they cost nothing and "
            "never reach detect/track/lift. The +0.1 fudge keeps nominal 10 Hz "
            "recordings (gaps ~99.999 ms) from being dropped wholesale. Pass "
            "`--fps 0` (or any value <= 0) to disable the throttle and process "
            "every frame."
        ),
    )
    runtime.add_argument(
        "--start-ts",
        type=float,
        default=0.0,
        help=(
            "Skip the first START_TS seconds of the recording (measured from "
            "the first frameset). Frames before the offset are dropped BEFORE "
            "H.265 decode, so they cost nothing. Default 0.0 (start at the "
            "beginning). Useful to skip a bad / irrelevant lead-in."
        ),
    )
    runtime.add_argument(
        "--vrs-decode-workers",
        type=int,
        default=4,
        help=(
            "Spawned worker processes for SLAM camera decode (default 4). "
            "Pass 1 to use sequential Project Aria decode."
        ),
    )

    # Visualizer
    viz = run.add_argument_group("visualizer")
    viz.add_argument(
        "--no-viser",
        action="store_true",
        help="Disable the Viser visualizer.",
    )
    viz.add_argument("--port", type=int, default=8080, help="Viser server port.")
    viz.add_argument(
        "--auto-start",
        action="store_true",
        help=(
            "Skip the UI gate and start the inference loop immediately. "
            "Default behavior (when the visualizer is enabled) waits until "
            "the user toggles `Auto play` in the Playback folder so they have "
            "time to open the browser tab before frames start streaming. "
            "Implied when `--no-viser` is set (nothing to gate on)."
        ),
    )

    # Pull defaults
    _pipe_defaults = LampPipelineSettings()
    _det_defaults = PeopleDetector2dSettings()
    _trk_defaults = LampTrackerSettings()
    _lift_defaults = LifterSettings()

    # Detector knobs
    det = run.add_argument_group("2D detection")
    det.add_argument(
        "--bbox-backend",
        choices=("rfdetr", "rfdetr-trt"),
        default=_det_defaults.bbox_backend,
        help=(
            "BBox stage. `rfdetr-trt` (default) is ONNX Runtime + TensorRT in "
            "FP32 -- correct scores AND fast (~4.3 ms forward); first run builds "
            "a TRT engine (~1-3 min, cached), falls back to ORT CUDA EP if TRT "
            "is unavailable. `rfdetr` is the eager BF16 torch path -- same "
            "correct scores, slower forward, no engine build / no TRT needed. "
            "RF-DETR weights auto-download to the model cache."
        ),
    )
    det.add_argument(
        "--kp-hf-model-id",
        type=str,
        default=_det_defaults.kp_hf_model_id,
        help="HuggingFace model id for the ViTPose keypoint detector.",
    )
    det.add_argument(
        "--min-box-conf",
        type=float,
        default=_det_defaults.min_box_conf,
        help="Minimum person detection confidence.",
    )
    det.add_argument(
        "--min-kp-conf",
        type=float,
        default=_det_defaults.min_kp_conf,
        help="Minimum per-keypoint confidence.",
    )
    det.add_argument(
        "--min-reliable-kp-num",
        type=int,
        default=_det_defaults.min_reliable_kp_num,
        help="Minimum number of keypoints with confidence above `--min-kp-conf`.",
    )
    det.add_argument(
        "--min-box-size-ratio",
        type=float,
        default=_det_defaults.min_box_size_ratio,
        help="Minimum bbox size as a ratio of `min(image_side)`.",
    )

    # Tracker knobs
    trk = run.add_argument_group("tracker")
    trk.add_argument(
        "--min-track-frame-ratio",
        type=float,
        default=_trk_defaults.min_track_frame_ratio,
        help="Minimum frame-coverage ratio before a track is eligible for 3D lifting.",
    )

    # Lifter knobs
    lift = run.add_argument_group("3D lifter")
    lift.add_argument(
        "--snippet-length",
        type=int,
        default=_lift_defaults.snippet_length,
        help="Number of frames in the per-track temporal snippet fed to the lifter.",
    )
    lift.add_argument(
        "--smpl-model-path",
        type=_path_arg,
        required=True,
        help=(
            "Path to the SMPL neutral .pkl. Download from https://smpl.is.tue.mpg.de."
        ),
    )
    lift.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=_pipe_defaults.cuda_graphs,
        help=(
            "Wrap the lifter forward in a torch.cuda.CUDAGraph capture on CUDA "
            "(default enabled). Pass --no-cuda-graphs to disable."
        ),
    )
    lift.add_argument(
        "--capture-batch-size",
        type=int,
        default=_pipe_defaults.capture_batch_size,
        help=(
            "Static batch size for the CUDA-Graph wrapper (default 8). Actual "
            "B<=this is padded into the captured shape; B>this falls back to "
            "eager. Only honored with CUDA Graphs enabled."
        ),
    )

    # Logging
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level).",
    )

    return parser
