# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Eager Torch RF-DETR person-box backend."""

# pyright: reportPrivateImportUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from lamp.core.types import CameraOrientation
from lamp.detection.geometry import (
    _boxes_oriented_to_original,
    _ensure_hwc3_uint8,
    _orient_image_for_detector,
    _resolve_detector_device,
)
from lamp.detection.rfdetr_common import (
    _resolve_rfdetr_weights,
    _RFDETR_PERSON_CLASS_ID,
)
from lamp.detection.types import BBoxDetector

logger: logging.Logger = logging.getLogger(__name__)


class _RfDetrBboxDetector(BBoxDetector):
    """RF-DETR-Nano backend (open-source bbox default)."""

    # ImageNet normalization — baked into the rfdetr training recipe
    # (rfdetr/datasets/coco.py); we have to apply it manually when we
    # bypass `predict()`.
    _IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str = "cuda:0",
        score_thresh: float = 0.85,
        min_box_size_ratio: float = 0.0,
        optimize_for_inference: bool = False,
        optimize_batch_size: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # None -> auto-download + cache the stock RF-DETR-Nano checkpoint.
        weights_path = _resolve_rfdetr_weights(weights_path)
        self._device: str = _resolve_detector_device(device)
        self._score_thresh: float = float(score_thresh)
        self._min_box_size_ratio: float = float(min_box_size_ratio)
        self._optimize_batch_size: int = int(optimize_batch_size)
        self._requested_dtype: torch.dtype = dtype
        try:
            from rfdetr import RFDETRNano  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR backend requires the `rfdetr` package. Install it with "
                "`uv pip install rfdetr` (then drop the full opencv "
                "via `uv pip uninstall opencv-python` and reinstate headless via "
                "`uv pip install --force-reinstall "
                "opencv-python-headless==4.13.0.92`)."
            ) from exc
        # `pretrain_weights=<path>` loads from the local file — here the
        # resolved/cached checkpoint (`_resolve_rfdetr_weights` above already
        # auto-downloaded it to the cache if no path was given).
        self._model = RFDETRNano(pretrain_weights=str(weights_path))

        if self._device.startswith("cuda"):
            inner = getattr(getattr(self._model, "model", None), "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(self._device)
                # Cast to the requested dtype (BF16 by default — see the
                # docstring on `from_rfdetr` for why). Skip the cast for
                # FP32 so we don't churn parameters unnecessarily.
                if self._requested_dtype != torch.float32:
                    inner.to(self._requested_dtype)
                # ModelContext.device drives the input-tensor `.to(...)`
                # in predict(); keep them aligned so we don't mix devices.
                import contextlib

                with contextlib.suppress(Exception):
                    self._model.model.device = torch.device(self._device)  # pyright: ignore[reportAttributeAccessIssue]
        if optimize_for_inference:
            try:
                # Compile for our actual 4-cam batch. Default is 1, which
                # would raise "Batch size mismatch" on every detect() call.
                self._model.optimize_for_inference(batch_size=self._optimize_batch_size)
            except Exception as exc:  # pyright: ignore[reportBroadExceptionCaught]
                logger.warning(
                    "RF-DETR optimize_for_inference(batch_size=%d) failed (continuing): %s",
                    self._optimize_batch_size,
                    exc,
                )
        # Resolve the LWDETR module + its postprocess once. After
        # `optimize_for_inference` the inner module may be a traced
        # ScriptModule (possibly FP16). `.eval()` is idempotent.
        self._inner_model = self._model.model.model  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(self._inner_model, "eval"):
            self._inner_model.eval()
        self._postprocess = self._model.model.postprocess  # pyright: ignore[reportAttributeAccessIssue]
        # Square input resolution the model was trained for (Nano=384).
        # Tolerate the attribute living at either nesting level.
        self._resolution: int = int(
            getattr(self._model.model, "resolution", None)
            or getattr(self._model, "resolution", 384)
        )
        # Match input dtype to the (possibly traced FP16) inner model.
        try:
            self._inner_dtype: torch.dtype = next(
                self._inner_model.parameters()  # pyright: ignore[reportArgumentType]
            ).dtype
        except (StopIteration, AttributeError):
            self._inner_dtype = torch.float32
        torch_device = torch.device(self._device)
        self._mean: torch.Tensor = torch.tensor(
            self._IMAGENET_MEAN, device=torch_device, dtype=self._inner_dtype
        ).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(
            self._IMAGENET_STD, device=torch_device, dtype=self._inner_dtype
        ).view(1, 3, 1, 1)

    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation] | None = None,
    ) -> dict[int, np.ndarray]:
        if not images:
            return {}
        cam_indices = sorted(images.keys())
        torch_device = torch.device(self._device)

        per_cam_meta: list[tuple[int, int, int, CameraOrientation, int, int]] = []
        hwc_uint8_list: list[np.ndarray] = []
        orientations_list: list[CameraOrientation] = []
        for cam_idx in cam_indices:
            image = _ensure_hwc3_uint8(images[cam_idx])
            orientation = (
                camera_orientations.get(cam_idx, CameraOrientation.UPRIGHT)
                if camera_orientations is not None
                else CameraOrientation.UPRIGHT
            )
            orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
            if orientation == CameraOrientation.ROTATE_90_CW:
                oh, ow = orig_w, orig_h
            else:
                oh, ow = orig_h, orig_w
            per_cam_meta.append((cam_idx, orig_h, orig_w, orientation, oh, ow))
            hwc_uint8_list.append(image)
            orientations_list.append(orientation)

        first_shape = hwc_uint8_list[0].shape
        first_orient = orientations_list[0]
        same_shape = all(img.shape == first_shape for img in hwc_uint8_list)
        same_orient = all(o == first_orient for o in orientations_list)
        if same_shape and same_orient:
            stacked = np.stack(hwc_uint8_list, axis=0)  # (B, H, W, 3) uint8
            batch = torch.from_numpy(stacked).to(torch_device, non_blocking=True)
            # (B, H, W, 3) uint8 -> (B, 3, H, W) float [0, 1]
            batch = batch.permute(0, 3, 1, 2).contiguous().to(self._inner_dtype) / 255.0
            if first_orient == CameraOrientation.ROTATE_90_CW:
                # `dims=(2, 3)` rotates the spatial axes of every image
                # in the batch independently — one launch instead of B.
                batch = torch.rot90(batch, k=-1, dims=(2, 3))
            resized = torch.nn.functional.interpolate(
                batch,
                size=(self._resolution, self._resolution),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )
        else:
            # Slow path: per-cam preprocess. Used when (e.g.) RGB + SLAM
            # cams arrive together with different shapes / orientations.
            chw_list: list[torch.Tensor] = []
            for img, orient in zip(hwc_uint8_list, orientations_list, strict=True):
                oriented_np = _orient_image_for_detector(img, orient)
                t = torch.from_numpy(oriented_np).to(torch_device, non_blocking=True)
                chw_list.append(
                    t.permute(2, 0, 1).contiguous().to(self._inner_dtype) / 255.0
                )
            resized = torch.stack(
                [
                    torch.nn.functional.interpolate(
                        t.unsqueeze(0),
                        size=(self._resolution, self._resolution),
                        mode="bilinear",
                        recompute_scale_factor=False,
                        align_corners=False,
                    ).squeeze(0)
                    for t in chw_list
                ],
                dim=0,
            )
        normalized = (resized - self._mean) / self._std

        # `target_sizes` drives the DETR postprocess: it scales the
        # normalized [0, 1] boxes back into the (oh, ow) frame so we can
        # un-rotate them via `_boxes_oriented_to_original`.
        target_sizes = torch.tensor(
            [[m[4], m[5]] for m in per_cam_meta],
            dtype=torch.long,
            device=torch_device,
        )
        with torch.no_grad():
            predictions = self._inner_model(normalized)
            # Some rfdetr variants return (boxes, logits); wrap to dict
            # so postprocess can read `pred_logits` / `pred_boxes`.
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
            results = self._postprocess(predictions, target_sizes=target_sizes)

        out: dict[int, np.ndarray] = {}
        for (cam_idx, orig_h, orig_w, orientation, _oh, _ow), res in zip(
            per_cam_meta, results, strict=True
        ):
            scores_t = res["scores"]  # (N,) on device
            labels_t = res["labels"]  # (N,) on device
            boxes_t = res["boxes"]  # (N, 4) on device, oriented frame
            # Person + score filter on-device before host transfer.
            keep_t = (labels_t == _RFDETR_PERSON_CLASS_ID) & (
                scores_t >= self._score_thresh
            )
            if not bool(keep_t.any()):
                out[cam_idx] = np.zeros((0, 5), dtype=np.float32)
                continue

            xyxy = boxes_t[keep_t].detach().cpu().float().numpy()
            confidence = scores_t[keep_t].detach().cpu().float().numpy()
            xyxy_orig = _boxes_oriented_to_original(xyxy, orig_h, orientation)
            if self._min_box_size_ratio > 0.0:
                min_side = float(min(orig_h, orig_w)) * self._min_box_size_ratio
                widths = xyxy_orig[:, 2] - xyxy_orig[:, 0]
                heights = xyxy_orig[:, 3] - xyxy_orig[:, 1]
                keep_arr = (widths >= min_side) & (heights >= min_side)
                xyxy_orig = xyxy_orig[keep_arr]
                confidence = confidence[keep_arr]
            arr = np.empty((xyxy_orig.shape[0], 5), dtype=np.float32)
            arr[:, :4] = xyxy_orig
            arr[:, 4] = confidence
            out[cam_idx] = arr
        return out
