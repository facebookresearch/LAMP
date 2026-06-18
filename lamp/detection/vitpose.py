# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ViTPose keypoint backend and two-stage detector composition."""

# pyright: reportPrivateImportUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision.ops as tv_ops  # pyright: ignore[reportMissingTypeStubs]
from lamp.core.types import CameraOrientation
from lamp.detection.geometry import (
    _boxes_original_to_oriented,
    _ensure_hwc3_uint8,
    _orient_image_for_detector,
    _points_oriented_to_original,
    _resolve_detector_device,
    empty_keypoint_result,
)
from lamp.detection.types import KeypointDetectionResult, KeypointDetector


class _HuggingFaceViTPoseKeypointDetector(KeypointDetector):
    """Standalone ViTPose via HuggingFace `transformers` (open-source default)."""

    # ViTPose-plus / -base / -small all expect 256-tall x 192-wide crops.
    _CROP_H: int = 256
    _CROP_W: int = 192

    _BOX_PADDING_FACTOR: float = 1.25
    _IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        dataset_index: int = 0,
    ) -> None:
        try:
            from transformers import (  # pyright: ignore[reportMissingImports]
                VitPoseForPoseEstimation,
            )
        except ImportError as exc:
            raise RuntimeError(
                "HF ViTPose backend requires `transformers` (already pulled "
                "in by `rfdetr`). Install with `uv pip install "
                "transformers` if missing."
            ) from exc
        self._device: torch.device = torch.device(_resolve_detector_device(device))
        # `from_pretrained` returns Self, but pyright doesn't have stubs
        # for `VitPoseForPoseEstimation` — `.to` / `.eval` are nn.Module.
        model: Any = VitPoseForPoseEstimation.from_pretrained(model_id)
        model = model.to(self._device).eval()
        if dtype != torch.float32:
            model = model.to(dtype)
        self._model = model
        self._dtype: torch.dtype = dtype
        self._dataset_index: int = int(dataset_index)
        # Cache dataset_index tensor + ImageNet mean/std once at init.
        self._dataset_index_t: torch.Tensor = torch.tensor(
            [self._dataset_index], device=self._device, dtype=torch.long
        )
        self._mean: torch.Tensor = torch.tensor(
            self._IMAGENET_MEAN, device=self._device, dtype=dtype
        ).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(
            self._IMAGENET_STD, device=self._device, dtype=dtype
        ).view(1, 3, 1, 1)

    def _expand_box_aspect_ratio(
        self, boxes_xyxy_oriented: torch.Tensor
    ) -> torch.Tensor:
        """Expand xyxy boxes to the model's input aspect ratio + 1.25 padding.

        Mirrors `box_to_center_and_scale` in HF's ViTPose processor — without
        the per-box Python loop. Operates on a `(N, 4)` GPU tensor.
        """
        x1 = boxes_xyxy_oriented[:, 0]
        y1 = boxes_xyxy_oriented[:, 1]
        x2 = boxes_xyxy_oriented[:, 2]
        y2 = boxes_xyxy_oriented[:, 3]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        aspect = self._CROP_W / self._CROP_H  # 0.75
        # If box is wider than aspect, grow height; else grow width.
        new_w = torch.where(w > aspect * h, w, h * aspect)
        new_h = torch.where(w > aspect * h, w / aspect, h)
        new_w = new_w * self._BOX_PADDING_FACTOR
        new_h = new_h * self._BOX_PADDING_FACTOR
        nx1 = cx - new_w * 0.5
        ny1 = cy - new_h * 0.5
        nx2 = cx + new_w * 0.5
        ny2 = cy + new_h * 0.5
        return torch.stack((nx1, ny1, nx2, ny2), dim=1)

    def detect(
        self,
        images: dict[int, np.ndarray],
        bboxes_per_cam: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation] | None = None,
    ) -> dict[int, KeypointDetectionResult]:
        if not images:
            return {}
        cam_indices = sorted(images.keys())
        orientations = camera_orientations or {}

        # Per-cam: rotate the image into the head-up frame + rotate the
        # supplied original-frame xyxy bboxes into that same frame. The
        # actual crop happens on GPU via roi_align below.
        per_cam_data: dict[int, dict[str, Any]] = {}
        for cam_idx in cam_indices:
            image = _ensure_hwc3_uint8(images[cam_idx])
            orientation = orientations.get(cam_idx, CameraOrientation.UPRIGHT)
            oriented_img = _orient_image_for_detector(image, orientation)
            np_box = bboxes_per_cam.get(cam_idx, np.zeros((0, 7), dtype=np.float32))
            n = int(np_box.shape[0])
            if n > 0:
                orig_xyxy = np_box[:, :4].astype(np.float32, copy=False)
                scores = (np_box[:, 4] * np_box[:, 5]).astype(np.float32, copy=False)
                oriented_xyxy = _boxes_original_to_oriented(
                    orig_xyxy, int(image.shape[0]), orientation
                )
            else:
                orig_xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                oriented_xyxy = np.zeros((0, 4), dtype=np.float32)
            per_cam_data[cam_idx] = {
                "oriented_img": oriented_img,
                "oriented_xyxy": oriented_xyxy,
                "orig_xyxy": orig_xyxy,
                "scores": scores,
                "orientation": orientation,
                "orig_h": int(image.shape[0]),
            }

        # Short-circuit cams with no boxes.
        with_box_cams = [
            c for c in cam_indices if per_cam_data[c]["oriented_xyxy"].shape[0] > 0
        ]
        out: dict[int, KeypointDetectionResult] = {}
        for c in cam_indices:
            if c not in with_box_cams:
                out[c] = empty_keypoint_result()
        if not with_box_cams:
            return out

        first_shape = per_cam_data[with_box_cams[0]]["oriented_img"].shape
        for c in with_box_cams[1:]:
            if per_cam_data[c]["oriented_img"].shape != first_shape:
                raise RuntimeError(
                    "HF ViTPose backend requires all detector cams to share "
                    f"the same (H, W, 3) shape after orientation; got "
                    f"{first_shape} and "
                    f"{per_cam_data[c]['oriented_img'].shape}."
                )
        np_stacked = np.stack(
            [per_cam_data[c]["oriented_img"] for c in with_box_cams], axis=0
        )
        imgs_chw = (
            torch.from_numpy(np_stacked)
            .to(self._device, non_blocking=True)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            / 255.0
        )

        # Build per-cam expanded xyxy in oriented coords (aspect + padding).
        expanded_boxes_per_cam: list[torch.Tensor] = []
        for c in with_box_cams:
            t = torch.from_numpy(per_cam_data[c]["oriented_xyxy"]).to(
                self._device, dtype=torch.float32, non_blocking=True
            )
            expanded_boxes_per_cam.append(self._expand_box_aspect_ratio(t))

        crops_f32 = tv_ops.roi_align(  # pyright: ignore[reportCallIssue]
            imgs_chw,
            expanded_boxes_per_cam,
            output_size=(self._CROP_H, self._CROP_W),
            spatial_scale=1.0,
            aligned=True,
        )  # (sum_N, 3, 256, 192) float32
        crops = crops_f32.to(self._dtype)
        crops_norm = (crops - self._mean) / self._std

        n_persons = int(crops_norm.shape[0])
        dataset_index = self._dataset_index_t.expand(n_persons)
        with torch.no_grad():
            outputs = self._model(pixel_values=crops_norm, dataset_index=dataset_index)

        heatmaps = outputs.heatmaps
        if heatmaps.dtype != torch.float32:
            heatmaps = heatmaps.float()
        _, _n_kp, h_hm, w_hm = heatmaps.shape
        flat = heatmaps.flatten(start_dim=2)
        kp_scores_t, indices = flat.max(dim=-1)  # (sum_N, 17)
        y_hm = (indices // w_hm).float()
        x_hm = (indices % w_hm).float()
        # Map heatmap-grid -> crop-pixel coords.
        x_crop = x_hm / float(w_hm - 1) * float(self._CROP_W - 1)
        y_crop = y_hm / float(h_hm - 1) * float(self._CROP_H - 1)
        # Map crop-pixel -> oriented-image-pixel using the per-person
        # expanded box. We need the box's x1, y1, w, h on device, aligned
        # to the same sum_N order as `heatmaps`.
        all_expanded = torch.cat(expanded_boxes_per_cam, dim=0).float()  # (sum_N, 4)
        bx1 = all_expanded[:, 0:1]  # (sum_N, 1) broadcasts over K=17
        by1 = all_expanded[:, 1:2]
        bw = (all_expanded[:, 2:3] - all_expanded[:, 0:1]).clamp(min=1.0)
        bh = (all_expanded[:, 3:4] - all_expanded[:, 1:2]).clamp(min=1.0)
        x_oriented = bx1 + x_crop / float(self._CROP_W - 1) * bw
        y_oriented = by1 + y_crop / float(self._CROP_H - 1) * bh
        kps_oriented_t = torch.stack((x_oriented, y_oriented), dim=-1)  # (sum_N,17,2)

        # One device->host transfer per logical array, then split per cam.
        kps_np = kps_oriented_t.cpu().numpy().astype(np.float32, copy=False)
        scores_np = kp_scores_t.cpu().numpy().astype(np.float32, copy=False)
        offset = 0
        for c in with_box_cams:
            n = per_cam_data[c]["oriented_xyxy"].shape[0]
            kps_xy_oriented = kps_np[offset : offset + n]
            kp_score = scores_np[offset : offset + n]
            offset += n
            data = per_cam_data[c]
            kps_xy_orig = _points_oriented_to_original(
                kps_xy_oriented,
                orig_h=data["orig_h"],
                orientation=data["orientation"],
            )
            out[c] = KeypointDetectionResult(
                kps_ij=kps_xy_orig,
                kp_scores=kp_score,
                bboxes_xyxy=data["orig_xyxy"],
                bbox_scores=data["scores"],
            )
        return out
