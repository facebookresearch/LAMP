# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ONNX Runtime TensorRT RF-DETR person-box backend."""

# pyright: reportPrivateImportUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

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


# Module-level cache so repeat instantiation of `_RfDetrTrtBboxDetector` is
# idempotent — the second class instance doesn't re-walk `site-packages` or
# re-`ctypes.CDLL` every NVIDIA shared object.
_TRT_NVIDIA_LIBS_LOADED: bool = False


def _preload_trt_nvidia_libs() -> bool:
    """Best-effort preload of bundled `nvidia/*` + `tensorrt_libs` .so files."""
    import ctypes
    import site

    global _TRT_NVIDIA_LIBS_LOADED
    if _TRT_NVIDIA_LIBS_LOADED:
        return True

    candidate_roots: list[Path] = []
    for site_dir in site.getsitepackages():
        candidate_roots.append(Path(site_dir))
    # User site-packages too, in case the user `pip install --user`'d.
    user_site = site.getusersitepackages()
    if user_site:
        candidate_roots.append(Path(user_site))

    # The set of nvidia submodules that bundle .so libs. We don't blow up if
    # any is missing -- onnxruntime-gpu only really needs cuda_runtime + cudnn
    # + cublas + the TRT libs; the rest are loaded transitively.
    nvidia_subdirs = (
        "cuda_runtime",
        "cudnn",
        "cublas",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nvjitlink",
        "cuda_cupti",
        "nccl",
    )

    loaded_count = 0
    seen: set[str] = set()
    for root in candidate_roots:
        cuda_dirs = [root / "nvidia" / sub / "lib" for sub in nvidia_subdirs]
        cuda_dirs.append(root / "tensorrt_libs")
        for lib_dir in cuda_dirs:
            if not lib_dir.is_dir():
                continue
            for so_path in sorted(lib_dir.glob("lib*.so*")):
                # Avoid loading the same lib twice from a duplicate site-dir.
                key = so_path.name
                if key in seen:
                    continue
                seen.add(key)
                try:
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                    loaded_count += 1
                except OSError as exc:
                    logger.debug("skipping ctypes.CDLL preload of %s: %s", so_path, exc)

    if loaded_count > 0:
        logger.info(
            "Preloaded %d nvidia/tensorrt_libs shared objects for ORT TRT EP",
            loaded_count,
        )
        _TRT_NVIDIA_LIBS_LOADED = True
        return True
    logger.warning(
        "No nvidia/tensorrt_libs .so files found via site-packages; "
        "TRT EP may fail to load. Set LD_LIBRARY_PATH to point at "
        "<venv>/lib/python*/site-packages/{nvidia/*/lib,tensorrt_libs} "
        "if the InferenceSession constructor crashes."
    )
    return False


class _RfDetrTrtBboxDetector(BBoxDetector):
    """RF-DETR-Nano via ONNX Runtime TensorRT EP."""

    # ImageNet normalization -- mirrors `_RfDetrBboxDetector._IMAGENET_*`.
    _IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # RF-DETR-Nano was trained at 384x384. The bench script + the
    # `RFDETRNano.export` call both fix this size.
    _RESOLUTION: int = 384

    def __init__(
        self,
        rfdetr_weights: Path | None = None,
        onnx_path: Path | None = None,
        engine_cache_dir: Path | None = None,
        device: str = "cuda:0",
        score_thresh: float = 0.85,
        min_box_size_ratio: float = 0.0,
        fp16: bool = False,
    ) -> None:
        self._device: str = _resolve_detector_device(device)
        self._score_thresh: float = float(score_thresh)
        self._min_box_size_ratio: float = float(min_box_size_ratio)

        self._fp16: bool = bool(fp16)

        _preload_trt_nvidia_libs()

        rfdetr_weights = _resolve_rfdetr_weights(rfdetr_weights)
        if onnx_path is None:
            onnx_path = rfdetr_weights.parent / "rfdetr-nano.onnx"
        else:
            onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            self._export_onnx(rfdetr_weights, onnx_path)
        self._onnx_path: Path = onnx_path

        # Resolve the engine cache directory (sibling of the ONNX file by
        # default). TRT serializes a .engine file per (model, GPU, opt config)
        # tuple into this dir on first init.
        if engine_cache_dir is None:
            engine_cache_dir = onnx_path.parent / "rfdetr_trt_cache"
        else:
            engine_cache_dir = Path(engine_cache_dir)
        engine_cache_dir.mkdir(parents=True, exist_ok=True)
        self._engine_cache_dir: Path = engine_cache_dir

        # Build the InferenceSession. Try TensorrtExecutionProvider first;
        # fall back to CUDA EP if TRT load fails (logged loudly so the user
        # knows they're not getting the 5.9x speedup).
        self._session = self._build_session()
        self._input_name: str = self._session.get_inputs()[0].name

        # Warm the engine: first call after Session creation triggers the
        # TRT engine build (or load-from-cache). Doing it here moves the cost
        # to init time rather than burying it in the first real `detect()`.
        dummy = np.zeros((4, 3, self._RESOLUTION, self._RESOLUTION), dtype=np.float32)
        t0 = time.perf_counter()
        self._session.run(None, {self._input_name: dummy})
        wall_s = time.perf_counter() - t0
        logger.info(
            "TRT engine ready (warmup forward took %.1f s%s)",
            wall_s,
            " -- engine cache hit" if wall_s < 30.0 else " -- cold engine build",
        )

        # Cache ImageNet mean/std + torch device for the per-call
        # preprocess (we run the preprocess on GPU via torch ops to match
        # eager parity, then numpy-copy into the ORT session).
        torch_device = torch.device(self._device)
        self._torch_device: torch.device = torch_device
        self._mean: torch.Tensor = torch.tensor(
            self._IMAGENET_MEAN, device=torch_device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(
            self._IMAGENET_STD, device=torch_device, dtype=torch.float32
        ).view(1, 3, 1, 1)

    def _export_onnx(self, weights_path: Path, onnx_path: Path) -> None:
        """One-time RF-DETR ONNX export. Takes ~35 s; logged so users see it."""
        try:
            from rfdetr import RFDETRNano  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR TRT backend requires the `rfdetr` package for the "
                "one-time ONNX export. Install it with "
                "`uv pip install rfdetr`."
            ) from exc
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Exporting RF-DETR-Nano to ONNX (one-time, ~35 s)\n"
            "  weights:    %s\n  onnx_path:  %s",
            weights_path,
            onnx_path,
        )
        t0 = time.perf_counter()
        model = RFDETRNano(pretrain_weights=str(weights_path))
        # `batch_size=4` matches the typical 4-cam SLAM batch; TRT will
        # static-shape the engine to this batch.
        model.export(
            format="onnx",
            quantization=None,
            output_dir=str(onnx_path.parent),
            batch_size=4,
        )
        # rfdetr writes the graph under a FIXED name `inference_model.onnx`
        # in `output_dir` (it ignores any requested basename). Move it to the
        # `onnx_path` we advertise so `_build_session` finds it.
        produced = onnx_path.parent / "inference_model.onnx"
        if produced.exists() and produced.resolve() != onnx_path.resolve():
            produced.replace(onnx_path)
        elapsed = time.perf_counter() - t0
        size_mb = onnx_path.stat().st_size / 1e6 if onnx_path.exists() else 0.0
        logger.info(
            "ONNX export complete (%.1f s, %.1f MB)",
            elapsed,
            size_mb,
        )

    def _build_session(self) -> Any:
        """Create the ORT InferenceSession with TRT + CUDA + CPU EPs."""
        try:
            import onnxruntime as ort  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR TRT backend requires `onnxruntime-gpu`. Install it "
                "with `uv pip install onnxruntime-gpu` (then "
                "`uv pip install --no-build-isolation "
                "'tensorrt-cu12<11'` for the TRT EP)."
            ) from exc

        providers: list[str] = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        provider_options: list[dict[str, Any]] = [
            {
                # FP32 by default (correct scores). FP16 is opt-in and broken
                # for RF-DETR — see __init__'s `_fp16` note.
                "trt_fp16_enable": self._fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self._engine_cache_dir),
            },
            {},
            {},
        ]
        try:
            sess = ort.InferenceSession(
                str(self._onnx_path),
                providers=providers,
                provider_options=provider_options,
            )
            active = sess.get_providers()
            if "TensorrtExecutionProvider" not in active:
                logger.warning(
                    "TensorrtExecutionProvider requested but not active "
                    "(got %s). Falling back to CUDA EP -- expect ~10 ms "
                    "forward instead of ~5 ms.",
                    active,
                )
            else:
                logger.info("ORT session active providers: %s", active)
            return sess
        except Exception as exc:  # pyright: ignore[reportBroadExceptionCaught]
            logger.warning(
                "TRT EP failed to initialize (%s) -- retrying with CUDA EP only",
                exc,
            )
            return ort.InferenceSession(
                str(self._onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def detect(
        self,
        images: dict[int, np.ndarray],
        camera_orientations: dict[int, CameraOrientation] | None = None,
    ) -> dict[int, np.ndarray]:
        if not images:
            return {}
        cam_indices = sorted(images.keys())

        # Per-cam metadata + HWC-uint8 normalization (parity with the eager
        # path -- same `_ensure_hwc3_uint8`, same orientation handling).
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

        normalized = self._preprocess_batch_gpu(hwc_uint8_list, orientations_list)
        # ORT FP32 input -- convert from torch GPU tensor to numpy CPU
        # contiguous FP32. TRT's input binding is FP32; it casts internally
        # to FP16 for the GPU compute.
        ort_input = normalized.cpu().numpy().astype(np.float32, copy=False)

        # ONNX forward. Outputs are `(dets, labels)`:
        #   dets:   (B, 300, 4) -- xyxy in normalized [0, 1] frame
        #   labels: (B, 300, 91) -- raw logits, sparse COCO classes
        dets_n, labels_n = self._session.run(None, {self._input_name: ort_input})

        # Postprocess on CPU -- the per-cam loop is tiny (B=4) and the boxes
        # are already cropped to top-300 by RF-DETR's head, so GPU acceleration
        # buys nothing here.
        return self._postprocess(dets_n, labels_n, per_cam_meta)

    def _preprocess_batch_gpu(
        self,
        hwc_uint8_list: list[np.ndarray],
        orientations_list: list[CameraOrientation],
    ) -> torch.Tensor:
        """Mirror `_RfDetrBboxDetector` preprocess: rotate+bilinear+normalize."""
        first_shape = hwc_uint8_list[0].shape
        first_orient = orientations_list[0]
        same_shape = all(img.shape == first_shape for img in hwc_uint8_list)
        same_orient = all(o == first_orient for o in orientations_list)

        if same_shape and same_orient:
            stacked = np.stack(hwc_uint8_list, axis=0)
            batch = torch.from_numpy(stacked).to(self._torch_device, non_blocking=True)
            batch = batch.permute(0, 3, 1, 2).contiguous().float() / 255.0
            if first_orient == CameraOrientation.ROTATE_90_CW:
                batch = torch.rot90(batch, k=-1, dims=(2, 3))
            resized = torch.nn.functional.interpolate(
                batch,
                size=(self._RESOLUTION, self._RESOLUTION),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )
        else:
            chw_list: list[torch.Tensor] = []
            for img, orient in zip(hwc_uint8_list, orientations_list, strict=True):
                oriented_np = _orient_image_for_detector(img, orient)
                t = torch.from_numpy(oriented_np).to(
                    self._torch_device, non_blocking=True
                )
                chw_list.append(t.permute(2, 0, 1).contiguous().float() / 255.0)
            resized = torch.stack(
                [
                    torch.nn.functional.interpolate(
                        t.unsqueeze(0),
                        size=(self._RESOLUTION, self._RESOLUTION),
                        mode="bilinear",
                        recompute_scale_factor=False,
                        align_corners=False,
                    ).squeeze(0)
                    for t in chw_list
                ],
                dim=0,
            )
        return (resized - self._mean) / self._std

    def _postprocess(
        self,
        dets_n: np.ndarray,
        labels_n: np.ndarray,
        per_cam_meta: list[tuple[int, int, int, CameraOrientation, int, int]],
    ) -> dict[int, np.ndarray]:
        """Decode `(B, 300, 4)` + `(B, 300, 91)` -> per-cam `(N, 5)` xyxy+score."""
        # Sigmoid on the person column only -- avoid sigmoid'ing the full
        # (B, 300, 91) tensor (~10x smaller op).
        person_logits = labels_n[..., _RFDETR_PERSON_CLASS_ID]  # (B, 300)
        person_scores = 1.0 / (1.0 + np.exp(-person_logits))

        out: dict[int, np.ndarray] = {}
        for b_idx, (cam_idx, orig_h, orig_w, orientation, oh, ow) in enumerate(
            per_cam_meta
        ):
            scores = person_scores[b_idx]  # (300,)
            boxes_cxcywh = dets_n[b_idx]  # (300, 4)
            keep_mask = scores >= self._score_thresh
            if not bool(keep_mask.any()):
                out[cam_idx] = np.zeros((0, 5), dtype=np.float32)
                continue

            kept_cxcywh = boxes_cxcywh[keep_mask]
            kept_scores = scores[keep_mask].astype(np.float32, copy=False)

            # cxcywh (normalized) -> xyxy (in oriented-image-pixel frame).
            cx = kept_cxcywh[:, 0] * float(ow)
            cy = kept_cxcywh[:, 1] * float(oh)
            w = kept_cxcywh[:, 2] * float(ow)
            h = kept_cxcywh[:, 3] * float(oh)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            xyxy_oriented = np.stack((x1, y1, x2, y2), axis=1).astype(
                np.float32, copy=False
            )
            xyxy_orig = _boxes_oriented_to_original(xyxy_oriented, orig_h, orientation)

            if self._min_box_size_ratio > 0.0:
                min_side = float(min(orig_h, orig_w)) * self._min_box_size_ratio
                widths = xyxy_orig[:, 2] - xyxy_orig[:, 0]
                heights = xyxy_orig[:, 3] - xyxy_orig[:, 1]
                keep_arr = (widths >= min_side) & (heights >= min_side)
                xyxy_orig = xyxy_orig[keep_arr]
                kept_scores = kept_scores[keep_arr]

            arr = np.empty((xyxy_orig.shape[0], 5), dtype=np.float32)
            arr[:, :4] = xyxy_orig
            arr[:, 4] = kept_scores
            out[cam_idx] = arr
        return out
