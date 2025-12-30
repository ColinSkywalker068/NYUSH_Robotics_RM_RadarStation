#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_report.py

Usage:
  python image_report.py path/to/image.jpg

Outputs:
  - pixel height / width
  - file size + basic metadata
  - "clearness" (sharpness) score(s)
  - estimated noise level (single-image, spatial noise estimate)

Dependencies:
  pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageReport:
    path: str
    exists: bool
    file_size_bytes: Optional[int]
    width_px: Optional[int]
    height_px: Optional[int]
    channels: Optional[int]
    dtype: Optional[str]

    sharpness_laplacian_var: Optional[float]
    sharpness_tenengrad_mean: Optional[float]
    clearness_label: Optional[str]

    noise_sigma_est: Optional[float]
    noise_sigma_est_label: Optional[str]


def _clearness_label(lap_var: float) -> str:
    # Heuristic thresholds; best used for comparing images from the same camera/settings.
    if lap_var < 50:
        return "very_blurry"
    if lap_var < 150:
        return "blurry"
    if lap_var < 400:
        return "ok"
    return "sharp"


def _noise_label(sigma: float) -> str:
    # Heuristic thresholds in grayscale intensity units [0..255] (roughly).
    if sigma < 2.0:
        return "very_low"
    if sigma < 5.0:
        return "low"
    if sigma < 10.0:
        return "medium"
    if sigma < 20.0:
        return "high"
    return "very_high"


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] >= 3:
        # OpenCV uses BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return gray.astype(np.float32)


def sharpness_metrics(gray_f32: np.ndarray) -> Tuple[float, float]:
    # 1) Variance of Laplacian: higher = sharper
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())

    # 2) Tenengrad (mean gradient magnitude^2): higher = sharper
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    ten = gx * gx + gy * gy
    ten_mean = float(ten.mean())

    return lap_var, ten_mean


def estimate_noise_sigma(gray_f32: np.ndarray) -> float:
    """
    Single-image noise estimate (spatial):
    - Use high-pass residual: gray - GaussianBlur(gray)
    - Estimate sigma via MAD on *low-gradient* pixels to avoid edges dominating.
    """
    # High-pass residual
    blur = cv2.GaussianBlur(gray_f32, (0, 0), sigmaX=1.0, sigmaY=1.0)
    resid = gray_f32 - blur

    # Gradient magnitude to find "flat" regions
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # Take the flattest ~30% pixels
    thresh = float(np.percentile(grad, 30))
    mask = grad <= thresh

    flat = resid[mask]
    if flat.size < 1000:
        # Fallback if the image is very textured
        flat = resid.reshape(-1)

    # Robust sigma estimate from MAD: sigma ≈ MAD / 0.6745
    mad = float(np.median(np.abs(flat - np.median(flat))))
    sigma = mad / 0.6745 if mad > 0 else 0.0

    # Clamp to sane range (helps if image is tiny or pathological)
    return float(max(0.0, min(sigma, 100.0)))


def build_report(path: str) -> ImageReport:
    exists = os.path.exists(path)
    file_size = os.path.getsize(path) if exists and os.path.isfile(path) else None

    if not exists:
        return ImageReport(
            path=path,
            exists=False,
            file_size_bytes=file_size,
            width_px=None,
            height_px=None,
            channels=None,
            dtype=None,
            sharpness_laplacian_var=None,
            sharpness_tenengrad_mean=None,
            clearness_label=None,
            noise_sigma_est=None,
            noise_sigma_est_label=None,
        )

    img = _read_image(path)

    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w = img.shape[:2]
        c = int(img.shape[2]) if img.ndim == 3 else None

    gray = _to_gray(img)

    lap_var, ten_mean = sharpness_metrics(gray)
    noise_sigma = estimate_noise_sigma(gray)

    return ImageReport(
        path=path,
        exists=True,
        file_size_bytes=file_size,
        width_px=int(w),
        height_px=int(h),
        channels=int(c) if c is not None else None,
        dtype=str(img.dtype),
        sharpness_laplacian_var=lap_var,
        sharpness_tenengrad_mean=ten_mean,
        clearness_label=_clearness_label(lap_var),
        noise_sigma_est=noise_sigma,
        noise_sigma_est_label=_noise_label(noise_sigma),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report image size, clearness, and noise level.")
    parser.add_argument("image_path", help="Path to an image file (jpg/png/bmp/tif, etc.)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    rep = build_report(args.image_path)
    data: Dict[str, Any] = asdict(rep)

    if args.pretty:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()

#e.g. .\.venv\Scripts\python.exe tools\image_info\image_report.py tools\image_info\test_images\test1.jpg --pretty