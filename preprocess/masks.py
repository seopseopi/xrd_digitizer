"""
$11.5: mask_A (color-distance foreground), mask_B (edge/gradient), combine.

combine = mask_A OR (mask_B AND dilate(mask_A, 3x3))
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, sobel


def _percentile_stretch_gray(gray: np.ndarray, p_lo: float = 3.0, p_hi: float = 97.0) -> np.ndarray:
    """저대비 ROI에서 Sobel 전 명도를 넓힌다 (Step 1b, scikit-image 불필요)."""
    g = gray.astype(np.float64)
    lo, hi = np.percentile(g, [p_lo, p_hi])
    if hi - lo < 1e-6:
        return g
    return np.clip((g - lo) / (hi - lo) * 255.0, 0.0, 255.0)


def build_mask_a(
    color_distance_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """color-distance < threshold -> foreground (curve-like pixel)."""
    return (color_distance_map < threshold).astype(np.uint8)


def build_mask_b(roi: np.ndarray, edge_threshold: Optional[float] = None) -> np.ndarray:
    """edge/gradient mask from grayscale Sobel magnitude.

    edge_threshold 가 None 이면: Step 1b 명도 스트레치 후 Step 1a 적응형 임계값.
    고정 임계값만 쓰려면 예: build_mask_b(roi, 30.0) — 이 경우 스트레치 생략(기존 동작).
    """
    if roi.ndim == 3:
        gray = np.mean(roi.astype(np.float64), axis=2)
    else:
        gray = roi.astype(np.float64)

    if edge_threshold is None:
        gray = _percentile_stretch_gray(gray)

    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    mag = np.sqrt(sx ** 2 + sy ** 2)

    if edge_threshold is not None:
        thr = float(edge_threshold)
    else:
        p70 = float(np.percentile(mag, 70.0))
        thr = float(np.clip(p70, 18.0, 48.0))

    return (mag > thr).astype(np.uint8)


def combine_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """$11.5 combine: mask = mask_A OR (mask_B AND dilate(mask_A, 3x3))."""
    struct_3x3 = np.ones((3, 3), dtype=bool)
    dilated_a = binary_dilation(mask_a.astype(bool), structure=struct_3x3).astype(np.uint8)
    combined = np.clip(mask_a + (mask_b & dilated_a), 0, 1).astype(np.uint8)
    return combined


def mask_axis_lines(
    mask: np.ndarray,
    margin: int = 3,
) -> np.ndarray:
    """Remove axis border pixels from mask (applied after morphology).

    Only zeroes out a thin hard strip at the very edges where the rendered
    rectangle border lives. DP border penalty handles the rest.
    """
    h, w = mask.shape[:2]
    out = mask.copy()
    out[:margin, :] = 0
    out[max(0, h - margin):, :] = 0
    out[:, :margin] = 0
    out[:, max(0, w - margin):] = 0
    return out
