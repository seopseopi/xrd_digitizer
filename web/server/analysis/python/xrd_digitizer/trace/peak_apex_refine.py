"""
major peak 세로 위치를 ROI 밝기 프로파일로 미세 보정 (열 고정).

detect_peaks 가 준 trace 기반 y 에 더해, 해당 열에서 국소 극값을 찾는다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _luma(roi_rgb: np.ndarray) -> np.ndarray:
    if roi_rgb.ndim == 2:
        return roi_rgb.astype(np.float64)
    rgb = roi_rgb[..., :3].astype(np.float64)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def refine_major_peaks_roi_profile(
    peak_result: dict,
    roi_rgb: np.ndarray,
    columns: List[int],
    *,
    search_radius_px: int = 5,
    enabled: bool = True,
) -> dict:
    """
    peaks / major_peaks 의 각 항목에 `y_pixel_roi_refined` 추가 (가능할 때만).

    곡선이 배경보다 어두운 경우가 많아 세로 구간에서 **최소** 밝기(어두운 픽셀)를 apex 로 본다.
    배경보다 밝은 곡선이면 자동으로 **최대**를 선택한다.
    """
    if not enabled:
        return peak_result
    peaks = peak_result.get("peaks") or []
    if not peaks or not columns:
        return peak_result

    h, w = roi_rgb.shape[:2]
    lum = _luma(roi_rgb)

    def _refine_one(pk: dict) -> None:
        idx = int(pk.get("index", -1))
        if idx < 0 or idx >= len(columns):
            return
        col = int(columns[idx])
        if col < 0 or col >= w:
            return

        y_base = pk.get("y_pixel_refined")
        if y_base is None:
            y_base = pk.get("y_pixel")
        if y_base is None:
            return
        y0 = int(round(float(y_base)))
        r = max(1, int(search_radius_px))
        y_lo = max(0, y0 - r)
        y_hi = min(h, y0 + r + 1)
        segment = lum[y_lo:y_hi, col]
        if segment.size < 3:
            return

        col_med = float(np.median(lum[:, col]))
        center = float(lum[y0, col]) if 0 <= y0 < h else col_med
        # 어두운 곡선: segment 최소 / 밝은 곡선: 최대
        if center <= col_med:
            iy = int(np.argmin(segment))
        else:
            iy = int(np.argmax(segment))
        y_new = float(y_lo + iy)
        pk["y_pixel_roi_refined"] = round(y_new, 4)

    for pk in peaks:
        _refine_one(pk)

    return peak_result
