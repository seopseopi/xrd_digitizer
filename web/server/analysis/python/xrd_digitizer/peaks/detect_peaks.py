"""
피크 검출용 trace(y 픽셀, 작을수록 위쪽 꼭짓점)에 대한 국소 적응 prominence + scipy find_peaks.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Set

import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None  # type: ignore

from trace.postprocess import (
    MAJOR_PEAK_CAP,
    MAJOR_PEAK_MIN,
    MAJOR_PEAK_RATIO,
    _refine_peak_y_subpixel,
)


def _mad_centered(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def compute_local_prominence_thresholds(
    y_valid: np.ndarray,
    global_y_range: float,
    *,
    global_prom_ratio: float,
    local_prom_window: int,
    local_prom_ratio: float,
    local_noise_k: float,
) -> np.ndarray:
    """각 유효 샘플 인덱스(0..len-1)에 대한 최소 prominence 임계값."""
    n = len(y_valid)
    half = max(1, int(local_prom_window) // 2)
    thresh = np.zeros(n, dtype=np.float64)
    g_floor = float(global_prom_ratio) * float(global_y_range)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = y_valid[lo:hi]
        if win.size < 2:
            thresh[i] = g_floor
            continue
        p95 = float(np.percentile(win, 95))
        p10 = float(np.percentile(win, 10))
        local_range = max(p95 - p10, 1e-9)
        d = np.diff(win.astype(np.float64))
        mad_d = _mad_centered(d)
        local_noise = float(local_noise_k) * 1.4826 * mad_d
        thresh[i] = max(
            g_floor,
            float(local_prom_ratio) * local_range,
            local_noise,
        )
    return thresh


def _nms_peaks_by_index(
    merged: Dict[int, tuple],
    min_sep: int,
) -> List[int]:
    """global trace 인덱스 기준 클러스터링 후 prominence 최대만 유지."""
    if not merged:
        return []
    keys = sorted(merged.keys())
    clusters: List[List[int]] = [[keys[0]]]
    for gi in keys[1:]:
        if gi - clusters[-1][-1] <= min_sep:
            clusters[-1].append(gi)
        else:
            clusters.append([gi])
    reps: List[int] = []
    for cl in clusters:
        reps.append(max(cl, key=lambda g: merged[g][0]))
    return reps


def detect_peaks_sharp(
    y_peak_smooth_full: np.ndarray,
    valid: np.ndarray,
    gap_filled_indices: Optional[Set[int]],
    columns: Optional[List[int]],
    y_final_for_refine: np.ndarray,
    *,
    global_prom_ratio: float = 0.015,
    local_prom_window: int = 61,
    local_prom_ratio: float = 0.12,
    local_noise_k: float = 3.0,
) -> Dict[str, object]:
    """
    y_peak_smooth_full: 길이 n 트레이스 배열(열 순서와 동일).
    반환 형식은 trace.postprocess.detect_peaks 와 호환(peaks, major_peaks, params).
    """
    if find_peaks is None:
        return {"peaks": [], "major_peaks": [], "params": {}}

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 5:
        return {"peaks": [], "major_peaks": [], "params": {}}

    y_valid = y_peak_smooth_full[valid_idx].astype(np.float64)
    y_inverted = -y_valid

    y_max = float(np.max(y_valid))
    y_min = float(np.min(y_valid))
    y_range = y_max - y_min
    if y_range < 1e-6:
        return {"peaks": [], "major_peaks": [], "params": {}}

    num_points = len(y_valid)
    prom_thresh_arr = compute_local_prominence_thresholds(
        y_valid,
        y_range,
        global_prom_ratio=global_prom_ratio,
        local_prom_window=local_prom_window,
        local_prom_ratio=local_prom_ratio,
        local_noise_k=local_noise_k,
    )

    min_peak_distance = max(3, round(0.004 * num_points))
    min_peak_height_abs = y_min + 0.03 * y_range
    max_height_inverted = -(min_peak_height_abs)

    peak_kw = dict(
        distance=min_peak_distance,
        height=(None, max_height_inverted),
    )

    min_prom_detect = float(max(np.min(prom_thresh_arr), 1e-6))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks_raw, props = find_peaks(
            y_inverted,
            prominence=min_prom_detect * 0.01,
            **peak_kw,
        )

    merged: Dict[int, tuple] = {}
    if peaks_raw.size:
        proms = props.get("prominences", np.zeros(len(peaks_raw)))
        lb = props.get("left_bases", np.zeros(len(peaks_raw), dtype=np.int64))
        rb = props.get("right_bases", np.zeros(len(peaks_raw), dtype=np.int64))
        for row_i, pk in enumerate(peaks_raw):
            pk_i = int(pk)
            if pk_i < 0 or pk_i >= len(prom_thresh_arr):
                continue
            scipy_p = float(proms[row_i]) if row_i < len(proms) else 0.0
            if scipy_p < float(prom_thresh_arr[pk_i]):
                continue
            gi = int(valid_idx[pk_i])
            prev = merged.get(gi)
            if prev is None or scipy_p > prev[0]:
                merged[gi] = (scipy_p, pk_i, props, row_i)

    winners = _nms_peaks_by_index(merged, min_peak_distance)

    peaks_info: List[dict] = []
    for global_idx in winners:
        prom, pk, props_full, row_i = merged[global_idx]
        y_val = float(y_peak_smooth_full[global_idx])

        is_gap_filled = bool(gap_filled_indices and global_idx in gap_filled_indices)

        left_base = int(props_full["left_bases"][row_i]) if "left_bases" in props_full else 0
        right_base = int(props_full["right_bases"][row_i]) if "right_bases" in props_full else len(y_valid) - 1
        half_width = (right_base - left_base) / 2.0
        symmetry = 1.0 - abs((pk - left_base) - (right_base - pk)) / max(half_width * 2, 1)
        sharpness = prom / max(half_width, 1)
        quality = 0.5 * min(prom / max(y_range * 0.1, 1), 1.0) + 0.3 * symmetry + 0.2 * min(sharpness / 5.0, 1.0)
        if is_gap_filled:
            quality *= 0.5

        y_refined = None
        if columns is not None and len(columns) == len(y_final_for_refine) and 0 < global_idx < len(y_final_for_refine) - 1:
            y_refined = _refine_peak_y_subpixel(y_final_for_refine, global_idx)

        entry = {
            "index": global_idx,
            "y_pixel": y_val,
            "prominence": round(float(prom), 4),
            "quality": round(float(quality), 4),
            "is_gap_filled": is_gap_filled,
        }
        if y_refined is not None:
            entry["y_pixel_refined"] = round(float(y_refined), 4)
        peaks_info.append(entry)

    peaks_info.sort(key=lambda p: -p["prominence"])

    n_detected = len(peaks_info)
    n_major = min(MAJOR_PEAK_CAP, max(MAJOR_PEAK_MIN, math.ceil(MAJOR_PEAK_RATIO * max(n_detected, 1))))
    major_peaks = peaks_info[:n_major]

    return {
        "peaks": peaks_info,
        "major_peaks": major_peaks,
        "params": {
            "y_range": round(y_range, 2),
            "prominence_mode": "local_adaptive",
            "global_prom_ratio": float(global_prom_ratio),
            "local_prom_window": int(local_prom_window),
            "local_prom_ratio": float(local_prom_ratio),
            "local_noise_k": float(local_noise_k),
            "min_peak_distance": min_peak_distance,
            "min_peak_height": round(float(min_peak_height_abs), 4),
            "num_points": num_points,
            "num_peaks_detected": n_detected,
            "num_major_peaks": len(major_peaks),
            "min_prominence_detect_floor": round(min_prom_detect, 6),
        },
    }
