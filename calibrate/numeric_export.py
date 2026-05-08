"""
$16: Numeric export - traced curve to final JSON output fields.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from calibrate.axis_mapping import (
    build_x_mapping,
    build_y_mapping,
    compute_roundtrip_error,
    convert_trace_to_numeric,
    compute_calibration_meta,
    pixel_x_to_value,
    pixel_y_to_value,
)
from calibrate.numeric_curve_peaks import detect_peaks_on_numeric_curve
from core.types import ManualInputs

# export_resample_points 상한 (과대 JSON 방지)
_EXPORT_RESAMPLE_CAP = 20000


def resample_two_theta_uniform(
    two_theta: List[float],
    intensities: List[float],
    x_lo: float,
    x_hi: float,
    n_target: int,
) -> Tuple[List[float], List[float]]:
    """
    x_lo~x_hi를 n_target개 균일 2θ 그리드로 나누고, (two_theta, intensities)에 선형 보간.
    트레이스 해상도는 그대로이며 JSON·UI용 샘플만 촘촘해진다.

    좁은 피크에서 격자점이 원본 샘플 사이에만 있으면 선형 보간이 꼭짓점보다 낮아진다.
    각 격자점 주변(이웃 간격의 절반)에 들어오는 원본 (tt,yy)의 최댓값과 upper envelope으로 보정한다.
    """
    if n_target < 2 or len(two_theta) < 2 or len(intensities) < 2:
        return two_theta, intensities
    n_target = min(int(n_target), _EXPORT_RESAMPLE_CAP)
    if n_target <= len(two_theta):
        return two_theta, intensities

    tt = np.asarray(two_theta, dtype=np.float64)
    yy = np.asarray(intensities, dtype=np.float64)
    order = np.argsort(tt)
    tt = tt[order]
    yy = yy[order]
    if np.any(np.diff(tt) <= 0):
        return two_theta, intensities

    xs = np.linspace(float(x_lo), float(x_hi), n_target, dtype=np.float64)
    yi = np.interp(xs, tt, yy, left=float(yy[0]), right=float(yy[-1]))
    if n_target >= 3:
        span = float(x_hi) - float(x_lo)
        half = 0.5 * span / float(n_target - 1)
        lo_b = np.maximum(float(x_lo), xs - half)
        hi_b = np.minimum(float(x_hi), xs + half)
        for i in range(n_target):
            m = (tt >= lo_b[i]) & (tt <= hi_b[i])
            if np.any(m):
                yi[i] = max(float(yi[i]), float(np.max(yy[m])))
    out_tt = [round(float(x), 6) for x in xs]
    out_y = [round(float(y), 6) for y in yi]
    return out_tt, out_y


def export_numeric(
    columns: List[int],
    y_smoothed: np.ndarray,
    valid: np.ndarray,
    mi: ManualInputs,
    peak_result: dict,
) -> dict:
    """
    Full pixel->numeric pipeline.
    Returns dict with: two_theta_values, intensities, x_range, y_range,
                       calibration_meta, peak_positions_2theta,
                       peaks_numeric_curve, peaks_numeric_curve_params
    """
    x_map = build_x_mapping(mi.x_axis_points, mi.x_axis_values, mi.plot_box)
    y_map = build_y_mapping(mi.y_axis_points, mi.y_axis_values, mi.plot_box)
    roundtrip = compute_roundtrip_error(x_map, y_map)

    two_theta, intensities = convert_trace_to_numeric(
        columns, y_smoothed, valid, x_map, y_map,
    )

    x_range = [float(mi.x_axis_values[0]), float(mi.x_axis_values[-1])]
    y_range = [float(mi.y_axis_values[0]), float(mi.y_axis_values[-1])]
    y_clip_lo = min(y_range[0], y_range[1])
    y_clip_hi = max(y_range[0], y_range[1])

    n_rs: Optional[int] = getattr(mi, "export_resample_points", None)
    if n_rs is not None and int(n_rs) > 0:
        two_theta, intensities = resample_two_theta_uniform(
            two_theta, intensities, x_range[0], x_range[1], int(n_rs),
        )
        intensities = [min(max(y_clip_lo, float(v)), y_clip_hi) for v in intensities]

    calibration_meta = compute_calibration_meta(
        x_map, y_map, roundtrip, two_theta, intensities,
    )

    major_idx = {p["index"] for p in peak_result.get("major_peaks", [])}
    peak_positions = []
    for pk in peak_result.get("peaks", []):
        idx = pk["index"]
        if idx >= len(columns):
            continue
        tt = pixel_x_to_value(float(columns[idx]), x_map)
        if pk.get("y_pixel_roi_refined") is not None:
            y_px = float(pk["y_pixel_roi_refined"])
        elif pk.get("y_pixel_refined") is not None:
            y_px = float(pk["y_pixel_refined"])
        else:
            y_px = float(pk.get("y_pixel", 0.0))
        inten = float(pixel_y_to_value(y_px, y_map))
        inten = min(max(y_clip_lo, inten), y_clip_hi)
        peak_positions.append({
            "two_theta": round(tt, 4),
            "intensity": round(inten, 6),
            "prominence": pk["prominence"],
            "quality": pk["quality"],
            "is_major": pk["index"] in major_idx,
        })

    peaks_nc, peaks_nc_params = detect_peaks_on_numeric_curve(two_theta, intensities)

    return {
        "two_theta_values": two_theta,
        "intensities": intensities,
        "x_range": x_range,
        "y_range": y_range,
        "calibration_meta": calibration_meta,
        "peak_positions_2theta": peak_positions,
        "peaks_numeric_curve": peaks_nc,
        "peaks_numeric_curve_params": peaks_nc_params,
    }
