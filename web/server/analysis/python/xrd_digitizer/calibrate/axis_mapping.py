"""
$16: Pixel -> numeric axis mapping.

- x mapping: LinearRegression (pixel x -> two_theta)
- y mapping: LinearRegression (pixel y -> intensity, y-axis inverted)
- Roundtrip error calculation

캘리브레이션 포인트(2점)를 학습 데이터로 삼아 sklearn LinearRegression 모델을
fitting하고, 추출된 픽셀 좌표 전체에 적용해 물리적 수치로 복원한다.

주의: manual_inputs 의 축 좌표는 plot_box 기준 원본 이미지 좌표이다. ROI crop 후
픽셀 열·행은 crop 기준(0,0)이며, perspective 보정이 켜지면 ROI가 워핑되므로
축 클릭이 워핑 전 좌표만 가리키면 이 매핑과 미세 불일치가 날 수 있다.
(perspective 가 없으면 trace 열·행과 일치.)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


def build_x_mapping(
    x_axis_points: List[List[int]],
    x_axis_values: List[float],
    plot_box: List[int],
) -> Dict:
    """
    X축 캘리브레이션: LinearRegression으로 pixel_x -> two_theta 모델 학습.
    x_axis_points: [[px1, py1], [px2, py2]]
    x_axis_values: [val1, val2]
    """
    x0, y0, x1, y1 = plot_box
    px1 = float(x_axis_points[0][0]) - x0
    px2 = float(x_axis_points[-1][0]) - x0
    v1 = float(x_axis_values[0])
    v2 = float(x_axis_values[-1])

    if abs(px2 - px1) < 1e-6:
        return {"scale": 0.0, "offset": v1, "px_ref": [px1, px2], "val_ref": [v1, v2]}

    model = LinearRegression()
    model.fit([[px1], [px2]], [v1, v2])
    scale = float(model.coef_[0])
    offset = float(model.intercept_)

    return {"scale": scale, "offset": offset, "px_ref": [px1, px2], "val_ref": [v1, v2]}


def build_y_mapping(
    y_axis_points: List[List[int]],
    y_axis_values: List[float],
    plot_box: List[int],
) -> Dict:
    """
    Y축 캘리브레이션: LinearRegression으로 pixel_y -> intensity 모델 학습.
    픽셀 y는 아래로 증가, intensity는 위로 증가 (역방향 자동 반영).
    """
    x0, y0, x1, y1 = plot_box
    py1 = float(y_axis_points[0][1]) - y0
    py2 = float(y_axis_points[-1][1]) - y0
    v1 = float(y_axis_values[0])
    v2 = float(y_axis_values[-1])

    if abs(py2 - py1) < 1e-6:
        return {"scale": 0.0, "offset": v1, "px_ref": [py1, py2], "val_ref": [v1, v2]}

    model = LinearRegression()
    model.fit([[py1], [py2]], [v1, v2])
    scale = float(model.coef_[0])
    offset = float(model.intercept_)

    return {"scale": scale, "offset": offset, "px_ref": [py1, py2], "val_ref": [v1, v2]}


def pixel_x_to_value(px: float, x_map: Dict) -> float:
    return x_map["scale"] * px + x_map["offset"]


def pixel_y_to_value(py: float, y_map: Dict) -> float:
    return y_map["scale"] * py + y_map["offset"]


def value_to_pixel_x(val: float, x_map: Dict) -> float:
    if abs(x_map["scale"]) < 1e-12:
        return x_map["px_ref"][0]
    return (val - x_map["offset"]) / x_map["scale"]


def value_to_pixel_y(val: float, y_map: Dict) -> float:
    if abs(y_map["scale"]) < 1e-12:
        return y_map["px_ref"][0]
    return (val - y_map["offset"]) / y_map["scale"]


def compute_roundtrip_error(
    x_map: Dict,
    y_map: Dict,
) -> Dict:
    """
    Roundtrip error: pixel -> value -> pixel, measure deviation at reference points.
    """
    x_errors = []
    for px, val in zip(x_map["px_ref"], x_map["val_ref"]):
        converted = pixel_x_to_value(px, x_map)
        back = value_to_pixel_x(converted, x_map)
        x_errors.append(abs(back - px))

    y_errors = []
    for py, val in zip(y_map["px_ref"], y_map["val_ref"]):
        converted = pixel_y_to_value(py, y_map)
        back = value_to_pixel_y(converted, y_map)
        y_errors.append(abs(back - py))

    x_mean = float(np.mean(x_errors)) if x_errors else 0.0
    y_mean = float(np.mean(y_errors)) if y_errors else 0.0
    total_mean = (x_mean + y_mean) / 2.0

    confidence = max(0.0, min(1.0, 1.0 - total_mean / 3.0))

    return {
        "x_roundtrip_px": [round(e, 6) for e in x_errors],
        "y_roundtrip_px": [round(e, 6) for e in y_errors],
        "x_mean_error_px": round(x_mean, 6),
        "y_mean_error_px": round(y_mean, 6),
        "total_mean_error_px": round(total_mean, 6),
        "calibration_confidence": round(confidence, 4),
    }


def convert_trace_to_numeric(
    columns: List[int],
    y_smoothed: np.ndarray,
    valid: np.ndarray,
    x_map: Dict,
    y_map: Dict,
) -> Tuple[List[float], List[float]]:
    """
    Pixel trace -> (two_theta_values, intensities).
    Only valid columns are converted.
    축 값(y_axis_values)이 음수를 포함하면 매핑 결과도 음수가 될 수 있으며 그대로 둔다.
    """
    two_theta = []
    intensities = []

    for i, col in enumerate(columns):
        if i >= len(valid) or not valid[i]:
            continue
        x_val = pixel_x_to_value(float(col), x_map)
        y_val = float(pixel_y_to_value(float(y_smoothed[i]), y_map))
        two_theta.append(round(x_val, 6))
        intensities.append(round(y_val, 6))

    return two_theta, intensities


def convert_trace_upscaled_roi_to_numeric(
    columns_up: List[int],
    y_smoothed_up: np.ndarray,
    valid_up: np.ndarray,
    x_map: Dict,
    y_map: Dict,
    upscale_factor: int,
) -> Tuple[List[float], List[float]]:
    """
    ROI가 균일 upscale 된 경우, 픽셀 (col_up, y_up)을 원본 ROI 해상도 좌표로 환산한 뒤
    기존 plot_box 기준 axis mapping을 적용한다.

    - 추가 interpolation 없음 (열 단위 샘플만 사용).
    - upscale_factor <= 1 이면 기존 ``convert_trace_to_numeric`` 와 동일.
    """
    if upscale_factor <= 1:
        return convert_trace_to_numeric(columns_up, y_smoothed_up, valid_up, x_map, y_map)
    inv = 1.0 / float(upscale_factor)
    ys = np.asarray(y_smoothed_up, dtype=np.float64)
    va = np.asarray(valid_up, dtype=bool)
    two_theta: List[float] = []
    intensities: List[float] = []
    for i, col in enumerate(columns_up):
        if i >= len(va) or not va[i]:
            continue
        cx = float(col) * inv
        cy = float(ys[i]) * inv
        two_theta.append(round(float(pixel_x_to_value(cx, x_map)), 6))
        intensities.append(round(float(pixel_y_to_value(cy, y_map)), 6))
    return two_theta, intensities


def compute_calibration_meta(
    x_map: Dict,
    y_map: Dict,
    roundtrip: Dict,
    two_theta: List[float],
    intensities: List[float],
) -> Dict:
    """Build calibration_meta for output."""
    i_max = max(intensities) if intensities else 0.0
    i_min = min(intensities) if intensities else 0.0
    i_range = i_max - i_min

    intensities_normalized = []
    if i_range > 1e-12:
        intensities_normalized = [round((v - i_min) / i_range * 100.0, 4) for v in intensities]

    return {
        "x_scale": round(x_map["scale"], 8),
        "x_offset": round(x_map["offset"], 6),
        "y_scale": round(y_map["scale"], 8),
        "y_offset": round(y_map["offset"], 6),
        "roundtrip_error": roundtrip,
        "num_points": len(two_theta),
        "intensities_normalized": intensities_normalized,
    }
