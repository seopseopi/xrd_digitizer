"""
세로 방향 변화(수평 능선) 응답 맵 — 마스크 후보가 격자·배경과 겹칠 때
국소적으로 '잉크가 세로로 바뀌는' 위치를 선호하기 위한 보조 신호.
"""

from __future__ import annotations

import numpy as np


def compute_vertical_ridge_response(roi: np.ndarray) -> np.ndarray:
    """
    각 열마다 |∂gray/∂y| 를 p95로 나눠 [0,1]로 정규화한 H×W float32.
    ChartLine 등에서 말하는 경계/곡선 대비를 비학습으로 근사.
    """
    if roi.ndim == 3:
        gray = np.mean(roi.astype(np.float64), axis=2)
    else:
        gray = roi.astype(np.float64)
    gy = np.abs(np.gradient(gray, axis=0)).astype(np.float32)
    p95 = np.percentile(gy, 95.0, axis=0, keepdims=True).astype(np.float32) + 1e-6
    return np.clip(gy / p95, 0.0, 1.0).astype(np.float32)
