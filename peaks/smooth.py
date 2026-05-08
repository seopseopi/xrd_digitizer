"""
곡선 안정화용 / 피크 검출용 SG 분리.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None  # type: ignore


def _effective_sg_window(desired: int, n_samples: int) -> int:
    """홀수이며 n_samples 이하; 최소 3 (불가 시 SG 스킵 호출부에서 처리)."""
    w = min(int(desired), int(n_samples))
    if w < 3:
        return 1
    if w % 2 == 0:
        w -= 1
    return max(3, w)


def _sg_on_valid_series(
    y_filled: np.ndarray,
    valid: np.ndarray,
    window: int,
    polyorder: int = 2,
) -> np.ndarray:
    """유효 인덱스만 이어 붙인 1D 시퀀스에 SG 적용 후 같은 위치에 되돌린다."""
    out = np.asarray(y_filled, dtype=np.float64).copy()
    if savgol_filter is None:
        return out
    vy = out[valid.astype(bool)]
    if vy.size < 3:
        return out
    w = _effective_sg_window(window, vy.size)
    if w < 3:
        return out
    try:
        sm = savgol_filter(vy, w, polyorder)
    except ValueError:
        return out
    out[valid.astype(bool)] = sm
    return out


def smooth_for_curve(
    y_filled: np.ndarray,
    valid: np.ndarray,
    curve_smooth_window: int = 9,
    polyorder: int = 2,
) -> np.ndarray:
    return _sg_on_valid_series(y_filled, valid, curve_smooth_window, polyorder)


def smooth_for_peak(
    y_filled: np.ndarray,
    valid: np.ndarray,
    peak_smooth_window: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    return _sg_on_valid_series(y_filled, valid, peak_smooth_window, polyorder)
