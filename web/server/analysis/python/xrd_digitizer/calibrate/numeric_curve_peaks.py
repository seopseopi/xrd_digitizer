"""
수출된 (2θ, intensity) 곡선만으로 국소 최대를 재검출.

분리 로드맵 1단계: 픽셀 경로 `detect_peaks`와 독립 — 근접 피크가 한 줄로 뭉개져도
곡선에 미세한 이중 돌출이 남아 있으면 별도 피크로 잡을 수 있다.

임계값은 트레이스 길이·Δ2θ·강도 범위에서 자동 산출 (패턴 ID 고정값 없음).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def refine_peak_parabolic_tt_yy(
    tt: np.ndarray,
    yy: np.ndarray,
    i: int,
) -> Tuple[float, float]:
    """
    이산 격자에서 피크 인덱스가 꼭짓점이 아닐 때(양옆 샘플 사이), 인덱스 -1,0,1 이차식으로
    2θ·강도 보정. ROI 표시가 곡선 꼭짓점에 가깝게 붙도록 한다.
    """
    if i < 1 or i >= len(yy) - 1:
        return float(tt[i]), float(yy[i])
    y0, y1, y2 = float(yy[i - 1]), float(yy[i]), float(yy[i + 1])
    t0, t1, t2 = float(tt[i - 1]), float(tt[i]), float(tt[i + 1])
    a = 0.5 * (y0 - 2.0 * y1 + y2)
    b = 0.5 * (y2 - y0)
    if abs(a) < 1e-14 or a >= 0.0:
        return t1, y1
    di = float(-b / (2.0 * a))
    di = float(np.clip(di, -0.95, 0.95))
    idx_f = float(i) + di
    t_ref = float(
        np.interp(
            idx_f,
            np.array([float(i - 1), float(i), float(i + 1)], dtype=np.float64),
            np.array([t0, t1, t2], dtype=np.float64),
        )
    )
    y_ref = a * di * di + b * di + y1
    y_ref = float(max(y_ref, y1))
    return t_ref, y_ref


try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None  # type: ignore

# 스캔 축에서의 상대 위치만 사용 (절대 2θ 고정 각도 없음). 마지막 일부 구간에서만 prominence 바닥 상향.
NUMERIC_PEAK_TAIL_SPAN_FRAC = 0.22


def _fallback_local_maxima(y: np.ndarray, distance: int, prominence: float) -> Tuple[np.ndarray, dict]:
    """scipy 없을 때 단순 국소 최대 + 거리·prominence 필터."""
    n = len(y)
    cand = [
        i
        for i in range(1, n - 1)
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > y[i - 1] + 1e-12
    ]
    if not cand:
        return np.array([], dtype=np.int64), {}
    cand.sort(key=lambda i: float(y[i]), reverse=True)
    taken: List[int] = []
    for i in cand:
        if all(abs(i - j) >= distance for j in taken):
            # 간이 prominence: 좌우 최저와의 차이
            lo = max(0, i - min(distance * 3, 15))
            hi = min(n - 1, i + min(distance * 3, 15))
            base = float(min(np.min(y[lo:i]), np.min(y[i + 1 : hi + 1])) if i + 1 <= hi else y[i])
            if float(y[i]) - base >= prominence:
                taken.append(i)
    return np.asarray(sorted(taken), dtype=np.int64), {}


def detect_peaks_on_numeric_curve(
    two_theta: List[float],
    intensities: List[float],
) -> Tuple[List[dict], Dict[str, float]]:
    """
    균일/비균일 샘플 곡선에서 scipy find_peaks (prominence + distance).

    반환:
      peaks: [{two_theta, intensity, prominence_est, width_samples}, ...]
      params: 디버그용 스케일 (min_sep_deg, distance_samples, prom_strict, ...)
    """
    if len(two_theta) < 7 or len(intensities) != len(two_theta):
        return [], {"reason": "too_few_points"}

    tt = np.asarray(two_theta, dtype=np.float64)
    yy = np.asarray(intensities, dtype=np.float64)
    order = np.argsort(tt)
    tt = tt[order]
    yy = yy[order]
    if np.any(np.diff(tt) <= 0):
        return [], {"reason": "non_monotonic_x"}

    n = len(tt)
    dtt = np.diff(tt)
    med_dt = float(np.median(dtt)) if dtt.size else 1e-6
    med_dt = max(med_dt, 1e-9)
    span_tt = float(tt[-1] - tt[0])
    ir = float(np.ptp(yy))
    if ir < 1e-12:
        return [], {"reason": "flat"}

    noise_i = float(np.median(np.abs(np.diff(yy)))) if n > 2 else 0.0
    noise_i = max(noise_i, ir * 1e-6)

    # 인접 피크 최소 각도 간격: 너무 크면 저강도 구간의 근접 피크가 한 덩어리로 합쳐짐
    min_sep_deg = float(max(1.35 * med_dt, 0.00035 * span_tt))
    distance_samples = int(max(2, round(min_sep_deg / med_dt)))
    distance_samples = min(distance_samples, max(2, n // 10))

    # 전역 ptp 기반은 주피크에 맞춰져 저강도 어깨를 놓침 → noise_i 기반 ultra 패스 병행
    prom_strict = max(0.0024 * ir, 1.75 * noise_i, ir * 1e-5)
    prom_loose = max(prom_strict * 0.42, 1.05 * noise_i)
    prom_ultra = max(0.82 * noise_i, 0.00012 * ir, ir * 1e-6)

    dist_loose = max(2, (distance_samples * 2) // 3)
    dist_ultra = max(2, distance_samples // 2)

    params: Dict[str, float] = {
        "span_2theta": span_tt,
        "median_dtheta": med_dt,
        "min_sep_deg": min_sep_deg,
        "distance_samples": float(distance_samples),
        "prominence_strict": float(prom_strict),
        "prominence_loose": float(prom_loose),
        "prominence_ultra": float(prom_ultra),
        "intensity_ptp": ir,
    }

    if find_peaks is not None:
        pk_s, _ = find_peaks(yy, prominence=prom_strict, distance=distance_samples)
        pk_l, _ = find_peaks(yy, prominence=prom_loose, distance=dist_loose)
        pk_u, _ = find_peaks(yy, prominence=prom_ultra, distance=dist_ultra)
        idx_set = set(int(x) for x in pk_s.tolist())
        for x in pk_l.tolist():
            idx_set.add(int(x))
        for x in pk_u.tolist():
            idx_set.add(int(x))
        merged_idx = np.asarray(sorted(idx_set), dtype=np.int64)
    else:
        merged_idx, _ = _fallback_local_maxima(yy, dist_ultra, prom_ultra)

    # NMS: 억제 반경을 줄여 근접 피크가 서로 잡히도록 (이전 //2 → //3)
    sep = max(2, (distance_samples + 2) // 3)
    keep: List[int] = []
    order_pk = sorted([int(i) for i in merged_idx], key=lambda i: float(yy[i]), reverse=True)
    used = np.zeros(n, dtype=bool)
    for i in order_pk:
        if used[i]:
            continue
        keep.append(i)
        lo = max(0, i - sep)
        hi = min(n - 1, i + sep)
        used[lo : hi + 1] = True

    keep.sort()
    peaks: List[dict] = []
    for i in keep:
        left_seg = yy[max(0, i - sep) : i]
        right_seg = yy[i + 1 : min(n, i + sep + 1)]
        base_l = float(np.min(left_seg)) if left_seg.size else float(yy[i])
        base_r = float(np.min(right_seg)) if right_seg.size else float(yy[i])
        prom_est = float(yy[i]) - min(base_l, base_r)
        tr, yr = refine_peak_parabolic_tt_yy(tt, yy, i)
        # 축 전체 span 대비 후단(tail)에서만 ultra-loose 오탐 억제 (패턴별 각도 하드코딩 없음)
        frac_along = (float(tt[i]) - float(tt[0])) / max(float(span_tt), 1e-12)
        if frac_along >= (1.0 - NUMERIC_PEAK_TAIL_SPAN_FRAC):
            min_prom_tail = max(1.45 * prom_strict, 4.2 * noise_i, 0.006 * ir)
            if prom_est < min_prom_tail:
                continue
        peaks.append(
            {
                "two_theta": round(tr, 6),
                "intensity": round(yr, 6),
                "prominence_est": round(max(prom_est, 0.0), 6),
                "width_samples": float(sep),
                "source": "numeric_curve",
            }
        )

    params["n_peaks_raw"] = float(len(merged_idx))
    params["n_peaks_after_nms"] = float(len(peaks))
    params["tail_prominence_span_frac"] = float(NUMERIC_PEAK_TAIL_SPAN_FRAC)
    return peaks, params

