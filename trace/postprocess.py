"""
$15: Gap fill, Savitzky-Golay smoothing, peak detection.

후처리 복원(스파이크·SG 꼭짓점) 임계값은 패턴 ID가 아니라 **당해 트레이스의 연속성·수직 span**
에서 자동 산출한다(단일 샘플 과적합 완화).
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter, find_peaks
except ImportError:
    savgol_filter = None  # type: ignore
    find_peaks = None  # type: ignore

MAX_GAP_PX = 10
SG_POLYORDER = 2
MAJOR_PEAK_MIN = 3
MAJOR_PEAK_RATIO = 0.10
MAJOR_PEAK_CAP = 8
# Step 3a: 둘째 패스 prominence 비율(첫 패스 대비). 낮을수록 더 많은 후보 후 NMS로 병합.
# 0.62→0.54: 미세 피크(저 prominence) 한 번 더 잡기.
LOOSE_PEAK_PROMINENCE_FACTOR = 0.54


def _peak_best_per_column(
    valid_idx: np.ndarray,
    peaks_inv: np.ndarray,
    props: dict,
) -> Dict[int, Tuple[float, int, dict, int]]:
    """global trace index -> (prominence, pk, props, row_i). 동일 열 중복 시 최대 prominence만 유지."""
    best: Dict[int, Tuple[float, int, dict, int]] = {}
    for row_i, pk in enumerate(peaks_inv):
        gi = int(valid_idx[int(pk)])
        prom = float(props["prominences"][row_i]) if "prominences" in props else 0.0
        prev = best.get(gi)
        if prev is None or prom > prev[0]:
            best[gi] = (prom, int(pk), props, row_i)
    return best


def _merge_peak_maps(
    a: Dict[int, Tuple[float, int, dict, int]],
    b: Dict[int, Tuple[float, int, dict, int]],
) -> Dict[int, Tuple[float, int, dict, int]]:
    out = dict(a)
    for gi, tup in b.items():
        if gi not in out or tup[0] > out[gi][0]:
            out[gi] = tup
    return out


def _nms_peaks_by_index(
    merged: Dict[int, Tuple[float, int, dict, int]],
    min_sep: int,
) -> List[int]:
    """인덱스 거리 min_sep 이내는 prominence 최대 1개만 대표로 남김."""
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


def _refine_peak_y_subpixel(y_trace: np.ndarray, idx: int) -> float:
    """
    trace 상에서만 이차 보간으로 피크 세로 위치 보정 (열은 그대로).
    two_theta 는 columns[idx] 로 유지해 major_peak_x 평가가 흔들리지 않게 한다.
    """
    if idx <= 0 or idx >= len(y_trace) - 1:
        return float(y_trace[idx])
    y0, y1, y2 = float(y_trace[idx - 1]), float(y_trace[idx]), float(y_trace[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return y1
    di = float(np.clip(0.5 * (y0 - y2) / denom, -0.85, 0.85))
    a = 0.5 * denom
    b = 0.5 * (y2 - y0)
    return float(a * di * di + b * di + y1)


def _nearest_odd(x: float) -> int:
    n = int(round(x))
    return n if n % 2 == 1 else n + 1


def _compute_sg_window(pw: int) -> int:
    # 계수 0.010·상한 7이던 SG 창을 최대 5로 한 번 더 제한 — 좁은 피크 합쳐짐 완화.
    raw = max(5, min(7, round(0.010 * pw)))
    capped = min(raw, 5)
    return _nearest_odd(max(5, capped))


def _trace_step_median_abs(y: np.ndarray, valid: np.ndarray) -> float:
    """유효 열을 따라 연속한 한 칸 |Δy| 중앙값 — 로컬 기울기·잡음 스케일."""
    y = np.asarray(y, dtype=np.float64)
    v = valid.astype(bool, copy=False)
    diffs: List[float] = []
    prev: Optional[int] = None
    for i in range(len(y)):
        if not v[i]:
            prev = None
            continue
        if prev is not None:
            diffs.append(abs(float(y[i]) - float(y[prev])))
        prev = i
    if not diffs:
        return 1.0
    return float(max(float(np.median(np.asarray(diffs, dtype=np.float64))), 1e-6))


def _trace_vertical_span_robust(y: np.ndarray, valid: np.ndarray) -> float:
    """y(행) 범위의 견고한 span — 극단 픽셀 한두 개 영향 완화."""
    yv = np.asarray(y, dtype=np.float64)[valid.astype(bool, copy=False)]
    if yv.size < 4:
        return 1.0
    lo, hi = np.percentile(yv, [4.0, 96.0])
    return float(max(hi - lo, 1e-6))


def fidelity_thresholds_from_trace(
    y_for_scale: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    """
    gap-fill / 스무딩 복원용 스케일. 패턴별 하드코딩 없이 트레이스 통계만 사용.

    반환: radius, min_prominence_px, min_lift_px, spike_delta_px, step_median, span
    """
    span = _trace_vertical_span_robust(y_for_scale, valid)
    noise_raw = _trace_step_median_abs(y_for_scale, valid)
    # 긴 평탄 구간만 있으면 median(|Δy|)→0 이므로 span 비례 바닥을 둔다.
    noise = float(max(noise_raw, 0.0035 * span, 0.20))
    n_valid = int(np.count_nonzero(valid))
    radius = int(np.clip(round(0.012 * max(n_valid, 12)), 4, 10))
    min_prom = max(1.28 * noise, 0.0016 * span, 0.38)
    min_lift = max(0.28 * noise, 0.08)
    spike_delta = float(np.clip(max(6.5, 5.0 * noise), 6.0, 20.0))
    return {
        "radius": float(radius),
        "min_prominence_px": float(min_prom),
        "min_lift_px": float(min_lift),
        "spike_delta_px": float(spike_delta),
        "step_median_abs": float(noise_raw),
        "continuity_scale_px": float(noise),
        "span_robust_px": float(span),
    }


def restore_peaks_lowered_by_smoothing(
    y_filled: np.ndarray,
    y_smooth: np.ndarray,
    valid: np.ndarray,
    *,
    radius: Optional[int] = None,
    min_prominence_px: Optional[float] = None,
    min_lift_px: Optional[float] = None,
) -> np.ndarray:
    """
    gap-fill(DP) 경로의 국소 최소 y(= 차트에서 피크 꼭짓점)에서 SG가 y를 키워
    작은 피크가 원본보다 낮게 깔리면, 해당 열은 gap-fill 높이로 맞춘다.

    베이스라인 잡음은 양옆 어깨 대비 prominence가 작으면 건너뛴다.

    radius / min_prominence_px / min_lift_px 가 None 이면 `y_filled`에서
    `fidelity_thresholds_from_trace`로 자동 산출(패턴별 튜닝 지양).
    """
    out = np.asarray(y_smooth, dtype=np.float64).copy()
    yf = np.asarray(y_filled, dtype=np.float64)
    v = valid.astype(bool, copy=False)
    n = len(out)
    if radius is None or min_prominence_px is None or min_lift_px is None:
        th = fidelity_thresholds_from_trace(yf, v)
        R = int(radius if radius is not None else th["radius"])
        mp = float(min_prominence_px if min_prominence_px is not None else th["min_prominence_px"])
        ml = float(min_lift_px if min_lift_px is not None else th["min_lift_px"])
    else:
        R = int(radius)
        mp = float(min_prominence_px)
        ml = float(min_lift_px)
    R = max(2, R)
    for i in range(n):
        if not v[i]:
            continue
        lo = max(0, i - R)
        hi = min(n - 1, i + R)
        yfi = float(yf[i])
        local_min = min(float(yf[j]) for j in range(lo, hi + 1) if v[j])
        if yfi > local_min + 1e-9:
            continue
        left_vals = [float(yf[j]) for j in range(lo, i) if v[j]]
        right_vals = [float(yf[j]) for j in range(i + 1, hi + 1) if v[j]]
        if not left_vals or not right_vals:
            continue
        shoulder = max(max(left_vals), max(right_vals))
        prom = shoulder - yfi
        if prom < mp:
            continue
        if float(out[i]) > yfi + ml:
            out[i] = yfi
    return out


def repair_isolated_spike_down_y(
    y: np.ndarray,
    valid: np.ndarray,
    *,
    delta_px: Optional[float] = None,
) -> np.ndarray:
    """
    ROI에서 y(행)가 클수록 차트 아래쪽(저강도). 한 칸만 이웃보다 크게 아래로 튀면
    플롯에 고립된 점/찍힘으로 보인다. 양옆이 유효할 때만 이웃 평균으로 메운다.

    delta_px 가 None 이면 `fidelity_thresholds_from_trace`의 spike_delta_px 사용.
    """
    out = np.asarray(y, dtype=np.float64).copy()
    v = valid.astype(bool, copy=False)
    if delta_px is None:
        delta_px = float(fidelity_thresholds_from_trace(out, v)["spike_delta_px"])
    dpx = float(delta_px)
    n = len(out)
    for i in range(1, n - 1):
        if not v[i] or not v[i - 1] or not v[i + 1]:
            continue
        a, b, c = float(out[i - 1]), float(out[i]), float(out[i + 1])
        m = 0.5 * (a + c)
        if b > a + dpx and b > c + dpx:
            out[i] = m
    return out


def _repair_sg_vs_gapfilled(
    y_filled: np.ndarray,
    y_sg: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    ROI 행 y는 작을수록 차트 위(고강도). SG가 피크를 뭉개면 y가 커진다.
    gap-fill과 SG 중 **더 작은 y**(강도 상한에 가까운 쪽)를 취해 주피크·미세 피크 높이를 보존한다.
    (잘못 np.maximum 을 쓰면 큰 y를 택해 강도가 전반적으로 낮아진다.)
    """
    out = y_sg.copy()
    v = valid.astype(bool, copy=False)
    out[v] = np.minimum(y_sg[v], y_filled[v])
    return out


def gap_fill(
    path: List[Optional[int]],
    columns: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    $15.1: linear interpolation for gaps <= 10 px.
    Returns: (y_filled, valid_mask, gap_ranges)
    """
    n = len(columns)
    y_arr = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i, y in enumerate(path):
        if y is not None:
            y_arr[i] = float(y)
            valid[i] = True

    gap_ranges: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not valid[i]:
            gap_start = i
            while i < n and not valid[i]:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len <= MAX_GAP_PX:
                left_val = y_arr[gap_start - 1] if gap_start > 0 and valid[gap_start - 1] else None
                right_val = y_arr[gap_end] if gap_end < n and valid[gap_end] else None

                if left_val is not None and right_val is not None:
                    for j in range(gap_start, gap_end):
                        t = (j - gap_start + 1) / (gap_len + 1)
                        y_arr[j] = left_val + t * (right_val - left_val)
                        valid[j] = True
                    gap_ranges.append((gap_start, gap_end))
                elif left_val is not None:
                    for j in range(gap_start, gap_end):
                        y_arr[j] = left_val
                        valid[j] = True
                    gap_ranges.append((gap_start, gap_end))
                elif right_val is not None:
                    for j in range(gap_start, gap_end):
                        y_arr[j] = right_val
                        valid[j] = True
                    gap_ranges.append((gap_start, gap_end))
        else:
            i += 1

    return y_arr, valid, gap_ranges


def smooth_trace(
    y_filled: np.ndarray,
    valid: np.ndarray,
    pw: int,
) -> Tuple[np.ndarray, int]:
    """
    $15.2: Savitzky-Golay smoothing with peak-preservation retry.
    Returns: (y_smoothed, final_window)
    """
    if savgol_filter is None:
        return y_filled.copy(), 0

    window = _compute_sg_window(pw)
    valid_y = y_filled[valid]

    if len(valid_y) < window:
        return y_filled.copy(), 0

    raw_peaks_before = _count_raw_peaks(valid_y)

    for attempt in range(4):
        if window < 5:
            break
        smoothed_valid = savgol_filter(valid_y, window, SG_POLYORDER)
        peaks_after = _count_raw_peaks(smoothed_valid)

        if peaks_after >= raw_peaks_before * 0.65:
            result = y_filled.copy()
            result[valid] = smoothed_valid
            return _repair_sg_vs_gapfilled(y_filled, result, valid), window

        window = max(5, window - 2)
        window = _nearest_odd(window)

    result = y_filled.copy()
    result[valid] = savgol_filter(valid_y, window, SG_POLYORDER)
    return _repair_sg_vs_gapfilled(y_filled, result, valid), window


def blend_sg_toward_gapfill_on_high_curvature(
    y_filled: np.ndarray,
    y_smooth: np.ndarray,
    valid: np.ndarray,
    *,
    strength: float = 0.32,
) -> np.ndarray:
    """
    gap-fill 경로의 이산 곡률(|Δ²y|)이 큰 인덱스에서 스무딩을 원 경로 쪽으로 당겨
    근접 피크가 한 덩어리로 보이는 현상을 완화한다 (로드맵 2단계 경량).

    strength 는 최대 블렌드 비율; 트레이스 전역 상수가 아닌 곡률 정규화만 사용.
    """
    out = np.asarray(y_smooth, dtype=np.float64).copy()
    yf = np.asarray(y_filled, dtype=np.float64)
    v = valid.astype(bool, copy=False)
    n = len(out)
    d2 = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        if not v[i] or not v[i - 1] or not v[i + 1]:
            continue
        d2[i] = abs(float(yf[i - 1]) - 2.0 * float(yf[i]) + float(yf[i + 1]))
    active = d2 > 0
    if not np.any(active):
        return out
    q95 = float(np.percentile(d2[active], 95))
    if q95 < 1e-9:
        return out
    st = float(np.clip(strength, 0.0, 1.0))
    for i in range(1, n - 1):
        if not v[i]:
            continue
        w = min(1.0, float(d2[i]) / q95)
        blend_w = st * (w**1.15)
        out[i] = (1.0 - blend_w) * float(out[i]) + blend_w * float(yf[i])
    return out


def _count_raw_peaks(y: np.ndarray) -> int:
    if find_peaks is None or len(y) < 5:
        return 0
    peaks, _ = find_peaks(y * -1, distance=3)
    return len(peaks)


def detect_peaks(
    y_smoothed: np.ndarray,
    valid: np.ndarray,
    gap_filled_indices: Optional[set] = None,
    columns: Optional[List[int]] = None,
    *,
    two_pass_prominence: bool = True,
    loose_prominence_factor: float = LOOSE_PEAK_PROMINENCE_FACTOR,
) -> dict:
    """
    $15.3-15.4: peak detection + major peak selection.
    Returns dict with peaks, major_peaks, and params.
    """
    if find_peaks is None:
        return {"peaks": [], "major_peaks": [], "params": {}}

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 5:
        return {"peaks": [], "major_peaks": [], "params": {}}

    y_valid = y_smoothed[valid_idx]
    y_inverted = -y_valid

    y_max = float(np.max(y_valid))
    y_min = float(np.min(y_valid))
    y_range = y_max - y_min
    if y_range < 1e-6:
        return {"peaks": [], "major_peaks": [], "params": {}}

    num_points = len(y_valid)
    sigma_local = _estimate_local_noise(y_valid)
    local_noise_floor = max(3.0 * sigma_local, 0.01 * y_range)
    prominence_thresh = max(0.07 * y_range, local_noise_floor)
    min_peak_distance = max(3, round(0.004 * num_points))
    min_peak_height_abs = y_min + 0.03 * y_range

    max_height_inverted = -(min_peak_height_abs)

    peak_kw = dict(
        distance=min_peak_distance,
        height=(None, max_height_inverted),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks_s, props_s = find_peaks(
            y_inverted,
            prominence=prominence_thresh,
            **peak_kw,
        )
        if two_pass_prominence:
            prom_loose = max(prominence_thresh * float(loose_prominence_factor), local_noise_floor * 0.85)
            peaks_l, props_l = find_peaks(
                y_inverted,
                prominence=prom_loose,
                **peak_kw,
            )
        else:
            peaks_l, props_l = np.array([], dtype=np.int64), {}

    merged = _peak_best_per_column(valid_idx, peaks_s, props_s)
    n_strict = len(merged)
    if two_pass_prominence and peaks_l.size:
        loose_map = _peak_best_per_column(valid_idx, peaks_l, props_l)
        merged = _merge_peak_maps(merged, loose_map)
    n_after_union = len(merged)
    winners = _nms_peaks_by_index(merged, min_peak_distance)
    n_after_nms = len(winners)

    peaks_info = []
    for global_idx in winners:
        prom, pk, props, row_i = merged[global_idx]
        y_val = float(y_valid[pk])

        is_gap_filled = False
        if gap_filled_indices and global_idx in gap_filled_indices:
            is_gap_filled = True

        left_base = int(props["left_bases"][row_i]) if "left_bases" in props else 0
        right_base = int(props["right_bases"][row_i]) if "right_bases" in props else len(y_valid) - 1
        half_width = (right_base - left_base) / 2.0
        symmetry = 1.0 - abs((pk - left_base) - (right_base - pk)) / max(half_width * 2, 1)
        sharpness = prom / max(half_width, 1)
        quality = 0.5 * min(prom / max(y_range * 0.1, 1), 1.0) + 0.3 * symmetry + 0.2 * min(sharpness / 5.0, 1.0)
        if is_gap_filled:
            quality *= 0.5

        y_refined: Optional[float] = None
        if columns is not None and len(columns) == len(y_smoothed) and 0 < global_idx < len(y_smoothed) - 1:
            y_refined = _refine_peak_y_subpixel(y_smoothed, global_idx)

        entry = {
            "index": global_idx,
            "y_pixel": y_val,
            "prominence": round(prom, 4),
            "quality": round(float(quality), 4),
            "is_gap_filled": is_gap_filled,
        }
        if y_refined is not None:
            entry["y_pixel_refined"] = round(float(y_refined), 4)
        peaks_info.append(entry)

    peaks_info.sort(key=lambda p: -p["prominence"])

    n_detected = len(peaks_info)
    n_major = min(MAJOR_PEAK_CAP, max(MAJOR_PEAK_MIN, math.ceil(MAJOR_PEAK_RATIO * n_detected)))
    major_peaks = peaks_info[:n_major]

    return {
        "peaks": peaks_info,
        "major_peaks": major_peaks,
        "params": {
            "y_range": round(y_range, 2),
            "sigma_local": round(sigma_local, 4),
            "local_noise_floor": round(local_noise_floor, 4),
            "prominence_thresh": round(prominence_thresh, 4),
            "min_peak_distance": min_peak_distance,
            "min_peak_height": round(float(min_peak_height_abs), 4),
            "num_points": num_points,
            "two_pass_prominence": bool(two_pass_prominence),
            "loose_prominence_factor": round(float(loose_prominence_factor), 4),
            "num_peaks_strict_unique": n_strict,
            "num_peaks_after_union": n_after_union,
            "num_peaks_after_nms": n_after_nms,
            "num_peaks_detected": n_detected,
            "num_major_peaks": len(major_peaks),
        },
    }


def _estimate_local_noise(y: np.ndarray, window: int = 30) -> float:
    """Adaptive local noise: windowed std, then median across windows."""
    if len(y) < window:
        return float(np.std(np.diff(y))) if len(y) > 2 else 0.0

    local_stds = []
    for i in range(0, len(y) - window + 1, window // 2):
        segment = y[i:i + window]
        diff = np.diff(segment)
        local_stds.append(float(np.std(diff)))

    return float(np.median(local_stds)) if local_stds else 0.0


def render_smoothed_trace(
    roi: np.ndarray,
    columns: List[int],
    y_filled: np.ndarray,
    y_smoothed: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """smoothed_trace.png: raw=blue, smoothed=green."""
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    h, w = overlay.shape[:2]

    for i, col in enumerate(columns):
        if col >= w or i >= len(valid) or not valid[i]:
            continue

        ry = int(round(y_filled[i]))
        if 0 <= ry < h:
            overlay[ry, col] = [100, 100, 255]

        sy = int(round(y_smoothed[i]))
        if 0 <= sy < h:
            overlay[sy, col] = [0, 220, 0]

    return overlay


def render_peaks_overlay(
    roi: np.ndarray,
    columns: List[int],
    y_smoothed: np.ndarray,
    valid: np.ndarray,
    peaks: List[dict],
    major_peaks: List[dict],
) -> np.ndarray:
    """peaks_overlay.png: smoothed curve + all peaks (yellow) + major peaks (red)."""
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    h, w = overlay.shape[:2]

    for i, col in enumerate(columns):
        if col >= w or i >= len(valid) or not valid[i]:
            continue
        sy = int(round(y_smoothed[i]))
        if 0 <= sy < h:
            overlay[sy, col] = [0, 200, 0]

    major_indices = {p["index"] for p in major_peaks}
    all_peak_indices = {p["index"] for p in peaks}

    for pk in peaks:
        idx = pk["index"]
        if idx >= len(columns):
            continue
        col = columns[idx]
        if col >= w:
            continue
        sy = int(round(y_smoothed[idx]))
        if not (0 <= sy < h):
            continue

        is_major = idx in major_indices
        color = [255, 50, 50] if is_major else [255, 220, 50]
        radius = 3 if is_major else 2

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    yy, xx = sy + dy, col + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        overlay[yy, xx] = color

    return overlay


def smooth_trace_experimental(
    y_filled: np.ndarray,
    valid: np.ndarray,
    pw: int,
    s_h: float,
    params: Optional[dict] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    pp = (params or {})
    smooth_k_thr = float(pp.get("smooth_k_thr", 1.5))
    smooth_p_thr = float(pp.get("smooth_p_thr", 0.40))
    """§9.3: 곡률/돌출도 gate 기반 선택적 Savitzky-Golay."""
    if savgol_filter is None:
        return y_filled.copy(), 0, np.zeros_like(valid, dtype=np.uint8)

    window = _compute_sg_window(pw)
    if np.sum(valid) < window:
        return y_filled.copy(), 0, np.zeros_like(valid, dtype=np.uint8)

    y = y_filled.copy()
    y_valid = y[valid]
    gate = np.zeros_like(valid, dtype=np.uint8)
    if len(y_valid) < 3:
        return y, 0, gate
    yr = float(np.nanmax(y_valid) - np.nanmin(y_valid) + 1e-6)
    r = max(5, round(0.006 * pw))

    idx = np.where(valid)[0]
    for i, col in enumerate(idx):
        if i <= 0 or i >= len(idx) - 1:
            continue
        k_t = abs(y[idx[i + 1]] - 2 * y[col] + y[idx[i - 1]]) / max(1.0, s_h)
        li = max(0, i - r)
        ri = min(len(idx) - 1, i + r)
        if li == i or ri == i:
            continue
        min_left = float(np.nanmin(y[idx[li:i]]))
        min_right = float(np.nanmin(y[idx[i + 1 : ri + 1]]))
        p_t = max(0.0, float(y[col]) - 0.5 * (min_left + min_right)) / yr
        if k_t < smooth_k_thr and p_t < smooth_p_thr:
            gate[col] = 1

    y_out = y.copy()
    start = None
    for i in range(len(valid) + 1):
        on = i < len(valid) and gate[i] == 1 and valid[i]
        if on and start is None:
            start = i
        if (not on) and start is not None:
            end = i - 1
            seg_idx = np.where(valid[start : end + 1])[0] + start
            if len(seg_idx) >= window:
                seg = y[seg_idx]
                sm = savgol_filter(seg, window, SG_POLYORDER)
                # 경계 2 샘플 블렌드
                for k, si in enumerate(seg_idx):
                    if k < 2:
                        a = (k + 1) / 3.0
                        y_out[si] = (1 - a) * y[si] + a * sm[k]
                    elif k >= len(seg_idx) - 2:
                        a = (len(seg_idx) - k) / 3.0
                        y_out[si] = (1 - a) * y[si] + a * sm[k]
                    else:
                        y_out[si] = sm[k]
            start = None

    return y_out, window, gate


def detect_peaks_experimental(
    y_raw: np.ndarray,
    y_smoothed: np.ndarray,
    valid: np.ndarray,
    gap_filled_indices: Optional[set] = None,
    params: Optional[dict] = None,
) -> dict:
    pp = (params or {})
    nms_x_scale = float(pp.get("nms_x_scale", 0.0045))
    raw_prom_ratio = float(pp.get("raw_prom_ratio", 0.85))
    """§9.4~9.5: raw/smoothed 동시 검출 후 NMS 병합."""
    if find_peaks is None:
        return {"peaks": [], "major_peaks": [], "params": {}}
    idx = np.where(valid)[0]
    if len(idx) < 5:
        return {"peaks": [], "major_peaks": [], "params": {}}

    def _detect(y_arr: np.ndarray, tag: str) -> List[dict]:
        yv = y_arr[idx]
        yr = float(np.max(yv) - np.min(yv))
        if yr < 1e-6:
            return []
        pp = params or {}
        pscale = float(pp.get("peak_prominence_scale", 1.0))
        nscale = float(pp.get("peak_noise_scale", 1.0))
        sigma_local = _estimate_local_noise(yv)
        local_noise_floor = max(3.0 * sigma_local * nscale, 0.01 * yr)
        prom_thr = max(0.07 * yr * pscale, local_noise_floor)
        min_dist = max(3, round(0.004 * len(yv)))
        min_h = float(np.min(yv) + 0.03 * yr)
        pk, props = find_peaks(-yv, prominence=prom_thr, distance=min_dist, height=(None, -min_h))
        out = []
        for i, p in enumerate(pk):
            gi = int(idx[p])
            prom = float(props["prominences"][i]) if "prominences" in props else 0.0
            out.append({"index": gi, "prominence": prom, "source": tag, "y_pixel": float(y_arr[gi])})
        return out

    raw = _detect(y_raw, "raw")
    sm = _detect(y_smoothed, "smoothed")
    merged = sorted(raw + sm, key=lambda p: p["index"])
    nms_x = max(3, round(nms_x_scale * len(idx)))
    clusters: List[List[dict]] = []
    for p in merged:
        if not clusters or abs(p["index"] - clusters[-1][-1]["index"]) > nms_x:
            clusters.append([p])
        else:
            clusters[-1].append(p)

    peaks = []
    for c in clusters:
        c_sorted = sorted(c, key=lambda x: -x["prominence"])
        cand = c_sorted[0]
        if len(c_sorted) >= 2:
            a, b = c_sorted[0], c_sorted[1]
            if a["source"] != b["source"]:
                if a["source"] == "raw":
                    cand = a if a["prominence"] >= raw_prom_ratio * b["prominence"] else b
                else:
                    cand = b if b["prominence"] >= raw_prom_ratio * a["prominence"] else a
        gi = int(cand["index"])
        peaks.append(
            {
                "index": gi,
                "y_pixel": float(y_raw[gi]),
                "prominence": round(float(cand["prominence"]), 4),
                "quality": 1.0 if cand["source"] == "raw" else 0.8,
                "is_gap_filled": bool(gap_filled_indices and gi in gap_filled_indices),
                "source": cand["source"],
            }
        )

    peaks.sort(key=lambda p: -p["prominence"])
    n_detected = len(peaks)
    n_major = min(MAJOR_PEAK_CAP, max(MAJOR_PEAK_MIN, math.ceil(MAJOR_PEAK_RATIO * n_detected)))
    major = peaks[:n_major]
    return {
        "peaks": peaks,
        "major_peaks": major,
        "params": {
            "num_points": int(len(idx)),
            "num_peaks_detected": n_detected,
            "num_major_peaks": len(major),
            "nms_x": int(nms_x),
        },
    }
