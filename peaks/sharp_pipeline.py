"""
sharp peak preserve: 곡선 SG / 피크 SG 분리, 국소 prominence, near-peak raw 블렌드, apex 스냅.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from core.sharp_peak_settings import SharpPeakPreserveSettings
from peaks.detect_peaks import detect_peaks_sharp
from peaks.smooth import smooth_for_curve, smooth_for_peak
from trace.postprocess import (
    _refine_peak_y_subpixel,
    _repair_sg_vs_gapfilled,
)


def _render_curve_overlay(
    roi: np.ndarray,
    columns: List[int],
    y_trace: np.ndarray,
    valid: np.ndarray,
    color: Tuple[int, int, int],
) -> np.ndarray:
    h, w = roi.shape[:2]
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    for i, col in enumerate(columns):
        if col >= w or i >= len(valid) or not valid[i]:
            continue
        yy = int(round(float(y_trace[i])))
        if 0 <= yy < h:
            overlay[yy, col] = color
    return overlay


def _render_dual_curve(
    roi: np.ndarray,
    columns: List[int],
    y_a: np.ndarray,
    y_b: np.ndarray,
    valid: np.ndarray,
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
) -> np.ndarray:
    """y_a, y_b 두 곡선."""
    h, w = roi.shape[:2]
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    for i, col in enumerate(columns):
        if col >= w or i >= len(valid) or not valid[i]:
            continue
        for yy, c in (
            (int(round(float(y_a[i]))), color_a),
            (int(round(float(y_b[i]))), color_b),
        ):
            if 0 <= yy < h:
                overlay[yy, col] = c
    return overlay


def _near_peak_mask_image(
    roi_h: int,
    roi_w: int,
    columns: List[int],
    near_peak: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    img = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for i, col in enumerate(columns):
        if col >= roi_w or i >= len(near_peak) or not valid[i]:
            continue
        if near_peak[i]:
            img[:, col] = 180
    return img


def _apex_overlay(
    roi: np.ndarray,
    columns: List[int],
    y_before: np.ndarray,
    y_after: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """스냅 전=자홍, 후=녹색."""
    return _render_dual_curve(
        roi, columns, y_before, y_after, valid,
        (255, 0, 255),
        (0, 255, 80),
    )


def _build_near_peak(global_n: int, peak_indices: List[int], radius: int, valid: np.ndarray) -> np.ndarray:
    m = np.zeros(global_n, dtype=bool)
    for p in peak_indices:
        for j in range(max(0, p - radius), min(global_n, p + radius + 1)):
            if valid[j]:
                m[j] = True
    return m


def _apex_snap_pass(
    y_work: np.ndarray,
    y_raw: np.ndarray,
    y_curve_smooth: np.ndarray,
    peak_indices: List[int],
    preserve_radius: int,
    valid: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """apex에서 raw 복원; apex±1 은 raw·curve 50:50."""
    y_out = np.asarray(y_work, dtype=np.float64).copy()
    deltas: List[float] = []
    n = len(y_raw)
    for p in peak_indices:
        lo = max(0, p - preserve_radius)
        hi = min(n, p + preserve_radius + 1)
        seg = np.asarray(y_raw[lo:hi], dtype=np.float64)
        if seg.size == 0:
            continue
        apex = lo + int(np.argmin(seg))
        old = float(y_out[apex])
        y_out[apex] = float(y_raw[apex])
        deltas.append(abs(float(y_out[apex]) - old))
        for off in (-1, 1):
            j = apex + off
            if 0 <= j < n and valid[j]:
                y_out[j] = 0.5 * float(y_raw[j]) + 0.5 * float(y_curve_smooth[j])
    return y_out, deltas


def _apex_snap_gap_repeat(
    y_work: np.ndarray,
    y_raw: np.ndarray,
    y_curve_smooth: np.ndarray,
    peak_indices: List[int],
    preserve_radius: int,
    gap_filled: Set[int],
    valid: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """피크 ±4 이내에 gap-fill 인덱스가 있으면 해당 피크만 다시 apex 스냅."""
    need: List[int] = []
    gap_adj = set()
    for g in gap_filled:
        for d in range(-4, 5):
            gap_adj.add(g + d)
    for p in peak_indices:
        for j in range(max(0, p - 4), min(len(valid), p + 5)):
            if j in gap_adj:
                need.append(p)
                break
    if not need:
        return y_work, []
    return _apex_snap_pass(
        y_work, y_raw, y_curve_smooth, need, preserve_radius, valid,
    )


def run_sharp_peak_preserve(
    roi: np.ndarray,
    y_filled: np.ndarray,
    valid_mask: np.ndarray,
    gap_filled_set: Set[int],
    columns: List[int],
    roi_w: int,
    roi_h: int,
    cfg: SharpPeakPreserveSettings,
) -> Tuple[np.ndarray, Dict, Dict[str, object]]:
    """
    반환: (y_final, peak_result, debug_artifacts dict for save_debug_files).
    """
    y_raw = np.asarray(y_filled, dtype=np.float64).copy()
    valid = valid_mask.astype(bool)

    y_curve = smooth_for_curve(y_raw, valid, cfg.curve_smooth_window)
    y_curve = _repair_sg_vs_gapfilled(y_raw, y_curve, valid)
    y_peak_s = smooth_for_peak(y_raw, valid, cfg.peak_smooth_window)
    y_peak_s = _repair_sg_vs_gapfilled(y_raw, y_peak_s, valid)

    peak_result = detect_peaks_sharp(
        y_peak_s,
        valid,
        gap_filled_set,
        columns,
        y_peak_s,
        global_prom_ratio=cfg.global_prom_ratio,
        local_prom_window=cfg.local_prom_window,
        local_prom_ratio=cfg.local_prom_ratio,
        local_noise_k=cfg.local_noise_k,
    )
    peak_ix = [int(p["index"]) for p in peak_result.get("peaks", [])]

    near_peak = _build_near_peak(len(y_raw), peak_ix, cfg.peak_preserve_radius, valid)

    w_raw = float(cfg.peak_blend_raw_weight)
    y_final = np.asarray(y_curve, dtype=np.float64).copy()
    for i in range(len(y_final)):
        if not valid[i]:
            continue
        if near_peak[i]:
            y_final[i] = w_raw * float(y_raw[i]) + (1.0 - w_raw) * float(y_curve[i])
        else:
            y_final[i] = float(y_curve[i])

    y_before_snap = y_final.copy()
    y_final, deltas1 = _apex_snap_pass(
        y_final, y_raw, y_curve, peak_ix, cfg.peak_preserve_radius, valid,
    )
    y_final, deltas2 = _apex_snap_gap_repeat(
        y_final,
        y_raw,
        y_curve,
        peak_ix,
        cfg.peak_preserve_radius,
        gap_filled_set,
        valid,
    )

    for p in peak_result.get("peaks", []):
        gi = int(p["index"])
        p["y_pixel"] = float(y_final[gi])
        if columns is not None and len(columns) == len(y_final) and 0 < gi < len(y_final) - 1:
            yr = _refine_peak_y_subpixel(y_final, gi)
            if yr is not None:
                p["y_pixel_refined"] = round(float(yr), 4)

    all_deltas = deltas1 + deltas2
    mean_d = float(np.mean(all_deltas)) if all_deltas else 0.0
    max_d = float(np.max(all_deltas)) if all_deltas else 0.0

    dbg_json = {
        "use_sharp_peak_preserve": True,
        "curve_smooth_window": int(cfg.curve_smooth_window),
        "peak_smooth_window": int(cfg.peak_smooth_window),
        "global_prom_ratio": float(cfg.global_prom_ratio),
        "local_prom_window": int(cfg.local_prom_window),
        "local_prom_ratio": float(cfg.local_prom_ratio),
        "local_noise_k": float(cfg.local_noise_k),
        "peak_preserve_radius": int(cfg.peak_preserve_radius),
        "peak_blend_raw_weight": float(cfg.peak_blend_raw_weight),
        "num_detected_peaks": int(len(peak_result.get("peaks", []))),
        "num_near_peak_columns": int(np.sum(near_peak & valid)),
        "num_apex_snapped": len(all_deltas),
        "mean_apex_delta_px": round(mean_d, 6),
        "max_apex_delta_px": round(max_d, 6),
    }

    dbg: Dict[str, object] = {
        "smooth_for_curve_overlay": _render_curve_overlay(
            roi, columns, y_curve, valid, (0, 200, 255),
        ),
        "smooth_for_peak_overlay": _render_curve_overlay(
            roi, columns, y_peak_s, valid, (255, 180, 0),
        ),
        "detected_peaks_peak_smooth": _render_curve_overlay(
            roi, columns, y_peak_s, valid, (120, 255, 120),
        ),
        "near_peak_mask": _near_peak_mask_image(roi_h, roi_w, columns, near_peak, valid),
        "apex_snapping_overlay": _apex_overlay(roi, columns, y_before_snap, y_final, valid),
        "sharp_peak_preserve_debug": dbg_json,
    }

    peak_ov = dbg["detected_peaks_peak_smooth"]
    if isinstance(peak_ov, np.ndarray) and peak_ix:
        h, w = peak_ov.shape[:2]
        major = {p["index"] for p in peak_result.get("major_peaks", [])}
        for pk in peak_result.get("peaks", []):
            idx = int(pk["index"])
            if idx >= len(columns):
                continue
            col = columns[idx]
            sy = int(round(float(y_peak_s[idx])))
            if not (0 <= sy < h and 0 <= col < w):
                continue
            rad = 3 if idx in major else 2
            color = [255, 40, 40] if idx in major else [255, 240, 60]
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if dy * dy + dx * dx <= rad * rad:
                        yy, xx = sy + dy, col + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            peak_ov[yy, xx] = color
        dbg["detected_peaks_peak_smooth"] = peak_ov

    return y_final, peak_result, dbg
