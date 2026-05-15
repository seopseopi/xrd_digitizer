"""
Edge-only trace repair v2: detect anomalies at ROI left/right margins and apply small corrections.

Uses only runtime-safe signals (selected path y, candidate counts, column geometry).
Does not use GT, source_numeric, or oracle labels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _finite_vals(path: Sequence[Optional[int]], cols: range) -> Tuple[np.ndarray, List[int]]:
    ys: List[float] = []
    idx: List[int] = []
    for c in cols:
        if c < 0 or c >= len(path):
            continue
        y = path[c]
        if y is None:
            continue
        ys.append(float(y))
        idx.append(int(c))
    return np.asarray(ys, dtype=np.float64), idx


def _robust_med_sigma(vals: np.ndarray) -> Tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    return med, sigma


def _detector_side(
    path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
    *,
    roi_w: int,
    edge_n: int,
    stable_gap: int,
    stable_width: int,
    side: str,
    min_abs_dev_px: float,
    z_thresh: float,
    min_stable_points: int,
    sigma_floor_px: float,
) -> Dict[str, Any]:
    """side in {'left','right'}"""
    edge_n = int(edge_n)
    stable_gap = int(stable_gap)
    stable_width = int(stable_width)
    if side == "left":
        edge_cols = range(0, min(edge_n, roi_w))
        s0 = edge_n + stable_gap
        s1 = min(s0 + stable_width, roi_w)
        stable_cols = range(s0, s1)
        boundary_edge = edge_n - 1 if edge_n > 0 else 0
        boundary_stable = s0 if s0 < roi_w else max(0, roi_w - 1)
    else:
        # Layout (left → right): … | stable_width | stable_gap | edge_n |
        edge_cols = range(max(0, roi_w - edge_n), roi_w)
        r_hi_excl = roi_w - edge_n - stable_gap
        r_lo = max(0, r_hi_excl - stable_width)
        stable_cols = range(r_lo, max(r_lo, r_hi_excl))
        boundary_stable = max(r_lo, r_hi_excl - 1) if r_hi_excl > r_lo else r_lo
        boundary_edge = max(0, roi_w - edge_n)

    edge_y, edge_ix = _finite_vals(path, edge_cols)
    stab_y, stab_ix = _finite_vals(path, stable_cols)

    cand_counts_edge = [len(final_cands.get(c, []) or []) for c in edge_cols]
    cand_counts_stable = [len(final_cands.get(c, []) or []) for c in stable_cols]

    warnings: List[str] = []
    if stab_y.size < int(min_stable_points):
        warnings.append(f"{side}: insufficient_stable_points ({stab_y.size}<{min_stable_points})")

    med_s, sig_s = _robust_med_sigma(stab_y)
    sigma_eff = float(max(sig_s, float(sigma_floor_px)))

    median_edge = float(np.median(edge_y)) if edge_y.size else float("nan")
    dev = float(median_edge - med_s) if np.isfinite(median_edge) and np.isfinite(med_s) else float("nan")

    be_y = path[boundary_edge] if 0 <= boundary_edge < len(path) else None
    bs_y = path[boundary_stable] if 0 <= boundary_stable < len(path) else None
    jump = (
        abs(float(be_y) - float(bs_y))
        if be_y is not None and bs_y is not None
        else float("nan")
    )

    slope_edge = float("nan")
    if edge_y.size >= 2:
        slope_edge = float((edge_y[-1] - edge_y[0]) / max(edge_ix[-1] - edge_ix[0], 1))

    low_cand_edge = int(np.sum(np.asarray(cand_counts_edge, dtype=np.int64) <= 1))
    if edge_y.size and low_cand_edge >= max(1, edge_n // 2):
        warnings.append(f"{side}: candidate_starvation_on_edge_columns ({low_cand_edge}/{len(cand_counts_edge)})")

    z_score = abs(dev) / sigma_eff if sigma_eff > 1e-9 and np.isfinite(dev) else float("nan")
    thresh = max(float(min_abs_dev_px), float(z_thresh) * sigma_eff)

    anomaly = False
    if (
        stab_y.size >= int(min_stable_points)
        and edge_y.size >= 2
        and np.isfinite(dev)
        and abs(dev) > thresh
    ):
        anomaly = True
    # Secondary: large boundary jump
    if np.isfinite(jump) and jump > max(float(min_abs_dev_px), float(z_thresh) * sigma_eff):
        anomaly = True

    return {
        "side": side,
        "edge_columns": [int(c) for c in edge_cols],
        "stable_columns": [int(c) for c in stable_cols],
        "stable_median_y": float(med_s) if np.isfinite(med_s) else None,
        "stable_sigma_robust": sig_s,
        "stable_sigma_effective": sigma_eff,
        "median_edge_y": median_edge,
        "deviation_median_edge_minus_stable": dev,
        "z_score_deviation": z_score,
        "threshold_abs": thresh,
        "boundary_edge_col": int(boundary_edge),
        "boundary_stable_col": int(boundary_stable),
        "boundary_jump_abs": jump,
        "edge_local_slope": slope_edge,
        "edge_candidate_counts": cand_counts_edge,
        "stable_candidate_counts_summary": {
            "mean": float(np.mean(cand_counts_stable)) if cand_counts_stable else float("nan"),
            "min": int(np.min(cand_counts_stable)) if cand_counts_stable else 0,
        },
        "low_candidate_columns_edge": int(low_cand_edge),
        "anomaly": bool(anomaly),
        "warnings": warnings,
    }


def detect_edge_trace_repair_v2(
    path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
    *,
    roi_w: int,
    roi_h: int,
    edge_n: int = 5,
    stable_gap: int = 5,
    stable_width: int = 32,
    min_abs_dev_px: float = 8.0,
    z_thresh: float = 3.0,
    min_stable_points: int = 12,
    sigma_floor_px: float = 3.0,
) -> Dict[str, Any]:
    left = _detector_side(
        path,
        final_cands,
        roi_w=roi_w,
        edge_n=edge_n,
        stable_gap=stable_gap,
        stable_width=stable_width,
        side="left",
        min_abs_dev_px=min_abs_dev_px,
        z_thresh=z_thresh,
        min_stable_points=min_stable_points,
        sigma_floor_px=sigma_floor_px,
    )
    right = _detector_side(
        path,
        final_cands,
        roi_w=roi_w,
        edge_n=edge_n,
        stable_gap=stable_gap,
        stable_width=stable_width,
        side="right",
        min_abs_dev_px=min_abs_dev_px,
        z_thresh=z_thresh,
        min_stable_points=min_stable_points,
        sigma_floor_px=sigma_floor_px,
    )
    return {
        "left": left,
        "right": right,
        "runtime_safe": True,
        "uses_gt": False,
        "uses_source_numeric": False,
        "roi_w": int(roi_w),
        "roi_h": int(roi_h),
        "edge_n": int(edge_n),
        "stable_gap": int(stable_gap),
        "stable_width": int(stable_width),
    }


def _fit_stable_slope_intercept(
    path: List[Optional[int]], stable_cols: range
) -> Tuple[float, float, bool]:
    xs: List[float] = []
    ys: List[float] = []
    for c in stable_cols:
        if c < 0 or c >= len(path):
            continue
        y = path[c]
        if y is None:
            continue
        xs.append(float(c))
        ys.append(float(y))
    if len(xs) < 3:
        med = float(np.median(np.asarray(ys, dtype=np.float64))) if ys else float("nan")
        return 0.0, med, False
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    m, b = np.polyfit(x_arr, y_arr, 1)
    slope = float(m)
    intercept = float(b)
    unstable = abs(slope) > 4.0  # px per column — avoid wild extrapolation
    return slope, intercept, unstable


def evaluate_peak_edge_guard_side(
    path: List[Optional[int]],
    det_side: Dict[str, Any],
    *,
    side: str,
    roi_w: int,
    edge_n: int,
    peak_edge_window: int,
    curvature_thresh: float,
    prominence_thresh: float,
) -> Dict[str, Any]:
    """
    Runtime-safe peak-like structure near an edge band (curvature + prominence vs stable).
    Does not use GT or source_numeric.
    """
    edge_n = int(edge_n)
    pw = max(3, int(peak_edge_window))
    roi_w = int(roi_w)
    if side == "left":
        lo = 0
        hi_excl = min(edge_n + pw, roi_w)
    else:
        lo = max(0, roi_w - edge_n - pw)
        hi_excl = roi_w

    cols_scan: List[int] = []
    for c in range(lo, hi_excl):
        if c < 0 or c >= len(path):
            continue
        if path[c] is None:
            continue
        cols_scan.append(int(c))

    med_s = det_side.get("stable_median_y")
    sig_s = float(det_side.get("stable_sigma_effective") or 0.0)
    sigma_eff = float(max(sig_s, 3.0))

    curvature_max = 0.0
    if len(cols_scan) >= 3:
        for a, b, cc in zip(cols_scan, cols_scan[1:], cols_scan[2:]):
            if b != a + 1 or cc != b + 1:
                continue
            ya = path[a]
            yb = path[b]
            yc = path[cc]
            if ya is None or yb is None or yc is None:
                continue
            d2 = abs(float(ya) - 2.0 * float(yb) + float(yc))
            curvature_max = max(curvature_max, d2)

    prominence_ratio = 0.0
    if cols_scan and med_s is not None and np.isfinite(float(med_s)):
        m0 = float(med_s)
        devs = []
        for c in cols_scan:
            y = path[c]
            if y is None:
                continue
            devs.append(abs(float(y) - m0))
        if devs:
            prominence_ratio = float(max(devs) / max(sigma_eff, 1e-9))

    peak_like = False
    reasons: List[str] = []
    if curvature_max >= float(curvature_thresh):
        peak_like = True
        reasons.append(f"high_curvature({curvature_max:.2f}>={curvature_thresh})")
    if prominence_ratio >= float(prominence_thresh):
        peak_like = True
        reasons.append(f"high_prominence_ratio({prominence_ratio:.2f}>={prominence_thresh})")

    score = max(
        curvature_max / max(float(curvature_thresh), 1e-9),
        prominence_ratio / max(float(prominence_thresh), 1e-9),
    )

    return {
        "side": side,
        "peak_like": bool(peak_like),
        "peak_like_score": float(score),
        "curvature_max_abs_d2": float(curvature_max),
        "prominence_ratio_vs_stable_sigma": float(prominence_ratio),
        "analysis_columns_span": [lo, hi_excl - 1],
        "reasons": reasons,
        "uses_gt": False,
        "uses_source_numeric": False,
    }


def _repair_side(
    path: List[Optional[int]],
    det: Dict[str, Any],
    *,
    side: str,
    mode: str,
    edge_n: int,
    roi_w: int,
    roi_h: int,
    max_repair_delta_px: float,
    med_stable: float,
) -> Tuple[List[Optional[int]], List[int], float, List[str]]:
    out = list(path)
    changed: List[int] = []
    max_d = 0.0
    warns: List[str] = []
    if not det.get("anomaly"):
        return out, changed, max_d, warns
    if not np.isfinite(float(med_stable)):
        warns.append(f"{side}: skip_repair_invalid_stable_median")
        return out, changed, max_d, warns

    edge_cols = det["edge_columns"]
    sc = det.get("stable_columns") or []
    if len(sc) >= 2:
        stable_cols = range(int(sc[0]), int(sc[-1]) + 1)
    elif len(sc) == 1:
        stable_cols = range(int(sc[0]), int(sc[0]) + 1)
    else:
        stable_cols = range(0, 0)

    slope, intercept, unstable = _fit_stable_slope_intercept(out, stable_cols)
    use_clamp = mode == "clamp_to_stable_median" or (mode == "hybrid" and unstable)

    for c in edge_cols:
        if c < 0 or c >= len(out):
            continue
        cur = out[c]
        if cur is None:
            continue
        cur_f = float(cur)
        if use_clamp or mode == "clamp_to_stable_median":
            tgt = float(med_stable)
        elif mode == "linear_extrapolate_from_stable":
            tgt = float(slope * float(c) + intercept)
        else:  # hybrid
            tgt = float(slope * float(c) + intercept)

        delta = tgt - cur_f
        if abs(delta) > float(max_repair_delta_px):
            delta = float(np.sign(delta) * max_repair_delta_px)
            warns.append(f"{side}: clamped_delta_at_col_{c}")
        new_y = int(round(cur_f + delta))
        new_y = int(np.clip(new_y, 0, max(0, int(roi_h) - 1)))
        if new_y != int(cur):
            changed.append(int(c))
            max_d = max(max_d, abs(float(new_y - cur_f)))
            out[c] = new_y

    return out, changed, max_d, warns


def apply_edge_trace_repair_v2(
    trace_path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
    *,
    roi_w: int,
    roi_h: int,
    edge_n: int = 5,
    stable_gap: int = 5,
    stable_width: int = 32,
    min_abs_dev_px: float = 8.0,
    z_thresh: float = 3.0,
    max_repair_delta_px: float = 80.0,
    min_stable_points: int = 12,
    mode: str = "hybrid",
    sigma_floor_px: float = 3.0,
    peak_edge_guard_enabled: bool = False,
    peak_edge_guard_mode: str = "skip_side",
    peak_edge_window: int = 16,
    peak_edge_curvature_thresh: float = 10.0,
    peak_edge_prominence_thresh: float = 4.5,
    peak_edge_guard_delta_cap_px: float = 20.0,
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """
    Returns updated path (copy) and debug metadata.
    """
    mode_l = str(mode).strip().lower()
    if mode_l not in {"clamp_to_stable_median", "linear_extrapolate_from_stable", "hybrid"}:
        mode_l = "hybrid"

    det_bundle = detect_edge_trace_repair_v2(
        trace_path,
        final_cands,
        roi_w=roi_w,
        roi_h=roi_h,
        edge_n=edge_n,
        stable_gap=stable_gap,
        stable_width=stable_width,
        min_abs_dev_px=min_abs_dev_px,
        z_thresh=z_thresh,
        min_stable_points=min_stable_points,
        sigma_floor_px=sigma_floor_px,
    )

    path_work = list(trace_path)
    meta_warnings: List[str] = []

    left_det = dict(det_bundle["left"])
    right_det = dict(det_bundle["right"])

    med_left = float(left_det.get("stable_median_y", float("nan")))
    med_right = float(right_det.get("stable_median_y", float("nan")))

    guard_mode = str(peak_edge_guard_mode).strip().lower()
    if guard_mode not in {"skip_side", "cap_delta"}:
        guard_mode = "skip_side"

    lg: Dict[str, Any] = {
        "enabled": bool(peak_edge_guard_enabled),
        "left_guard_triggered": False,
        "right_guard_triggered": False,
        "left_guard_reason": "",
        "right_guard_reason": "",
        "guard_mode": guard_mode,
        "left_peak_like_score": None,
        "right_peak_like_score": None,
        "uses_gt": False,
        "uses_source_numeric": False,
    }

    left_delta_use = float(max_repair_delta_px)
    right_delta_use = float(max_repair_delta_px)
    skip_left = False
    skip_right = False

    if peak_edge_guard_enabled:
        gl = evaluate_peak_edge_guard_side(
            trace_path,
            left_det,
            side="left",
            roi_w=roi_w,
            edge_n=edge_n,
            peak_edge_window=int(peak_edge_window),
            curvature_thresh=float(peak_edge_curvature_thresh),
            prominence_thresh=float(peak_edge_prominence_thresh),
        )
        gr = evaluate_peak_edge_guard_side(
            trace_path,
            right_det,
            side="right",
            roi_w=roi_w,
            edge_n=edge_n,
            peak_edge_window=int(peak_edge_window),
            curvature_thresh=float(peak_edge_curvature_thresh),
            prominence_thresh=float(peak_edge_prominence_thresh),
        )
        lg["left_peak_like_score"] = gl.get("peak_like_score")
        lg["right_peak_like_score"] = gr.get("peak_like_score")
        lg["left_peak_detail"] = gl
        lg["right_peak_detail"] = gr

        if bool(left_det.get("anomaly")) and bool(gl.get("peak_like")):
            lg["left_guard_triggered"] = True
            lg["left_guard_reason"] = ";".join(gl.get("reasons") or []) or "peak_like"
            if guard_mode == "skip_side":
                skip_left = True
            else:
                left_delta_use = float(min(max_repair_delta_px, float(peak_edge_guard_delta_cap_px)))

        if bool(right_det.get("anomaly")) and bool(gr.get("peak_like")):
            lg["right_guard_triggered"] = True
            lg["right_guard_reason"] = ";".join(gr.get("reasons") or []) or "peak_like"
            if guard_mode == "skip_side":
                skip_right = True
            else:
                right_delta_use = float(min(max_repair_delta_px, float(peak_edge_guard_delta_cap_px)))

    left_det_repair = dict(left_det)
    right_det_repair = dict(right_det)
    if skip_left:
        left_det_repair["anomaly"] = False
        meta_warnings.append("peak_edge_guard:left_skip_side")
    if skip_right:
        right_det_repair["anomaly"] = False
        meta_warnings.append("peak_edge_guard:right_skip_side")

    path_work, ch_l, md_l, w_l = _repair_side(
        path_work,
        left_det_repair,
        side="left",
        mode=mode_l,
        edge_n=edge_n,
        roi_w=roi_w,
        roi_h=roi_h,
        max_repair_delta_px=left_delta_use,
        med_stable=med_left,
    )
    meta_warnings.extend(w_l)

    path_work, ch_r, md_r, w_r = _repair_side(
        path_work,
        right_det_repair,
        side="right",
        mode=mode_l,
        edge_n=edge_n,
        roi_w=roi_w,
        roi_h=roi_h,
        max_repair_delta_px=right_delta_use,
        med_stable=med_right,
    )
    meta_warnings.extend(w_r)

    left_applied = bool(left_det.get("anomaly")) and bool(ch_l)
    right_applied = bool(right_det.get("anomaly")) and bool(ch_r)

    edge_lo = set(range(0, min(edge_n, roi_w)))
    edge_hi = set(range(max(0, roi_w - edge_n), roi_w))
    bad = [c for c in ch_l + ch_r if c not in edge_lo and c not in edge_hi]
    if bad:
        meta_warnings.append(f"repair_touch_non_edge_columns:{bad[:16]}")

    repaired_cols = sorted(set(ch_l + ch_r))
    max_abs = float(max(md_l, md_r))

    detector_debug = {
        "left": det_bundle["left"],
        "right": det_bundle["right"],
        "runtime_safe": True,
        "uses_gt": False,
        "uses_source_numeric": False,
    }

    out_meta = {
        "enabled": True,
        "applied": bool(left_applied or right_applied),
        "left_applied": left_applied,
        "right_applied": right_applied,
        "left_anomaly": bool(left_det.get("anomaly")),
        "right_anomaly": bool(right_det.get("anomaly")),
        "repaired_columns": repaired_cols,
        "changed_columns": int(len(repaired_cols)),
        "max_abs_delta_y": max_abs,
        "mode": mode_l,
        "edge_n": int(edge_n),
        "stable_windows": {
            "left_stable": [int(left_det["stable_columns"][0]), int(left_det["stable_columns"][-1])]
            if left_det.get("stable_columns")
            else [],
            "right_stable": [int(right_det["stable_columns"][0]), int(right_det["stable_columns"][-1])]
            if right_det.get("stable_columns")
            else [],
        },
        "uses_gt": False,
        "uses_source_numeric": False,
        "warnings": meta_warnings + left_det.get("warnings", []) + right_det.get("warnings", []),
        "edge_trace_repair_v2_detector": detector_debug,
        "edge_trace_repair_v2_peak_edge_guard": lg,
    }

    return path_work, out_meta
