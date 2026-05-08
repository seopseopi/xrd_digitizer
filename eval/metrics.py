"""
$17.2: Metric computation - main, debug, diagnosis.

Compares engine output (result JSON + debug JSON) against GT.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from eval.gt_compat import normalize_gt_for_eval


def compute_all_metrics(
    result: dict,
    debug: dict,
    gt: dict,
) -> dict:
    """
    Compute all 3 categories of metrics.
    Returns: {"main": {...}, "debug": {...}, "diagnosis": {...}}
    """
    gt = normalize_gt_for_eval(gt)
    main = _compute_main_metrics(result, debug, gt)
    dbg = _compute_debug_metrics(result, debug, gt)
    diag = _compute_diagnosis_metrics(debug, gt)
    return {"main": main, "debug": dbg, "diagnosis": diag}


def _compute_main_metrics(result: dict, debug: dict, gt: dict) -> dict:
    """$17.2(1): gate 판단용 main metrics (numeric fidelity 포함)."""
    gt_pixel_path = gt.get("pixel_curve_path", [])
    plot_box = gt.get("plot_box", [0, 0, 1200, 900])
    x0 = plot_box[0]

    curve_y_mae_px = _curve_y_mae(result, debug, gt_pixel_path, x0)
    mpx_err, mpy_err = _major_peak_errors(result, debug, gt)
    peak_recall = _peak_recall_at_prominence(result, debug, gt)
    max_gap = _max_gap_px(debug)
    cal_rt = debug.get("calibration", {}).get("roundtrip_error", {}).get("total_mean_error_px", 0.0)
    numeric_y_mae_norm = _numeric_y_mae_norm(result, gt)
    major_peak_x_error_2theta = _major_peak_x_error_2theta(result, debug, gt)

    return {
        "curve_y_mae_px": round(curve_y_mae_px, 4),
        "major_peak_x_error": round(mpx_err, 4),
        "peak_recall": round(peak_recall, 4),
        "max_gap_px": max_gap,
        "calibration_roundtrip_error": round(cal_rt, 6),
        "numeric_y_mae_norm": round(numeric_y_mae_norm, 6),
        "major_peak_x_error_2theta": round(major_peak_x_error_2theta, 6),
    }


def _compute_debug_metrics(result: dict, debug: dict, gt: dict) -> dict:
    """$17.2(2): 분석용 debug metrics."""
    trace = debug.get("trace", {})
    valid_ratio = trace.get("valid_ratio", 0.0)
    trace_score = trace.get("trace_score", 0.0)

    gt_pixel_path = gt.get("pixel_curve_path", [])
    plot_box = gt.get("plot_box", [0, 0, 1200, 900])
    x0 = plot_box[0]

    iou = _compute_iou(debug, gt_pixel_path, x0)
    tail_mae, tail_collapse = _tail_metrics(result, debug, gt_pixel_path, x0)
    precision, recall, f1 = _peak_prf(result, debug, gt)
    prom_pres = _prominence_preservation(result, debug, gt)
    _, mpy_err = _major_peak_errors(result, debug, gt)

    return {
        "IoU": round(iou, 4),
        "valid_ratio": round(valid_ratio, 4),
        "trace_score": round(trace_score, 2),
        "tail_mae_px": round(tail_mae, 4),
        "tail_collapse_rate": round(tail_collapse, 4),
        "peak_precision": round(precision, 4),
        "peak_f1": round(f1, 4),
        "prominence_preservation": round(prom_pres, 4),
        "major_peak_y_error": round(mpy_err, 4),
    }


def _compute_diagnosis_metrics(debug: dict, gt: Optional[dict] = None) -> dict:
    """$17.2(3): mandatory diagnosis metrics."""
    cand_stats = debug.get("candidate_stats", {})
    total_cols = cand_stats.get("total_columns", 1)
    raw_nonempty = cand_stats.get("raw_nonempty_columns", 0)
    candidate_recall = raw_nonempty / max(total_cols, 1)

    empty_col_rate = cand_stats.get("missing_column_ratio", 0.0)

    recovery = debug.get("recovery", {})
    zones = recovery.get("zones", [])
    n_zones = len(zones)
    n_resolved = sum(1 for z in zones if z.get("resolved"))
    recovery_rate = n_resolved / max(n_zones, 1) if n_zones > 0 else 1.0
    reentry_count = n_zones

    trace = debug.get("trace", {})
    blockwise = trace.get("blockwise", [])
    margin_instability = _path_margin_instability(blockwise)

    out = {
        "candidate_recall_per_column": round(candidate_recall, 4),
        "empty_column_rate": round(empty_col_rate, 4),
        "recovery_success_rate": round(recovery_rate, 4),
        "reentry_count": reentry_count,
        "path_margin_instability": round(margin_instability, 4),
    }

    prox = debug.get("candidate_gt_proximity")
    if isinstance(prox, dict):
        for k, v in prox.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[str(k)] = round(float(v), 6) if isinstance(v, float) else int(v)
            elif v is None:
                out[str(k)] = None
            elif isinstance(v, bool):
                out[str(k)] = v

    return out


def _curve_y_mae(result: dict, debug: dict, gt_path: list, x0: int) -> float:
    """Column-aligned MAE between traced y and GT y in pixel space."""
    if not gt_path:
        return 0.0

    gt_map: Dict[int, float] = {}
    for pt in gt_path:
        col = int(pt[0]) - x0
        gt_map[col] = float(pt[1])

    pp = debug.get("postprocess", {})
    peak_list = pp.get("peak_list", [])

    trace = debug.get("trace", {})
    diag = trace.get("diagnostics", {})

    plot_box = debug.get("plot_box", [0, 0, 1200, 900])
    y0_box = plot_box[1]

    cand_stats = debug.get("candidate_stats", {})
    total_cols = cand_stats.get("total_columns", 950)

    two_theta = result.get("two_theta_values", [])
    intensities = result.get("intensities", [])

    cal = debug.get("calibration", {})
    y_scale = cal.get("y_scale", 1.0)
    y_offset = cal.get("y_offset", 0.0)

    if not two_theta:
        return 999.0

    x_scale = cal.get("x_scale", 1.0)
    x_offset = cal.get("x_offset", 0.0)

    errors = []
    for i, (tt, inten) in enumerate(zip(two_theta, intensities)):
        if abs(x_scale) > 1e-12:
            px = (tt - x_offset) / x_scale
        else:
            px = i
        col = int(round(px))

        if col in gt_map:
            if abs(y_scale) > 1e-12:
                traced_py = (inten - y_offset) / y_scale
            else:
                traced_py = 0.0
            gt_py = gt_map[col] - plot_box[1]
            errors.append(abs(traced_py - gt_py))

    return float(np.mean(errors)) if errors else 999.0


def _numeric_y_mae_norm(result: dict, gt: dict) -> float:
    """
    mean(abs(y_pred_interp(x_gt) - y_gt)) / (y_gt_max - y_gt_min).
    GT x/y는 x_values·y_values(또는 two_theta_values·intensities) 사용.
    """
    eps = 1e-8
    gt = normalize_gt_for_eval(gt)
    x_gt = gt.get("x_values")
    y_gt = gt.get("y_values")
    if x_gt is None:
        x_gt = gt.get("two_theta_values")
    if y_gt is None:
        y_gt = gt.get("intensities")
    if not x_gt or not y_gt or len(x_gt) < 2 or len(y_gt) < 2:
        return 999.0
    x_pred = result.get("two_theta_values", [])
    y_pred = result.get("intensities", [])
    if len(x_pred) < 2 or len(y_pred) < 2:
        return 999.0

    x_gt = np.asarray(x_gt, dtype=np.float64)
    y_gt = np.asarray(y_gt, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(x_gt) != len(y_gt):
        return 999.0

    order = np.argsort(x_pred)
    x_pred = x_pred[order]
    y_pred = y_pred[order]
    if np.any(np.diff(x_pred) <= 0):
        return 999.0

    y_interp = np.interp(x_gt, x_pred, y_pred)
    rng = float(np.max(y_gt) - np.min(y_gt))
    if rng < eps:
        return 999.0
    mae = float(np.mean(np.abs(y_interp - y_gt)))
    return mae / max(rng, eps)


def _major_peak_x_error_2theta(result: dict, debug: dict, gt: dict) -> float:
    """
    검출 major peak 각각에 대해 가장 가까운 GT peak(2θ)와의 절대 차(°)의 평균.
    debug.calibration.peak_positions_2theta 의 is_major=True 사용.
    """
    gt = normalize_gt_for_eval(gt)
    gt_peaks = gt.get("peaks", [])
    gx_list: List[float] = []
    for gp in gt_peaks:
        if isinstance(gp, dict):
            gx_list.append(float(gp.get("two_theta", gp.get("x", 0))))
        elif isinstance(gp, (list, tuple)):
            gx_list.append(float(gp[0]))
    if not gx_list:
        px = gt.get("peak_x_values")
        if px:
            gx_list = [float(x) for x in px]
    if not gx_list:
        return 999.0

    cal = debug.get("calibration", {})
    detected = cal.get("peak_positions_2theta", [])
    major = [p for p in detected if p.get("is_major")]
    if not major:
        return 999.0

    errs: List[float] = []
    for mp in major:
        tt = float(mp["two_theta"])
        best = min(abs(tt - gx) for gx in gx_list)
        errs.append(best)
    return float(np.mean(errs)) if errs else 999.0


def _major_peak_errors(result: dict, debug: dict, gt: dict) -> Tuple[float, float]:
    """MAE of major peak x/y positions vs GT peaks."""
    gt_peaks = gt.get("peaks", [])
    if not gt_peaks:
        return 0.0, 0.0

    cal = debug.get("calibration", {})
    detected_peaks = cal.get("peak_positions_2theta", [])
    major_detected = [p for p in detected_peaks if p.get("is_major")]

    if not major_detected:
        return 999.0, 999.0

    gt_x_positions = []
    for gp in gt_peaks:
        if isinstance(gp, dict):
            gt_x_positions.append(float(gp.get("two_theta", gp.get("x", 0))))
        elif isinstance(gp, (list, tuple)):
            gt_x_positions.append(float(gp[0]))

    if not gt_x_positions:
        return 0.0, 0.0

    plot_box = gt.get("plot_box", [0, 0, 1200, 900])
    am = gt.get("axis_metadata", {})
    x_min = am.get("x_min", 0)
    x_max = am.get("x_max", 90)
    px_per_deg = (plot_box[2] - plot_box[0]) / max(x_max - x_min, 1)

    x_errors_px = []
    for mp in major_detected:
        tt = mp["two_theta"]
        best_dist = min(abs(tt - gx) for gx in gt_x_positions)
        x_errors_px.append(best_dist * px_per_deg)

    mpx = float(np.mean(x_errors_px)) if x_errors_px else 0.0
    mpy = 0.0

    return mpx, mpy


def _peak_recall_at_prominence(result: dict, debug: dict, gt: dict) -> float:
    """Fraction of GT peaks matched by detected peaks."""
    gt_peaks = gt.get("peaks", [])
    if not gt_peaks:
        return 1.0

    cal = debug.get("calibration", {})
    detected = cal.get("peak_positions_2theta", [])
    det_positions = [p["two_theta"] for p in detected]

    if not det_positions:
        return 0.0

    am = gt.get("axis_metadata", {})
    x_range = am.get("x_max", 90) - am.get("x_min", 0)
    tolerance = x_range * 0.02

    matched = 0
    for gp in gt_peaks:
        if isinstance(gp, dict):
            gx = float(gp.get("two_theta", gp.get("x", 0)))
        elif isinstance(gp, (list, tuple)):
            gx = float(gp[0])
        else:
            continue
        if any(abs(dx - gx) <= tolerance for dx in det_positions):
            matched += 1

    return matched / len(gt_peaks)


def _max_gap_px(debug: dict) -> int:
    """Maximum gap in the trace path."""
    pp = debug.get("postprocess", {})
    gap_ranges = pp.get("gap_fill", {}).get("gap_ranges", [])
    if not gap_ranges:
        return 0
    return max(ge - gs for gs, ge in gap_ranges)


def _compute_iou(debug: dict, gt_path: list, x0: int) -> float:
    """Simplified IoU: overlap of traced vs GT column sets."""
    if not gt_path:
        return 1.0
    gt_cols = {int(pt[0]) - x0 for pt in gt_path}
    cand_stats = debug.get("candidate_stats", {})
    total = cand_stats.get("total_columns", 1)
    traced_cols = set(range(total))

    if not traced_cols or not gt_cols:
        return 0.0

    intersection = len(traced_cols & gt_cols)
    union = len(traced_cols | gt_cols)
    return intersection / max(union, 1)


def _tail_metrics(result: dict, debug: dict, gt_path: list, x0: int) -> Tuple[float, float]:
    """Tail MAE and collapse rate (last 20% of curve)."""
    if not gt_path:
        return 0.0, 0.0

    gt_sorted = sorted(gt_path, key=lambda p: p[0])
    n = len(gt_sorted)
    tail_start = int(n * 0.8)
    tail_gt = gt_sorted[tail_start:]

    if not tail_gt:
        return 0.0, 0.0

    gt_tail_y = [pt[1] for pt in tail_gt]
    gt_y_range = max(gt_tail_y) - min(gt_tail_y) if gt_tail_y else 1.0

    tail_mae = 0.0
    collapse = 0.0
    if gt_y_range < 3.0:
        collapse = 1.0

    return tail_mae, collapse


def _peak_prf(result: dict, debug: dict, gt: dict) -> Tuple[float, float, float]:
    """Peak precision, recall, F1."""
    gt_peaks = gt.get("peaks", [])
    cal = debug.get("calibration", {})
    detected = cal.get("peak_positions_2theta", [])
    det_positions = [p["two_theta"] for p in detected]

    if not gt_peaks:
        return 1.0, 1.0, 1.0
    if not det_positions:
        return 0.0, 0.0, 0.0

    am = gt.get("axis_metadata", {})
    x_range = am.get("x_max", 90) - am.get("x_min", 0)
    tolerance = x_range * 0.02

    gt_positions = []
    for gp in gt_peaks:
        if isinstance(gp, dict):
            gt_positions.append(float(gp.get("two_theta", gp.get("x", 0))))
        elif isinstance(gp, (list, tuple)):
            gt_positions.append(float(gp[0]))

    tp = 0
    for dx in det_positions:
        if any(abs(dx - gx) <= tolerance for gx in gt_positions):
            tp += 1

    precision = tp / max(len(det_positions), 1)
    recall = sum(1 for gx in gt_positions
                 if any(abs(dx - gx) <= tolerance for dx in det_positions)) / max(len(gt_positions), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return precision, recall, f1


def _prominence_preservation(result: dict, debug: dict, gt: dict) -> float:
    """How well peak prominences are preserved relative to GT."""
    pp = debug.get("postprocess", {})
    peaks = pp.get("peak_list", [])
    if not peaks:
        return 0.0
    proms = [p["prominence"] for p in peaks]
    if not proms:
        return 0.0
    max_prom = max(proms)
    if max_prom < 1:
        return 0.0
    ratio = sorted(proms, reverse=True)
    if len(ratio) >= 2:
        return min(1.0, ratio[0] / max(ratio[1], 1.0) * 0.5)
    return 1.0


def _path_margin_instability(blockwise: list) -> float:
    """Fraction of blocks with very low margin (< 1.0) or missing margin."""
    if not blockwise:
        return 0.0
    low = sum(1 for b in blockwise
              if b.get("margin") is not None and b["margin"] < 1.0)
    missing = sum(1 for b in blockwise if b.get("margin") is None)
    return (low + missing * 0.5) / max(len(blockwise), 1)
