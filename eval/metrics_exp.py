"""§5: Evaluator v2 — strict MAE, band-aware MAE, peak_recall_fixed."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from core.plot_scale import compute_plot_scale
from eval.gt_compat import normalize_gt_for_eval


def _gt_column_map(gt_path: list, x0: int) -> Dict[int, float]:
    m: Dict[int, float] = {}
    for pt in gt_path:
        col = int(pt[0]) - x0
        m[col] = float(pt[1])
    return m


def _build_gt_y_per_x(gt_path: list, x0: int, y0: int, plot_w: int) -> np.ndarray:
    """Interpolated y_gt per column in ROI coords (y relative to plot top)."""
    if not gt_path:
        return np.zeros(plot_w)
    pts = sorted([(int(p[0]) - x0, float(p[1]) - y0) for p in gt_path], key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    out = np.zeros(plot_w)
    for col in range(plot_w):
        if col <= xs[0]:
            out[col] = ys[0]
        elif col >= xs[-1]:
            out[col] = ys[-1]
        else:
            for i in range(len(xs) - 1):
                if xs[i] <= col <= xs[i + 1]:
                    t = (col - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
                    out[col] = ys[i] * (1 - t) + ys[i + 1] * t
                    break
    return out


def compute_band_metrics(
    y_gt_per_col: np.ndarray,
    y_pred_per_col: np.ndarray,
    valid_mask: np.ndarray,
    s_h: float,
) -> Tuple[float, float]:
    """Returns curve_band_mae_px, band_hit_rate."""
    plot_w = len(y_gt_per_col)
    errs = []
    hits = 0
    total = 0
    for x in range(plot_w):
        if not valid_mask[x]:
            continue
        yg = y_gt_per_col[x]
        yp = y_pred_per_col[x]
        xm = max(0, x - 1)
        xp = min(plot_w - 1, x + 1)
        slope_t = 0.5 * abs(y_gt_per_col[xp] - y_gt_per_col[xm])
        curv_t = abs(y_gt_per_col[xp] - 2 * y_gt_per_col[x] + y_gt_per_col[xm])
        hw0 = 1.25 * s_h
        hw_slope = 0.35 * min(slope_t, 4.0 * s_h)
        hw_curv = 0.20 * min(curv_t, 3.0 * s_h)
        half_w = float(np.clip(hw0 + hw_slope + hw_curv, 1.25 * s_h, 2.75 * s_h))
        lo, hi = yg - half_w, yg + half_w
        if lo <= yp <= hi:
            errs.append(0.0)
            hits += 1
        else:
            errs.append(min(abs(yp - lo), abs(yp - hi)))
        total += 1
    mae_band = float(np.mean(errs)) if errs else 0.0
    hit_rate = float(hits) / max(total, 1)
    return mae_band, hit_rate


def compute_metrics_v2(
    result: dict,
    debug: dict,
    gt: dict,
) -> dict:
    """Strict curve MAE (same as main), band metrics, peak_recall_fixed."""
    gt = normalize_gt_for_eval(gt)
    plot_box = gt.get("plot_box", [0, 0, 1200, 900])
    x0, y0, x1, y1 = plot_box
    roi_w = max(1, int(x1 - x0))
    roi_h = max(1, int(y1 - y0))
    sc = compute_plot_scale((0, 0, roi_w - 1, roi_h - 1))
    plot_w = int(debug.get("candidate_stats", {}).get("total_columns", roi_w))
    plot_h = int(sc["plot_h"])
    s_h = float(sc["s_h"])
    s_w = float(sc["s_w"])
    s = float(sc["s"])

    gt_path = gt.get("pixel_curve_path", [])
    y_gt = _build_gt_y_per_x(gt_path, x0, y0, plot_w)

    cal = debug.get("calibration", {})
    x_scale = cal.get("x_scale", 1.0)
    x_offset = cal.get("x_offset", 0.0)
    y_scale = cal.get("y_scale", 1.0)
    y_offset = cal.get("y_offset", 0.0)

    two_theta = result.get("two_theta_values", [])
    intensities = result.get("intensities", [])

    y_pred = np.full(plot_w, np.nan)
    for i, tt in enumerate(two_theta):
        if abs(x_scale) < 1e-12:
            continue
        col = int(round((tt - x_offset) / x_scale))
        if 0 <= col < plot_w and i < len(intensities):
            py = (intensities[i] - y_offset) / y_scale if abs(y_scale) > 1e-12 else 0.0
            y_pred[col] = py

    valid_mask = np.isfinite(y_pred)
    errs_strict = []
    for x in range(plot_w):
        if not valid_mask[x]:
            continue
        errs_strict.append(abs(float(y_pred[x]) - float(y_gt[x])))

    strict_mae = float(np.mean(errs_strict)) if errs_strict else 999.0
    band_mae, band_hit = compute_band_metrics(y_gt, y_pred, valid_mask, s_h)

    # peak_recall_fixed: pixel x window
    win = max(4, int(round(0.006 * plot_w)))
    gt_peaks = gt.get("peaks", [])
    detected = cal.get("peak_positions_2theta", [])
    det_x_px = []
    for p in detected:
        tt = p.get("two_theta", 0)
        col = int(round((tt - x_offset) / x_scale)) if abs(x_scale) > 1e-12 else 0
        det_x_px.append(col)

    gt_x_px = []
    am = gt.get("axis_metadata", {})
    x_min = am.get("x_min", 0)
    x_max = am.get("x_max", 90)
    for gp in gt_peaks:
        if isinstance(gp, dict):
            gx = float(gp.get("two_theta", gp.get("x", 0)))
        else:
            gx = float(gp[0])
        col = int(round((gx - x_min) / max(x_max - x_min, 1e-9) * (plot_w - 1)))
        gt_x_px.append(col)

    matched = 0
    used_det = set()
    for gcol in gt_x_px:
        best_j = None
        best_d = 1e9
        for j, dcol in enumerate(det_x_px):
            if j in used_det:
                continue
            d = abs(dcol - gcol)
            if d < best_d and d <= win:
                best_d = d
                best_j = j
        if best_j is not None:
            matched += 1
            used_det.add(best_j)

    peak_recall_fixed = float(matched / max(len(gt_x_px), 1)) if gt_x_px else 1.0

    return {
        "plot_w": plot_w,
        "plot_h": plot_h,
        "s_w": s_w,
        "s_h": s_h,
        "s": s,
        "strict_curve_y_mae_px": round(strict_mae, 4),
        "curve_band_mae_px": round(band_mae, 4),
        "band_hit_rate": round(band_hit, 4),
        "peak_recall_fixed": round(peak_recall_fixed, 4),
    }


def merge_main_with_v2(main: dict, v2: dict) -> dict:
    out = dict(main)
    out["strict_curve_y_mae_px"] = v2["strict_curve_y_mae_px"]
    out["curve_band_mae_px"] = v2["curve_band_mae_px"]
    out["band_hit_rate"] = v2["band_hit_rate"]
    out["peak_recall_fixed"] = v2["peak_recall_fixed"]
    return out
