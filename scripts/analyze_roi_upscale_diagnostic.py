#!/usr/bin/env python3
"""Analyze ROI 1x vs 2x diagnostic outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.gt_compat import normalize_gt_for_eval  # noqa: E402
from eval.report import evaluate_single  # noqa: E402
from eval.candidate_gt_proximity import compute_candidate_gt_proximity_diag  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _path_upper_fraction(path_vals: List[Any], roi_h: int) -> Optional[float]:
    ys = []
    for y in path_vals:
        if y is None:
            continue
        fy = _safe_float(y)
        if fy is not None:
            ys.append(fy)
    if not ys or roi_h <= 0:
        return None
    upper_thr = float(roi_h) * 0.2
    return float(sum(1 for y in ys if y < upper_thr) / len(ys))


def _scaled_candidates_to_original(
    cands: Mapping[Any, List[Dict[str, Any]]],
    *,
    factor: int,
    roi_w_orig: int,
    roi_h_orig: int,
) -> Dict[int, List[Dict[str, Any]]]:
    if factor <= 1:
        out: Dict[int, List[Dict[str, Any]]] = {}
        for k, lst in cands.items():
            try:
                col = int(k)
            except Exception:
                continue
            out[col] = [dict(c) for c in lst]
        return out
    bucket: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for k, lst in cands.items():
        try:
            col_s = int(k)
        except Exception:
            continue
        col_o = max(0, min(int(roi_w_orig) - 1, int(round(float(col_s) / float(factor)))))
        col_bucket = bucket.setdefault(col_o, {})
        for c in lst:
            y = _safe_float(c.get("y"))
            if y is None:
                continue
            y_o = max(0, min(int(roi_h_orig) - 1, int(round(float(y) / float(factor)))))
            new_c = dict(c)
            new_c["y"] = int(y_o)
            prev = col_bucket.get(int(y_o))
            if prev is None or float(new_c.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                col_bucket[int(y_o)] = new_c
    out2: Dict[int, List[Dict[str, Any]]] = {}
    for col, d in bucket.items():
        vals = list(d.values())
        vals.sort(key=lambda c: -float(c.get("confidence", 0.0)))
        out2[int(col)] = vals
    return out2


def _numeric_rmse_norm(result: Dict[str, Any], gt: Dict[str, Any]) -> float:
    x_gt = gt.get("x_values") or gt.get("two_theta_values")
    y_gt = gt.get("y_values") or gt.get("intensities")
    x_pred = result.get("two_theta_values", [])
    y_pred = result.get("intensities", [])
    if not x_gt or not y_gt or len(x_gt) < 2 or len(y_gt) < 2:
        return 999.0
    if len(x_pred) < 2 or len(y_pred) < 2:
        return 999.0
    x_gt = np.asarray(x_gt, dtype=np.float64)
    y_gt = np.asarray(y_gt, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    order = np.argsort(x_pred)
    x_pred = x_pred[order]
    y_pred = y_pred[order]
    if np.any(np.diff(x_pred) <= 0):
        return 999.0
    y_interp = np.interp(x_gt, x_pred, y_pred)
    rng = float(np.max(y_gt) - np.min(y_gt))
    if rng <= 1e-8:
        return 999.0
    rmse = float(np.sqrt(np.mean((y_interp - y_gt) ** 2)))
    return rmse / rng


def _y_amplitude_metrics(result: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, float]:
    x_gt = gt.get("x_values") or gt.get("two_theta_values")
    y_gt = gt.get("y_values") or gt.get("intensities")
    x_pred = result.get("two_theta_values", [])
    y_pred = result.get("intensities", [])
    if not x_gt or not y_gt or len(x_gt) < 2 or len(y_gt) < 2 or len(x_pred) < 2 or len(y_pred) < 2:
        return {
            "amplitude_ratio_pred_over_gt": 999.0,
            "amplitude_error_norm": 999.0,
            "baseline_offset_norm": 999.0,
            "peak_height_error_norm_top3": 999.0,
        }
    x_gt = np.asarray(x_gt, dtype=np.float64)
    y_gt = np.asarray(y_gt, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    order = np.argsort(x_pred)
    x_pred = x_pred[order]
    y_pred = y_pred[order]
    if np.any(np.diff(x_pred) <= 0):
        return {
            "amplitude_ratio_pred_over_gt": 999.0,
            "amplitude_error_norm": 999.0,
            "baseline_offset_norm": 999.0,
            "peak_height_error_norm_top3": 999.0,
        }
    y_interp = np.interp(x_gt, x_pred, y_pred)
    gt_rng = float(np.max(y_gt) - np.min(y_gt))
    if gt_rng <= 1e-8:
        return {
            "amplitude_ratio_pred_over_gt": 999.0,
            "amplitude_error_norm": 999.0,
            "baseline_offset_norm": 999.0,
            "peak_height_error_norm_top3": 999.0,
        }
    pr_rng = float(np.max(y_interp) - np.min(y_interp))
    amp_ratio = pr_rng / gt_rng
    amp_err = abs(pr_rng - gt_rng) / gt_rng
    q = 0.1
    gt_base = float(np.mean(np.sort(y_gt)[: max(1, int(len(y_gt) * q))]))
    pr_base = float(np.mean(np.sort(y_interp)[: max(1, int(len(y_interp) * q))]))
    baseline_offset = abs(pr_base - gt_base) / gt_rng
    gt_top = np.sort(y_gt)[-3:]
    pr_top = np.sort(y_interp)[-3:]
    peak_h_err = float(np.mean(np.abs(gt_top - pr_top))) / gt_rng
    return {
        "amplitude_ratio_pred_over_gt": float(amp_ratio),
        "amplitude_error_norm": float(amp_err),
        "baseline_offset_norm": float(baseline_offset),
        "peak_height_error_norm_top3": float(peak_h_err),
    }


def _aligned_pred_gt_y_px(result: Dict[str, Any], debug: Dict[str, Any], gt_norm: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    gt_path = gt_norm.get("pixel_curve_path", [])
    if not gt_path:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    plot_box = debug.get("plot_box", gt_norm.get("plot_box", [0, 0, 1200, 900]))
    x0 = int(plot_box[0])
    y0 = int(plot_box[1])
    gt_cols = []
    gt_y = []
    for p in gt_path:
        col = int(p[0]) - x0
        gt_cols.append(col)
        gt_y.append(float(p[1]) - y0)
    if len(gt_cols) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    cal = debug.get("calibration", {})
    x_scale = float(cal.get("x_scale", 1.0))
    x_offset = float(cal.get("x_offset", 0.0))
    y_scale = float(cal.get("y_scale", 1.0))
    y_offset = float(cal.get("y_offset", 0.0))
    tt = np.asarray(result.get("two_theta_values", []), dtype=np.float64)
    yy = np.asarray(result.get("intensities", []), dtype=np.float64)
    if tt.size < 2 or yy.size < 2 or abs(x_scale) < 1e-12 or abs(y_scale) < 1e-12:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    col_pred = (tt - x_offset) / x_scale
    y_pred_px = (yy - y_offset) / y_scale
    order = np.argsort(col_pred)
    col_pred = col_pred[order]
    y_pred_px = y_pred_px[order]
    if np.any(np.diff(col_pred) <= 0):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    gt_cols_arr = np.asarray(gt_cols, dtype=np.float64)
    gt_y_arr = np.asarray(gt_y, dtype=np.float64)
    pred_interp = np.interp(gt_cols_arr, col_pred, y_pred_px)
    return pred_interp, gt_y_arr


def _edge_metrics(result: Dict[str, Any], debug: Dict[str, Any], gt_norm: Dict[str, Any]) -> Dict[str, Any]:
    pred, gt = _aligned_pred_gt_y_px(result, debug, gt_norm)
    if pred.size < 10:
        return {
            "start_edge_y_error_px": None,
            "end_edge_y_error_px": None,
            "start_edge_upturn_flag": None,
            "end_edge_upturn_flag": None,
            "edge_window_mean_error_px": None,
            "edge_window_slope_diff": None,
            "edge_window_curvature_proxy": None,
        }
    err = pred - gt
    n = int(pred.size)
    w = max(3, int(round(n * 0.05)))
    s_pred, s_gt, s_err = pred[:w], gt[:w], err[:w]
    e_pred, e_gt, e_err = pred[-w:], gt[-w:], err[-w:]
    start_mean = float(np.mean(s_err))
    end_mean = float(np.mean(e_err))
    # y-axis pixel 기준: y가 작을수록 위로 들림.
    start_up = float(np.mean(s_gt - s_pred)) > 1.0
    end_up = float(np.mean(e_gt - e_pred)) > 1.0
    edge_mean_abs = float((np.mean(np.abs(s_err)) + np.mean(np.abs(e_err))) * 0.5)
    s_slope_pred = float(s_pred[-1] - s_pred[0]) / max(len(s_pred) - 1, 1)
    s_slope_gt = float(s_gt[-1] - s_gt[0]) / max(len(s_gt) - 1, 1)
    e_slope_pred = float(e_pred[-1] - e_pred[0]) / max(len(e_pred) - 1, 1)
    e_slope_gt = float(e_gt[-1] - e_gt[0]) / max(len(e_gt) - 1, 1)
    slope_diff = float((abs(s_slope_pred - s_slope_gt) + abs(e_slope_pred - e_slope_gt)) * 0.5)
    def _curv(x: np.ndarray) -> float:
        if x.size < 3:
            return 0.0
        return float(np.mean(np.abs(np.diff(x, n=2))))
    curv_proxy = float((abs(_curv(s_pred) - _curv(s_gt)) + abs(_curv(e_pred) - _curv(e_gt))) * 0.5)
    return {
        "start_edge_y_error_px": round(start_mean, 6),
        "end_edge_y_error_px": round(end_mean, 6),
        "start_edge_upturn_flag": bool(start_up),
        "end_edge_upturn_flag": bool(end_up),
        "edge_window_mean_error_px": round(edge_mean_abs, 6),
        "edge_window_slope_diff": round(slope_diff, 6),
        "edge_window_curvature_proxy": round(curv_proxy, 6),
    }


def _copy_review_assets(src_debug_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "11_trace_path.png",
        "13_smoothed_trace.png",
        "14_peaks_overlay.png",
        "15_numeric_curve_peaks_roi.png",
        "debug.json",
    ]
    for n in names:
        src = src_debug_dir / n
        if src.is_file():
            shutil.copy2(src, dst_dir / n)


def _extract_variant_row(root: Path, row: pd.Series, variant: str) -> Dict[str, Any]:
    sid = str(row["sample_id"])
    dom = str(row["domain"])
    run_key = f"{dom}_{sid}"
    run_root = root / "runs" / run_key / variant
    result_json = run_root / f"{sid}_result.json"
    debug_dir = run_root / f"debug_{sid}_global"
    debug_json = debug_dir / "debug.json"
    gt_path = Path(str(row["gt_json"]))
    if not gt_path.is_absolute():
        gt_path = (ROOT / str(gt_path)).resolve()
    if not result_json.is_file() or not debug_json.is_file():
        raise FileNotFoundError(f"missing outputs for {run_key}/{variant}")

    ev = evaluate_single(
        str(result_json),
        str(debug_json),
        str(gt_path),
        gate_type=dom if dom in ("clean", "styled") else "clean",
        gate_level="development",
    )
    result = _read_json(result_json)
    debug = _read_json(debug_json)
    gt_norm = normalize_gt_for_eval(_read_json(gt_path))

    roi_up = debug.get("roi_upscale", {}) if isinstance(debug.get("roi_upscale"), dict) else {}
    roi_up_factor = int(roi_up.get("factor", 1))
    roi_w_orig = int((roi_up.get("original_roi_size") or [debug["plot_box"][2] - debug["plot_box"][0]])[0])
    roi_h_orig = int((roi_up.get("original_roi_size") or [None, debug["plot_box"][3] - debug["plot_box"][1]])[1])

    final_candidates_path = debug_dir / "final_candidates.json"
    prox_diag = {
        "candidate_gt_near_recall_px3": None,
        "candidate_gt_near_recall_px5": None,
        "candidate_gt_near_recall_px10": None,
        "mean_nearest_candidate_gt_dist_px": None,
    }
    if final_candidates_path.is_file():
        final_candidates = _read_json(final_candidates_path)
        final_candidates = _scaled_candidates_to_original(
            final_candidates,
            factor=roi_up_factor,
            roi_w_orig=roi_w_orig,
            roi_h_orig=roi_h_orig,
        )
        prox_full = compute_candidate_gt_proximity_diag(
            final_candidates,
            str(gt_path),
            tuple(debug.get("plot_box", [0, 0, 0, 0])),
            int(roi_h_orig),
            int(roi_w_orig),
        )
        prox_diag.update({
            "candidate_gt_near_recall_px3": prox_full.get("candidate_gt_near_recall_px3"),
            "candidate_gt_near_recall_px5": prox_full.get("candidate_gt_near_recall_px5"),
            "candidate_gt_near_recall_px10": prox_full.get("candidate_gt_near_recall_px10"),
            "mean_nearest_candidate_gt_dist_px": prox_full.get("mean_nearest_candidate_gt_dist_px"),
        })

    main = ev["metrics"]["main"]
    dbg = ev["metrics"].get("debug", {})
    edge = _edge_metrics(result, debug, gt_norm)
    amp = _y_amplitude_metrics(result, gt_norm)
    numeric_rmse_norm = _numeric_rmse_norm(result, gt_norm)
    path_upper = _path_upper_fraction(
        debug.get("trace", {}).get("path", []),
        int(roi_h_orig),
    )
    st = debug.get("stage_timings", {})
    run_time = 0.0
    for v in st.values():
        fv = _safe_float(v)
        if fv is not None:
            run_time += fv
    cstats = debug.get("candidate_stats", {})

    out = {
        "sample_id": sid,
        "domain": dom,
        "variant": variant,
        "roi_upscale_factor": roi_up_factor,
        "roi_upscale_method": str(roi_up.get("method", "none")),
        "curve_y_mae_px": _safe_float(main.get("curve_y_mae_px")),
        "numeric_y_mae_norm": _safe_float(main.get("numeric_y_mae_norm")),
        "numeric_y_rmse_norm": float(round(numeric_rmse_norm, 6)),
        "peak_f1": _safe_float(dbg.get("peak_f1")),
        "peak_recall": _safe_float(main.get("peak_recall")),
        "candidate_gt_near_recall_px3": _safe_float(prox_diag.get("candidate_gt_near_recall_px3")),
        "candidate_gt_near_recall_px5": _safe_float(prox_diag.get("candidate_gt_near_recall_px5")),
        "candidate_gt_near_recall_px10": _safe_float(prox_diag.get("candidate_gt_near_recall_px10")),
        "mean_nearest_candidate_gt_dist_px": _safe_float(prox_diag.get("mean_nearest_candidate_gt_dist_px")),
        "mean_oracle_dist_px": _safe_float(
            debug.get("model_assist", {}).get("oracle_rerank", {}).get("oracle_score_summary", {}).get("mean_oracle_dist_px")
        ),
        "final_path_upper_band_fraction": _safe_float(path_upper),
        "raw_candidates_total": _safe_float(cstats.get("raw_candidates_total")),
        "final_candidates_total": _safe_float(cstats.get("final_candidates_total")),
        "runtime_total_sec_est": float(round(run_time, 6)),
        "amplitude_ratio_pred_over_gt": float(round(amp["amplitude_ratio_pred_over_gt"], 6)),
        "amplitude_error_norm": float(round(amp["amplitude_error_norm"], 6)),
        "baseline_offset_norm": float(round(amp["baseline_offset_norm"], 6)),
        "peak_height_error_norm_top3": float(round(amp["peak_height_error_norm_top3"], 6)),
    }
    out.update(edge)
    return out


def _classify(summary: Dict[str, Any]) -> str:
    cand = float(summary.get("delta_candidate_gt_near_recall_px5_mean", 0.0))
    y_mae = float(summary.get("delta_numeric_y_mae_norm_mean", 0.0))
    edge = float(summary.get("delta_edge_window_mean_error_px_mean", 0.0))
    peak_f1 = float(summary.get("delta_peak_f1_mean", 0.0))
    peak_recall = float(summary.get("delta_peak_recall_mean", 0.0))
    runtime = float(summary.get("runtime_ratio_2x_over_1x_mean", 1.0))
    if peak_f1 < -1e-6 or peak_recall < -1e-6 or runtime > 2.5:
        return "ROI_UPSCALE_HURTS_OR_UNSTABLE"
    cand_help = cand > 1e-4
    y_help = y_mae < -1e-4
    edge_help = edge < -1e-4
    if cand_help and y_help and edge_help:
        return "ROI_UPSCALE_HELPS_OVERALL"
    if cand_help and not y_help and not edge_help:
        return "ROI_UPSCALE_HELPS_RAW_CANDIDATES"
    if y_help and not edge_help:
        return "ROI_UPSCALE_HELPS_Y_FIDELITY"
    if edge_help and not y_help:
        return "ROI_UPSCALE_HELPS_EDGE_ARTIFACT"
    if abs(cand) < 1e-4 and abs(y_mae) < 1e-4 and abs(edge) < 1e-4:
        return "ROI_UPSCALE_NO_EFFECT"
    return "ROI_UPSCALE_HURTS_OR_UNSTABLE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "_roi_upscale_diag")
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    manifest = args.manifest or (root / "diag_manifest.csv")
    df = pd.read_csv(manifest)
    need = {"sample_id", "domain", "gt_json"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(_extract_variant_row(root, r, "baseline_1x"))
        rows.append(_extract_variant_row(root, r, "roi_upscale_2x"))
    comp = pd.DataFrame(rows)
    out_csv = root / "roi_upscale_comparison.csv"
    out_json = root / "roi_upscale_summary.json"
    comp.to_csv(out_csv, index=False)

    piv = comp.pivot_table(index=["sample_id", "domain"], columns="variant", aggfunc="first")
    deltas = []
    review_root = root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    for (sid, dom), _ in comp.groupby(["sample_id", "domain"]):
        r1 = comp[(comp["sample_id"] == sid) & (comp["domain"] == dom) & (comp["variant"] == "baseline_1x")].iloc[0]
        r2 = comp[(comp["sample_id"] == sid) & (comp["domain"] == dom) & (comp["variant"] == "roi_upscale_2x")].iloc[0]
        d = {
            "sample_id": sid,
            "domain": dom,
            "delta_curve_y_mae_px": float(r2["curve_y_mae_px"] - r1["curve_y_mae_px"]),
            "delta_numeric_y_mae_norm": float(r2["numeric_y_mae_norm"] - r1["numeric_y_mae_norm"]),
            "delta_numeric_y_rmse_norm": float(r2["numeric_y_rmse_norm"] - r1["numeric_y_rmse_norm"]),
            "delta_candidate_gt_near_recall_px5": float(r2["candidate_gt_near_recall_px5"] - r1["candidate_gt_near_recall_px5"]),
            "delta_mean_nearest_candidate_gt_dist_px": float(
                r2["mean_nearest_candidate_gt_dist_px"] - r1["mean_nearest_candidate_gt_dist_px"]
            ),
            "delta_peak_f1": float(r2["peak_f1"] - r1["peak_f1"]),
            "delta_peak_recall": float(r2["peak_recall"] - r1["peak_recall"]),
            "delta_edge_window_mean_error_px": float(r2["edge_window_mean_error_px"] - r1["edge_window_mean_error_px"]),
            "runtime_ratio_2x_over_1x": float(r2["runtime_total_sec_est"] / max(float(r1["runtime_total_sec_est"]), 1e-9)),
        }
        deltas.append(d)

        sample_review = review_root / f"{dom}_{sid}"
        base_debug = root / "runs" / f"{dom}_{sid}" / "baseline_1x" / f"debug_{sid}_global"
        up_debug = root / "runs" / f"{dom}_{sid}" / "roi_upscale_2x" / f"debug_{sid}_global"
        _copy_review_assets(base_debug, sample_review / "baseline_1x")
        _copy_review_assets(up_debug, sample_review / "roi_upscale_2x")
        (sample_review / "result_compare_summary.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    ddf = pd.DataFrame(deltas)
    summary = {
        "n_samples": int(len(ddf)),
        "delta_curve_y_mae_px_mean": float(ddf["delta_curve_y_mae_px"].mean()),
        "delta_numeric_y_mae_norm_mean": float(ddf["delta_numeric_y_mae_norm"].mean()),
        "delta_numeric_y_rmse_norm_mean": float(ddf["delta_numeric_y_rmse_norm"].mean()),
        "delta_candidate_gt_near_recall_px5_mean": float(ddf["delta_candidate_gt_near_recall_px5"].mean()),
        "delta_mean_nearest_candidate_gt_dist_px_mean": float(ddf["delta_mean_nearest_candidate_gt_dist_px"].mean()),
        "delta_peak_f1_mean": float(ddf["delta_peak_f1"].mean()),
        "delta_peak_recall_mean": float(ddf["delta_peak_recall"].mean()),
        "delta_edge_window_mean_error_px_mean": float(ddf["delta_edge_window_mean_error_px"].mean()),
        "runtime_ratio_2x_over_1x_mean": float(ddf["runtime_ratio_2x_over_1x"].mean()),
        "final_preserve_enabled_for_both_variants": True,
    }
    summary["classification"] = _classify(summary)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {review_root}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
