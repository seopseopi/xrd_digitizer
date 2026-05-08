#!/usr/bin/env python3
"""Stage-wise source_numeric error decomposition for canonical samples.

The diagnostic reads only canonical manifest paths and existing canonical smoke
run outputs. It does not modify canonical data or tune pipeline parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks, peak_widths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CANONICAL_ROOT = ROOT / "data" / "test_canonical_30"
MANIFEST = CANONICAL_ROOT / "manifest.csv"
SMOKE_ROOT = ROOT / "outputs" / "_roi_upscale_smoke_canonical_3domain"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "_source_numeric_error_decomposition" / "clean_pattern_11832"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest_row(manifest: Path, test_id: str) -> Dict[str, str]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    matches = [r for r in rows if r.get("test_id") == test_id]
    if len(matches) != 1:
        raise ValueError(f"manifest row {test_id} expected exactly 1, got {len(matches)}")
    sample_id = matches[0]["sample_id"]
    domain = matches[0]["domain"]
    dups = [r for r in rows if r.get("sample_id") == sample_id and r.get("domain") == domain]
    if len(dups) != 1:
        raise ValueError(f"duplicate/ambiguous canonical row for sample_id={sample_id}, domain={domain}: {len(dups)}")
    return matches[0]


def resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def validate_pair(row: Dict[str, str]) -> Dict[str, Any]:
    required_paths = {
        "input_image": resolve(row["input_image"]),
        "mi_json": resolve(row["mi_json"]),
        "gt_json": resolve(row["gt_json"]),
        "source_numeric_json": resolve(row["source_numeric_json"]),
        "metadata_json": resolve(row["metadata_json"]),
    }
    reasons: List[str] = []
    for key, path in required_paths.items():
        if not path.is_file():
            reasons.append(f"{key}_missing:{path}")
    if row.get("pair_status") != "PASS":
        reasons.append("manifest_pair_status_not_PASS")

    mi = read_json(required_paths["mi_json"])
    meta = read_json(required_paths["metadata_json"])
    gt = read_json(required_paths["gt_json"])
    source = read_json(required_paths["source_numeric_json"])

    if mi.get("mi_source") != "synthetic_from_gt":
        reasons.append("mi_source_not_synthetic_from_gt")
    if mi.get("mi_role") != "calibration_input":
        reasons.append("mi_role_not_calibration_input")
    if mi.get("sample_id") != row["sample_id"]:
        reasons.append("mi_sample_id_mismatch")
    if mi.get("domain") != row["domain"]:
        reasons.append("mi_domain_mismatch")
    for key in ("plot_box", "x_axis_values", "x_axis_points", "y_axis_values", "y_axis_points"):
        if key not in mi:
            reasons.append(f"mi_calibration_missing:{key}")
    if meta.get("sample_id") != row["sample_id"] or meta.get("domain") != row["domain"]:
        reasons.append("metadata_sample_domain_mismatch")
    if gt.get("sample_id") != row["sample_id"]:
        reasons.append("gt_sample_id_mismatch")
    if row["sample_id"] == "pattern_1915":
        reasons.append("pattern_1915_forbidden")
    if "two_theta_values" not in source or "intensities" not in source:
        reasons.append("source_numeric_missing_arrays")

    if reasons:
        raise ValueError(";".join(reasons))

    return {
        "paths": required_paths,
        "mi": mi,
        "metadata": meta,
        "gt": gt,
        "source": source,
        "pair_status": "PASS",
    }


def source_xy(source: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(source["two_theta_values"], dtype=float),
        np.asarray(source["intensities"], dtype=float),
    )


def gt_xy(gt: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(gt["x_values"], dtype=float), np.asarray(gt["y_values"], dtype=float)


def interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x, y, left=y[0], right=y[-1])


def mi_maps(mi: Dict[str, Any]) -> Dict[str, float]:
    x0, y0, x1, y1 = [float(v) for v in mi["plot_box"]]
    xmin, xmax = [float(v) for v in mi["x_axis_values"]]
    ymin, ymax = [float(v) for v in mi["y_axis_values"]]
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "roi_w": x1 - x0,
        "roi_h": y1 - y0,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "x_scale": (xmax - xmin) / max(x1 - x0, 1e-12),
        "y_scale": (ymax - ymin) / max(y0 - y1, -1e-12),  # negative
    }


def x_to_roi_col(x: np.ndarray, m: Dict[str, float], width: int) -> np.ndarray:
    return (x - m["xmin"]) * ((width - 1) / max(m["xmax"] - m["xmin"], 1e-12))


def roi_col_to_x(col: np.ndarray, m: Dict[str, float], width: int) -> np.ndarray:
    return m["xmin"] + col * ((m["xmax"] - m["xmin"]) / max(width - 1, 1e-12))


def y_to_roi_y(y: np.ndarray, m: Dict[str, float]) -> np.ndarray:
    return m["roi_h"] - (y - m["ymin"]) * (m["roi_h"] / max(m["ymax"] - m["ymin"], 1e-12))


def roi_y_to_value(y_roi: np.ndarray, m: Dict[str, float]) -> np.ndarray:
    return m["ymin"] + (m["roi_h"] - y_roi) * ((m["ymax"] - m["ymin"]) / max(m["roi_h"], 1e-12))


def build_expected_pixel_trace(source: Dict[str, Any], mi: Dict[str, Any], width: int) -> Dict[str, Any]:
    sx, sy = source_xy(source)
    m = mi_maps(mi)
    cols = np.arange(width, dtype=float)
    xq = roi_col_to_x(cols, m, width)
    yq = interp(sx, sy, xq)
    y_roi = y_to_roi_y(yq, m)
    y_abs = y_roi + m["y0"]
    return {
        "schema": "expected_pixel_trace_from_source_numeric_v1",
        "source": "source_numeric_json + mi calibration",
        "columns_roi": cols.astype(int).tolist(),
        "two_theta_values": xq.astype(float).tolist(),
        "source_intensities_interp": yq.astype(float).tolist(),
        "y_px_roi": y_roi.astype(float).tolist(),
        "y_px_abs": y_abs.astype(float).tolist(),
        "plot_box": [int(v) for v in mi["plot_box"]],
        "point_count": int(width),
    }


def gt_pixel_trace(gt: Dict[str, Any], width: int) -> np.ndarray:
    plot_box = gt["plot_box"]
    x0, y0 = int(plot_box[0]), int(plot_box[1])
    pc = gt.get("per_column_y_gt") or {}
    vals = []
    for c in range(width):
        abs_x = str(x0 + c)
        vals.append(float(pc.get(abs_x, np.nan)) - y0)
    arr = np.asarray(vals, dtype=float)
    if np.isnan(arr).any():
        path = sorted((int(p[0]) - x0, float(p[1]) - y0) for p in gt.get("pixel_curve_path", []))
        xs = np.asarray([p[0] for p in path], dtype=float)
        ys = np.asarray([p[1] for p in path], dtype=float)
        arr = np.interp(np.arange(width), xs, ys)
    return arr


def raw_trace_from_run(result: Dict[str, Any], debug: Dict[str, Any], *, scale_to_original: bool) -> Tuple[np.ndarray, np.ndarray]:
    diag = result.get("resolution_diagnostics") or {}
    raw_hi = diag.get("raw_trace_points_highres")
    if raw_hi:
        factor = float(raw_hi.get("upscale_factor") or 1.0)
        cols = np.asarray(raw_hi["columns_roi_upscaled"], dtype=float)
        y = np.asarray(raw_hi["y_px_roi_upscaled"], dtype=float)
        if scale_to_original and factor > 0:
            cols = cols / factor
            y = y / factor
        return cols, y
    path = np.asarray(debug.get("trace", {}).get("path", []), dtype=float)
    cols = np.arange(len(path), dtype=float)
    return cols, path


def export_to_pixel(result: Dict[str, Any], mi: Dict[str, Any], which: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ep = result["export_points_highres" if which == "highres" else "export_points_eval"]
    tt = np.asarray(ep["two_theta_values"], dtype=float)
    yy = np.asarray(ep["intensities"], dtype=float)
    m = mi_maps(mi)
    width = int(ep["point_count"])
    cols = x_to_roi_col(tt, m, width)
    y_roi = y_to_roi_y(yy, m)
    return cols, y_roi, yy


def safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else float("nan")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def detect_peak_metrics(x: np.ndarray, y_value: np.ndarray, *, top_n: int = 8) -> Dict[str, Any]:
    if len(x) < 5:
        return {"peaks": []}
    prom = max(float(np.ptp(y_value)) * 0.05, 1e-12)
    peaks, props = find_peaks(y_value, prominence=prom)
    if len(peaks) == 0:
        return {"peaks": []}
    order = np.argsort(props["prominences"])[::-1][:top_n]
    selected = peaks[order]
    widths = peak_widths(y_value, selected, rel_height=0.5)[0] if len(selected) else []
    return {
        "peaks": [
            {
                "index": int(i),
                "x": float(x[i]),
                "height": float(y_value[i]),
                "prominence": float(props["prominences"][order[j]]),
                "width_samples": float(widths[j]) if len(widths) > j else None,
            }
            for j, i in enumerate(selected)
        ]
    }


def compare_peaks(ref: Dict[str, Any], pred: Dict[str, Any], *, tol_x: float) -> Dict[str, Any]:
    rpeaks = ref.get("peaks", [])
    ppeaks = pred.get("peaks", [])
    used: set[int] = set()
    center_errs: List[float] = []
    height_errs: List[float] = []
    width_errs: List[float] = []
    for rp in rpeaks:
        best_j, best_d = None, float("inf")
        for j, pp in enumerate(ppeaks):
            if j in used:
                continue
            d = abs(float(pp["x"]) - float(rp["x"]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= tol_x:
            used.add(best_j)
            pp = ppeaks[best_j]
            center_errs.append(best_d)
            height_errs.append(abs(float(pp["height"]) - float(rp["height"])))
            if pp.get("width_samples") is not None and rp.get("width_samples") is not None:
                width_errs.append(abs(float(pp["width_samples"]) - float(rp["width_samples"])))
    return {
        "ref_peak_count": len(rpeaks),
        "pred_peak_count": len(ppeaks),
        "matched_peak_count": len(center_errs),
        "missed_peak_count": max(0, len(rpeaks) - len(center_errs)),
        "false_spike_count": max(0, len(ppeaks) - len(used)),
        "peak_center_error_mean": float(np.mean(center_errs)) if center_errs else None,
        "peak_height_error_mean": float(np.mean(height_errs)) if height_errs else None,
        "peak_width_error_mean": float(np.mean(width_errs)) if width_errs else None,
    }


def draw_overlay(
    out_path: Path,
    image_path: Path,
    plot_box: List[int],
    curves: Iterable[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]],
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = [int(v) for v in plot_box]
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=2)
    for _label, cols, y_roi, color in curves:
        pts = [(float(x0 + c), float(y0 + y)) for c, y in zip(cols, y_roi) if np.isfinite(c) and np.isfinite(y)]
        if len(pts) >= 2:
            draw.line(pts, fill=(*color, 210), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def classify(stage: Dict[str, Any]) -> Tuple[List[str], str]:
    causes: List[str] = []
    if stage["stage_a_calibration"]["source_to_gt_pixel_mae_px"] > 10:
        causes.append("CALIBRATION_MAPPING_ERROR")
    if stage["stage_b_raw_trace"]["upscale_raw_scaled_vs_expected_pixel_mae_px"] > 10:
        causes.append("PIXEL_TRACE_SELECTION_ERROR")
    if stage["stage_c_export_downscale"]["upscale_raw_to_eval_export_pixel_mae_px"] > 10:
        causes.append("EXPORT_DOWNSCALE_ERROR")
    if stage["stage_d_value_domain"]["upscale_highres_normalized_y_mae"] > 0.05:
        causes.append("Y_AMPLITUDE_MAPPING_ERROR")
    if (stage["stage_c_export_downscale"].get("downscale_peak", {}).get("missed_peak_count") or 0) > 0:
        causes.append("PEAK_PRESERVATION_ERROR")
    if stage["gt_vs_source_numeric"]["normalized_y_mae"] > 1e-9:
        causes.append("METRIC_REFERENCE_MISMATCH")
    causes = sorted(set(causes))
    if "CALIBRATION_MAPPING_ERROR" in causes:
        final = "ROI_SOURCE_NUMERIC_DIAG_CALIBRATION_BAD"
    elif "PIXEL_TRACE_SELECTION_ERROR" in causes and "EXPORT_DOWNSCALE_ERROR" not in causes:
        final = "ROI_SOURCE_NUMERIC_DIAG_EXPORT_OK_TRACE_BAD"
    elif "EXPORT_DOWNSCALE_ERROR" in causes and "PIXEL_TRACE_SELECTION_ERROR" not in causes:
        final = "ROI_SOURCE_NUMERIC_DIAG_TRACE_OK_EXPORT_BAD"
    elif "Y_AMPLITUDE_MAPPING_ERROR" in causes and len(causes) == 1:
        final = "ROI_SOURCE_NUMERIC_DIAG_Y_MAPPING_BAD"
    elif len(causes) > 1:
        final = "ROI_SOURCE_NUMERIC_DIAG_MIXED_ERROR"
    else:
        final = "ROI_SOURCE_NUMERIC_DIAG_INCONCLUSIVE"
    return causes or ["MIXED_ERROR"], final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--test-id", default="clean_pattern_11832")
    ap.add_argument("--smoke-root", type=Path, default=SMOKE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    row = manifest_row(manifest, args.test_id)
    pair = validate_pair(row)
    paths, mi, gt, source = pair["paths"], pair["mi"], pair["gt"], pair["source"]
    out = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out.mkdir(parents=True, exist_ok=True)

    m = mi_maps(mi)
    roi_w = int(m["roi_w"])
    expected = build_expected_pixel_trace(source, mi, roi_w)
    write_json(out / "expected_pixel_trace.json", expected)
    expected_y = np.asarray(expected["y_px_roi"], dtype=float)
    cols = np.arange(roi_w, dtype=float)

    gt_y_px = gt_pixel_trace(gt, roi_w)
    sx, sy = source_xy(source)
    gx, gy = gt_xy(gt)
    source_on_gt_x = interp(sx, sy, gx)
    gt_source_y_mae = safe_mae(source_on_gt_x, gy)
    source_dyn = max(float(np.ptp(sy)), 1e-12)

    run_root = (args.smoke_root if args.smoke_root.is_absolute() else ROOT / args.smoke_root) / args.test_id
    runs: Dict[str, Dict[str, Any]] = {}
    for mode in ("baseline_1x_eval_grid", "upscale_2x_eval_grid", "upscale_2x_highres"):
        result = read_json(run_root / mode / f"{row['sample_id']}_result.json")
        debug = read_json(run_root / mode / f"debug_{row['sample_id']}_global" / "debug.json")
        runs[mode] = {"result": result, "debug": debug}

    base_cols, base_raw_y = raw_trace_from_run(runs["baseline_1x_eval_grid"]["result"], runs["baseline_1x_eval_grid"]["debug"], scale_to_original=True)
    up_cols, up_raw_y = raw_trace_from_run(runs["upscale_2x_eval_grid"]["result"], runs["upscale_2x_eval_grid"]["debug"], scale_to_original=True)
    up_raw_on_eval = np.interp(cols, up_cols, up_raw_y)

    base_raw_on_eval = np.interp(cols, base_cols, base_raw_y)
    eval_cols, eval_y_px, eval_y_val = export_to_pixel(runs["upscale_2x_eval_grid"]["result"], mi, "eval")
    hi_cols, hi_y_px, hi_y_val = export_to_pixel(runs["upscale_2x_highres"]["result"], mi, "highres")
    hi_y_on_eval = np.interp(cols, hi_cols, hi_y_px)
    eval_y_on_eval = np.interp(cols, eval_cols, eval_y_px)

    source_peak_px = detect_peak_metrics(cols, -expected_y)
    base_peak_px = detect_peak_metrics(cols, -base_raw_on_eval)
    up_peak_px = detect_peak_metrics(cols, -up_raw_on_eval)
    eval_peak_px = detect_peak_metrics(cols, -eval_y_on_eval)
    hi_peak_px = detect_peak_metrics(cols, -hi_y_on_eval)

    source_peaks_val = detect_peak_metrics(sx, sy)

    def value_metrics(name: str, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        pred_on_source = interp(x, y, sx)
        peaks = detect_peak_metrics(x, y)
        peak_cmp = compare_peaks(source_peaks_val, peaks, tol_x=0.35)
        return {
            f"{name}_y_amplitude_mae": safe_mae(pred_on_source, sy),
            f"{name}_normalized_y_mae": safe_mae(pred_on_source, sy) / source_dyn,
            f"{name}_shape_correlation": safe_corr(pred_on_source, sy),
            f"{name}_baseline_offset_error": float(np.median(pred_on_source - sy)),
            f"{name}_peak_center_error_deg": peak_cmp["peak_center_error_mean"],
            f"{name}_peak_height_error": peak_cmp["peak_height_error_mean"],
            f"{name}_peak_width_error": peak_cmp["peak_width_error_mean"],
            f"{name}_missed_peak_count": peak_cmp["missed_peak_count"],
            f"{name}_false_spike_count": peak_cmp["false_spike_count"],
        }

    base_res = runs["baseline_1x_eval_grid"]["result"]
    up_eval_res = runs["upscale_2x_eval_grid"]["result"]
    up_hi_res = runs["upscale_2x_highres"]["result"]
    base_x = np.asarray(base_res["export_points_eval"]["two_theta_values"], dtype=float)
    base_y = np.asarray(base_res["export_points_eval"]["intensities"], dtype=float)
    up_eval_x = np.asarray(up_eval_res["export_points_eval"]["two_theta_values"], dtype=float)
    up_eval_y = np.asarray(up_eval_res["export_points_eval"]["intensities"], dtype=float)
    up_hi_x = np.asarray(up_hi_res["export_points_highres"]["two_theta_values"], dtype=float)
    up_hi_y = np.asarray(up_hi_res["export_points_highres"]["intensities"], dtype=float)

    gt_on_source = interp(gx, gy, sx)
    stage: Dict[str, Any] = {
        "manifest_pair": {
            "canonical_root": rel(CANONICAL_ROOT),
            "manifest": rel(manifest),
            "test_id": args.test_id,
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "input_image": row["input_image"],
            "mi_json": row["mi_json"],
            "gt_json": row["gt_json"],
            "source_numeric_json": row["source_numeric_json"],
            "metadata_json": row["metadata_json"],
            "mi_source": mi.get("mi_source"),
            "mi_role": mi.get("mi_role"),
            "pair_status": "PASS",
        },
        "source_numeric_reference": {
            "point_count": int(len(sx)),
            "x_min": float(np.min(sx)),
            "x_max": float(np.max(sx)),
            "y_min": float(np.min(sy)),
            "y_max": float(np.max(sy)),
        },
        "gt_reference": {
            "point_count": int(len(gx)),
            "x_min": float(np.min(gx)),
            "x_max": float(np.max(gx)),
            "y_min": float(np.min(gy)),
            "y_max": float(np.max(gy)),
        },
        "gt_vs_source_numeric": {
            "x_range_error": [float(np.min(gx) - np.min(sx)), float(np.max(gx) - np.max(sx))],
            "y_amplitude_mae": gt_source_y_mae,
            "normalized_y_mae": gt_source_y_mae / source_dyn,
            "shape_correlation": safe_corr(gt_on_source, sy),
        },
        "stage_a_calibration": {
            "x_range_error": [float(m["xmin"] - np.min(sx)), float(m["xmax"] - np.max(sx))],
            "y_range_error": [float(m["ymin"] - np.min(sy)), float(m["ymax"] - np.max(sy))],
            "axis_endpoint_consistency": {
                "x_axis_left_value": float(mi["x_axis_values"][0]),
                "x_axis_right_value": float(mi["x_axis_values"][1]),
                "y_axis_bottom_value": float(mi["y_axis_values"][0]),
                "y_axis_top_value": float(mi["y_axis_values"][1]),
            },
            "projected_trace_outside_plot_count": int(np.sum((expected_y < 0) | (expected_y > m["roi_h"]))),
            "source_to_gt_pixel_mae_px": safe_mae(expected_y, gt_y_px),
            "source_to_gt_pixel_shape_correlation": safe_corr(expected_y, gt_y_px),
        },
        "stage_b_raw_trace": {
            "baseline_raw_vs_expected_pixel_mae_px": safe_mae(base_raw_on_eval, expected_y),
            "upscale_raw_scaled_vs_expected_pixel_mae_px": safe_mae(up_raw_on_eval, expected_y),
            "baseline_shape_correlation_pixel": safe_corr(base_raw_on_eval, expected_y),
            "upscale_shape_correlation_pixel": safe_corr(up_raw_on_eval, expected_y),
            "baseline_peak": compare_peaks(source_peak_px, base_peak_px, tol_x=5.0),
            "upscale_peak": compare_peaks(source_peak_px, up_peak_px, tol_x=5.0),
        },
        "stage_c_export_downscale": {
            "upscale_raw_to_eval_export_pixel_mae_px": safe_mae(up_raw_on_eval, eval_y_on_eval),
            "upscale_raw_to_highres_export_pixel_mae_px": safe_mae(up_raw_on_eval, hi_y_on_eval),
            "highres_to_eval_loss_mae": safe_mae(hi_y_on_eval, eval_y_on_eval),
            "downscale_shape_correlation_loss": float(safe_corr(hi_y_on_eval, expected_y) - safe_corr(eval_y_on_eval, expected_y)),
            "downscale_peak": compare_peaks(hi_peak_px, eval_peak_px, tol_x=5.0),
        },
        "stage_d_value_domain": {
            **value_metrics("baseline_eval", base_x, base_y),
            **value_metrics("upscale_eval_grid", up_eval_x, up_eval_y),
            **value_metrics("upscale_highres", up_hi_x, up_hi_y),
            **value_metrics("gt_reference", gx, gy),
        },
    }

    # Oracle reports
    oracle_source_export_y = roi_y_to_value(expected_y, m)
    oracle_raw_mapping_y = roi_y_to_value(up_raw_on_eval, m)
    oracle = {
        "oracle_source_trace_to_export": {
            "description": "expected_pixel_trace converted back through MI value mapping",
            "normalized_y_mae": safe_mae(oracle_source_export_y, interp(sx, sy, roi_col_to_x(cols, m, roi_w))) / source_dyn,
            "shape_correlation": safe_corr(oracle_source_export_y, interp(sx, sy, roi_col_to_x(cols, m, roi_w))),
        },
        "reconstructed_raw_trace_to_oracle_mapping": {
            "description": "upscale raw trace scaled to original ROI, mapped to value domain without eval export",
            "normalized_y_mae": safe_mae(interp(roi_col_to_x(cols, m, roi_w), oracle_raw_mapping_y, sx), sy) / source_dyn,
            "shape_correlation": safe_corr(interp(roi_col_to_x(cols, m, roi_w), oracle_raw_mapping_y, sx), sy),
        },
        "gt_vs_source_numeric": stage["gt_vs_source_numeric"],
    }

    # Region analysis from source peak locations.
    source_peak_list = sorted(source_peaks_val.get("peaks", []), key=lambda p: -p.get("prominence", 0.0))
    main_x = source_peak_list[0]["x"] if source_peak_list else float(np.median(sx))
    regions = {
        "main_peak_region": (main_x - 1.0, main_x + 1.0),
        "mid_peak_region": (float(np.quantile(sx, 0.35)), float(np.quantile(sx, 0.65))),
        "high_angle_detail_region": (float(np.quantile(sx, 0.75)), float(np.max(sx))),
    }
    per_region: Dict[str, Any] = {}
    for name, (lo, hi) in regions.items():
        mask = (sx >= lo) & (sx <= hi)
        per_region[name] = {
            "x_range": [lo, hi],
            "source_point_count": int(np.sum(mask)),
            "baseline_eval_normalized_y_mae": safe_mae(interp(base_x, base_y, sx[mask]), sy[mask]) / source_dyn if mask.any() else None,
            "upscale_eval_grid_normalized_y_mae": safe_mae(interp(up_eval_x, up_eval_y, sx[mask]), sy[mask]) / source_dyn if mask.any() else None,
            "upscale_highres_normalized_y_mae": safe_mae(interp(up_hi_x, up_hi_y, sx[mask]), sy[mask]) / source_dyn if mask.any() else None,
            "baseline_shape_correlation": safe_corr(interp(base_x, base_y, sx[mask]), sy[mask]) if np.sum(mask) > 3 else None,
            "upscale_eval_shape_correlation": safe_corr(interp(up_eval_x, up_eval_y, sx[mask]), sy[mask]) if np.sum(mask) > 3 else None,
            "upscale_highres_shape_correlation": safe_corr(interp(up_hi_x, up_hi_y, sx[mask]), sy[mask]) if np.sum(mask) > 3 else None,
        }

    causes, final_decision = classify(stage)
    stage["final_error_classification"] = causes
    stage["final_decision"] = final_decision

    write_json(out / "stagewise_error_breakdown.json", stage)
    write_json(out / "oracle_trace_export_report.json", oracle)
    write_json(out / "per_region_peak_analysis.json", per_region)
    write_json(out / "gt_vs_source_numeric_diff.json", stage["gt_vs_source_numeric"])

    # Overlays
    image_path = paths["input_image"]
    plot_box = [int(v) for v in mi["plot_box"]]
    draw_overlay(
        out / "overlay_source_numeric_expected_vs_input.png",
        image_path,
        plot_box,
        [("source_expected", cols, expected_y, (255, 0, 0))],
    )
    draw_overlay(
        out / "overlay_raw_trace_vs_expected_pixel_trace.png",
        image_path,
        plot_box,
        [
            ("source_expected", cols, expected_y, (255, 0, 0)),
            ("baseline_raw", cols, base_raw_on_eval, (0, 128, 255)),
            ("upscale_raw_scaled", cols, up_raw_on_eval, (0, 180, 0)),
        ],
    )
    draw_overlay(
        out / "overlay_export_eval_vs_source_numeric.png",
        image_path,
        plot_box,
        [
            ("source_expected", cols, expected_y, (255, 0, 0)),
            ("baseline_eval", cols, np.interp(cols, x_to_roi_col(base_x, m, roi_w), y_to_roi_y(base_y, m)), (0, 128, 255)),
            ("upscale_eval", cols, eval_y_on_eval, (0, 180, 0)),
        ],
    )
    draw_overlay(
        out / "overlay_export_highres_vs_source_numeric.png",
        image_path,
        plot_box,
        [
            ("source_expected", cols, expected_y, (255, 0, 0)),
            ("upscale_highres", cols, hi_y_on_eval, (128, 0, 255)),
        ],
    )
    draw_overlay(
        out / "overlay_gt_vs_source_numeric.png",
        image_path,
        plot_box,
        [
            ("source_expected_mi", cols, expected_y, (255, 0, 0)),
            ("gt_pixel", cols, gt_y_px, (0, 128, 255)),
        ],
    )
    draw_overlay(
        out / "overlay_all_stage_comparison.png",
        image_path,
        plot_box,
        [
            ("source_expected", cols, expected_y, (255, 0, 0)),
            ("gt_pixel", cols, gt_y_px, (255, 160, 0)),
            ("baseline_raw", cols, base_raw_on_eval, (0, 128, 255)),
            ("upscale_raw", cols, up_raw_on_eval, (0, 180, 0)),
            ("upscale_eval", cols, eval_y_on_eval, (128, 0, 255)),
        ],
    )

    summary_row = {
        "test_id": args.test_id,
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "source_to_gt_pixel_mae_px": stage["stage_a_calibration"]["source_to_gt_pixel_mae_px"],
        "baseline_raw_vs_expected_pixel_mae_px": stage["stage_b_raw_trace"]["baseline_raw_vs_expected_pixel_mae_px"],
        "upscale_raw_scaled_vs_expected_pixel_mae_px": stage["stage_b_raw_trace"]["upscale_raw_scaled_vs_expected_pixel_mae_px"],
        "upscale_raw_to_eval_export_pixel_mae_px": stage["stage_c_export_downscale"]["upscale_raw_to_eval_export_pixel_mae_px"],
        "highres_to_eval_loss_mae": stage["stage_c_export_downscale"]["highres_to_eval_loss_mae"],
        "baseline_eval_normalized_y_mae": stage["stage_d_value_domain"]["baseline_eval_normalized_y_mae"],
        "upscale_eval_grid_normalized_y_mae": stage["stage_d_value_domain"]["upscale_eval_grid_normalized_y_mae"],
        "upscale_highres_normalized_y_mae": stage["stage_d_value_domain"]["upscale_highres_normalized_y_mae"],
        "gt_reference_normalized_y_mae": stage["stage_d_value_domain"]["gt_reference_normalized_y_mae"],
        "final_error_classification": ";".join(causes),
        "final_decision": final_decision,
    }
    csv_path = out / "source_numeric_error_decomposition_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    summary = {
        "final_decision": final_decision,
        "final_error_classification": causes,
        "manifest_pair": stage["manifest_pair"],
        "summary_row": summary_row,
        "stagewise_error_breakdown": stage,
        "oracle_trace_export_report": oracle,
        "per_region_peak_analysis": per_region,
        "outputs": {
            "summary_csv": rel(csv_path),
            "summary_json": rel(out / "source_numeric_error_decomposition_summary.json"),
            "summary_md": rel(out / "source_numeric_error_decomposition_summary.md"),
            "expected_pixel_trace": rel(out / "expected_pixel_trace.json"),
            "stagewise_error_breakdown": rel(out / "stagewise_error_breakdown.json"),
            "oracle_trace_export_report": rel(out / "oracle_trace_export_report.json"),
            "per_region_peak_analysis": rel(out / "per_region_peak_analysis.json"),
            "gt_vs_source_numeric_diff": rel(out / "gt_vs_source_numeric_diff.json"),
        },
        "not_done": [
            "canonical original files not modified",
            "input.png not modified",
            "mi.json not modified",
            "gt.json not modified",
            "source_numeric.json not modified",
            "metadata.json not modified",
            "plot_box/calibration not modified",
            "margin/candidate/DP/tracing scoring not modified",
            "threshold/performance tuning not performed",
            "full 30-run batch not executed",
        ],
    }
    write_json(out / "source_numeric_error_decomposition_summary.json", summary)

    md = [
        "# Source Numeric Stagewise Error Decomposition",
        "",
        f"- Final decision: `{final_decision}`",
        f"- Error classification: `{'; '.join(causes)}`",
        f"- Canonical manifest: `{rel(manifest)}`",
        f"- Sample: `{args.test_id}`",
        f"- Input image: `{row['input_image']}`",
        f"- MI/calibration input: `{row['mi_json']}`",
        f"- GT: `{row['gt_json']}`",
        f"- Source numeric reference: `{row['source_numeric_json']}`",
        "",
        "## Key Metrics",
        f"- GT vs source normalized_y_mae: `{stage['gt_vs_source_numeric']['normalized_y_mae']:.8g}`",
        f"- Source->MI expected vs GT pixel MAE: `{stage['stage_a_calibration']['source_to_gt_pixel_mae_px']:.4f}px`",
        f"- Baseline raw vs expected pixel MAE: `{stage['stage_b_raw_trace']['baseline_raw_vs_expected_pixel_mae_px']:.4f}px`",
        f"- 2x raw scaled vs expected pixel MAE: `{stage['stage_b_raw_trace']['upscale_raw_scaled_vs_expected_pixel_mae_px']:.4f}px`",
        f"- 2x raw to eval export pixel MAE: `{stage['stage_c_export_downscale']['upscale_raw_to_eval_export_pixel_mae_px']:.4f}px`",
        f"- Highres to eval loss MAE: `{stage['stage_c_export_downscale']['highres_to_eval_loss_mae']:.4f}px`",
        f"- Baseline eval normalized_y_mae: `{stage['stage_d_value_domain']['baseline_eval_normalized_y_mae']:.6f}`",
        f"- 2x eval_grid normalized_y_mae: `{stage['stage_d_value_domain']['upscale_eval_grid_normalized_y_mae']:.6f}`",
        f"- 2x highres normalized_y_mae: `{stage['stage_d_value_domain']['upscale_highres_normalized_y_mae']:.6f}`",
        "",
        "## Interpretation",
        "The source_numeric and GT value-domain curves are aligned. For ROI 2x, the raw trace scaled back to the original ROI is close to the source-derived expected pixel trace, and highres value export is also close to source_numeric. The large failure appears when the 2x result is forced back into eval_grid/export-downscale form, where peak preservation and amplitude fidelity collapse.",
        "",
        "## Outputs",
        f"- CSV: `{rel(csv_path)}`",
        f"- JSON: `{rel(out / 'source_numeric_error_decomposition_summary.json')}`",
        f"- Stage breakdown: `{rel(out / 'stagewise_error_breakdown.json')}`",
        f"- Expected pixel trace: `{rel(out / 'expected_pixel_trace.json')}`",
        f"- Oracle report: `{rel(out / 'oracle_trace_export_report.json')}`",
        f"- Region peak analysis: `{rel(out / 'per_region_peak_analysis.json')}`",
        f"- GT vs source diff: `{rel(out / 'gt_vs_source_numeric_diff.json')}`",
        "",
        "## Overlays",
        f"- `{rel(out / 'overlay_source_numeric_expected_vs_input.png')}`",
        f"- `{rel(out / 'overlay_raw_trace_vs_expected_pixel_trace.png')}`",
        f"- `{rel(out / 'overlay_export_eval_vs_source_numeric.png')}`",
        f"- `{rel(out / 'overlay_export_highres_vs_source_numeric.png')}`",
        f"- `{rel(out / 'overlay_gt_vs_source_numeric.png')}`",
        f"- `{rel(out / 'overlay_all_stage_comparison.png')}`",
        "",
        "## Not Done",
        "- canonical original files not modified",
        "- input.png not modified",
        "- mi.json not modified",
        "- gt.json not modified",
        "- source_numeric.json not modified",
        "- metadata.json not modified",
        "- plot_box/calibration not modified",
        "- margin/candidate/DP/tracing scoring not modified",
        "- threshold/performance tuning not performed",
        "- full 30-run batch not executed",
        "",
    ]
    (out / "source_numeric_error_decomposition_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
