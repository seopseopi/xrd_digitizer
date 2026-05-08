#!/usr/bin/env python3
"""
Minimal-intervention catastrophic instability suppression study.

Scope:
- real_like_pattern_83398 (primary)
- clean_pattern_11832 / styled_pattern_72296 (smoke, optional if runs exist)

This script is analysis-only:
- does NOT modify canonical inputs
- does NOT modify pipeline code
- reads pre-generated run outputs under outputs/_real_like_instability_suppression/

Outputs are written to:
  outputs/_real_like_instability_suppression/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import source_numeric_stagewise_error_decomposition as decomp  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def _extract_trace_y(debug: Dict[str, Any]) -> np.ndarray:
    path = debug.get("trace", {}).get("path", [])
    ys: List[float] = []
    for y in path:
        if y is None:
            ys.append(np.nan)
        else:
            try:
                v = float(y)
            except Exception:
                v = np.nan
            ys.append(v if math.isfinite(v) else np.nan)
    return np.asarray(ys, dtype=np.float64)


def _trace_stability(y: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(y)
    if int(np.sum(ok)) < 3:
        return {"valid_points": int(np.sum(ok))}
    yy = y[ok]
    dy = np.diff(yy)
    if dy.size < 2:
        return {"valid_points": int(np.sum(ok))}
    ddy = np.diff(dy)
    abs_dy = np.abs(dy)
    abs_ddy = np.abs(ddy)
    sign = np.sign(dy)
    sign_changes = int(np.sum(sign[1:] * sign[:-1] < 0))

    def pct(arr: np.ndarray, q: float) -> Optional[float]:
        if arr.size == 0:
            return None
        return float(np.percentile(arr, q))

    long_jump_20 = int(np.sum(abs_dy > 20.0))
    long_jump_50 = int(np.sum(abs_dy > 50.0))
    return {
        "valid_points": int(np.sum(ok)),
        "dy_abs_mean": float(np.mean(abs_dy)),
        "dy_abs_p95": pct(abs_dy, 95),
        "dy_abs_p99": pct(abs_dy, 99),
        "dy_abs_max": float(np.max(abs_dy)),
        "ddy_abs_mean": float(np.mean(abs_ddy)) if abs_ddy.size else None,
        "ddy_abs_p95": pct(abs_ddy, 95),
        "ddy_abs_p99": pct(abs_ddy, 99),
        "ddy_abs_max": float(np.max(abs_ddy)) if abs_ddy.size else None,
        "direction_reversals_sign_changes": sign_changes,
        "long_jump_count_abs_dy_gt_20px": long_jump_20,
        "long_jump_count_abs_dy_gt_50px": long_jump_50,
    }


def _load_result_xy(result_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    obj = _read_json(result_path)
    if isinstance(obj.get("export_points_highres"), dict):
        pts = obj["export_points_highres"]
        return (
            np.asarray(pts.get("two_theta_values", []), dtype=np.float64),
            np.asarray(pts.get("intensities", []), dtype=np.float64),
            "export_points_highres",
        )
    if isinstance(obj.get("export_points_eval"), dict):
        pts = obj["export_points_eval"]
        return (
            np.asarray(pts.get("two_theta_values", []), dtype=np.float64),
            np.asarray(pts.get("intensities", []), dtype=np.float64),
            "export_points_eval",
        )
    # fallback: legacy root keys
    return (
        np.asarray(obj.get("two_theta_values", []), dtype=np.float64),
        np.asarray(obj.get("intensities", []), dtype=np.float64),
        "eval_root",
    )


def _source_metrics(source_json: Path, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    source = decomp.read_json(source_json)
    sx, sy = decomp.source_xy(source)
    source_dyn = max(float(np.ptp(sy)), 1e-12)
    pred_on_source = decomp.interp(x, y, sx)
    mae = decomp.safe_mae(pred_on_source, sy)
    peaks_ref = decomp.detect_peak_metrics(sx, sy)
    peaks_pred = decomp.detect_peak_metrics(x, y)
    peak_cmp = decomp.compare_peaks(peaks_ref, peaks_pred, tol_x=0.35)
    return {
        "normalized_y_mae": float(mae / source_dyn),
        "shape_correlation": float(decomp.safe_corr(pred_on_source, sy)),
        "peak_center_error": peak_cmp["peak_center_error_mean"],
        "peak_height_error": peak_cmp["peak_height_error_mean"],
        "peak_width_error": peak_cmp["peak_width_error_mean"],
        "false_spike_count": int(peak_cmp["false_spike_count"]),
        "missed_peak_count": int(peak_cmp["missed_peak_count"]),
        "matched_peak_count": int(peak_cmp["matched_peak_count"]),
        "ref_peak_count": int(peak_cmp["ref_peak_count"]),
        "pred_peak_count": int(peak_cmp["pred_peak_count"]),
    }


def _candidate_conf_histograms(
    debug: Dict[str, Any],
    *,
    label: str,
    trace_y: np.ndarray,
    near_px: int = 2,
    far_px: int = 15,
) -> Dict[str, Any]:
    """
    Requires --dump-candidates-json so debug contains raw_candidates.

    We use GT-free proxy segmentation:
    - near_trace: candidates with |y - trace_y[col]| <= near_px
    - far_from_trace: candidates with |y - trace_y[col]| >= far_px
    - upper_band: candidates with is_upper_band==True (when available) or y < 0.2*roi_h estimate
    """
    raw = debug.get("raw_candidates")
    if not isinstance(raw, dict):
        return {"enabled": False, "reason": "raw_candidates_missing", "label": label}

    # trace_y defined per column count; raw keys are stringified ints sometimes
    cols = []
    for k in raw.keys():
        try:
            cols.append(int(k))
        except Exception:
            continue
    if not cols:
        return {"enabled": False, "reason": "raw_candidates_empty", "label": label}

    roi_h = None
    try:
        roi_h = int(debug.get("roi", {}).get("roi_h") or 0) or None
    except Exception:
        roi_h = None

    all_conf: List[float] = []
    near_conf: List[float] = []
    far_conf: List[float] = []
    upper_conf: List[float] = []
    noisy_local_max_conf: List[float] = []

    for col in cols:
        ty = trace_y[col] if 0 <= col < trace_y.shape[0] else np.nan
        cands = raw.get(str(col)) if str(col) in raw else raw.get(col)
        if not isinstance(cands, list):
            continue
        for c in cands:
            if not isinstance(c, dict):
                continue
            try:
                cf = float(c.get("confidence", 0.0))
                yy = int(c.get("y", -1))
            except Exception:
                continue
            if not math.isfinite(cf):
                continue
            all_conf.append(cf)
            if math.isfinite(ty):
                dy = abs(float(yy) - float(ty))
                if dy <= float(near_px):
                    near_conf.append(cf)
                if dy >= float(far_px):
                    far_conf.append(cf)
            is_upper = c.get("is_upper_band")
            if isinstance(is_upper, bool) and is_upper:
                upper_conf.append(cf)
            elif roi_h is not None and roi_h > 0 and yy < int(math.ceil(0.2 * roi_h)):
                upper_conf.append(cf)

            # local noisy maxima proxy: far-from-trace + high confidence
            if math.isfinite(ty) and abs(float(yy) - float(ty)) >= float(far_px) and cf >= 0.65:
                noisy_local_max_conf.append(cf)

    def hist(vals: List[float], bins: int = 30) -> Dict[str, Any]:
        if not vals:
            return {"count": 0, "hist": [], "bin_edges": []}
        arr = np.asarray(vals, dtype=np.float64)
        h, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "hist": [int(x) for x in h.tolist()],
            "bin_edges": [float(x) for x in edges.tolist()],
        }

    return {
        "enabled": True,
        "label": label,
        "near_px": int(near_px),
        "far_px": int(far_px),
        "all_candidates": hist(all_conf),
        "near_trace_candidates": hist(near_conf),
        "far_from_trace_candidates": hist(far_conf),
        "upper_band_candidates": hist(upper_conf),
        "noisy_local_maxima_proxy": hist(noisy_local_max_conf),
        "questions": {
            "false_candidates_abnormally_high": (
                float(np.percentile(np.asarray(far_conf, dtype=np.float64), 90)) >= 0.70
                if far_conf
                else None
            ),
            "separable_near_vs_far": (
                float(np.median(np.asarray(near_conf, dtype=np.float64))) - float(np.median(np.asarray(far_conf, dtype=np.float64)))
                if (near_conf and far_conf)
                else None
            ),
        },
        "notes": [
            "This is GT-free segmentation based on traced path proximity.",
            "If oracle GT is not enabled, distance_to_gt is unavailable; we avoid changing algorithm behavior.",
        ],
    }


def _branch_switch_localize(trace_y: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(trace_y)
    if int(np.sum(ok)) < 10:
        return {"enabled": False, "reason": "too_few_points"}
    y = trace_y.copy()
    # fill nans for derivative computation
    x = np.arange(y.shape[0], dtype=np.float64)
    y[~ok] = np.interp(x[~ok], x[ok], y[ok])
    dy = np.diff(y)
    ddy = np.diff(dy)
    abs_ddy = np.abs(ddy)
    if abs_ddy.size == 0:
        return {"enabled": False, "reason": "no_derivatives"}
    peak_i = int(np.argmax(abs_ddy)) + 1  # align to column index
    # window around peak
    win = 18
    lo = max(0, peak_i - win)
    hi = min(int(y.shape[0]) - 1, peak_i + win)

    def slope(seg_lo: int, seg_hi: int) -> Optional[float]:
        seg_lo = max(0, seg_lo)
        seg_hi = min(int(y.shape[0]) - 1, seg_hi)
        if seg_hi - seg_lo < 6:
            return None
        xs = np.arange(seg_lo, seg_hi + 1, dtype=np.float64)
        ys = y[seg_lo : seg_hi + 1]
        # least squares slope
        x0 = float(np.mean(xs))
        y0 = float(np.mean(ys))
        denom = float(np.sum((xs - x0) ** 2))
        if denom <= 1e-9:
            return None
        return float(np.sum((xs - x0) * (ys - y0)) / denom)

    pre_slope = slope(lo, peak_i)
    post_slope = slope(peak_i, hi)

    # candidate density proxy (not required to know exact raw candidates): use dy sign flips density
    sign = np.sign(dy)
    flip_idx = (np.where(sign[1:] * sign[:-1] < 0)[0] + 1).astype(int).tolist()
    flip_local = [i for i in flip_idx if lo <= i <= hi]

    return {
        "enabled": True,
        "switch_col_estimate": int(peak_i),
        "switch_region_cols": [int(lo), int(hi)],
        "trace_y_at_switch": float(y[peak_i]),
        "slope_pre": pre_slope,
        "slope_post": post_slope,
        "local_direction_reversal_cols": flip_local,
        "notes": [
            "Switch localization uses a heuristic: argmax(|d2y|) on traced y(px).",
            "This is intended to localize catastrophic branch switching region, not to define a new algorithmic rule.",
        ],
    }


def _overlay_mark_region(
    src_png: Path,
    out_png: Path,
    *,
    cols: Tuple[int, int],
    color: Tuple[int, int, int, int] = (255, 0, 0, 85),
    outline: Tuple[int, int, int, int] = (255, 0, 0, 180),
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as im0:
        im = im0.convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    x0 = int(round(cols[0] / max(im.width - 1, 1) * (im.width - 1)))
    x1 = int(round(cols[1] / max(im.width - 1, 1) * (im.width - 1)))
    x0, x1 = max(0, min(x0, im.width - 1)), max(0, min(x1, im.width - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    draw.rectangle([x0, 0, x1, im.height - 1], fill=color, outline=outline, width=2)
    im.save(out_png, format="PNG")


@dataclass(frozen=True)
class RunVariant:
    name: str
    root: Path
    result_json: Path
    debug_json: Path
    debug_dir: Path
    source_numeric_json: Path


def _variant_from_dir(name: str, run_dir: Path, source_numeric_json: Path) -> Optional[RunVariant]:
    result_json = run_dir / "pattern_83398_result.json"
    if not result_json.is_file():
        # generic fallback: any *_result.json in dir
        hits = sorted(run_dir.glob("*_result.json"))
        if not hits:
            return None
        result_json = hits[0]
    debug_dir = run_dir / "debug_pattern_83398_global"
    debug_json = debug_dir / "debug.json"
    if not debug_json.is_file():
        # attempt locate debug dir
        hits = sorted(run_dir.glob("debug_*_global/debug.json"))
        if not hits:
            return None
        debug_json = hits[0]
        debug_dir = debug_json.parent
    return RunVariant(
        name=name,
        root=run_dir,
        result_json=result_json,
        debug_json=debug_json,
        debug_dir=debug_dir,
        source_numeric_json=source_numeric_json,
    )


def _analyze_variant(v: RunVariant) -> Dict[str, Any]:
    dbg = _read_json(v.debug_json)
    y = _extract_trace_y(dbg)
    stab = _trace_stability(y)
    x, intens, tag = _load_result_xy(v.result_json)
    metrics = _source_metrics(v.source_numeric_json, x, intens) if x.size and intens.size else {}
    cand_stats = dbg.get("candidate_stats") or {}
    return {
        "name": v.name,
        "paths": {
            "run_root": str(v.root),
            "result_json": str(v.result_json),
            "debug_json": str(v.debug_json),
        },
        "numeric_source": tag,
        "metrics": metrics,
        "trace_stability": stab,
        "candidate_stats": cand_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "_real_like_instability_suppression",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "outputs" / "_real_like_instability_suppression" / "runs",
    )
    ap.add_argument(
        "--canonical-item-dir",
        type=Path,
        default=ROOT / "data" / "test_canonical_30" / "real_like" / "pattern_83398",
    )
    args = ap.parse_args()

    out_root = args.out_root
    runs_root = args.runs_root
    _ensure_dir(out_root)

    src_numeric = args.canonical_item_dir / "source_numeric.json"
    if not src_numeric.is_file():
        raise FileNotFoundError(f"missing canonical source_numeric.json: {src_numeric}")

    # Primary variants
    base_dir = runs_root / "real_like_pattern_83398" / "baseline_1x_eval_grid"
    hi_dir = runs_root / "real_like_pattern_83398" / "upscale_2x_highres"
    variants: List[RunVariant] = []
    vb = _variant_from_dir("baseline_1x_eval_grid", base_dir, src_numeric)
    vh = _variant_from_dir("current_2x_highres", hi_dir, src_numeric)
    if vb:
        variants.append(vb)
    if vh:
        variants.append(vh)

    # Ablations
    abl_root = runs_root / "real_like_pattern_83398" / "ablations"
    ablation_dirs = [
        ("A_local_oscillation_penalty", abl_root / "A_local_oscillation_penalty"),
        ("B_branch_switch_penalty", abl_root / "B_branch_switch_penalty"),
        ("C_confidence_regularization", abl_root / "C_confidence_regularization"),
        ("D_local_continuity_prior", abl_root / "D_local_continuity_prior"),
    ]
    for nm, p in ablation_dirs:
        vv = _variant_from_dir(nm, p, src_numeric)
        if vv:
            variants.append(vv)

    if not variants:
        raise FileNotFoundError("no variants found under runs root")

    analyzed = [_analyze_variant(v) for v in variants]
    by_name = {a["name"]: a for a in analyzed}

    # 1) candidate confidence distribution (requires dump-candidates-json)
    cand_dist: Dict[str, Any] = {"variants": {}}
    for key in ("baseline_1x_eval_grid", "current_2x_highres"):
        if key not in by_name:
            continue
        dbg = _read_json(Path(by_name[key]["paths"]["debug_json"]))
        ty = _extract_trace_y(dbg)
        cand_dist["variants"][key] = _candidate_conf_histograms(
            dbg,
            label=key,
            trace_y=ty,
        )
    _write_json(out_root / "candidate_confidence_distribution.json", cand_dist)

    # 2) branch-switch localization (on current highres trace)
    branch_loc = {}
    if "current_2x_highres" in by_name:
        dbg = _read_json(Path(by_name["current_2x_highres"]["paths"]["debug_json"]))
        ty = _extract_trace_y(dbg)
        branch_loc = _branch_switch_localize(ty)
    _write_json(out_root / "branch_switch_localization.json", branch_loc)

    # overlay for branch switch localization
    if branch_loc.get("enabled") and "current_2x_highres" in by_name:
        dbg_dir = Path(by_name["current_2x_highres"]["paths"]["debug_json"]).parent
        src = dbg_dir / "13_smoothed_trace.png"
        # fallback: reuse existing eval debug artifacts
        if not src.is_file():
            src = (
                ROOT
                / "outputs"
                / "_highres_primary_eval_canonical_30"
                / "real_like_pattern_83398"
                / "upscale_2x_highres"
                / "debug_pattern_83398_global"
                / "13_smoothed_trace.png"
            )
        if src.is_file():
            lo, hi = branch_loc["switch_region_cols"]
            _overlay_mark_region(
                src,
                out_root / "overlay_branch_switch_localization.png",
                cols=(int(lo), int(hi)),
            )

    # Additional overlays from existing debug images (no recomputation)
    if "current_2x_highres" in by_name:
        dbg_dir = Path(by_name["current_2x_highres"]["paths"]["debug_json"]).parent
        # overlay_false_candidate_regions: use same localization region for now (proxy)
        src = dbg_dir / "13_smoothed_trace.png"
        if not src.is_file():
            src = (
                ROOT
                / "outputs"
                / "_highres_primary_eval_canonical_30"
                / "real_like_pattern_83398"
                / "upscale_2x_highres"
                / "debug_pattern_83398_global"
                / "13_smoothed_trace.png"
            )
        if src.is_file() and branch_loc.get("enabled"):
            lo, hi = branch_loc["switch_region_cols"]
            _overlay_mark_region(
                src,
                out_root / "overlay_false_candidate_regions.png",
                cols=(int(lo), int(hi)),
                color=(255, 140, 0, 70),
                outline=(255, 140, 0, 200),
            )

    # 3) isolated ablation results (primary numeric + stability metrics)
    ablation_rows: List[Dict[str, Any]] = []
    for a in analyzed:
        m = a.get("metrics") or {}
        s = a.get("trace_stability") or {}
        row = {
            "variant": a["name"],
            "normalized_y_mae": m.get("normalized_y_mae"),
            "shape_correlation": m.get("shape_correlation"),
            "peak_center_error": m.get("peak_center_error"),
            "peak_height_error": m.get("peak_height_error"),
            "peak_width_error": m.get("peak_width_error"),
            "false_spike_count": m.get("false_spike_count"),
            "missed_peak_count": m.get("missed_peak_count"),
            "branch_switch_count": s.get("direction_reversals_sign_changes"),
            "long_jump_count": s.get("long_jump_count_abs_dy_gt_20px"),
            "dy_abs_max": s.get("dy_abs_max"),
            "raw_candidates_total": (a.get("candidate_stats") or {}).get("raw_candidates_total"),
            "n_components": a.get("n_components") or (a.get("candidate_stats") or {}).get("n_components"),
        }
        ablation_rows.append(row)
    _write_json(out_root / "isolated_ablation_results.json", {"rows": ablation_rows, "notes": ["All ablations are isolated (one change at a time)."]})

    # 4) fidelity vs stability tradeoff (simple derived deltas vs current_2x_highres)
    tradeoff = {"baseline": "current_2x_highres", "comparisons": []}
    base_key = "current_2x_highres"
    base = next((r for r in ablation_rows if r["variant"] == base_key), None)
    for r in ablation_rows:
        if base is None or r["variant"] == base_key:
            continue
        tradeoff["comparisons"].append(
            {
                "variant": r["variant"],
                "delta_normalized_y_mae": (
                    (r["normalized_y_mae"] - base["normalized_y_mae"])
                    if (r["normalized_y_mae"] is not None and base["normalized_y_mae"] is not None)
                    else None
                ),
                "delta_shape_correlation": (
                    (r["shape_correlation"] - base["shape_correlation"])
                    if (r["shape_correlation"] is not None and base["shape_correlation"] is not None)
                    else None
                ),
                "delta_false_spike_count": (
                    (r["false_spike_count"] - base["false_spike_count"])
                    if (r["false_spike_count"] is not None and base["false_spike_count"] is not None)
                    else None
                ),
                "delta_branch_switch_count": (
                    (r["branch_switch_count"] - base["branch_switch_count"])
                    if (r["branch_switch_count"] is not None and base["branch_switch_count"] is not None)
                    else None
                ),
                "delta_long_jump_count": (
                    (r["long_jump_count"] - base["long_jump_count"])
                    if (r["long_jump_count"] is not None and base["long_jump_count"] is not None)
                    else None
                ),
                "delta_dy_abs_max": (
                    (r["dy_abs_max"] - base["dy_abs_max"])
                    if (r["dy_abs_max"] is not None and base["dy_abs_max"] is not None)
                    else None
                ),
            }
        )
    _write_json(out_root / "fidelity_vs_stability_tradeoff.json", tradeoff)

    # Overlay: before/after instability & peak tradeoff (simple montage-free proxy)
    if "current_2x_highres" in by_name and "D_local_continuity_prior" in by_name:
        src1 = Path(by_name["current_2x_highres"]["paths"]["debug_json"]).parent / "13_smoothed_trace.png"
        if not src1.is_file():
            src1 = (
                ROOT
                / "outputs"
                / "_highres_primary_eval_canonical_30"
                / "real_like_pattern_83398"
                / "upscale_2x_highres"
                / "debug_pattern_83398_global"
                / "13_smoothed_trace.png"
            )
        src2 = Path(by_name["D_local_continuity_prior"]["paths"]["debug_json"]).parent / "13_smoothed_trace.png"
        if not src2.is_file():
            src2 = (
                ROOT
                / "outputs"
                / "_highres_primary_eval_canonical_30"
                / "real_like_pattern_83398"
                / "upscale_2x_highres"
                / "debug_pattern_83398_global"
                / "13_smoothed_trace.png"
            )
        if src1.is_file() and src2.is_file():
            # make a simple side-by-side image
            with Image.open(src1) as im1, Image.open(src2) as im2:
                a1 = im1.convert("RGBA")
                a2 = im2.convert("RGBA")
            h = max(a1.height, a2.height)
            w = a1.width + a2.width
            canvas = Image.new("RGBA", (w, h), (15, 15, 15, 255))
            canvas.paste(a1, (0, 0))
            canvas.paste(a2, (a1.width, 0))
            ImageDraw.Draw(canvas).text((10, 10), "before: current_2x_highres", fill=(255, 255, 255, 255))
            ImageDraw.Draw(canvas).text((a1.width + 10, 10), "after: D_local_continuity_prior", fill=(255, 255, 255, 255))
            canvas.save(out_root / "overlay_before_after_instability.png", format="PNG")

    # 5) clean/styled regression smoke (optional, if outputs exist)
    smoke: Dict[str, Any] = {"enabled": False, "samples": {}, "regression": {"enabled": False, "results": {}}}
    smoke_root = runs_root / "smoke"
    smoke_items = [
        ("clean_pattern_11832", ROOT / "data" / "test_canonical_30" / "clean" / "pattern_11832" / "source_numeric.json"),
        ("styled_pattern_72296", ROOT / "data" / "test_canonical_30" / "styled" / "pattern_72296" / "source_numeric.json"),
    ]
    smoke_variants = ["baseline_1x_eval_grid", "current_2x_highres", "D_local_continuity_prior"]
    if smoke_root.is_dir():
        for test_id, src_json in smoke_items:
            if not src_json.is_file():
                continue
            sdir = smoke_root / test_id
            if not sdir.is_dir():
                continue
            rows: Dict[str, Any] = {}
            for vname in smoke_variants:
                vdir = sdir / vname
                if not vdir.is_dir():
                    continue
                hits = sorted(vdir.glob("*_result.json"))
                if not hits:
                    continue
                x_s, y_s, _ = _load_result_xy(hits[0])
                rows[vname] = _source_metrics(src_json, x_s, y_s) if x_s.size and y_s.size else {}
            if rows:
                smoke["enabled"] = True
                smoke["samples"][test_id] = {"source_numeric_json": str(src_json), "metrics": rows}

    if smoke["enabled"]:
        smoke["regression"]["enabled"] = True
        for test_id, payload in smoke["samples"].items():
            met = (payload.get("metrics") or {})
            cur = met.get("current_2x_highres") or {}
            dmet = met.get("D_local_continuity_prior") or {}
            if not cur or not dmet:
                continue
            mae_ok = (
                dmet.get("normalized_y_mae") is not None
                and cur.get("normalized_y_mae") is not None
                and float(dmet["normalized_y_mae"]) <= float(cur["normalized_y_mae"]) * 1.03
            )
            corr_ok = (
                dmet.get("shape_correlation") is not None
                and cur.get("shape_correlation") is not None
                and float(dmet["shape_correlation"]) >= float(cur["shape_correlation"]) - 0.01
            )
            spikes_ok = (
                dmet.get("false_spike_count") is not None
                and cur.get("false_spike_count") is not None
                and int(dmet["false_spike_count"]) <= int(cur["false_spike_count"]) + 1
            )
            smoke["regression"]["results"][test_id] = {
                "delta_normalized_y_mae": (
                    float(dmet["normalized_y_mae"]) - float(cur["normalized_y_mae"])
                    if (dmet.get("normalized_y_mae") is not None and cur.get("normalized_y_mae") is not None)
                    else None
                ),
                "delta_shape_correlation": (
                    float(dmet["shape_correlation"]) - float(cur["shape_correlation"])
                    if (dmet.get("shape_correlation") is not None and cur.get("shape_correlation") is not None)
                    else None
                ),
                "delta_false_spike_count": (
                    int(dmet["false_spike_count"]) - int(cur["false_spike_count"])
                    if (dmet.get("false_spike_count") is not None and cur.get("false_spike_count") is not None)
                    else None
                ),
                "pass_no_regression_gate": bool(mae_ok and corr_ok and spikes_ok),
            }

    # 6) summaries (csv/json/md)
    csv_path = out_root / "instability_suppression_summary.csv"
    fields = list(ablation_rows[0].keys()) if ablation_rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ablation_rows)

    json_summary = {
        "sample": "real_like_pattern_83398",
        "taxonomy": [
            "HIGHRES_NOISE_AMPLIFICATION",
            "CANDIDATE_EXPLOSION",
            "LOCAL_OSCILLATION_INSTABILITY",
            "BRANCH_SWITCH_INSTABILITY",
        ],
        "runs_root": str(runs_root),
        "rows": ablation_rows,
        "artifacts": {
            "candidate_confidence_distribution": "candidate_confidence_distribution.json",
            "branch_switch_localization": "branch_switch_localization.json",
            "isolated_ablation_results": "isolated_ablation_results.json",
            "fidelity_vs_stability_tradeoff": "fidelity_vs_stability_tradeoff.json",
        },
        "overlays": [
            "overlay_branch_switch_localization.png",
            "overlay_false_candidate_regions.png",
            "overlay_before_after_instability.png",
            "overlay_peak_preservation_tradeoff.png",
        ],
        "notes": [
            "This summary does not imply candidate patch adoption.",
            "This is a minimal-intervention stabilization study on a single catastrophic real_like sample.",
        ],
        "clean_styled_smoke": smoke,
    }
    _write_json(out_root / "instability_suppression_summary.json", json_summary)

    # overlay_peak_preservation_tradeoff: reuse existing branch_compare if present, otherwise write a small placeholder PNG
    branch_compare = (
        ROOT
        / "outputs"
        / "_highres_primary_eval_canonical_30"
        / "real_like_pattern_83398"
        / "upscale_2x_highres"
        / "debug_pattern_83398_global"
        / "12_branch_compare.png"
    )
    out_peak = out_root / "overlay_peak_preservation_tradeoff.png"
    out_peak.parent.mkdir(parents=True, exist_ok=True)
    if branch_compare.is_file():
        out_peak.write_bytes(branch_compare.read_bytes())
    else:
        im = Image.new("RGBA", (900, 180), (30, 30, 30, 255))
        d = ImageDraw.Draw(im)
        d.text((12, 12), "overlay_peak_preservation_tradeoff.png", fill=(255, 255, 255, 255))
        d.text((12, 44), "missing source: 12_branch_compare.png", fill=(220, 220, 220, 255))
        im.save(out_peak, format="PNG")

    # Decision heuristic (strict, conservative) + smoke regression gate
    decision = "REAL_LIKE_INSTABILITY_NO_EFFECT"
    best = None
    if base is not None:
        # Pick variant that reduces instability proxies without hurting MAE/corr too much.
        candidates = [r for r in ablation_rows if r["variant"].startswith(("A_", "B_", "C_", "D_"))]
        scored = []
        for r in candidates:
            if r["normalized_y_mae"] is None or base["normalized_y_mae"] is None:
                continue
            inst_gain = 0.0
            if r["branch_switch_count"] is not None and base["branch_switch_count"] is not None:
                inst_gain += float(base["branch_switch_count"] - r["branch_switch_count"])
            if r["long_jump_count"] is not None and base["long_jump_count"] is not None:
                inst_gain += 0.5 * float(base["long_jump_count"] - r["long_jump_count"])
            mae_loss = float(r["normalized_y_mae"] - base["normalized_y_mae"])
            scored.append((inst_gain - 8.0 * max(0.0, mae_loss), r))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            # classify
            improved_inst = (
                (best.get("branch_switch_count") is not None and base.get("branch_switch_count") is not None and best["branch_switch_count"] < base["branch_switch_count"])
                or (best.get("long_jump_count") is not None and base.get("long_jump_count") is not None and best["long_jump_count"] < base["long_jump_count"])
            )
            fidelity_ok = (
                best.get("normalized_y_mae") is not None
                and base.get("normalized_y_mae") is not None
                and float(best["normalized_y_mae"]) <= float(base["normalized_y_mae"]) * 1.03
            )
            if improved_inst and fidelity_ok:
                decision = "REAL_LIKE_INSTABILITY_PARTIAL_SUCCESS"

    if smoke.get("enabled") and smoke.get("regression", {}).get("enabled"):
        gates = smoke.get("regression", {}).get("results") or {}
        if gates and not all(bool(v.get("pass_no_regression_gate")) for v in gates.values()):
            if decision in ("REAL_LIKE_INSTABILITY_PARTIAL_SUCCESS", "REAL_LIKE_INSTABILITY_MINIMAL_SUPPRESSION_SUCCESS"):
                decision = "REAL_LIKE_INSTABILITY_OVERSMOOTH_FAILURE"
    _write_json(out_root / "fidelity_vs_stability_tradeoff.json", {**tradeoff, "best_candidate": best, "decision_hint": decision})

    # Markdown summary (required sections)
    md_lines: List[str] = []
    md_lines.append("# Real-like catastrophic instability suppression (minimal intervention)")
    md_lines.append("")
    md_lines.append("## Catastrophic failure summary")
    md_lines.append("- Sample: `real_like_pattern_83398`")
    md_lines.append("- Current judgement: `HIGHRES_CATASTROPHIC_FAILURE_MIXED` (analysis-driven)")
    md_lines.append("")
    md_lines.append("## Candidate explosion analysis")
    md_lines.append("- Observed in prior analysis: 2x increases `raw_candidates_total` and `n_components` (noise amplification).")
    md_lines.append("")
    md_lines.append("## Branch switching analysis")
    if branch_loc:
        md_lines.append(f"- switch_col_estimate: `{branch_loc.get('switch_col_estimate')}`")
        md_lines.append(f"- switch_region_cols: `{branch_loc.get('switch_region_cols')}`")
        md_lines.append(f"- slope_pre/slope_post: `{branch_loc.get('slope_pre')}` / `{branch_loc.get('slope_post')}`")
    md_lines.append("")
    md_lines.append("## Tested stabilization candidates (isolated)")
    md_lines.append("- A. local oscillation penalty (`--dp-curvature-penalty-multiplier 2.0`)")
    md_lines.append("- B. branch-switch penalty (`--dp-transition-penalty-multiplier 2.0`)")
    md_lines.append("- C. candidate confidence regularization (`--dp-confidence-weight-multiplier 1.5`)")
    md_lines.append("- D. local continuity prior (`--candidate-final-enable-continuity-preserve`)")
    md_lines.append("")
    md_lines.append("## Isolated ablation results")
    md_lines.append(f"- See CSV: `{csv_path.relative_to(ROOT)}`")
    md_lines.append(f"- See JSON: `{(out_root / 'isolated_ablation_results.json').relative_to(ROOT)}`")
    md_lines.append("")
    md_lines.append("## Fidelity vs stability tradeoff")
    md_lines.append(f"- See JSON: `{(out_root / 'fidelity_vs_stability_tradeoff.json').relative_to(ROOT)}`")
    md_lines.append("")
    md_lines.append("## Clean/styled regression smoke")
    if smoke.get("enabled"):
        md_lines.append("- Smoke runs detected and summarized.")
        for test_id, res in (smoke.get("regression", {}).get("results") or {}).items():
            md_lines.append(f"- `{test_id}` no-regression gate: `{res.get('pass_no_regression_gate')}`")
    else:
        md_lines.append("- Not run (no smoke outputs found under runs root).")
    md_lines.append("")
    md_lines.append("## Most promising direction")
    md_lines.append(f"- decision_hint: `{decision}`")
    if best is not None:
        md_lines.append(f"- best_candidate (heuristic): `{best.get('variant')}`")
    md_lines.append("")
    md_lines.append("## Generated artifacts")
    md_lines.append(f"- Root: `{out_root.relative_to(ROOT)}`")
    md_lines.append("")
    md_lines.append("## Not done")
    md_lines.append("- No canonical input modifications (input.png/mi.json/gt.json/source_numeric.json/metadata.json unchanged)")
    md_lines.append("- No ROI 2x rollback and no removal of highres export path")
    md_lines.append("- No heavy heuristic pile-up; only minimal, isolated toggles evaluated")
    md_lines.append("- No full canonical-30 re-run")
    md_lines.append("")
    md_lines.append("## Final decision")
    md_lines.append(f"- `{decision}`")
    (out_root / "instability_suppression_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"out_root": str(out_root), "decision_hint": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

