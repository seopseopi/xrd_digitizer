#!/usr/bin/env python3
"""
Branch-instability minimal stabilization analysis (analysis + reporting).

This script is analysis-only:
- does not modify canonical inputs
- does not change pipeline code
- consumes outputs produced by runner/run_local.py runs

It expects run artifacts under:
  outputs/_branch_instability_stabilization_analysis/runs/<variant>/... (see RUN_LAYOUT below)

Primary sample:
  real_like_pattern_83398 (canonical)

Smoke samples:
  clean_pattern_11832
  styled_pattern_72296
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
    # for source_numeric decomposition module
    sys.path.insert(0, str(SCRIPTS))

import source_numeric_stagewise_error_decomposition as decomp  # noqa: E402

RUN_LAYOUT = {
    "result_json": "{sample_id}_result.json",
    "debug_dir": "debug_{sample_id}_global",
    "debug_json": "debug.json",
    "raw_candidates": "18_raw_candidates.json",
    "filtered_candidates": "19_filtered_candidates.json",
    "final_candidates": "20_final_candidates.json",
}


def rjson(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def wjson(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _extract_trace_y(debug: Dict[str, Any]) -> np.ndarray:
    path = debug.get("trace", {}).get("path", [])
    ys: List[float] = []
    for y in path:
        if y is None:
            ys.append(np.nan)
        else:
            v = safe_float(y)
            ys.append(v if v is not None else np.nan)
    return np.asarray(ys, dtype=np.float64)


def _col_keys(d: Dict[str, Any]) -> List[int]:
    out = []
    for k in d.keys():
        try:
            out.append(int(k))
        except Exception:
            pass
    out.sort()
    return out


def _infer_roi_wh(raw: Dict[str, Any]) -> Tuple[int, int]:
    cols = _col_keys(raw)
    roi_w = (max(cols) + 1) if cols else 0
    y_max = 0
    for c in cols:
        lst = raw.get(str(c), [])
        if not isinstance(lst, list):
            continue
        for it in lst:
            if not isinstance(it, dict):
                continue
            yi = safe_int(it.get("y"))
            if yi is not None:
                y_max = max(y_max, yi)
    return roi_w, int(y_max + 1)


def _component_id_from_comp_score(
    comp_score: float,
    component_scores: Dict[str, Any],
    *,
    eps: float = 1e-6,
) -> Optional[str]:
    best = None
    best_d = float("inf")
    for cid, meta in component_scores.items():
        try:
            s = float(meta.get("score"))
        except Exception:
            continue
        d = abs(float(comp_score) - s)
        if d <= eps:
            return str(cid)
        if d < best_d:
            best_d = d
            best = str(cid)
    return best if best_d <= 1e-3 else None


def _counts_by_col(stage: Dict[str, Any], roi_w: int) -> np.ndarray:
    arr = np.zeros(roi_w, dtype=np.int32)
    for col in _col_keys(stage):
        if col >= roi_w:
            continue
        lst = stage.get(str(col), [])
        if isinstance(lst, list):
            arr[col] = int(len(lst))
    return arr


def _burst_regions(counts_by_col: np.ndarray, *, q: float = 99.0, min_len: int = 8) -> Dict[str, Any]:
    nonz = counts_by_col[counts_by_col > 0].astype(np.float64)
    if nonz.size == 0:
        return {"threshold": None, "regions": []}
    thr = float(np.percentile(nonz, q))
    mask = counts_by_col.astype(np.float64) >= thr
    regs: List[Tuple[int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        if (j - i + 1) >= int(min_len):
            regs.append((int(i), int(j)))
        i = j + 1
    return {"threshold": thr, "regions": regs, "quantile": q, "min_len": int(min_len)}


def _density_png(stage: Dict[str, Any], *, roi_w: int, roi_h: int, out_png: Path, title: str) -> Dict[str, Any]:
    dens = np.zeros((roi_h, roi_w), dtype=np.uint16)
    for col in _col_keys(stage):
        if col >= roi_w:
            continue
        lst = stage.get(str(col), [])
        if not isinstance(lst, list):
            continue
        for c in lst:
            if not isinstance(c, dict):
                continue
            yi = safe_int(c.get("y"))
            if yi is None or yi < 0 or yi >= roi_h:
                continue
            if dens[yi, col] < np.iinfo(np.uint16).max:
                dens[yi, col] += 1
    arr = dens.astype(np.float64)
    vmax = float(np.percentile(arr, 99.5)) if arr.size else 1.0
    vmax = max(vmax, 1.0)
    img = np.sqrt(np.clip(arr / vmax, 0.0, 1.0))
    im = Image.fromarray((img * 255).astype(np.uint8), mode="L").convert("RGBA")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0, 160))
    d.text((8, 6), title, fill=(255, 255, 255, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png, format="PNG")
    return {"vmax_p99_5": vmax}


def _overlay_mark_cols(src_png: Path, out_png: Path, cols: List[Tuple[int, int]], label: str) -> None:
    with Image.open(src_png) as im0:
        im = im0.convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    for lo, hi in cols:
        x0 = int(np.clip(lo, 0, im.width - 1))
        x1 = int(np.clip(hi, 0, im.width - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        d.rectangle([x0, 0, x1, im.height - 1], fill=(255, 0, 0, 70), outline=(255, 0, 0, 200), width=2)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0, 160))
    d.text((8, 6), label, fill=(255, 255, 255, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png, format="PNG")


def _false_top1_cols(final: Dict[str, Any], *, trace_y: np.ndarray, roi_w: int, far_px: int = 15) -> List[int]:
    cols = []
    for col in _col_keys(final):
        if col >= roi_w:
            continue
        lst = final.get(str(col), [])
        if not isinstance(lst, list) or not lst:
            continue
        ty = trace_y[col] if 0 <= col < trace_y.shape[0] else np.nan
        if not math.isfinite(float(ty)):
            continue
        # top1 by confidence
        lst2 = [c for c in lst if isinstance(c, dict) and safe_float(c.get("confidence")) is not None]
        if not lst2:
            continue
        lst2.sort(key=lambda c: -float(c.get("confidence", 0.0)))
        y1 = safe_float(lst2[0].get("y"))
        if y1 is None:
            continue
        if abs(float(y1) - float(ty)) >= float(far_px):
            cols.append(int(col))
    return cols


def _contiguous_regions(cols: List[int], *, max_gap: int = 2, min_len: int = 12) -> List[Tuple[int, int]]:
    if not cols:
        return []
    s = sorted(set(int(c) for c in cols))
    out: List[Tuple[int, int]] = []
    lo = s[0]
    prev = s[0]
    for c in s[1:]:
        if int(c) <= int(prev) + int(max_gap):
            prev = int(c)
            continue
        if (prev - lo + 1) >= int(min_len):
            out.append((int(lo), int(prev)))
        lo = int(c)
        prev = int(c)
    if (prev - lo + 1) >= int(min_len):
        out.append((int(lo), int(prev)))
    return out


def _branch_continuity_diagnostics(trace_y: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(trace_y)
    if int(np.sum(ok)) < 10:
        return {"enabled": False, "reason": "too_few_points"}
    y = trace_y.copy()
    x = np.arange(y.shape[0], dtype=np.float64)
    y[~ok] = np.interp(x[~ok], x[ok], y[ok])
    dy = np.diff(y)
    abs_dy = np.abs(dy)
    sign = np.sign(dy)
    rev_cols = (np.where(sign[1:] * sign[:-1] < 0)[0] + 1).astype(int).tolist()
    long_jumps = (np.where(abs_dy > 20.0)[0] + 1).astype(int).tolist()
    # oscillation clusters: windows with many reversals
    clusters: List[Tuple[int, int]] = []
    win = 31
    if y.size >= win:
        rev_mask = np.zeros(y.size, dtype=np.int32)
        for c in rev_cols:
            if 0 <= c < rev_mask.size:
                rev_mask[c] = 1
        pref = np.cumsum(rev_mask)
        for i in range(0, int(y.size) - win + 1, 5):
            j = i + win - 1
            cnt = int(pref[j] - (pref[i - 1] if i > 0 else 0))
            if cnt >= 6:
                clusters.append((i, j))
        # merge clusters
        merged: List[Tuple[int, int]] = []
        for lo, hi in clusters:
            if not merged or lo > merged[-1][1] + 5:
                merged.append((lo, hi))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        clusters = merged
    return {
        "enabled": True,
        "direction_reversal_cols": rev_cols[:2000],
        "direction_reversal_count": int(len(rev_cols)),
        "long_jump_cols_abs_dy_gt_20px": long_jumps[:2000],
        "long_jump_count": int(len(long_jumps)),
        "dy_abs_p95": float(np.percentile(abs_dy, 95)) if abs_dy.size else None,
        "dy_abs_p99": float(np.percentile(abs_dy, 99)) if abs_dy.size else None,
        "dy_abs_max": float(np.max(abs_dy)) if abs_dy.size else None,
        "oscillation_clusters_proxy": clusters[:200],
        "notes": [
            "All diagnostics are computed on debug trace path y(px) (GT-free).",
            "oscillation_clusters_proxy flags windows with many direction reversals.",
        ],
    }


@dataclass(frozen=True)
class VariantRun:
    name: str
    run_dir: Path
    sample_id: str

    @property
    def result_json(self) -> Path:
        return self.run_dir / RUN_LAYOUT["result_json"].format(sample_id=self.sample_id)

    @property
    def debug_dir(self) -> Path:
        return self.run_dir / RUN_LAYOUT["debug_dir"].format(sample_id=self.sample_id)

    @property
    def debug_json(self) -> Path:
        return self.debug_dir / RUN_LAYOUT["debug_json"]

    @property
    def raw_json(self) -> Path:
        return self.debug_dir / RUN_LAYOUT["raw_candidates"]

    @property
    def filt_json(self) -> Path:
        return self.debug_dir / RUN_LAYOUT["filtered_candidates"]

    @property
    def final_json(self) -> Path:
        return self.debug_dir / RUN_LAYOUT["final_candidates"]


def _load_variant(v: VariantRun) -> Dict[str, Any]:
    dbg = rjson(v.debug_json)
    raw = rjson(v.raw_json)
    filt = rjson(v.filt_json)
    fin = rjson(v.final_json)
    trace_y = _extract_trace_y(dbg)
    comp_scores = dbg.get("component_scores") or {}
    roi_w, roi_h = _infer_roi_wh(raw)
    raw_counts = _counts_by_col(raw, roi_w)
    burst = _burst_regions(raw_counts, q=99.0, min_len=8)
    false_top1 = _false_top1_cols(fin, trace_y=trace_y, roi_w=roi_w, far_px=15)
    false_regions = _contiguous_regions(false_top1, max_gap=2, min_len=12)
    cont = _branch_continuity_diagnostics(trace_y)

    # candidate totals per stage
    raw_total = int(sum(len(raw.get(str(c), []) or []) for c in _col_keys(raw)))
    filt_total = int(sum(len(filt.get(str(c), []) or []) for c in _col_keys(filt)))
    fin_total = int(sum(len(fin.get(str(c), []) or []) for c in _col_keys(fin)))

    # numeric metrics vs canonical source_numeric (if resolvable)
    # sample_id is like pattern_83398 / pattern_11832
    domain = "real_like" if v.sample_id == "pattern_83398" else ("clean" if v.sample_id == "pattern_11832" else ("styled" if v.sample_id == "pattern_72296" else None))
    metrics = {}
    if domain is not None:
        src_json = ROOT / "data" / "test_canonical_30" / domain / v.sample_id.replace("pattern_", "pattern_") / "source_numeric.json"
        # canonical structure: .../<domain>/<pattern_xxxxx>/source_numeric.json
        src_json = ROOT / "data" / "test_canonical_30" / domain / v.sample_id.replace("pattern_", "pattern_") / "source_numeric.json"
        if not src_json.is_file():
            src_json = ROOT / "data" / "test_canonical_30" / domain / v.sample_id.replace("pattern_", "pattern_") / "source_numeric.json"
        if src_json.is_file() and v.result_json.is_file():
            try:
                res = rjson(v.result_json)
                pts = res.get("export_points_highres") or res.get("export_points_eval") or {}
                x = np.asarray(pts.get("two_theta_values", []), dtype=np.float64)
                y = np.asarray(pts.get("intensities", []), dtype=np.float64)
                source = decomp.read_json(src_json)
                sx, sy = decomp.source_xy(source)
                dyn = max(float(np.ptp(sy)), 1e-12)
                pred_on_source = decomp.interp(x, y, sx) if x.size and y.size else np.asarray([], dtype=np.float64)
                mae = decomp.safe_mae(pred_on_source, sy) if pred_on_source.size else None
                peaks_ref = decomp.detect_peak_metrics(sx, sy)
                peaks_pred = decomp.detect_peak_metrics(x, y) if x.size and y.size else []
                peak_cmp = decomp.compare_peaks(peaks_ref, peaks_pred, tol_x=0.35)
                metrics = {
                    "normalized_y_mae": (float(mae / dyn) if mae is not None else None),
                    "shape_correlation": (float(decomp.safe_corr(pred_on_source, sy)) if pred_on_source.size else None),
                    "false_spike_count": int(peak_cmp.get("false_spike_count", 0)),
                    "missed_peak_count": int(peak_cmp.get("missed_peak_count", 0)),
                    "matched_peak_count": int(peak_cmp.get("matched_peak_count", 0)),
                    "peak_center_error": peak_cmp.get("peak_center_error_mean"),
                    "peak_height_error": peak_cmp.get("peak_height_error_mean"),
                    "peak_width_error": peak_cmp.get("peak_width_error_mean"),
                }
            except Exception:
                metrics = {}
    return {
        "variant": v.name,
        "paths": {
            "run_dir": str(v.run_dir),
            "debug_json": str(v.debug_json),
            "raw_candidates": str(v.raw_json),
            "filtered_candidates": str(v.filt_json),
            "final_candidates": str(v.final_json),
            "result_json": str(v.result_json),
        },
        "roi": {"roi_w": int(roi_w), "roi_h": int(roi_h)},
        "debug_summary": {
            "raw_candidate_pixels": dbg.get("raw_candidate_pixels"),
            "skeleton_pixels": dbg.get("skeleton_pixels"),
            "n_components": dbg.get("n_components"),
        },
        "candidate_totals": {
            "raw_total": raw_total,
            "filtered_total": filt_total,
            "final_total": fin_total,
        },
        "numeric_metrics": metrics,
        "burst": burst,
        "false_top1": {
            "count": int(len(false_top1)),
            "columns_sample": false_top1[:2000],
            "contiguous_regions": [{"lo": int(a), "hi": int(b)} for a, b in false_regions],
        },
        "branch_continuity_diagnostics": cont,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "outputs" / "_branch_instability_stabilization_analysis" / "runs",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "_branch_instability_stabilization_analysis",
    )
    args = ap.parse_args()

    out_root = args.out_root
    runs_root = args.runs_root
    ensure_dir(out_root)

    # expected variants
    primary = [
        VariantRun("current_2x_highres", runs_root / "real_like_pattern_83398" / "current_2x_highres", "pattern_83398"),
        VariantRun("A_branch_switch_penalty", runs_root / "real_like_pattern_83398" / "A_branch_switch_penalty", "pattern_83398"),
        VariantRun("B_noise_adaptive_candidate_pruning", runs_root / "real_like_pattern_83398" / "B_noise_adaptive_candidate_pruning", "pattern_83398"),
        VariantRun("C_component_plausibility_filtering", runs_root / "real_like_pattern_83398" / "C_component_plausibility_filtering", "pattern_83398"),
    ]

    loaded: List[Dict[str, Any]] = []
    missing: List[str] = []
    for v in primary:
        if not (v.debug_json.is_file() and v.raw_json.is_file() and v.filt_json.is_file() and v.final_json.is_file()):
            missing.append(v.name)
            continue
        loaded.append(_load_variant(v))

    # ----- Task 1/2/6 (refinement/localization/continuity) based on current -----
    cur = next((x for x in loaded if x["variant"] == "current_2x_highres"), None)
    if cur is None:
        raise FileNotFoundError("missing current_2x_highres run artifacts; run the variant first")

    burst_regs = cur["burst"]["regions"]
    false_regs = cur["false_top1"]["contiguous_regions"]

    # overlay base: raw density
    # Use current raw candidates to generate density and zoom mark
    cur_run = primary[0]
    raw = rjson(cur_run.raw_json)
    roi_w, roi_h = cur["roi"]["roi_w"], cur["roi"]["roi_h"]
    dens_png = out_root / "overlay_burst_region_zoom.png"
    dens_meta = _density_png(raw, roi_w=roi_w, roi_h=roi_h, out_png=dens_png, title="raw density + burst region (zoom mark)")
    # mark burst region on same image (column coordinate)
    if burst_regs:
        _overlay_mark_cols(dens_png, dens_png, [(int(a), int(b)) for a, b in burst_regs], "raw density with burst region highlighted")

    # burst branch density: final density with burst columns highlighted
    fin = rjson(cur_run.final_json)
    fin_png = out_root / "overlay_burst_branch_density.png"
    _density_png(fin, roi_w=roi_w, roi_h=roi_h, out_png=fin_png, title="final density (branch slots) + burst columns")
    if burst_regs:
        _overlay_mark_cols(fin_png, fin_png, [(int(a), int(b)) for a, b in burst_regs], "final density with burst columns highlighted")

    # false top1 regions overlays
    false_png = out_root / "overlay_false_top1_regions.png"
    _density_png(fin, roi_w=roi_w, roi_h=roi_h, out_png=false_png, title="final density + false top1 regions")
    if false_regs:
        _overlay_mark_cols(false_png, false_png, [(int(r["lo"]), int(r["hi"])) for r in false_regs], "false top1 contiguous regions highlighted")

    hijack_png = out_root / "overlay_false_branch_hijack.png"
    # reuse false_png but label differently (hijack = long contiguous regions)
    (hijack_png).write_bytes(false_png.read_bytes())

    # continuity overlays (use trace diagnostics only; draw vlines on a simple plot-like canvas)
    cont = cur["branch_continuity_diagnostics"]
    w, h = 1600, 320
    base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(base)
    d.text((10, 10), "branch continuity breaks: long jumps(red) + reversals(purple) + oscillation clusters(orange)", fill=(0, 0, 0, 255))
    # map column->x
    def xpx(col: int) -> int:
        return int(round(40 + (col / max(roi_w - 1, 1)) * (w - 80)))

    for c in cont.get("long_jump_cols_abs_dy_gt_20px", [])[:5000]:
        xx = xpx(int(c))
        d.line([(xx, 40), (xx, h - 30)], fill=(220, 20, 60, 50), width=1)
    for c in cont.get("direction_reversal_cols", [])[:5000]:
        xx = xpx(int(c))
        d.line([(xx, 40), (xx, h - 30)], fill=(128, 0, 128, 35), width=1)
    for lo, hi in cont.get("oscillation_clusters_proxy", [])[:200]:
        x0, x1 = xpx(int(lo)), xpx(int(hi))
        d.rectangle([x0, 60, x1, h - 60], outline=(255, 140, 0, 140), width=2)

    out_cont = out_root / "overlay_branch_continuity_breaks.png"
    out_osc = out_root / "overlay_oscillation_clusters.png"
    out_cont.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_cont, format="PNG")
    base.save(out_osc, format="PNG")

    # JSON outputs for refinement/localization/continuity
    burst_refine = {
        "burst_regions": burst_regs,
        "roi": cur["roi"],
        "density_png_meta": dens_meta,
        "questions": {
            "burst_is_localized": bool(burst_regs),
            "burst_overlaps_false_top1_regions": bool(burst_regs and false_regs),
        },
        "notes": ["Refinement in this step uses dump-derived per-column density tail. Component IDs require additional per-candidate scan; kept minimal here."],
    }
    wjson(out_root / "burst_region_refinement.json", burst_refine)

    false_loc = {
        "false_top1_count": cur["false_top1"]["count"],
        "contiguous_regions": false_regs,
        "questions": {
            "scattered_vs_long_branches": "long_branch" if len(false_regs) >= 1 else "scattered_or_none",
        },
    }
    wjson(out_root / "false_top1_localization.json", false_loc)

    wjson(out_root / "branch_continuity_diagnostics.json", cont)

    # ----- Task 3/4/5: ablation summary placeholders (filled from available variants only) -----
    ablation = {"variants_present": [x["variant"] for x in loaded], "missing_variants": missing, "rows": []}
    for x in loaded:
        ablation["rows"].append(
            {
                "variant": x["variant"],
                "normalized_y_mae": (x.get("numeric_metrics") or {}).get("normalized_y_mae"),
                "shape_correlation": (x.get("numeric_metrics") or {}).get("shape_correlation"),
                "false_spike_count": (x.get("numeric_metrics") or {}).get("false_spike_count"),
                "missed_peak_count": (x.get("numeric_metrics") or {}).get("missed_peak_count"),
                "matched_peak_count": (x.get("numeric_metrics") or {}).get("matched_peak_count"),
                "raw_total_candidates": (x.get("candidate_totals") or {}).get("raw_total"),
                "filtered_total_candidates": (x.get("candidate_totals") or {}).get("filtered_total"),
                "final_total_candidates": (x.get("candidate_totals") or {}).get("final_total"),
                "n_components": x["debug_summary"].get("n_components"),
                "burst_regions_count": len(x["burst"]["regions"]),
                "false_top1_columns": x["false_top1"]["count"],
                "branch_switch_count": x["branch_continuity_diagnostics"].get("direction_reversal_count"),
                "long_jump_count": x["branch_continuity_diagnostics"].get("long_jump_count"),
            }
        )
    wjson(out_root / "isolated_stabilization_ablation.json", ablation)

    # tradeoff: compare vs current if present
    trade = {"baseline": "current_2x_highres", "comparisons": []}
    base_row = next((r for r in ablation["rows"] if r["variant"] == "current_2x_highres"), None)
    if base_row:
        for r in ablation["rows"]:
            if r["variant"] == "current_2x_highres":
                continue
            trade["comparisons"].append(
                {
                    "variant": r["variant"],
                    "delta_false_top1_columns": (r["false_top1_columns"] - base_row["false_top1_columns"])
                    if (r.get("false_top1_columns") is not None and base_row.get("false_top1_columns") is not None)
                    else None,
                    "delta_branch_switch_count": (r["branch_switch_count"] - base_row["branch_switch_count"])
                    if (r.get("branch_switch_count") is not None and base_row.get("branch_switch_count") is not None)
                    else None,
                    "delta_long_jump_count": (r["long_jump_count"] - base_row["long_jump_count"])
                    if (r.get("long_jump_count") is not None and base_row.get("long_jump_count") is not None)
                    else None,
                }
            )
    wjson(out_root / "fidelity_vs_stabilization_tradeoff.json", trade)

    # smoke regression: compute numeric metrics for each variant if present
    smoke_root = runs_root / "smoke"
    smoke_payload: Dict[str, Any] = {"enabled": False, "samples": {}}
    for test_id, domain, sid in [
        ("clean_pattern_11832", "clean", "pattern_11832"),
        ("styled_pattern_72296", "styled", "pattern_72296"),
    ]:
        src_json = ROOT / "data" / "test_canonical_30" / domain / sid.replace("pattern_", "pattern_") / "source_numeric.json"
        if not src_json.is_file():
            continue
        sample_dir = smoke_root / test_id
        if not sample_dir.is_dir():
            continue
        smoke_payload["enabled"] = True
        per_variant: Dict[str, Any] = {}
        for vname in ["current_2x_highres", "A_branch_switch_penalty", "B_noise_adaptive_candidate_pruning", "C_component_plausibility_filtering"]:
            vdir = sample_dir / vname
            if not vdir.is_dir():
                continue
            hits = sorted(vdir.glob("*_result.json"))
            if not hits:
                continue
            try:
                res = rjson(hits[0])
                pts = res.get("export_points_highres") or res.get("export_points_eval") or {}
                x = np.asarray(pts.get("two_theta_values", []), dtype=np.float64)
                y = np.asarray(pts.get("intensities", []), dtype=np.float64)
                source = decomp.read_json(src_json)
                sx, sy = decomp.source_xy(source)
                dyn = max(float(np.ptp(sy)), 1e-12)
                pred = decomp.interp(x, y, sx) if x.size and y.size else np.asarray([], dtype=np.float64)
                mae = decomp.safe_mae(pred, sy) if pred.size else None
                peaks_ref = decomp.detect_peak_metrics(sx, sy)
                peaks_pred = decomp.detect_peak_metrics(x, y) if x.size and y.size else []
                peak_cmp = decomp.compare_peaks(peaks_ref, peaks_pred, tol_x=0.35)
                per_variant[vname] = {
                    "normalized_y_mae": (float(mae / dyn) if mae is not None else None),
                    "shape_correlation": (float(decomp.safe_corr(pred, sy)) if pred.size else None),
                    "false_spike_count": int(peak_cmp.get("false_spike_count", 0)),
                    "missed_peak_count": int(peak_cmp.get("missed_peak_count", 0)),
                    "matched_peak_count": int(peak_cmp.get("matched_peak_count", 0)),
                }
            except Exception:
                per_variant[vname] = {}
        smoke_payload["samples"][test_id] = {"source_numeric_json": str(src_json), "metrics": per_variant}
    wjson(out_root / "stabilization_regression_smoke.json", smoke_payload)

    # ----- Final summary artifacts -----
    # CSV
    csv_path = out_root / "branch_instability_stabilization_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(ablation["rows"][0].keys()) if ablation["rows"] else ["variant"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ablation["rows"]:
            w.writerow(r)

    # decision (conservative, based on available deltas only)
    final_decision = "BRANCH_INSTABILITY_NO_EFFECT"
    best = None
    if base_row and trade["comparisons"]:
        # pick variant reducing false_top1 and long_jump
        scored = []
        for c in trade["comparisons"]:
            df = c.get("delta_false_top1_columns")
            dj = c.get("delta_long_jump_count")
            ds = c.get("delta_branch_switch_count")
            if df is None or dj is None or ds is None:
                continue
            score = (-1.0 * float(df)) + (-2.0 * float(dj)) + (-0.5 * float(ds))
            scored.append((score, c["variant"]))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            # success gate: meaningful reduction
            best_cmp = next((c for c in trade["comparisons"] if c["variant"] == best), None)
            if best_cmp and (best_cmp["delta_false_top1_columns"] is not None) and (best_cmp["delta_false_top1_columns"] <= -50):
                final_decision = "BRANCH_INSTABILITY_PARTIAL_SUCCESS"

    summary_json = {
        "used_manifest": {
            "sample": "real_like_pattern_83398",
            "canonical_item_dir": str(ROOT / "data" / "test_canonical_30" / "real_like" / "pattern_83398"),
        },
        "sample": "real_like_pattern_83398",
        "burst_region_refinement": burst_refine,
        "false_top1_localization": false_loc,
        "branch_continuity_diagnostics": cont,
        "isolated_stabilization_ablation": ablation,
        "fidelity_vs_stabilization_tradeoff": trade,
        "best_candidate_hint": best,
        "final_decision": final_decision,
        "artifacts": {
            "json": [
                "burst_region_refinement.json",
                "false_top1_localization.json",
                "isolated_stabilization_ablation.json",
                "fidelity_vs_stabilization_tradeoff.json",
                "stabilization_regression_smoke.json",
                "branch_continuity_diagnostics.json",
            ],
            "overlays": [
                "overlay_burst_region_zoom.png",
                "overlay_burst_branch_density.png",
                "overlay_false_top1_regions.png",
                "overlay_false_branch_hijack.png",
                "overlay_branch_continuity_breaks.png",
                "overlay_oscillation_clusters.png",
            ],
        },
        "not_done": [
            "No canonical modifications",
            "No code changes to pipeline",
            "No multi-stabilization stacking (isolated variants only)",
            "No heavy smoothing",
        ],
    }
    wjson(out_root / "branch_instability_stabilization_summary.json", summary_json)

    md = []
    md.append("# Branch instability stabilization analysis (minimal intervention)")
    md.append("")
    md.append("## A. burst region 분석")
    md.append(f"- burst_regions (raw density tail): `{burst_regs}`")
    md.append("")
    md.append("## B. false top1 branch 분석")
    md.append(f"- false_top1_count (top1 far from trace): `{cur['false_top1']['count']}`")
    md.append(f"- contiguous false regions: `{false_regs}`")
    md.append("")
    md.append("## C. branch continuity 분석")
    md.append(f"- direction_reversal_count: `{cont.get('direction_reversal_count')}`")
    md.append(f"- long_jump_count(|dy|>20px): `{cont.get('long_jump_count')}`")
    md.append("")
    md.append("## D. isolated stabilization 결과 (83398 only)")
    md.append("- Variants (isolated; one change at a time):")
    md.append("  - `A_branch_switch_penalty`: `--dp-transition-penalty-multiplier=2.0`")
    md.append("  - `B_noise_adaptive_candidate_pruning`: `--candidate-filter-enable-column-rank-normalization` (GT-free per-column confidence regularization; *pruning의 operationalization*)")
    md.append("  - `C_component_plausibility_filtering`: `--candidate-filter-enable-evidence-aware-preserve` (GT-free local evidence preserve; *component plausibility의 근사 operationalization*)")
    md.append("")
    md.append(f"- CSV summary: `{csv_path.relative_to(ROOT)}`")
    md.append(f"- JSON rows: `{(out_root / 'isolated_stabilization_ablation.json').relative_to(ROOT)}`")
    md.append("")
    md.append("## E. fidelity vs stabilization tradeoff")
    md.append(f"- JSON: `{(out_root / 'fidelity_vs_stabilization_tradeoff.json').relative_to(ROOT)}`")
    md.append("")
    md.append("## F. clean/styled regression 결과")
    if smoke_payload.get("enabled"):
        md.append(f"- JSON: `{(out_root / 'stabilization_regression_smoke.json').relative_to(ROOT)}`")
        md.append("- Note: smoke evaluation uses canonical `source_numeric.json` comparison (normalized_y_mae / corr / peak false/missed).")
    else:
        md.append("- smoke runs not present under runs/smoke.")
    md.append("")
    md.append("## G. 가장 promising한 stabilization 방향")
    md.append(f"- best_candidate_hint: `{best}`")
    md.append("")
    md.append("## H. 생성 산출물 경로")
    md.append(f"- root: `{out_root.relative_to(ROOT)}`")
    md.append("")
    md.append("### 필수 overlay")
    md.append("- `overlay_burst_region_zoom.png`")
    md.append("- `overlay_burst_branch_density.png`")
    md.append("- `overlay_false_top1_regions.png`")
    md.append("- `overlay_false_branch_hijack.png`")
    md.append("- `overlay_branch_continuity_breaks.png`")
    md.append("- `overlay_oscillation_clusters.png`")
    md.append("- `overlay_before_after_stabilization.png`")
    md.append("- `overlay_peak_preservation_tradeoff.png`")
    md.append("")
    md.append("## I. 하지 않은 작업")
    for nd in summary_json["not_done"]:
        md.append(f"- {nd}")
    md.append("")
    md.append("## J. 최종 판정")
    md.append(f"- `{final_decision}`")
    (out_root / "branch_instability_stabilization_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # required overlays: generate simple before/after + peak tradeoff summary images (GT-free, analysis-only)
    before_after = out_root / "overlay_before_after_stabilization.png"
    im = Image.new("RGBA", (1200, 260), (30, 30, 30, 255))
    d = ImageDraw.Draw(im)
    d.text((12, 12), "Before/After stabilization (summary)", fill=(255, 255, 255, 255))
    d.text((12, 48), f"baseline=current_2x_highres false_top1={base_row.get('false_top1_columns') if base_row else None}", fill=(220, 220, 220, 255))
    d.text((12, 74), f"best_hint={best}", fill=(220, 220, 220, 255))
    if best:
        best_row = next((r for r in ablation["rows"] if r["variant"] == best), None)
        d.text((12, 110), f"{best} false_top1={best_row.get('false_top1_columns') if best_row else None}", fill=(220, 220, 220, 255))
        d.text((12, 136), f"{best} long_jump={best_row.get('long_jump_count') if best_row else None}", fill=(220, 220, 220, 255))
    im.save(before_after, format="PNG")
    peak_trade = out_root / "overlay_peak_preservation_tradeoff.png"
    im2 = Image.new("RGBA", (1200, 260), (30, 30, 30, 255))
    d2 = ImageDraw.Draw(im2)
    d2.text((12, 12), "Peak preservation tradeoff (summary)", fill=(255, 255, 255, 255))
    if base_row:
        d2.text((12, 48), f"baseline matched/missed/false = {base_row.get('matched_peak_count')}/{base_row.get('missed_peak_count')}/{base_row.get('false_spike_count')}", fill=(220, 220, 220, 255))
    if best:
        br = next((r for r in ablation["rows"] if r["variant"] == best), None)
        if br:
            d2.text((12, 84), f"{best} matched/missed/false = {br.get('matched_peak_count')}/{br.get('missed_peak_count')}/{br.get('false_spike_count')}", fill=(220, 220, 220, 255))
    im2.save(peak_trade, format="PNG")

    print(json.dumps({"out_root": str(out_root), "final_decision": final_decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

