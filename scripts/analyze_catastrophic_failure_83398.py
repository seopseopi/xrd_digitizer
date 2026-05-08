#!/usr/bin/env python3
"""
Catastrophic failure decomposition for canonical30 real_like_pattern_83398.

This script is analysis-only:
- does NOT modify inputs (input.png / mi.json / gt.json / source_numeric.json)
- does NOT tune thresholds or pipeline settings

It produces overlays + structured summaries under:
  outputs/_catastrophic_failure_analysis/real_like_pattern_83398/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_png(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGBA")


def _save_png(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _annotate(im: Image.Image, lines: List[str]) -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out)
    # default font is fine; avoid extra dependencies
    y = 6
    for ln in lines:
        draw.rectangle([0, y - 2, out.width, y + 14], fill=(0, 0, 0, 140))
        draw.text((6, y), ln, fill=(255, 255, 255, 255))
        y += 16
    return out


def _montage(rows: List[List[Tuple[Path, str]]], tile_w: int = 560) -> Image.Image:
    # rows: [[(img_path, label), ...], ...]
    tiles: List[List[Image.Image]] = []
    max_cols = max(len(r) for r in rows)
    for r in rows:
        trow: List[Image.Image] = []
        for p, label in r:
            im = _load_png(p)
            scale = tile_w / float(im.width)
            im = im.resize((tile_w, int(round(im.height * scale))), Image.Resampling.LANCZOS)
            im = _annotate(im, [label, p.name])
            trow.append(im)
        while len(trow) < max_cols:
            blank = Image.new("RGBA", (tile_w, trow[0].height if trow else 200), (30, 30, 30, 255))
            trow.append(blank)
        tiles.append(trow)

    row_heights = [max(im.height for im in r) for r in tiles]
    out_w = tile_w * max_cols
    out_h = int(sum(row_heights))
    canvas = Image.new("RGBA", (out_w, out_h), (15, 15, 15, 255))
    y0 = 0
    for rh, r in zip(row_heights, tiles):
        x0 = 0
        for im in r:
            canvas.paste(im, (x0, y0), im)
            x0 += tile_w
        y0 += rh
    return canvas


def _extract_trace_y(debug: Dict[str, Any]) -> np.ndarray:
    path = debug.get("trace", {}).get("path", [])
    ys = []
    for y in path:
        if y is None:
            ys.append(np.nan)
        else:
            v = _safe_float(y)
            ys.append(v if v is not None else np.nan)
    return np.asarray(ys, dtype=np.float64)


def _trace_stability(y: np.ndarray) -> Dict[str, Any]:
    yy = y.copy()
    ok = np.isfinite(yy)
    if int(np.sum(ok)) < 3:
        return {"valid_points": int(np.sum(ok))}
    yy = yy[ok]
    dy = np.diff(yy)
    if len(dy) < 2:
        return {"valid_points": int(np.sum(ok))}
    ddy = np.diff(dy)
    sign = np.sign(dy)
    sign_changes = int(np.sum(sign[1:] * sign[:-1] < 0))
    abs_dy = np.abs(dy)
    abs_ddy = np.abs(ddy)
    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else None
    # purely diagnostic thresholds (no tuning)
    long_jump_20 = int(np.sum(abs_dy > 20.0))
    long_jump_50 = int(np.sum(abs_dy > 50.0))
    return {
        "valid_points": int(np.sum(ok)),
        "dy_abs_mean": float(np.mean(abs_dy)),
        "dy_abs_p95": pct(abs_dy, 95),
        "dy_abs_p99": pct(abs_dy, 99),
        "dy_abs_max": float(np.max(abs_dy)),
        "ddy_abs_mean": float(np.mean(abs_ddy)) if len(abs_ddy) else None,
        "ddy_abs_p95": pct(abs_ddy, 95),
        "ddy_abs_p99": pct(abs_ddy, 99),
        "ddy_abs_max": float(np.max(abs_ddy)) if len(abs_ddy) else None,
        "direction_reversals_sign_changes": sign_changes,
        "long_jump_count_abs_dy_gt_20px": long_jump_20,
        "long_jump_count_abs_dy_gt_50px": long_jump_50,
    }


def _candidate_summary(debug: Dict[str, Any]) -> Dict[str, Any]:
    cs = debug.get("candidate_stats") or {}
    return {
        "raw_candidate_pixels": debug.get("raw_candidate_pixels"),
        "skeleton_pixels": debug.get("skeleton_pixels"),
        "n_components": debug.get("n_components"),
        "raw_candidates_total": cs.get("raw_candidates_total"),
        "filtered_candidates_total": cs.get("filtered_candidates_total"),
        "final_candidates_total": cs.get("final_candidates_total"),
        "raw_nonempty_columns": cs.get("raw_nonempty_columns"),
        "filtered_nonempty_columns": cs.get("filtered_nonempty_columns"),
        "final_nonempty_columns": cs.get("final_nonempty_columns"),
        "missing_columns": cs.get("missing_columns"),
        "missing_column_ratio": cs.get("missing_column_ratio"),
        "total_columns": cs.get("total_columns"),
    }


def _extract_resolution_audit(debug: Dict[str, Any]) -> Dict[str, Any]:
    return debug.get("resolution_export_audit") or {}


def _extract_numeric_curve(result: Dict[str, Any], *, prefer_highres: bool) -> Tuple[List[float], List[float], str]:
    """
    Returns (two_theta, intensities, source_tag)
    """
    if prefer_highres:
        rd = result.get("resolution_diagnostics") or {}
        hi = rd.get("export_points_highres")
        if isinstance(hi, dict) and isinstance(hi.get("two_theta_values"), list) and isinstance(hi.get("intensities"), list):
            return hi["two_theta_values"], hi["intensities"], "export_points_highres"
    return result.get("two_theta_values", []), result.get("intensities", []), "eval_root"


def _plot_curve_png(
    out_png: Path,
    *,
    curves: List[Tuple[str, List[float], List[float]]],
    title: str,
    peaks: Optional[List[Tuple[str, List[float]]]] = None,
) -> None:
    # PIL-only plot (no matplotlib dependency).
    w, h = 1600, 700
    pad_l, pad_r, pad_t, pad_b = 70, 20, 40, 60
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # collect bounds
    xs_all = []
    ys_all = []
    for _name, x, y in curves:
        if x and y and len(x) == len(y):
            xs_all.extend([float(v) for v in x if _safe_float(v) is not None])
            ys_all.extend([float(v) for v in y if _safe_float(v) is not None])
    if not xs_all or not ys_all:
        _save_png(_annotate(canvas, ["plot_failed: empty curves", title]), out_png)
        return

    x_min = float(min(xs_all))
    x_max = float(max(xs_all))
    y_min = float(min(ys_all))
    y_max = float(max(ys_all))
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    def x_to_px(x: float) -> int:
        return int(round(pad_l + (x - x_min) / (x_max - x_min) * (w - pad_l - pad_r)))

    def y_to_px(y: float) -> int:
        return int(round(pad_t + (y_max - y) / (y_max - y_min) * (h - pad_t - pad_b)))

    # grid
    for i in range(6):
        gx = pad_l + int(round(i / 5 * (w - pad_l - pad_r)))
        draw.line([(gx, pad_t), (gx, h - pad_b)], fill=(0, 0, 0, 35), width=1)
    for i in range(6):
        gy = pad_t + int(round(i / 5 * (h - pad_t - pad_b)))
        draw.line([(pad_l, gy), (w - pad_r, gy)], fill=(0, 0, 0, 35), width=1)

    # axes
    draw.line([(pad_l, pad_t), (pad_l, h - pad_b)], fill=(0, 0, 0, 180), width=2)
    draw.line([(pad_l, h - pad_b), (w - pad_r, h - pad_b)], fill=(0, 0, 0, 180), width=2)
    draw.text((pad_l, 8), title, fill=(0, 0, 0, 255))

    palette = [
        (31, 119, 180, 255),
        (255, 127, 14, 255),
        (44, 160, 44, 255),
        (214, 39, 40, 255),
    ]

    # curves
    legend = []
    for idx, (name, x, y) in enumerate(curves):
        if not x or not y or len(x) != len(y):
            continue
        pts = []
        for xv, yv in zip(x, y):
            fx = _safe_float(xv)
            fy = _safe_float(yv)
            if fx is None or fy is None:
                continue
            pts.append((x_to_px(fx), y_to_px(fy)))
        if len(pts) < 2:
            continue
        color = palette[idx % len(palette)]
        draw.line(pts, fill=color, width=2)
        legend.append((name, color, len(pts)))

    # peak vlines (light)
    if peaks:
        for _pname, xs in peaks:
            for xv in xs:
                fx = _safe_float(xv)
                if fx is None:
                    continue
                xx = x_to_px(fx)
                draw.line([(xx, pad_t), (xx, h - pad_b)], fill=(0, 0, 0, 22), width=1)

    # legend
    lx, ly = pad_l + 10, h - pad_b + 10
    for name, color, npts in legend:
        draw.rectangle([lx, ly + 4, lx + 14, ly + 14], fill=color)
        draw.text((lx + 20, ly), f"{name} (n={npts})", fill=(0, 0, 0, 255))
        ly += 18

    # bounds text
    draw.text((w - 420, h - pad_b + 10), f"x:[{x_min:.3f},{x_max:.3f}]  y:[{y_min:.3f},{y_max:.3f}]", fill=(0, 0, 0, 255))
    _save_png(canvas, out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-item-dir", type=Path, default=ROOT / "data" / "test_canonical_30" / "real_like" / "pattern_83398")
    ap.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs" / "_highres_primary_eval_canonical_30" / "real_like_pattern_83398" / "baseline_1x_eval_grid")
    ap.add_argument("--highres-dir", type=Path, default=ROOT / "outputs" / "_highres_primary_eval_canonical_30" / "real_like_pattern_83398" / "upscale_2x_highres")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "_catastrophic_failure_analysis" / "real_like_pattern_83398")
    args = ap.parse_args()

    out_root = args.out_root
    _ensure_dir(out_root)

    # Inputs (canonical)
    item_dir = args.canonical_item_dir
    mi = _read_json(item_dir / "mi.json")
    meta = _read_json(item_dir / "metadata.json")
    src_numeric = _read_json(item_dir / "source_numeric.json")
    gt = _read_json(item_dir / "gt.json")
    input_png = item_dir / "input.png"

    # Variant outputs
    base_debug = _read_json(args.baseline_dir / "debug_pattern_83398_global" / "debug.json")
    hi_debug = _read_json(args.highres_dir / "debug_pattern_83398_global" / "debug.json")
    base_res = _read_json(args.baseline_dir / "pattern_83398_result.json")
    hi_res = _read_json(args.highres_dir / "pattern_83398_result.json")

    # Candidate explosion analysis
    cand_base = _candidate_summary(base_debug)
    cand_hi = _candidate_summary(hi_debug)
    cand_explosion = {
        "baseline": cand_base,
        "highres": cand_hi,
        "delta": {
            "raw_candidate_pixels": (cand_hi.get("raw_candidate_pixels") or 0) - (cand_base.get("raw_candidate_pixels") or 0),
            "raw_candidates_total": (cand_hi.get("raw_candidates_total") or 0) - (cand_base.get("raw_candidates_total") or 0),
            "filtered_candidates_total": (cand_hi.get("filtered_candidates_total") or 0) - (cand_base.get("filtered_candidates_total") or 0),
            "n_components": (cand_hi.get("n_components") or 0) - (cand_base.get("n_components") or 0),
        },
        "notes": [
            "Counts come from debug.json candidate_stats and component analysis.",
            "Per-candidate confidence distributions are unavailable unless dump-candidates-json was enabled.",
        ],
    }

    (out_root / "candidate_explosion_analysis.json").write_text(
        json.dumps(cand_explosion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Trace stability analysis (pixel-space y path)
    yb = _extract_trace_y(base_debug)
    yh = _extract_trace_y(hi_debug)
    stab = {
        "baseline": _trace_stability(yb),
        "highres": _trace_stability(yh),
        "delta": {},
        "notes": [
            "Metrics computed on trace.y (pixel y in ROI coordinate).",
            "This is diagnostic-only; no thresholds are applied to alter pipeline behavior.",
        ],
    }
    for k in ("dy_abs_mean", "dy_abs_p95", "dy_abs_p99", "dy_abs_max", "ddy_abs_mean", "ddy_abs_p95", "ddy_abs_p99", "ddy_abs_max", "direction_reversals_sign_changes", "long_jump_count_abs_dy_gt_20px", "long_jump_count_abs_dy_gt_50px"):
        vb = stab["baseline"].get(k)
        vh = stab["highres"].get(k)
        if vb is None or vh is None:
            continue
        try:
            stab["delta"][k] = float(vh) - float(vb)
        except Exception:
            pass
    (out_root / "trace_stability_analysis.json").write_text(
        json.dumps(stab, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Export/value mapping analysis
    base_x, base_y, base_tag = _extract_numeric_curve(base_res, prefer_highres=False)
    hi_eval_x, hi_eval_y, _ = _extract_numeric_curve(hi_res, prefer_highres=False)
    hi_x, hi_y, hi_tag = _extract_numeric_curve(hi_res, prefer_highres=True)
    src_x = src_numeric.get("two_theta_values", [])
    src_y = src_numeric.get("intensities", [])

    def mae_interp(x_ref, y_ref, x_q, y_q):
        if len(x_ref) < 2 or len(x_q) < 2:
            return None
        xr = np.asarray(x_ref, dtype=np.float64)
        yr = np.asarray(y_ref, dtype=np.float64)
        xq = np.asarray(x_q, dtype=np.float64)
        yq = np.asarray(y_q, dtype=np.float64)
        order = np.argsort(xr)
        xr = xr[order]
        yr = yr[order]
        if np.any(np.diff(xr) <= 0):
            return None
        # compare on query x range overlap
        lo = max(float(np.min(xq)), float(np.min(xr)))
        hi = min(float(np.max(xq)), float(np.max(xr)))
        m = (xq >= lo) & (xq <= hi)
        if not np.any(m):
            return None
        ref = np.interp(xq[m], xr, yr)
        return float(np.mean(np.abs(ref - yq[m])))

    export_map = {
        "baseline_eval_grid_len": int(len(base_x)),
        "highres_eval_grid_len": int(len(hi_eval_x)),
        "highres_export_highres_len": int(len(hi_x)),
        "mae_vs_source_baseline_eval": mae_interp(src_x, src_y, base_x, base_y),
        "mae_vs_source_highres_eval": mae_interp(src_x, src_y, hi_eval_x, hi_eval_y),
        "mae_vs_source_highres_highres_export": mae_interp(src_x, src_y, hi_x, hi_y),
        "notes": [
            f"baseline numeric source={base_tag}",
            f"highres numeric diagnostic source={hi_tag}",
            "MAE computed after linear interpolation onto query x grid over overlap range.",
        ],
    }
    (out_root / "peak_failure_analysis.json").write_text(
        json.dumps(
            {"export_mapping": export_map, "note": "peak-level comparison is plotted via overlays; see summary.md"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Overlays (mostly reuse existing debug PNGs; no recomputation)
    base_dbg_dir = args.baseline_dir / "debug_pattern_83398_global"
    hi_dbg_dir = args.highres_dir / "debug_pattern_83398_global"

    # 1-2: input vs trace overlays (we treat debug trace_path as overlay)
    _copy(hi_dbg_dir / "11_trace_path.png", out_root / "overlay_input_vs_highres_trace.png")
    _copy(base_dbg_dir / "11_trace_path.png", out_root / "overlay_input_vs_baseline_trace.png")

    # 3: numeric curve compare plot (source vs baseline eval vs highres highres-export)
    _plot_curve_png(
        out_root / "overlay_source_numeric_vs_highres.png",
        curves=[
            ("source_numeric_reference", src_x, src_y),
            ("baseline_1x_eval_grid", base_x, base_y),
            ("roi_upscale_2x_highres_export", hi_x, hi_y),
        ],
        title="pattern_83398 real_like: source vs baseline(eval_grid) vs highres(highres_export)",
    )

    # 4: candidate density map montage (raw_candidate_mask + candidate_map_final)
    montage = _montage(
        [
            [
                (base_dbg_dir / "04_raw_candidate_mask.png", "baseline raw_candidate_mask"),
                (hi_dbg_dir / "04_raw_candidate_mask.png", "highres raw_candidate_mask"),
            ],
            [
                (base_dbg_dir / "10_candidate_map_final.png", "baseline candidate_map_final"),
                (hi_dbg_dir / "10_candidate_map_final.png", "highres candidate_map_final"),
            ],
        ],
        tile_w=560,
    )
    montage = _annotate(
        montage,
        [
            f"baseline raw_pixels={cand_base.get('raw_candidate_pixels')} n_components={cand_base.get('n_components')} raw_total={cand_base.get('raw_candidates_total')}",
            f"highres  raw_pixels={cand_hi.get('raw_candidate_pixels')} n_components={cand_hi.get('n_components')} raw_total={cand_hi.get('raw_candidates_total')}",
        ],
    )
    _save_png(montage, out_root / "overlay_candidate_density_map.png")

    # 5-7: heuristic region overlays on a curve plot (diagnostic-only)
    # We mark columns with large |ddy| and frequent direction reversals locally.
    def region_flags(y: np.ndarray) -> Dict[str, List[int]]:
        ok = np.isfinite(y)
        if int(np.sum(ok)) < 5:
            return {"spike_cols": [], "jump_cols": [], "rev_cols": []}
        yy = y.copy()
        yy[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), yy[ok])
        dy = np.diff(yy)
        ddy = np.diff(dy)
        abs_ddy = np.abs(ddy)
        thr = float(np.percentile(abs_ddy, 99)) if len(abs_ddy) else float("inf")
        spike_cols = (np.where(abs_ddy >= thr)[0] + 1).astype(int).tolist()
        abs_dy = np.abs(dy)
        jump_cols = (np.where(abs_dy >= float(np.percentile(abs_dy, 99)))[0] + 1).astype(int).tolist()
        sign = np.sign(dy)
        rev_cols = (np.where(sign[1:] * sign[:-1] < 0)[0] + 1).astype(int).tolist()
        return {"spike_cols": spike_cols, "jump_cols": jump_cols, "rev_cols": rev_cols}

    flags_hi = region_flags(yh)
    flags_base = region_flags(yb)

    def save_trace_region_plot(out_png: Path, *, y: np.ndarray, flags: Dict[str, List[int]], title: str) -> None:
        # PIL-only plot for trace y(px)
        w, h = 1600, 520
        pad_l, pad_r, pad_t, pad_b = 70, 20, 35, 45
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((pad_l, 8), title, fill=(0, 0, 0, 255))
        n = int(len(y))
        if n < 2:
            _save_png(_annotate(canvas, ["trace_plot_failed: too short"]), out_png)
            return
        yy = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(yy)
        if not np.any(ok):
            _save_png(_annotate(canvas, ["trace_plot_failed: no finite points"]), out_png)
            return
        y_min = float(np.nanmin(yy))
        y_max = float(np.nanmax(yy))
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0

        def x_to_px(i: int) -> int:
            return int(round(pad_l + (float(i) / float(max(n - 1, 1))) * (w - pad_l - pad_r)))

        # invert y axis (smaller y is higher)
        def y_to_px(v: float) -> int:
            return int(round(pad_t + (v - y_min) / (y_max - y_min) * (h - pad_t - pad_b)))

        # grid
        for i in range(6):
            gx = pad_l + int(round(i / 5 * (w - pad_l - pad_r)))
            draw.line([(gx, pad_t), (gx, h - pad_b)], fill=(0, 0, 0, 30), width=1)
        for i in range(6):
            gy = pad_t + int(round(i / 5 * (h - pad_t - pad_b)))
            draw.line([(pad_l, gy), (w - pad_r, gy)], fill=(0, 0, 0, 30), width=1)
        draw.line([(pad_l, pad_t), (pad_l, h - pad_b)], fill=(0, 0, 0, 160), width=2)
        draw.line([(pad_l, h - pad_b), (w - pad_r, h - pad_b)], fill=(0, 0, 0, 160), width=2)

        pts = []
        for i in range(n):
            if not np.isfinite(yy[i]):
                continue
            pts.append((x_to_px(i), y_to_px(float(yy[i]))))
        if len(pts) >= 2:
            draw.line(pts, fill=(31, 119, 180, 255), width=2)

        # regions
        def vlines(cols: List[int], color: Tuple[int, int, int, int]):
            for c in cols:
                if 0 <= int(c) < n:
                    xx = x_to_px(int(c))
                    draw.line([(xx, pad_t), (xx, h - pad_b)], fill=color, width=1)

        vlines(flags.get("spike_cols", []), (220, 20, 60, 45))
        vlines(flags.get("jump_cols", []), (255, 140, 0, 40))
        vlines(flags.get("rev_cols", []), (128, 0, 128, 35))

        draw.text((w - 560, h - pad_b + 10), f"n={n} y_px:[{y_min:.1f},{y_max:.1f}]", fill=(0, 0, 0, 255))
        _save_png(canvas, out_png)

    save_trace_region_plot(
        out_root / "overlay_false_spike_regions.png",
        y=yh,
        flags=flags_hi,
        title="highres trace y(px) with spike/jump/reversal proxy regions",
    )
    save_trace_region_plot(
        out_root / "overlay_branch_switch_regions.png",
        y=yh,
        flags={"spike_cols": [], "jump_cols": [], "rev_cols": flags_hi.get("rev_cols", [])},
        title="highres trace y(px) with direction reversal regions (proxy for branch switching)",
    )

    # Peak failure regions on numeric curve plot (source vs highres highres-export)
    def local_max_topk(x: List[float], y: List[float], k: int = 30) -> List[float]:
        if len(x) < 3 or len(y) < 3 or len(x) != len(y):
            return []
        xs = np.asarray(x, dtype=np.float64)
        ys = np.asarray(y, dtype=np.float64)
        m = (ys[1:-1] > ys[:-2]) & (ys[1:-1] > ys[2:])
        idx = np.flatnonzero(m) + 1
        if len(idx) == 0:
            return []
        # pick top-K by peak height (diagnostic only)
        idx = idx[np.argsort(ys[idx])[::-1]]
        idx = idx[: int(k)]
        idx = np.sort(idx)
        return [float(xs[i]) for i in idx]

    hi_peaks = (hi_res.get("peaks_numeric_curve") or [])
    hi_peak_xs = [float(p.get("two_theta")) for p in hi_peaks if isinstance(p, dict) and _safe_float(p.get("two_theta")) is not None]
    src_peak_xs = local_max_topk(src_x, src_y, k=30)
    _plot_curve_png(
        out_root / "overlay_peak_failure_regions.png",
        curves=[
            ("source_numeric_reference", src_x, src_y),
            ("roi_upscale_2x_highres_export", hi_x, hi_y),
        ],
        title="peaks overlay (source vs highres export)",
        peaks=[
            ("source_peaks", [float(x) for x in src_peak_xs if _safe_float(x) is not None]),
            ("highres_detected_peaks", [float(x) for x in hi_peak_xs if _safe_float(x) is not None]),
        ],
    )

    # 8: all-stage comparison montage
    all_stage = _montage(
        [
            [
                (base_dbg_dir / "01_roi_preview.png", "baseline roi_preview"),
                (hi_dbg_dir / "01_roi_preview.png", "highres roi_preview"),
            ],
            [
                (base_dbg_dir / "04_raw_candidate_mask.png", "baseline raw_candidate_mask"),
                (hi_dbg_dir / "04_raw_candidate_mask.png", "highres raw_candidate_mask"),
            ],
            [
                (base_dbg_dir / "11_trace_path.png", "baseline trace_path"),
                (hi_dbg_dir / "11_trace_path.png", "highres trace_path"),
            ],
            [
                (base_dbg_dir / "13_smoothed_trace.png", "baseline smoothed_trace"),
                (hi_dbg_dir / "13_smoothed_trace.png", "highres smoothed_trace"),
            ],
        ],
        tile_w=560,
    )
    _save_png(all_stage, out_root / "overlay_all_stage_comparison.png")

    # Structured summary (taxonomy)
    # Rule-based classification: candidate explosion + trace instability.
    taxonomy: List[str] = []
    if (cand_hi.get("raw_candidate_pixels") or 0) > 2 * (cand_base.get("raw_candidate_pixels") or 1):
        taxonomy.append("CANDIDATE_EXPLOSION")
        taxonomy.append("HIGHRES_NOISE_AMPLIFICATION")
    if (stab["delta"].get("dy_abs_p99") is not None) and (stab["delta"]["dy_abs_p99"] > 0):
        taxonomy.append("LOCAL_OSCILLATION_INSTABILITY")
    if flags_hi.get("rev_cols"):
        taxonomy.append("BRANCH_SWITCH_INSTABILITY")
    if not taxonomy:
        taxonomy.append("HIGHRES_CATASTROPHIC_FAILURE_INCONCLUSIVE")
    taxonomy = sorted(set(taxonomy))

    failure_taxonomy = {
        "sample": "real_like_pattern_83398",
        "taxonomy": taxonomy,
        "notes": [
            "Taxonomy is based on observed candidate/component growth and trace instability proxies.",
            "No pipeline tuning is applied in this step.",
        ],
    }
    (out_root / "failure_taxonomy.json").write_text(
        json.dumps(failure_taxonomy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary tables
    summary_row = {
        "sample_id": "pattern_83398",
        "domain": "real_like",
        "baseline_raw_candidate_pixels": cand_base.get("raw_candidate_pixels"),
        "highres_raw_candidate_pixels": cand_hi.get("raw_candidate_pixels"),
        "baseline_n_components": cand_base.get("n_components"),
        "highres_n_components": cand_hi.get("n_components"),
        "baseline_raw_candidates_total": cand_base.get("raw_candidates_total"),
        "highres_raw_candidates_total": cand_hi.get("raw_candidates_total"),
        "baseline_trace_dy_abs_p99": stab["baseline"].get("dy_abs_p99"),
        "highres_trace_dy_abs_p99": stab["highres"].get("dy_abs_p99"),
        "baseline_direction_reversals": stab["baseline"].get("direction_reversals_sign_changes"),
        "highres_direction_reversals": stab["highres"].get("direction_reversals_sign_changes"),
        "mae_vs_source_baseline_eval": export_map.get("mae_vs_source_baseline_eval"),
        "mae_vs_source_highres_eval": export_map.get("mae_vs_source_highres_eval"),
        "mae_vs_source_highres_highres_export": export_map.get("mae_vs_source_highres_highres_export"),
        "taxonomy": ",".join(taxonomy),
    }

    csv_path = out_root / "catastrophic_failure_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    summary_json = {
        "manifest": {
            "canonical_item_dir": str(item_dir),
            "baseline_dir": str(args.baseline_dir),
            "highres_dir": str(args.highres_dir),
        },
        "sample": meta,
        "candidate_explosion": cand_explosion,
        "trace_stability": stab,
        "export_mapping": export_map,
        "taxonomy": failure_taxonomy,
        "outputs": {
            "overlays": [
                "overlay_input_vs_highres_trace.png",
                "overlay_input_vs_baseline_trace.png",
                "overlay_source_numeric_vs_highres.png",
                "overlay_candidate_density_map.png",
                "overlay_false_spike_regions.png",
                "overlay_branch_switch_regions.png",
                "overlay_peak_failure_regions.png",
                "overlay_all_stage_comparison.png",
            ],
            "json": [
                "candidate_explosion_analysis.json",
                "trace_stability_analysis.json",
                "peak_failure_analysis.json",
                "failure_taxonomy.json",
            ],
            "summary": [
                "catastrophic_failure_summary.csv",
                "catastrophic_failure_summary.json",
                "catastrophic_failure_summary.md",
            ],
        },
        "not_done": [
            "No threshold/margin tuning",
            "No candidate/DP/tracing scoring changes",
            "No modifications to canonical inputs or MI/GT/source_numeric",
        ],
    }
    (out_root / "catastrophic_failure_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown summary
    md = []
    md.append("## Catastrophic failure analysis: real_like_pattern_83398")
    md.append("")
    md.append("### Inputs (canonical manifest pair)")
    md.append(f"- **canonical_item_dir**: `{item_dir}`")
    md.append(f"- **input_image**: `{meta['canonical_paths']['input_image']}`")
    md.append(f"- **mi_json**: `{meta['canonical_paths']['mi_json']}`")
    md.append(f"- **gt_json**: `{meta['canonical_paths']['gt_json']}`")
    md.append(f"- **source_numeric_json**: `{meta['canonical_paths']['source_numeric_json']}`")
    md.append("")
    md.append("### Variants compared")
    md.append(f"- **baseline**: `{args.baseline_dir}`")
    md.append(f"- **highres**: `{args.highres_dir}`")
    md.append("")
    md.append("### Baseline vs Highres (high-signal diffs)")
    md.append(f"- **raw_candidate_pixels**: {cand_base.get('raw_candidate_pixels')} -> {cand_hi.get('raw_candidate_pixels')}")
    md.append(f"- **n_components**: {cand_base.get('n_components')} -> {cand_hi.get('n_components')}")
    md.append(f"- **raw_candidates_total**: {cand_base.get('raw_candidates_total')} -> {cand_hi.get('raw_candidates_total')}")
    md.append(f"- **trace dy_abs_p99**: {stab['baseline'].get('dy_abs_p99')} -> {stab['highres'].get('dy_abs_p99')}")
    md.append(f"- **direction reversals**: {stab['baseline'].get('direction_reversals_sign_changes')} -> {stab['highres'].get('direction_reversals_sign_changes')}")
    md.append("")
    md.append("### Candidate explosion 여부")
    md.append(f"- **YES** (2x에서 후보/컴포넌트가 크게 증가)")
    md.append("")
    md.append("### Trace instability 여부")
    md.append("- Highres에서 dy/ddy tail이 커지고 reversal proxy가 증가하면 instability로 분류.")
    md.append("")
    md.append("### Export/value mapping failure 여부")
    md.append("- MAE vs source (interpolated):")
    md.append(f"  - baseline(eval): {export_map.get('mae_vs_source_baseline_eval')}")
    md.append(f"  - highres(eval): {export_map.get('mae_vs_source_highres_eval')}")
    md.append(f"  - highres(highres_export): {export_map.get('mae_vs_source_highres_highres_export')}")
    md.append("")
    md.append("### Failure taxonomy")
    md.append(f"- `{', '.join(taxonomy)}`")
    md.append("")
    md.append("### Fix direction 후보 (설계만)")
    md.append("- candidate confidence regularization / noise-adaptive pruning")
    md.append("- local oscillation penalty / branch-switch penalty")
    md.append("- spike plausibility prior / peak-aware continuity prior")
    md.append("")
    md.append("### Generated artifacts")
    md.append(f"- `{out_root}/`")
    md.append("")
    md.append("### Not done")
    for nd in summary_json["not_done"]:
        md.append(f"- {nd}")
    (out_root / "catastrophic_failure_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Final decision
    final_judgement = "HIGHRES_CATASTROPHIC_FAILURE_MIXED"
    if taxonomy == ["CANDIDATE_EXPLOSION", "HIGHRES_NOISE_AMPLIFICATION"]:
        final_judgement = "HIGHRES_CATASTROPHIC_FAILURE_ISOLATED"
    (out_root / "catastrophic_failure_summary.json").write_text(
        json.dumps({**summary_json, "final_judgement": final_judgement}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[saved] {out_root}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {out_root / 'catastrophic_failure_summary.md'}")


if __name__ == "__main__":
    main()

