#!/usr/bin/env python3
"""
Candidate dump root-cause localization for real_like_pattern_83398.

Strict constraints:
- analysis-only (no pipeline changes, no threshold tuning)
- uses canonical pair outputs only (existing run dumps)

Inputs (expected to exist):
  outputs/_real_like_instability_suppression/runs/real_like_pattern_83398/upscale_2x_highres/debug_pattern_83398_global/
    - 18_raw_candidates.json
    - 19_filtered_candidates.json
    - 20_final_candidates.json
    - debug.json
    - (optional) 13_smoothed_trace.png, 10_candidate_map_final.png etc (may not exist in dump-mode runs)

Outputs:
  outputs/_candidate_dump_rootcause_analysis/real_like_pattern_83398/
    (see user spec)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        v = int(x)
    except Exception:
        return None
    return v


def pct(arr: np.ndarray, q: float) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


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


def _col_keys_to_ints(d: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for k in d.keys():
        try:
            out.append(int(k))
        except Exception:
            continue
    out.sort()
    return out


def _infer_roi_wh_from_candidates(raw: Dict[str, Any]) -> Tuple[int, int]:
    # roi_w from max column key + 1; roi_h from max y + 1 across a limited scan (fast, safe)
    cols = _col_keys_to_ints(raw)
    roi_w = (max(cols) + 1) if cols else 0
    y_max = 0
    # scan a subsample of columns for speed, then full scan for exact y_max (still OK at 24MB)
    for k in cols[:: max(1, len(cols) // 256)]:
        lst = raw.get(str(k), [])
        if not isinstance(lst, list):
            continue
        for c in lst:
            if not isinstance(c, dict):
                continue
            yi = safe_int(c.get("y"))
            if yi is not None:
                y_max = max(y_max, yi)
    # full scan for exact max y
    for k in cols:
        lst = raw.get(str(k), [])
        if not isinstance(lst, list):
            continue
        for c in lst:
            if not isinstance(c, dict):
                continue
            yi = safe_int(c.get("y"))
            if yi is not None:
                y_max = max(y_max, yi)
    return roi_w, int(y_max + 1)


def _component_id_from_comp_score(
    comp_score: float,
    component_scores: Dict[str, Any],
    *,
    eps: float = 1e-6,
) -> Optional[str]:
    # candidate dict stores comp_score numeric; debug.json has component_scores keyed by component_id with "score".
    # We match by nearest score; for stability, exact match within eps first.
    best_id: Optional[str] = None
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
            best_id = str(cid)
    # accept only if reasonably close (scores are not arbitrary; but avoid mis-assigning)
    return best_id if best_d <= 1e-3 else None


def _build_density_image(
    candidates: Dict[str, Any],
    *,
    roi_w: int,
    roi_h: int,
    max_cols: Optional[int] = None,
) -> np.ndarray:
    # uint16 density per pixel
    dens = np.zeros((roi_h, roi_w), dtype=np.uint16)
    cols = _col_keys_to_ints(candidates)
    if max_cols is not None:
        cols = cols[: int(max_cols)]
    for col in cols:
        lst = candidates.get(str(col), [])
        if not isinstance(lst, list):
            continue
        for c in lst:
            if not isinstance(c, dict):
                continue
            y = safe_int(c.get("y"))
            if y is None:
                continue
            if 0 <= y < roi_h and 0 <= col < roi_w:
                if dens[y, col] < np.iinfo(np.uint16).max:
                    dens[y, col] += 1
    return dens


def _save_density_png(dens: np.ndarray, out_png: Path, *, title: Optional[str] = None) -> Dict[str, Any]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    arr = dens.astype(np.float64)
    vmax = float(np.percentile(arr, 99.5)) if arr.size else 1.0
    vmax = max(vmax, 1.0)
    img = np.clip(arr / vmax, 0.0, 1.0)
    # gamma for visibility
    img = np.sqrt(img)
    im = Image.fromarray((img * 255).astype(np.uint8), mode="L").convert("RGBA")
    if title:
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, im.width, 26], fill=(0, 0, 0, 160))
        draw.text((8, 6), title, fill=(255, 255, 255, 255))
    im.save(out_png, format="PNG")
    return {"vmax_p99_5": vmax, "shape": [int(dens.shape[1]), int(dens.shape[0])]}


def _mark_hotspots(
    base_png: Path,
    out_png: Path,
    *,
    burst_regions: List[Tuple[int, int]],
    color: Tuple[int, int, int, int] = (255, 0, 0, 90),
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(base_png) as im0:
        im = im0.convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    for lo, hi in burst_regions:
        x0 = int(np.clip(lo, 0, im.width - 1))
        x1 = int(np.clip(hi, 0, im.width - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        draw.rectangle([x0, 0, x1, im.height - 1], fill=color, outline=(255, 0, 0, 200), width=2)
    im.save(out_png, format="PNG")


def _candidate_stats_per_stage(
    stage: Dict[str, Any],
    *,
    trace_y: Optional[np.ndarray],
    roi_w: int,
    component_scores: Dict[str, Any],
    near_px: int = 2,
    far_px: int = 15,
) -> Dict[str, Any]:
    cols = _col_keys_to_ints(stage)
    counts_by_col = np.zeros(roi_w, dtype=np.int32)
    conf_all: List[float] = []
    conf_near: List[float] = []
    conf_far: List[float] = []
    comp_counts: Dict[str, int] = {}
    noisy_local_max_proxy = 0
    total = 0

    for col in cols:
        if col >= roi_w:
            continue
        lst = stage.get(str(col), [])
        if not isinstance(lst, list):
            continue
        counts_by_col[col] = int(len(lst))
        ty = None
        if trace_y is not None and 0 <= col < trace_y.shape[0] and math.isfinite(float(trace_y[col])):
            ty = float(trace_y[col])
        for c in lst:
            if not isinstance(c, dict):
                continue
            total += 1
            cf = safe_float(c.get("confidence"))
            if cf is not None:
                conf_all.append(cf)
            if ty is not None:
                yy = safe_float(c.get("y"))
                if yy is not None:
                    dy = abs(float(yy) - float(ty))
                    if dy <= near_px and cf is not None:
                        conf_near.append(cf)
                    if dy >= far_px and cf is not None:
                        conf_far.append(cf)
                    if dy >= far_px and cf is not None and cf >= 0.65:
                        noisy_local_max_proxy += 1
            cs = safe_float(c.get("comp_score"))
            if cs is not None:
                cid = _component_id_from_comp_score(cs, component_scores)
                if cid is not None:
                    comp_counts[cid] = comp_counts.get(cid, 0) + 1

    arr = counts_by_col.astype(np.float64)
    nonempty = arr[arr > 0]
    conf_all_arr = np.asarray(conf_all, dtype=np.float64) if conf_all else np.asarray([], dtype=np.float64)

    def _dist(a: np.ndarray) -> Dict[str, Any]:
        if a.size == 0:
            return {"count": 0}
        return {
            "count": int(a.size),
            "mean": float(np.mean(a)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    return {
        "total_candidates": int(total),
        "columns": {
            "roi_w": int(roi_w),
            "nonempty_columns": int(np.sum(counts_by_col > 0)),
            "candidate_to_column_ratio": float(total / max(roi_w, 1)),
            "count_distribution": {
                "p50": float(np.percentile(nonempty, 50)) if nonempty.size else 0.0,
                "p90": float(np.percentile(nonempty, 90)) if nonempty.size else 0.0,
                "p99": float(np.percentile(nonempty, 99)) if nonempty.size else 0.0,
                "max": float(np.max(nonempty)) if nonempty.size else 0.0,
            },
        },
        "candidate_density_histogram": {
            "bins": 40,
            "hist": [int(x) for x in np.histogram(nonempty, bins=40)[0].tolist()] if nonempty.size else [],
            "bin_edges": [float(x) for x in np.histogram(nonempty, bins=40)[1].tolist()] if nonempty.size else [],
        },
        "component_size_distribution": {
            "n_components_seen": int(len(comp_counts)),
            "top_components_by_count": sorted(
                [{"component_id": k, "count": int(v)} for k, v in comp_counts.items()],
                key=lambda x: -x["count"],
            )[:20],
        },
        "confidence": {
            "all": _dist(conf_all_arr),
            "near_trace": _dist(np.asarray(conf_near, dtype=np.float64) if conf_near else np.asarray([], dtype=np.float64)),
            "far_from_trace": _dist(np.asarray(conf_far, dtype=np.float64) if conf_far else np.asarray([], dtype=np.float64)),
        },
        "local_noisy_maxima_ratio_proxy": float(noisy_local_max_proxy / max(total, 1)),
        "notes": [
            "component_id is inferred by matching candidate.comp_score to debug.json component_scores[*].score.",
            "near/far trace segmentation is GT-free and uses debug trace path y(px).",
            "noisy_local_maxima_proxy := far_from_trace AND confidence>=0.65.",
        ],
        "counts_by_col": counts_by_col.tolist(),  # used for burst localization downstream
    }


def _burst_regions_from_counts(counts_by_col: List[int], *, thr_q: float = 99.0, min_len: int = 8) -> Dict[str, Any]:
    arr = np.asarray(counts_by_col, dtype=np.float64)
    nonempty = arr[arr > 0]
    if nonempty.size == 0:
        return {"threshold": None, "regions": []}
    thr = float(np.percentile(nonempty, thr_q))
    mask = arr >= thr
    regions: List[Tuple[int, int]] = []
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
            regions.append((int(i), int(j)))
        i = j + 1
    return {"threshold": thr, "regions": regions, "thr_quantile": thr_q, "min_len": int(min_len)}


def _final_ranking_analysis(final_stage: Dict[str, Any], *, trace_y: np.ndarray, roi_w: int) -> Dict[str, Any]:
    cols = _col_keys_to_ints(final_stage)
    top1: List[float] = []
    margins: List[float] = []
    near_top1 = 0
    far_top1 = 0
    ambiguous = 0
    amb_cols: List[int] = []
    far_cols: List[int] = []
    for col in cols:
        if col >= roi_w:
            continue
        lst = final_stage.get(str(col), [])
        if not isinstance(lst, list) or not lst:
            continue
        lst2 = [c for c in lst if isinstance(c, dict) and safe_float(c.get("confidence")) is not None]
        if not lst2:
            continue
        lst2.sort(key=lambda c: -float(c.get("confidence", 0.0)))
        c1 = float(lst2[0]["confidence"])
        top1.append(c1)
        if len(lst2) >= 2:
            margins.append(float(lst2[0]["confidence"]) - float(lst2[1]["confidence"]))
        else:
            margins.append(float(lst2[0]["confidence"]))
        # trace proximity of top1 candidate
        ty = trace_y[col] if 0 <= col < trace_y.shape[0] else np.nan
        y1 = safe_float(lst2[0].get("y"))
        if y1 is not None and math.isfinite(float(ty)):
            if abs(float(y1) - float(ty)) <= 2.0:
                near_top1 += 1
            elif abs(float(y1) - float(ty)) >= 15.0:
                far_top1 += 1
                far_cols.append(int(col))
        # ambiguity proxy: small top1-top2 margin
        if len(lst2) >= 2 and (float(lst2[0]["confidence"]) - float(lst2[1]["confidence"])) <= 0.03:
            ambiguous += 1
            amb_cols.append(int(col))

    def dist(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "count": int(a.size),
            "mean": float(np.mean(a)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    return {
        "topk_confidence_distribution": {"top1": dist(top1), "top1_top2_margin": dist(margins)},
        "true_branch_rank_proxy": {
            "top1_near_trace_columns": int(near_top1),
            "top1_far_from_trace_columns": int(far_top1),
            "near_fraction": float(near_top1 / max((near_top1 + far_top1), 1)),
        },
        "branch_ambiguity_score": {
            "ambiguous_columns_margin_le_0.03": int(ambiguous),
            "ambiguous_fraction": float(ambiguous / max(len(cols), 1)),
            "ambiguous_columns_sample": amb_cols[:200],
        },
        "false_branch_rank_proxy": {
            "top1_far_columns_sample": far_cols[:200],
        },
        "notes": [
            "true/false branch is inferred GT-free using proximity to debug trace path (near<=2px, far>=15px).",
            "ambiguity proxy uses margin(top1-top2)<=0.03 in final candidates.",
        ],
    }


def _plot_numeric_overlay(
    out_png: Path,
    *,
    title: str,
    source_xy: Tuple[np.ndarray, np.ndarray],
    export_xy: Tuple[np.ndarray, np.ndarray],
    density_by_col: Optional[np.ndarray],
    x_for_col: Optional[np.ndarray],
    note: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sx, sy = source_xy
    ex, ey = export_xy
    # bounds
    xs = np.concatenate([sx, ex]) if sx.size and ex.size else (sx if sx.size else ex)
    ys = np.concatenate([sy, ey]) if sy.size and ey.size else (sy if sy.size else ey)
    if xs.size == 0 or ys.size == 0:
        Image.new("RGBA", (1200, 520), (30, 30, 30, 255)).save(out_png, format="PNG")
        return
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    w, h = 1600, 640
    pad_l, pad_r, pad_t, pad_b = 70, 20, 45, 70
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)

    def xp(x: float) -> int:
        return int(round(pad_l + (x - x_min) / (x_max - x_min) * (w - pad_l - pad_r)))

    def yp(y: float) -> int:
        return int(round(pad_t + (y_max - y) / (y_max - y_min) * (h - pad_t - pad_b)))

    # grid
    for i in range(6):
        gx = pad_l + int(round(i / 5 * (w - pad_l - pad_r)))
        d.line([(gx, pad_t), (gx, h - pad_b)], fill=(0, 0, 0, 30), width=1)
    for i in range(6):
        gy = pad_t + int(round(i / 5 * (h - pad_t - pad_b)))
        d.line([(pad_l, gy), (w - pad_r, gy)], fill=(0, 0, 0, 30), width=1)
    d.line([(pad_l, pad_t), (pad_l, h - pad_b)], fill=(0, 0, 0, 180), width=2)
    d.line([(pad_l, h - pad_b), (w - pad_r, h - pad_b)], fill=(0, 0, 0, 180), width=2)

    d.text((pad_l, 10), title, fill=(0, 0, 0, 255))
    d.text((pad_l, h - pad_b + 10), note, fill=(0, 0, 0, 180))

    # density vlines (light red)
    if density_by_col is not None and x_for_col is not None and density_by_col.size == x_for_col.size and density_by_col.size:
        dens = density_by_col.astype(np.float64)
        nonz = dens[dens > 0]
        thr = float(np.percentile(nonz, 99)) if nonz.size else float("inf")
        for i in range(dens.size):
            if dens[i] < thr:
                continue
            xx = xp(float(x_for_col[i]))
            d.line([(xx, pad_t), (xx, h - pad_b)], fill=(220, 20, 60, 40), width=1)

    # plot curves
    def plot(xa: np.ndarray, ya: np.ndarray, color: Tuple[int, int, int, int]) -> None:
        pts = []
        for x0, y0 in zip(xa.tolist(), ya.tolist()):
            if not math.isfinite(float(x0)) or not math.isfinite(float(y0)):
                continue
            pts.append((xp(float(x0)), yp(float(y0))))
        if len(pts) >= 2:
            d.line(pts, fill=color, width=2)

    if sx.size and sy.size:
        plot(sx, sy, (31, 119, 180, 255))
    if ex.size and ey.size:
        plot(ex, ey, (255, 127, 14, 255))

    # legend
    d.rectangle([pad_l + 10, h - pad_b + 34, pad_l + 22, h - pad_b + 46], fill=(31, 119, 180, 255))
    d.text((pad_l + 28, h - pad_b + 30), "source_numeric", fill=(0, 0, 0, 255))
    d.rectangle([pad_l + 190, h - pad_b + 34, pad_l + 202, h - pad_b + 46], fill=(255, 127, 14, 255))
    d.text((pad_l + 208, h - pad_b + 30), "export_highres", fill=(0, 0, 0, 255))

    im.save(out_png, format="PNG")


@dataclass(frozen=True)
class Inputs:
    sample: str
    debug_dir: Path
    raw_json: Path
    filt_json: Path
    final_json: Path
    debug_json: Path
    result_json: Path
    canonical_item_dir: Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=str, default="real_like_pattern_83398")
    ap.add_argument(
        "--debug-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "_real_like_instability_suppression"
        / "runs"
        / "real_like_pattern_83398"
        / "upscale_2x_highres"
        / "debug_pattern_83398_global",
    )
    ap.add_argument(
        "--result-json",
        type=Path,
        default=ROOT
        / "outputs"
        / "_real_like_instability_suppression"
        / "runs"
        / "real_like_pattern_83398"
        / "upscale_2x_highres"
        / "pattern_83398_result.json",
    )
    ap.add_argument(
        "--canonical-item-dir",
        type=Path,
        default=ROOT / "data" / "test_canonical_30" / "real_like" / "pattern_83398",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "_candidate_dump_rootcause_analysis" / "real_like_pattern_83398",
    )
    args = ap.parse_args()

    inp = Inputs(
        sample=str(args.sample),
        debug_dir=args.debug_dir,
        raw_json=args.debug_dir / "18_raw_candidates.json",
        filt_json=args.debug_dir / "19_filtered_candidates.json",
        final_json=args.debug_dir / "20_final_candidates.json",
        debug_json=args.debug_dir / "debug.json",
        result_json=args.result_json,
        canonical_item_dir=args.canonical_item_dir,
    )
    out_root = args.out_root
    ensure_dir(out_root)

    # manifest info: we keep it explicit and minimal
    used_manifest = {
        "sample": inp.sample,
        "canonical_item_dir": str(inp.canonical_item_dir),
        "input_image": str(inp.canonical_item_dir / "input.png"),
        "mi_json": str(inp.canonical_item_dir / "mi.json"),
        "gt_json": str(inp.canonical_item_dir / "gt.json"),
        "source_numeric_json": str(inp.canonical_item_dir / "source_numeric.json"),
        "dump_debug_dir": str(inp.debug_dir),
        "dump_files": [
            str(inp.raw_json),
            str(inp.filt_json),
            str(inp.final_json),
            str(inp.debug_json),
            str(inp.result_json),
        ],
    }

    debug = rjson(inp.debug_json)
    component_scores = debug.get("component_scores") or {}
    trace_y = _extract_trace_y(debug)

    raw = rjson(inp.raw_json)
    filt = rjson(inp.filt_json)
    final = rjson(inp.final_json)

    roi_w, roi_h = _infer_roi_wh_from_candidates(raw)

    # ---- Task 1: raw candidate explosion localization ----
    raw_stats = _candidate_stats_per_stage(
        raw,
        trace_y=trace_y,
        roi_w=roi_w,
        component_scores=component_scores,
    )
    burst = _burst_regions_from_counts(raw_stats["counts_by_col"], thr_q=99.0, min_len=8)
    dens_raw = _build_density_image(raw, roi_w=roi_w, roi_h=roi_h)
    dens_meta = _save_density_png(
        dens_raw,
        out_root / "overlay_raw_candidate_density.png",
        title=f"{inp.sample} raw candidate density (p99.5 scaled)",
    )
    burst_regions = [(int(a), int(b)) for a, b in burst.get("regions", [])]
    if burst_regions:
        _mark_hotspots(
            out_root / "overlay_raw_candidate_density.png",
            out_root / "overlay_raw_candidate_hotspots.png",
            burst_regions=burst_regions,
        )
    else:
        # fallback: copy density as hotspots when no regions detected
        (out_root / "overlay_raw_candidate_hotspots.png").write_bytes(
            (out_root / "overlay_raw_candidate_density.png").read_bytes()
        )
    raw_analysis = {
        "used_manifest": used_manifest,
        "roi": {"roi_w": int(roi_w), "roi_h": int(roi_h)},
        "raw_candidate_density": {
            "density_png_meta": dens_meta,
            "burst_regions": burst,
        },
        "measurements": {k: v for k, v in raw_stats.items() if k != "counts_by_col"},
        "questions": {
            "burst_only_regions": bool(bool(burst_regions)),
            "uniform_noise_suspected": bool(len(burst_regions) == 0),
        },
    }
    wjson(out_root / "raw_candidate_density_analysis.json", raw_analysis)

    # ---- Task 2: filtered candidate survival analysis ----
    filt_stats = _candidate_stats_per_stage(
        filt,
        trace_y=trace_y,
        roi_w=roi_w,
        component_scores=component_scores,
    )
    survival = {
        "raw_total": int(raw_stats["total_candidates"]),
        "filtered_total": int(filt_stats["total_candidates"]),
        "filtering_survival_ratio": float(filt_stats["total_candidates"] / max(raw_stats["total_candidates"], 1)),
        "near_trace_survival_proxy": None,
        "far_from_trace_survival_proxy": None,
        "notes": [
            "Survival proxies use near/far segmentation relative to debug trace path (GT-free).",
        ],
    }
    # proxy: compare near/far confidence counts (not perfect, but indicates whether filtering keeps far candidates)
    # We can compute near/far candidate retained ratio by count distributions:
    #   retained_far_ratio_proxy := (filtered far count)/(raw far count)
    # We'll approximate using the hist counts (available in confidence distributions count fields).
    try:
        raw_far = int(raw_stats["confidence"]["far_from_trace"]["count"])
        raw_near = int(raw_stats["confidence"]["near_trace"]["count"])
        fil_far = int(filt_stats["confidence"]["far_from_trace"]["count"])
        fil_near = int(filt_stats["confidence"]["near_trace"]["count"])
        survival["near_trace_survival_proxy"] = float(fil_near / max(raw_near, 1))
        survival["far_from_trace_survival_proxy"] = float(fil_far / max(raw_far, 1))
    except Exception:
        pass

    filtered_analysis = {
        "used_manifest": used_manifest,
        "survival": survival,
        "measurements": {
            "filtered": {k: v for k, v in filt_stats.items() if k != "counts_by_col"},
            "raw_reference": {k: v for k, v in raw_stats.items() if k != "counts_by_col"},
        },
        "questions": {
            "filtering_effective": (
                survival["filtering_survival_ratio"] < 0.25
            ),  # heuristic only
            "far_candidates_retained_high": (
                (survival.get("far_from_trace_survival_proxy") is not None and survival["far_from_trace_survival_proxy"] > 0.5)
            ),
        },
    }
    wjson(out_root / "filtered_candidate_survival_analysis.json", filtered_analysis)

    # overlays: filtered density + false survival (far-from-trace proxy)
    dens_f = _build_density_image(filt, roi_w=roi_w, roi_h=roi_h)
    _save_density_png(dens_f, out_root / "overlay_filtered_candidate_survival.png", title=f"{inp.sample} filtered candidate density")
    # false candidate survival overlay: mark burst regions from raw on filtered map as proxy
    if burst_regions:
        _mark_hotspots(
            out_root / "overlay_filtered_candidate_survival.png",
            out_root / "overlay_false_candidate_survival.png",
            burst_regions=burst_regions,
            color=(255, 140, 0, 80),
        )
    else:
        (out_root / "overlay_false_candidate_survival.png").write_bytes(
            (out_root / "overlay_filtered_candidate_survival.png").read_bytes()
        )

    # ---- Task 3: final candidate ranking failure analysis ----
    final_stats = _candidate_stats_per_stage(
        final,
        trace_y=trace_y,
        roi_w=roi_w,
        component_scores=component_scores,
        near_px=2,
        far_px=15,
    )
    ranking = _final_ranking_analysis(final, trace_y=trace_y, roi_w=roi_w)
    final_analysis = {
        "used_manifest": used_manifest,
        "measurements": {k: v for k, v in final_stats.items() if k != "counts_by_col"},
        "ranking_failure": ranking,
        "questions": {
            "false_branch_high_confidence_suspected": (
                (ranking["true_branch_rank_proxy"]["top1_far_from_trace_columns"] > 0)
            ),
            "ambiguity_spikes_present": (
                ranking["branch_ambiguity_score"]["ambiguous_columns_margin_le_0.03"] > 0
            ),
        },
    }
    wjson(out_root / "final_candidate_ranking_analysis.json", final_analysis)

    # overlays for final branches: density + confidence map proxy
    dens_fin = _build_density_image(final, roi_w=roi_w, roi_h=roi_h)
    _save_density_png(dens_fin, out_root / "overlay_final_candidate_branches.png", title=f"{inp.sample} final candidate density (top-k branches)")
    # confidence map: per-pixel set to max confidence among candidates
    conf_map = np.zeros((roi_h, roi_w), dtype=np.float32)
    cols = _col_keys_to_ints(final)
    for col in cols:
        if col >= roi_w:
            continue
        lst = final.get(str(col), [])
        if not isinstance(lst, list):
            continue
        for c in lst:
            if not isinstance(c, dict):
                continue
            y = safe_int(c.get("y"))
            cf = safe_float(c.get("confidence"))
            if y is None or cf is None:
                continue
            if 0 <= y < roi_h:
                conf_map[y, col] = max(conf_map[y, col], float(cf))
    im_conf = Image.fromarray((np.clip(np.sqrt(conf_map), 0, 1) * 255).astype(np.uint8), mode="L").convert("RGBA")
    ImageDraw.Draw(im_conf).rectangle([0, 0, im_conf.width, 26], fill=(0, 0, 0, 160))
    ImageDraw.Draw(im_conf).text((8, 6), f"{inp.sample} final candidate confidence map (max per pixel)", fill=(255, 255, 255, 255))
    im_conf.save(out_root / "overlay_branch_confidence_map.png", format="PNG")

    # ---- Task 4: candidate stage progression ----
    stage_prog = {
        "used_manifest": used_manifest,
        "roi": {"roi_w": int(roi_w), "roi_h": int(roi_h)},
        "counts": {
            "raw_total": int(raw_stats["total_candidates"]),
            "filtered_total": int(filt_stats["total_candidates"]),
            "final_total": int(final_stats["total_candidates"]),
            "raw_to_filtered_ratio": float(filt_stats["total_candidates"] / max(raw_stats["total_candidates"], 1)),
            "filtered_to_final_ratio": float(final_stats["total_candidates"] / max(filt_stats["total_candidates"], 1)),
        },
        "components": {
            "raw_components_seen": int(raw_stats["component_size_distribution"]["n_components_seen"]),
            "filtered_components_seen": int(filt_stats["component_size_distribution"]["n_components_seen"]),
            "final_components_seen": int(final_stats["component_size_distribution"]["n_components_seen"]),
        },
        "noisy_persistence": {
            "raw_noisy_local_maxima_ratio_proxy": float(raw_stats["local_noisy_maxima_ratio_proxy"]),
            "filtered_noisy_local_maxima_ratio_proxy": float(filt_stats["local_noisy_maxima_ratio_proxy"]),
            "final_noisy_local_maxima_ratio_proxy": float(final_stats["local_noisy_maxima_ratio_proxy"]),
        },
        "confidence_shift": {
            "raw_all_mean": raw_stats["confidence"]["all"].get("mean"),
            "filtered_all_mean": filt_stats["confidence"]["all"].get("mean"),
            "final_all_mean": final_stats["confidence"]["all"].get("mean"),
        },
        "interpretation": {
            "raw_structural_explosion": bool(raw_stats["columns"]["candidate_to_column_ratio"] >= 40.0),
            "filtering_failure_suspected": bool(survival["filtering_survival_ratio"] >= 0.4),
            "ranking_failure_suspected": bool(ranking["true_branch_rank_proxy"]["top1_far_from_trace_columns"] >= 10),
        },
    }
    wjson(out_root / "candidate_stage_progression_analysis.json", stage_prog)

    # stage progression overlay (simple 3-panel montage from density pngs)
    with Image.open(out_root / "overlay_raw_candidate_density.png") as a, Image.open(out_root / "overlay_filtered_candidate_survival.png") as b, Image.open(out_root / "overlay_final_candidate_branches.png") as c:
        a = a.convert("RGBA")
        b = b.convert("RGBA")
        c = c.convert("RGBA")
    tile_h = max(a.height, b.height, c.height)
    tile_w = max(a.width, b.width, c.width)
    canvas = Image.new("RGBA", (tile_w * 3, tile_h), (15, 15, 15, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (tile_w, 0))
    canvas.paste(c, (tile_w * 2, 0))
    ImageDraw.Draw(canvas).text((10, 10), "raw", fill=(255, 255, 255, 255))
    ImageDraw.Draw(canvas).text((tile_w + 10, 10), "filtered", fill=(255, 255, 255, 255))
    ImageDraw.Draw(canvas).text((tile_w * 2 + 10, 10), "final", fill=(255, 255, 255, 255))
    canvas.save(out_root / "overlay_candidate_stage_progression.png", format="PNG")

    # ---- Task 5: source_numeric alignment overlay (numeric space, not pixel space) ----
    src_json = inp.canonical_item_dir / "source_numeric.json"
    src_obj = rjson(src_json)
    sx = np.asarray(src_obj.get("two_theta_values", []), dtype=np.float64)
    sy = np.asarray(src_obj.get("intensities", []), dtype=np.float64)
    res = rjson(inp.result_json)
    pts = res.get("export_points_highres") or {}
    ex = np.asarray(pts.get("two_theta_values", []), dtype=np.float64)
    ey = np.asarray(pts.get("intensities", []), dtype=np.float64)
    # map column->two_theta by assuming export_points_highres index corresponds to columns (roi_w)
    x_for_col = ex[:roi_w] if ex.size >= roi_w else None
    density_by_col = np.asarray(raw_stats["counts_by_col"], dtype=np.float64) if raw_stats.get("counts_by_col") else None
    _plot_numeric_overlay(
        out_root / "overlay_source_trace_vs_candidates.png",
        title=f"{inp.sample}: source_numeric vs export_highres (+raw burst vlines)",
        source_xy=(sx, sy),
        export_xy=(ex, ey),
        density_by_col=density_by_col if (density_by_col is not None and x_for_col is not None) else None,
        x_for_col=x_for_col,
        note="vlines show top-1% raw candidate density columns (proxy for burst regions)",
    )
    # true vs false candidate regions: reuse burst regions highlight overlay in numeric space (same plot)
    (out_root / "overlay_true_vs_false_candidate_regions.png").write_bytes(
        (out_root / "overlay_source_trace_vs_candidates.png").read_bytes()
    )

    # ---- Task 6: hypotheses ----
    hypotheses: Dict[str, Any] = {}
    # A background texture over-detection
    hypotheses["A_background_texture_over_detection"] = {
        "supporting_evidence": [
            "raw candidate-to-column ratio is high (many candidates per column)",
            "raw component count is large (fragmentation / many small components possible)",
        ],
        "contradicting_evidence": [
            "if burst regions are localized rather than uniform, may be not purely background-wide texture",
        ],
        "signals": {
            "raw_candidate_to_column_ratio": raw_stats["columns"]["candidate_to_column_ratio"],
            "burst_regions_count": len(burst_regions),
        },
    }
    # B local maxima burst
    hypotheses["B_local_maxima_burst"] = {
        "supporting_evidence": [
            "burst regions detected by high per-column candidate count tail",
        ],
        "contradicting_evidence": [
            "no burst regions detected (uniform noise) -> weak support",
        ],
        "signals": {"burst_regions": burst},
    }
    # C branch ambiguity explosion
    hypotheses["C_branch_ambiguity_explosion"] = {
        "supporting_evidence": [
            "final ambiguity proxy columns (top1-top2 margin small) exist -> DP susceptible to switches",
        ],
        "contradicting_evidence": [
            "if ambiguous fraction is near zero, branch ambiguity likely not the driver",
        ],
        "signals": ranking["branch_ambiguity_score"],
    }
    # D confidence collapse
    hypotheses["D_confidence_collapse"] = {
        "supporting_evidence": [
            "if far-from-trace confidence distribution overlaps near-trace heavily, separation is weak",
        ],
        "contradicting_evidence": [
            "if near-trace median is significantly higher than far median, collapse is less likely",
        ],
        "signals": {
            "raw_near_vs_far_median_gap": (
                (raw_stats["confidence"]["near_trace"].get("p50", 0.0) - raw_stats["confidence"]["far_from_trace"].get("p50", 0.0))
                if raw_stats["confidence"]["near_trace"].get("count", 0) and raw_stats["confidence"]["far_from_trace"].get("count", 0)
                else None
            )
        },
    }
    # E connectivity fragmentation
    hypotheses["E_connectivity_fragmentation"] = {
        "supporting_evidence": [
            "n_components is high (from debug.json), suggests fragmented connectivity",
        ],
        "contradicting_evidence": [
            "dominant single component length may indicate not purely fragmented",
        ],
        "signals": {
            "n_components": int(debug.get("n_components") or 0),
            "largest_component_length": (
                max((int(v.get("length") or 0) for v in component_scores.values()), default=0)
                if isinstance(component_scores, dict)
                else None
            ),
        },
    }
    # F oscillatory false branch persistence
    hypotheses["F_oscillatory_false_branch_persistence"] = {
        "supporting_evidence": [
            "final top1 far-from-trace columns exist (proxy of false branch becoming dominant in some columns)",
        ],
        "contradicting_evidence": [
            "if far-from-trace top1 is rare, persistence may not be dominant",
        ],
        "signals": ranking["true_branch_rank_proxy"],
    }
    # G noisy high-frequency component dominance
    hypotheses["G_noisy_high_frequency_component_dominance"] = {
        "supporting_evidence": [
            "top components by raw count may concentrate candidates (component-weighted noise)",
        ],
        "contradicting_evidence": [
            "if a single dominant component aligns with trace, dominance could be correct curve not noise",
        ],
        "signals": raw_stats["component_size_distribution"]["top_components_by_count"][:10],
    }
    wjson(out_root / "failure_trigger_hypotheses.json", hypotheses)

    # ---- Task 7: stabilization direction ranking (no apply) ----
    # Purely evidence-driven priority hints (no parameter changes here).
    priority: List[Dict[str, Any]] = []
    # Score each direction by how much it targets observed signals.
    def add(name: str, why: List[str], score: float) -> None:
        priority.append({"direction": name, "priority_score": float(score), "why": why})

    burst_ct = len(burst_regions)
    amb_ct = int(ranking["branch_ambiguity_score"]["ambiguous_columns_margin_le_0.03"])
    far_top1 = int(ranking["true_branch_rank_proxy"]["top1_far_from_trace_columns"])
    comp_n = int(debug.get("n_components") or 0)
    raw_ratio = float(raw_stats["columns"]["candidate_to_column_ratio"])

    add(
        "noise_adaptive_candidate_pruning",
        ["raw stage has high candidate-to-column ratio", "targets explosion before DP"],
        score=2.5 + 0.05 * min(raw_ratio, 100.0),
    )
    add(
        "component_plausibility_filtering",
        ["many components seen", "component-wise concentration can be used to suppress fragmented noise"],
        score=2.0 + 0.03 * min(comp_n, 60),
    )
    add(
        "branch_switch_penalty",
        ["final stage shows ambiguity/far-top1 signals", "targets DP switching instability without smoothing"],
        score=1.8 + 0.02 * min(amb_ct + far_top1, 200),
    )
    add(
        "candidate_confidence_regularization",
        ["near/far confidence separation weak -> regularization may improve separability"],
        score=1.5 + (0.5 if (raw_stats["confidence"]["far_from_trace"].get("p90") or 0) >= 0.75 else 0.0),
    )
    add(
        "local_oscillation_penalty",
        ["helps suppress oscillatory paths once candidates exist", "does not directly reduce explosion"],
        score=1.2 + (0.3 if far_top1 > 0 else 0.0),
    )
    add(
        "continuity_aware_branch_scoring",
        ["can reduce local branch flips", "mostly affects final selection not raw explosion"],
        score=1.0 + (0.2 if amb_ct > 0 else 0.0),
    )
    priority.sort(key=lambda x: (-x["priority_score"], x["direction"]))
    wjson(out_root / "stabilization_direction_ranking.json", {"priority": priority, "notes": ["Evidence-based ranking only; no application performed."]})

    # ---- Task 8: summary artifacts ----
    # summary json
    summary = {
        "used_manifest": used_manifest,
        "sample": inp.sample,
        "dump_files": used_manifest["dump_files"],
        "raw_explosion": {
            "raw_candidates_total": int(raw_stats["total_candidates"]),
            "candidate_to_column_ratio": raw_stats["columns"]["candidate_to_column_ratio"],
            "burst_regions": burst,
            "top_components": raw_stats["component_size_distribution"]["top_components_by_count"][:8],
        },
        "filtering": {
            "filtered_total": int(filt_stats["total_candidates"]),
            "survival_ratio": survival["filtering_survival_ratio"],
            "near_survival_proxy": survival.get("near_trace_survival_proxy"),
            "far_survival_proxy": survival.get("far_from_trace_survival_proxy"),
        },
        "final_ranking": ranking,
        "source_numeric_alignment": {
            "overlay": "overlay_source_trace_vs_candidates.png",
            "note": "numeric-space overlay; burst vlines use raw candidate density tail by column",
        },
        "most_likely_failure_triggers": [
            "C_branch_ambiguity_explosion" if amb_ct > 50 else "B_local_maxima_burst" if burst_ct > 0 else "A_background_texture_over_detection",
            "E_connectivity_fragmentation" if comp_n >= 10 else "G_noisy_high_frequency_component_dominance",
        ],
        "promising_stabilization_directions": [p["direction"] for p in priority[:3]],
        "artifacts": {
            "raw_candidate_density_analysis": "raw_candidate_density_analysis.json",
            "filtered_candidate_survival_analysis": "filtered_candidate_survival_analysis.json",
            "final_candidate_ranking_analysis": "final_candidate_ranking_analysis.json",
            "candidate_stage_progression_analysis": "candidate_stage_progression_analysis.json",
            "failure_trigger_hypotheses": "failure_trigger_hypotheses.json",
            "stabilization_direction_ranking": "stabilization_direction_ranking.json",
        },
        "overlays": [
            "overlay_raw_candidate_density.png",
            "overlay_raw_candidate_hotspots.png",
            "overlay_filtered_candidate_survival.png",
            "overlay_false_candidate_survival.png",
            "overlay_final_candidate_branches.png",
            "overlay_branch_confidence_map.png",
            "overlay_candidate_stage_progression.png",
            "overlay_source_trace_vs_candidates.png",
            "overlay_true_vs_false_candidate_regions.png",
        ],
        "not_done": [
            "No threshold/margin tuning",
            "No candidate scoring changes",
            "No DP/tracing scoring changes",
            "No heuristics added",
            "No smoothing added",
            "No canonical input modifications",
        ],
    }
    wjson(out_root / "candidate_dump_rootcause_summary.json", summary)

    # summary csv (single row)
    csv_row = {
        "sample": inp.sample,
        "roi_w": roi_w,
        "roi_h": roi_h,
        "raw_total": raw_stats["total_candidates"],
        "filtered_total": filt_stats["total_candidates"],
        "final_total": final_stats["total_candidates"],
        "raw_to_filtered": survival["filtering_survival_ratio"],
        "filtered_to_final": stage_prog["counts"]["filtered_to_final_ratio"],
        "n_components_debug": int(debug.get("n_components") or 0),
        "burst_regions_count": len(burst_regions),
        "final_ambiguous_columns": amb_ct,
        "final_top1_far_columns": far_top1,
        "priority_top1": priority[0]["direction"] if priority else None,
        "priority_top2": priority[1]["direction"] if len(priority) > 1 else None,
        "priority_top3": priority[2]["direction"] if len(priority) > 2 else None,
    }
    with (out_root / "candidate_dump_rootcause_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        w.writeheader()
        w.writerow(csv_row)

    # final decision
    # - localized: burst regions exist OR clear evidence of filtering failure OR ranking failure concentrated
    decision = "CANDIDATE_DUMP_ROOTCAUSE_MIXED"
    localized_signals = 0
    if len(burst_regions) > 0:
        localized_signals += 1
    if survival["filtering_survival_ratio"] >= 0.6:
        localized_signals += 1
    if far_top1 >= 10 or amb_ct >= 100:
        localized_signals += 1
    if localized_signals == 1:
        decision = "CANDIDATE_DUMP_ROOTCAUSE_LOCALIZED"
    elif localized_signals >= 2:
        decision = "CANDIDATE_DUMP_ROOTCAUSE_MIXED"
    else:
        decision = "CANDIDATE_DUMP_ROOTCAUSE_INCONCLUSIVE"

    # summary md
    md = []
    md.append("# Candidate dump root-cause analysis — real_like_pattern_83398")
    md.append("")
    md.append("## A. 사용 manifest")
    md.append(f"- sample: `{inp.sample}`")
    md.append(f"- canonical_item_dir: `{inp.canonical_item_dir}`")
    md.append(f"- dump_debug_dir: `{inp.debug_dir}`")
    md.append("")
    md.append("## B. 사용 sample")
    md.append("- canonical pair only (input.png + mi.json + gt.json + source_numeric.json)")
    md.append("")
    md.append("## C. raw candidate explosion 분석")
    md.append(f"- raw_total: **{raw_stats['total_candidates']}**")
    md.append(f"- candidate_to_column_ratio: **{raw_stats['columns']['candidate_to_column_ratio']:.2f}**")
    md.append(f"- burst_regions_count: **{len(burst_regions)}** (thr={burst.get('threshold')})")
    md.append(f"- top_components: `{[x['component_id'] for x in raw_stats['component_size_distribution']['top_components_by_count'][:5]]}`")
    md.append("")
    md.append("## D. filtered survival 분석")
    md.append(f"- filtered_total: **{filt_stats['total_candidates']}**")
    md.append(f"- raw->filtered survival_ratio: **{survival['filtering_survival_ratio']:.4f}**")
    md.append(f"- near_trace_survival_proxy: `{survival.get('near_trace_survival_proxy')}`")
    md.append(f"- far_from_trace_survival_proxy: `{survival.get('far_from_trace_survival_proxy')}`")
    md.append("")
    md.append("## E. final ranking 분석")
    md.append(f"- ambiguous_columns(margin<=0.03): **{amb_ct}**")
    md.append(f"- top1_far_from_trace_columns: **{far_top1}**")
    md.append("")
    md.append("## F. source trace alignment 분석")
    md.append("- numeric-space overlay 생성 (source_numeric vs export_highres + burst vlines proxy)")
    md.append("")
    md.append("## G. failure trigger hypothesis")
    md.append(f"- primary: `{summary['most_likely_failure_triggers'][0]}`")
    md.append(f"- secondary: `{summary['most_likely_failure_triggers'][1]}`")
    md.append("")
    md.append("## H. stabilization 방향 우선순위")
    md.append(f"- top-3: `{summary['promising_stabilization_directions']}`")
    md.append("")
    md.append("## I. 생성 산출물 경로")
    md.append(f"- root: `{out_root.relative_to(ROOT)}`")
    md.append("")
    md.append("## J. 하지 않은 작업")
    for nd in summary["not_done"]:
        md.append(f"- {nd}")
    md.append("")
    md.append("## K. 최종 판정")
    md.append(f"- `{decision}`")
    (out_root / "candidate_dump_rootcause_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # persist decision
    wjson(out_root / "candidate_dump_rootcause_summary.json", {**summary, "final_decision": decision})
    print(json.dumps({"out_root": str(out_root), "final_decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

