#!/usr/bin/env python3
"""Diagnose why DP selects an upper-band path despite GT-near final candidates.

This is a single-run diagnostic script. It reads an existing debug directory
containing debug.json and 20_final_candidates.json and uses GT only for
offline analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from trace.dp_trace import ALPHA, BETA, DELTA, GAMMA  # noqa: E402
from trace.oracle_rerank import build_gt_y_roi_per_column  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_candidates(raw: Dict[str, Any]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for k, vals in raw.items():
        try:
            col = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(vals, list):
            out[col] = [dict(v) for v in vals if isinstance(v, dict)]
    return out


def _summary(vals: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _run_lengths(mask: List[bool]) -> List[int]:
    out: List[int] = []
    cur = 0
    for flag in mask:
        if flag:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _rank_of_y(cands: List[dict], y: Optional[int]) -> Optional[int]:
    if y is None:
        return None
    yi = int(y)
    for idx, cand in enumerate(cands, 1):
        if int(cand.get("y", -999999)) == yi:
            return idx
    return None


def _selected_candidate(cands: List[dict], y: Optional[int]) -> Optional[dict]:
    if y is None:
        return None
    yi = int(y)
    for cand in cands:
        if int(cand.get("y", -999999)) == yi:
            return cand
    return None


def _best_gt_near(
    cands: List[dict],
    gt_y: Optional[float],
    near_px: float,
) -> Tuple[Optional[dict], Optional[int], Optional[float]]:
    if gt_y is None:
        return None, None, None
    near: List[Tuple[int, dict, float]] = []
    for idx, cand in enumerate(cands, 1):
        dist = abs(float(cand.get("y", 0.0)) - float(gt_y))
        if dist <= near_px:
            near.append((idx, cand, dist))
    if not near:
        return None, None, None
    rank, cand, dist = max(near, key=lambda row: float(row[1].get("confidence", 0.0)))
    return cand, rank, dist


def _path_stats(path: List[Optional[int]]) -> Dict[str, Any]:
    jumps: List[float] = []
    curv: List[float] = []
    prev: Optional[int] = None
    prev2: Optional[int] = None
    for y in path:
        if y is None:
            prev = None
            prev2 = None
            continue
        yi = int(y)
        if prev is not None:
            jumps.append(abs(float(yi - prev)))
        if prev is not None and prev2 is not None:
            curv.append(abs(float((yi - prev) - (prev - prev2))))
        prev2 = prev
        prev = yi
    return {
        "jump_abs": _summary(jumps),
        "curvature_abs": _summary(curv),
    }


def _classify(report: Dict[str, Any]) -> str:
    total = max(int(report["total_columns"]), 1)
    recall5 = report["gt_near_candidate_columns"]["px5"] / total
    gap_p90 = report["gt_near_missing_gap_lengths_px5_summary"].get("p90")
    run_med = report["gt_near_run_lengths_px5_summary"].get("median")
    gt_conf = report["confidence"]["best_gt_near_px5"].get("mean")
    sel_conf = report["confidence"]["dp_selected"].get("mean")
    selected_gt5 = report["dp_selected_gt_near_columns"]["px5"] / total
    upper_frac = report["dp_selected_upper_band_columns"] / total

    if recall5 < 0.35 or (gap_p90 is not None and gap_p90 >= 12) or (run_med is not None and run_med <= 2):
        return "DP_LOCK_DUE_TO_GT_NEAR_DISCONTINUITY"
    if gt_conf is not None and sel_conf is not None and gt_conf < sel_conf - 0.10:
        return "DP_CONFIDENCE_SIGNAL_WEAK"
    if upper_frac > 0.65 and selected_gt5 < 0.15 and gt_conf is not None and sel_conf is not None and gt_conf >= sel_conf - 0.05:
        return "DP_SMOOTHNESS_OVERPOWERS_CONFIDENCE"
    return "INCONCLUSIVE_DP_LOCK_DIAGNOSIS"


def analyze(run_dir: Path, gt_json: Path) -> Dict[str, Any]:
    debug = _read_json(run_dir / "debug.json")
    cand_path = run_dir / "20_final_candidates.json"
    if not cand_path.exists():
        matches = sorted(run_dir.glob("*final_candidates.json"))
        if not matches:
            raise FileNotFoundError(f"final candidates JSON not found in {run_dir}")
        cand_path = matches[-1]
    final = _as_candidates(_read_json(cand_path))
    gt = _read_json(gt_json)

    plot_box = debug.get("plot_box") or gt.get("plot_box")
    if not isinstance(plot_box, list) or len(plot_box) != 4:
        raise ValueError("plot_box missing from debug/GT")
    x0, y0, x1, y1 = [int(v) for v in plot_box]
    roi_w = int(debug.get("candidate_stats", {}).get("total_columns") or (x1 - x0))
    roi_h = int(y1 - y0)
    path = debug.get("trace", {}).get("path", [])
    if not isinstance(path, list):
        raise ValueError("debug.trace.path missing")
    if len(path) < roi_w:
        path = list(path) + [None] * (roi_w - len(path))
    elif len(path) > roi_w:
        path = path[:roi_w]

    gt_by_col, gt_meta = build_gt_y_roi_per_column(gt, tuple(plot_box), roi_h, roi_w)
    upper_y = float(roi_h) * 0.2

    near_exists = {3: [], 5: [], 10: []}
    selected_near = {3: [], 5: [], 10: []}
    gt_near_cols = {3: [], 5: [], 10: []}
    selected_gt_cols = {3: [], 5: [], 10: []}
    selected_upper: List[bool] = []
    selected_upper_cols: List[int] = []
    gt_near_but_upper_cols: List[int] = []

    gt_near_conf_px5: List[float] = []
    dp_selected_conf: List[float] = []
    upper_candidate_conf: List[float] = []
    selected_upper_conf: List[float] = []
    confidence_gap_gt_minus_selected: List[float] = []
    gt_near_rank_px5: List[int] = []
    selected_rank: List[int] = []
    gt_greedy_path_px5: List[Optional[int]] = []
    gt_greedy_conf_px5: List[float] = []
    upper_path: List[Optional[int]] = []

    per_col: List[Dict[str, Any]] = []

    for col in range(roi_w):
        cands = final.get(col, [])
        gt_y = gt_by_col.get(col)
        sel_y = path[col]
        if sel_y is not None:
            try:
                sel_y = int(sel_y)
            except (TypeError, ValueError):
                sel_y = None
        sel = _selected_candidate(cands, sel_y)
        sel_rank = _rank_of_y(cands, sel_y)
        if sel_rank is not None:
            selected_rank.append(sel_rank)
        if sel is not None:
            dp_selected_conf.append(float(sel.get("confidence", 0.0)))

        is_sel_upper = bool(sel_y is not None and float(sel_y) < upper_y)
        selected_upper.append(is_sel_upper)
        if is_sel_upper:
            selected_upper_cols.append(col)
            upper_path.append(sel_y)
            if sel is not None:
                selected_upper_conf.append(float(sel.get("confidence", 0.0)))
        else:
            upper_path.append(None if sel_y is None else int(sel_y))

        for cand in cands:
            if float(cand.get("y", 0.0)) < upper_y:
                upper_candidate_conf.append(float(cand.get("confidence", 0.0)))

        best5, rank5, dist5 = _best_gt_near(cands, gt_y, 5)
        gt_greedy_path_px5.append(None if best5 is None else int(best5.get("y", 0)))
        if best5 is not None:
            gt_near_conf_px5.append(float(best5.get("confidence", 0.0)))
            gt_greedy_conf_px5.append(float(best5.get("confidence", 0.0)))
            if rank5 is not None:
                gt_near_rank_px5.append(rank5)
            if sel is not None:
                confidence_gap_gt_minus_selected.append(
                    float(best5.get("confidence", 0.0)) - float(sel.get("confidence", 0.0))
                )
        if best5 is not None and is_sel_upper:
            gt_near_but_upper_cols.append(col)

        row: Dict[str, Any] = {
            "col": col,
            "gt_y": None if gt_y is None else round(float(gt_y), 4),
            "selected_y": sel_y,
            "selected_confidence": None if sel is None else round(float(sel.get("confidence", 0.0)), 8),
            "selected_rank": sel_rank,
            "selected_is_upper_band": is_sel_upper,
            "best_gt_near_px5_y": None if best5 is None else int(best5.get("y", 0)),
            "best_gt_near_px5_confidence": None if best5 is None else round(float(best5.get("confidence", 0.0)), 8),
            "best_gt_near_px5_rank": rank5,
            "best_gt_near_px5_dist": None if dist5 is None else round(float(dist5), 4),
        }
        per_col.append(row)

        for px in (3, 5, 10):
            best, _rank, _dist = _best_gt_near(cands, gt_y, px)
            exists = best is not None
            near_exists[px].append(exists)
            if exists:
                gt_near_cols[px].append(col)
            chosen = bool(sel_y is not None and gt_y is not None and abs(float(sel_y) - float(gt_y)) <= px)
            selected_near[px].append(chosen)
            if chosen:
                selected_gt_cols[px].append(col)

    gt_path_stats = _path_stats(gt_greedy_path_px5)
    upper_path_stats = _path_stats(path)

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "gt_json": str(gt_json),
        "total_columns": int(roi_w),
        "roi_height": int(roi_h),
        "upper_band_y_threshold": round(float(upper_y), 4),
        "gt_curve_meta": gt_meta,
        "gt_near_candidate_columns": {f"px{px}": int(sum(near_exists[px])) for px in (3, 5, 10)},
        "dp_selected_gt_near_columns": {f"px{px}": int(sum(selected_near[px])) for px in (3, 5, 10)},
        "dp_selected_upper_band_columns": int(sum(selected_upper)),
        "gt_near_exists_but_dp_selected_upper_columns": int(len(gt_near_but_upper_cols)),
        "confidence": {
            "best_gt_near_px5": _summary(gt_near_conf_px5),
            "dp_selected": _summary(dp_selected_conf),
            "upper_band_candidates_all": _summary(upper_candidate_conf),
            "upper_band_selected": _summary(selected_upper_conf),
            "gt_near_minus_dp_selected_px5": _summary(confidence_gap_gt_minus_selected),
        },
        "rank": {
            "best_gt_near_px5": _summary(gt_near_rank_px5),
            "dp_selected": _summary(selected_rank),
        },
        "gt_near_run_lengths_px5": _run_lengths(near_exists[5]),
        "upper_band_selected_run_lengths": _run_lengths(selected_upper),
        "gt_near_missing_gap_lengths_px5": _run_lengths([not b for b in near_exists[5]]),
        "gt_near_run_lengths_px5_summary": _summary(_run_lengths(near_exists[5])),
        "upper_band_selected_run_lengths_summary": _summary(_run_lengths(selected_upper)),
        "gt_near_missing_gap_lengths_px5_summary": _summary(_run_lengths([not b for b in near_exists[5]])),
        "gt_near_feasible_path_px5": {
            "jump_abs": gt_path_stats["jump_abs"],
            "curvature_abs": gt_path_stats["curvature_abs"],
            "confidence": _summary(gt_greedy_conf_px5),
        },
        "dp_selected_path": {
            "jump_abs": upper_path_stats["jump_abs"],
            "curvature_abs": upper_path_stats["curvature_abs"],
            "confidence": _summary(dp_selected_conf),
        },
        "cost_formula": {
            "transition_cost": "ALPHA*|dy| + BETA*|d2y| + GAMMA*(1-confidence) + DELTA*component_switch + border_penalty",
            "ALPHA_dy": float(ALPHA),
            "BETA_curvature": float(BETA),
            "GAMMA_confidence": float(GAMMA),
            "DELTA_component_switch": float(DELTA),
            "confidence_note": "confidence can change total cost by at most GAMMA per column at multiplier 1.0; dy of 1px costs ALPHA.",
        },
        "example_columns_gt_near_but_upper": gt_near_but_upper_cols[:40],
        "per_column": per_col,
    }
    report["primary_diagnosis"] = _classify(report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze DP upper-band lock for one debug run")
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--gt-json", required=True, type=str)
    ap.add_argument("--out-json", default=None, type=str)
    ap.add_argument("--include-per-column", action="store_true")
    args = ap.parse_args()

    report = analyze(Path(args.run_dir), Path(args.gt_json))
    printable = dict(report)
    if not args.include_per_column:
        printable.pop("per_column", None)
    text = json.dumps(printable, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
