#!/usr/bin/env python3
"""Analyze final candidate continuity for one diagnostic run.

GT is used only for offline diagnosis. The runtime candidate preservation path
must remain GT-free.
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
    arr = np.asarray(
        [float(v) for v in vals if v is not None and math.isfinite(float(v))],
        dtype=np.float64,
    )
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


def _gap_ranges(mask: List[bool]) -> List[Tuple[int, int]]:
    gaps: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if not flag and start is None:
            start = idx
        elif flag and start is not None:
            gaps.append((start, idx - 1))
            start = None
    if start is not None:
        gaps.append((start, len(mask) - 1))
    return gaps


def _best_near(
    cands: List[dict],
    gt_y: Optional[float],
    near_px: float,
) -> Tuple[Optional[dict], Optional[float]]:
    if gt_y is None:
        return None, None
    near: List[Tuple[float, dict]] = []
    for cand in cands:
        dist = abs(float(cand.get("y", 0.0)) - float(gt_y))
        if dist <= near_px:
            near.append((dist, cand))
    if not near:
        return None, None
    dist, cand = min(near, key=lambda row: (row[0], -float(row[1].get("confidence", 0.0))))
    return cand, float(dist)


def _has_near(cands: List[dict], gt_y: Optional[float], near_px: float) -> bool:
    cand, _ = _best_near(cands, gt_y, near_px)
    return cand is not None


def _nearest_y(cands: List[dict], gt_y: Optional[float]) -> Tuple[Optional[int], Optional[float]]:
    if gt_y is None or not cands:
        return None, None
    rows = [
        (abs(float(c.get("y", 0.0)) - float(gt_y)), int(c.get("y", 0)))
        for c in cands
    ]
    dist, y = min(rows, key=lambda row: row[0])
    return int(y), float(dist)


def _path_jump_summary(path: List[Optional[int]]) -> Dict[str, Any]:
    jumps: List[float] = []
    prev: Optional[int] = None
    for y in path:
        if y is None:
            prev = None
            continue
        yi = int(y)
        if prev is not None:
            jumps.append(abs(float(yi - prev)))
        prev = yi
    return _summary(jumps)


def _continuous_branch_stats(
    candidates: Dict[int, List[dict]],
    roi_w: int,
    *,
    window: int,
    max_jump: int,
) -> Dict[str, Any]:
    win = max(1, int(window))
    mj = max(1, int(max_jump))
    rows_by_col: Dict[int, List[dict]] = {
        col: list(candidates.get(col, [])) for col in range(roi_w)
    }
    dp: Dict[Tuple[int, int], Tuple[int, float, Optional[Tuple[int, int]]]] = {}
    for col in range(roi_w):
        rows = rows_by_col.get(col, [])
        for idx, cand in enumerate(rows):
            y = float(cand.get("y", 0.0))
            best_len = 1
            best_score = float(cand.get("confidence", 0.0))
            best_prev: Optional[Tuple[int, int]] = None
            for dcol in range(1, win + 1):
                pcol = col - dcol
                if pcol < 0:
                    break
                allowed = float(mj * dcol)
                for pidx, prev_cand in enumerate(rows_by_col.get(pcol, [])):
                    jump = abs(y - float(prev_cand.get("y", 0.0)))
                    if jump > allowed:
                        continue
                    pkey = (pcol, pidx)
                    if pkey not in dp:
                        continue
                    plen, pscore, _ = dp[pkey]
                    cand_len = int(plen) + 1
                    cand_score = float(pscore) + float(cand.get("confidence", 0.0)) - jump / max(allowed, 1e-6)
                    if (cand_len, cand_score) > (best_len, best_score):
                        best_len = cand_len
                        best_score = cand_score
                        best_prev = pkey
            dp[(col, idx)] = (best_len, best_score, best_prev)
    if not dp:
        return {
            "exists": False,
            "longest_branch_columns": 0,
            "longest_branch_cover_ratio": 0.0,
        }
    end_key = max(dp.keys(), key=lambda key: (dp[key][0], dp[key][1]))
    path_keys: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = end_key
    seen: set[Tuple[int, int]] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        path_keys.append(cur)
        cur = dp.get(cur, (0, 0.0, None))[2]
    path_keys.reverse()
    ys = [int(rows_by_col[col][idx].get("y", 0)) for col, idx in path_keys]
    cols = [int(col) for col, _ in path_keys]
    jumps = [abs(float(ys[i] - ys[i - 1])) for i in range(1, len(ys))]
    gaps = [int(cols[i] - cols[i - 1]) for i in range(1, len(cols))]
    return {
        "exists": bool(path_keys),
        "longest_branch_columns": int(len(path_keys)),
        "longest_branch_cover_ratio": round(float(len(path_keys)) / float(max(roi_w, 1)), 6),
        "longest_branch_col_start": int(cols[0]) if cols else None,
        "longest_branch_col_end": int(cols[-1]) if cols else None,
        "longest_branch_y_start": int(ys[0]) if ys else None,
        "longest_branch_y_end": int(ys[-1]) if ys else None,
        "longest_branch_jump_abs": _summary(jumps),
        "longest_branch_column_gap": _summary(gaps),
    }


def _gap_detail(
    *,
    gaps: List[Tuple[int, int]],
    final_best_y: List[Optional[int]],
    gt_y_by_col: Dict[int, float],
    filtered_near5: List[bool],
    filtered_near10: List[bool],
    final: Dict[int, List[dict]],
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    n = len(final_best_y)
    for start, end in gaps:
        left = start - 1 if start > 0 else None
        right = end + 1 if end + 1 < n else None
        left_y = final_best_y[left] if left is not None else None
        right_y = final_best_y[right] if right is not None else None
        left_gt = gt_y_by_col.get(left) if left is not None else None
        right_gt = gt_y_by_col.get(right) if right is not None else None
        count_filtered5 = sum(1 for col in range(start, end + 1) if filtered_near5[col])
        count_filtered10 = sum(1 for col in range(start, end + 1) if filtered_near10[col])
        count_final_nonempty = sum(1 for col in range(start, end + 1) if final.get(col))
        details.append(
            {
                "start_col": int(start),
                "end_col": int(end),
                "length": int(end - start + 1),
                "left_final_gt_near_y": None if left_y is None else int(left_y),
                "right_final_gt_near_y": None if right_y is None else int(right_y),
                "left_gt_y": None if left_gt is None else round(float(left_gt), 4),
                "right_gt_y": None if right_gt is None else round(float(right_gt), 4),
                "boundary_gt_y_delta": (
                    None if left_gt is None or right_gt is None else round(float(right_gt - left_gt), 4)
                ),
                "boundary_final_gt_near_y_delta": (
                    None if left_y is None or right_y is None else int(right_y - left_y)
                ),
                "gap_columns_with_filtered_gt_near_px5": int(count_filtered5),
                "gap_columns_with_filtered_gt_near_px10": int(count_filtered10),
                "gap_columns_with_any_final_candidate": int(count_final_nonempty),
            }
        )
    return details


def _classify(report: Dict[str, Any]) -> str:
    final5 = int(report["final_gt_near_candidate_columns"]["px5"])
    filtered5 = int(report["filtered_gt_near_candidate_columns"]["px5"])
    filtered_lost = int(report["gap_diagnostics"]["gap_columns_filtered_has_gt_near_px5"])
    jump_p90 = report["best_final_gt_near_px5_path_jump_abs"].get("p90")
    branch_ratio = float(report["final_y_continuous_branch"]["longest_branch_cover_ratio"])
    if filtered_lost > 0:
        return "FINAL_CANDIDATE_CONTINUITY_PROBLEM"
    if filtered5 <= final5:
        return "FILTER_PRESERVE_INSUFFICIENT"
    if jump_p90 is not None and float(jump_p90) > 8.0:
        return "CONTINUITY_AWARE_SELECTION_NEEDED"
    if branch_ratio >= 0.90 and final5 < 0.35 * int(report["total_columns"]):
        return "FINAL_BRANCH_EXISTS_BUT_GT_NEAR_BRANCH_SPARSE"
    return "INCONCLUSIVE_FINAL_CONTINUITY_DIAGNOSIS"


def analyze(
    run_dir: Path,
    gt_json: Path,
    *,
    continuity_window: int = 3,
    continuity_max_jump: int = 8,
    include_per_column: bool = False,
) -> Dict[str, Any]:
    debug = _read_json(run_dir / "debug.json")
    final_path = run_dir / "20_final_candidates.json"
    filtered_path = run_dir / "19_filtered_candidates.json"
    if not final_path.exists():
        matches = sorted(run_dir.glob("*final_candidates.json"))
        if not matches:
            raise FileNotFoundError(f"final candidates JSON not found in {run_dir}")
        final_path = matches[-1]
    if not filtered_path.exists():
        matches = sorted(run_dir.glob("*filtered_candidates.json"))
        if not matches:
            raise FileNotFoundError(f"filtered candidates JSON not found in {run_dir}")
        filtered_path = matches[-1]

    final = _as_candidates(_read_json(final_path))
    filtered = _as_candidates(_read_json(filtered_path))
    gt = _read_json(gt_json)
    plot_box = debug.get("plot_box") or gt.get("plot_box")
    if not isinstance(plot_box, list) or len(plot_box) != 4:
        raise ValueError("plot_box missing from debug/GT")
    x0, y0, x1, y1 = [int(v) for v in plot_box]
    roi_w = int(debug.get("candidate_stats", {}).get("total_columns") or (x1 - x0))
    roi_h = int(y1 - y0)
    gt_by_col, gt_meta = build_gt_y_roi_per_column(gt, tuple(plot_box), roi_h, roi_w)
    upper_y = float(roi_h) * 0.2

    final_near = {5: [], 10: []}
    filtered_near = {5: [], 10: []}
    final_best_y_px5: List[Optional[int]] = []
    final_nearest_dist: List[float] = []
    per_column: List[Dict[str, Any]] = []
    upper_mask: List[bool] = []

    for col in range(roi_w):
        gt_y = gt_by_col.get(col)
        fc = final.get(col, [])
        flt = filtered.get(col, [])
        best5, dist5 = _best_near(fc, gt_y, 5.0)
        best10, dist10 = _best_near(fc, gt_y, 10.0)
        f_best5, _ = _best_near(flt, gt_y, 5.0)
        f_best10, _ = _best_near(flt, gt_y, 10.0)
        final_near[5].append(best5 is not None)
        final_near[10].append(best10 is not None)
        filtered_near[5].append(f_best5 is not None)
        filtered_near[10].append(f_best10 is not None)
        final_best_y_px5.append(None if best5 is None else int(best5.get("y", 0)))
        _, nd = _nearest_y(fc, gt_y)
        if nd is not None:
            final_nearest_dist.append(float(nd))
        upper_mask.append(any(float(c.get("y", 0.0)) < upper_y for c in fc))
        if include_per_column:
            per_column.append(
                {
                    "col": int(col),
                    "gt_y": None if gt_y is None else round(float(gt_y), 4),
                    "final_y_list": [int(c.get("y", 0)) for c in fc],
                    "filtered_y_list": [int(c.get("y", 0)) for c in flt],
                    "best_final_gt_near_px5_y": None if best5 is None else int(best5.get("y", 0)),
                    "best_final_gt_near_px5_dist": None if dist5 is None else round(float(dist5), 4),
                    "best_final_gt_near_px10_y": None if best10 is None else int(best10.get("y", 0)),
                    "best_final_gt_near_px10_dist": None if dist10 is None else round(float(dist10), 4),
                    "filtered_has_gt_near_px5": bool(f_best5 is not None),
                    "filtered_has_gt_near_px10": bool(f_best10 is not None),
                    "final_has_upper_band_candidate": bool(upper_mask[-1]),
                }
            )

    gaps5 = _gap_ranges(final_near[5])
    gap_details5 = _gap_detail(
        gaps=gaps5,
        final_best_y=final_best_y_px5,
        gt_y_by_col=gt_by_col,
        filtered_near5=filtered_near[5],
        filtered_near10=filtered_near[10],
        final=final,
    )
    gap_cols_filtered5 = sum(
        int(row["gap_columns_with_filtered_gt_near_px5"]) for row in gap_details5
    )
    gap_cols_filtered10 = sum(
        int(row["gap_columns_with_filtered_gt_near_px10"]) for row in gap_details5
    )

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "gt_json": str(gt_json),
        "total_columns": int(roi_w),
        "roi_height": int(roi_h),
        "gt_mapped_meta": gt_meta,
        "final_gt_near_candidate_columns": {
            "px5": int(sum(final_near[5])),
            "px10": int(sum(final_near[10])),
        },
        "filtered_gt_near_candidate_columns": {
            "px5": int(sum(filtered_near[5])),
            "px10": int(sum(filtered_near[10])),
        },
        "final_gt_near_candidate_recall": {
            "px5": round(float(sum(final_near[5])) / float(max(roi_w, 1)), 6),
            "px10": round(float(sum(final_near[10])) / float(max(roi_w, 1)), 6),
        },
        "final_gt_near_run_lengths_px5": _run_lengths(final_near[5]),
        "final_gt_near_run_lengths_px5_summary": _summary(_run_lengths(final_near[5])),
        "final_gt_near_run_lengths_px10": _run_lengths(final_near[10]),
        "final_gt_near_run_lengths_px10_summary": _summary(_run_lengths(final_near[10])),
        "final_gt_near_missing_gap_lengths_px5": [end - start + 1 for start, end in gaps5],
        "final_gt_near_missing_gap_lengths_px5_summary": _summary(
            [end - start + 1 for start, end in gaps5]
        ),
        "best_final_gt_near_px5_path_jump_abs": _path_jump_summary(final_best_y_px5),
        "mean_nearest_candidate_gt_dist_px": (
            None if not final_nearest_dist else round(float(np.mean(final_nearest_dist)), 6)
        ),
        "upper_band_candidate_run_lengths": _run_lengths(upper_mask),
        "upper_band_candidate_run_lengths_summary": _summary(_run_lengths(upper_mask)),
        "final_y_continuous_branch": _continuous_branch_stats(
            final,
            roi_w,
            window=int(continuity_window),
            max_jump=int(continuity_max_jump),
        ),
        "filtered_y_continuous_branch": _continuous_branch_stats(
            filtered,
            roi_w,
            window=int(continuity_window),
            max_jump=int(continuity_max_jump),
        ),
        "gap_diagnostics": {
            "gap_count_px5": int(len(gaps5)),
            "gap_columns_filtered_has_gt_near_px5": int(gap_cols_filtered5),
            "gap_columns_filtered_has_gt_near_px10": int(gap_cols_filtered10),
            "gap_details_px5": gap_details5,
        },
    }
    report["classification"] = _classify(report)
    if include_per_column:
        report["per_column"] = per_column
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    print(json.dumps({
        "classification": report["classification"],
        "total_columns": report["total_columns"],
        "final_gt_near_candidate_columns": report["final_gt_near_candidate_columns"],
        "filtered_gt_near_candidate_columns": report["filtered_gt_near_candidate_columns"],
        "final_gt_near_run_lengths_px5_summary": report["final_gt_near_run_lengths_px5_summary"],
        "final_gt_near_missing_gap_lengths_px5_summary": report[
            "final_gt_near_missing_gap_lengths_px5_summary"
        ],
        "best_final_gt_near_px5_path_jump_abs": report["best_final_gt_near_px5_path_jump_abs"],
        "upper_band_candidate_run_lengths_summary": report[
            "upper_band_candidate_run_lengths_summary"
        ],
        "final_y_continuous_branch": report["final_y_continuous_branch"],
        "filtered_y_continuous_branch": report["filtered_y_continuous_branch"],
        "gap_diagnostics_summary": {
            "gap_count_px5": report["gap_diagnostics"]["gap_count_px5"],
            "gap_columns_filtered_has_gt_near_px5": report["gap_diagnostics"][
                "gap_columns_filtered_has_gt_near_px5"
            ],
            "gap_columns_filtered_has_gt_near_px10": report["gap_diagnostics"][
                "gap_columns_filtered_has_gt_near_px10"
            ],
        },
    }, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--gt-json", required=True, type=Path)
    ap.add_argument("--continuity-window", type=int, default=3)
    ap.add_argument("--continuity-max-jump", type=int, default=8)
    ap.add_argument("--include-per-column", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    report = analyze(
        args.run_dir,
        args.gt_json,
        continuity_window=int(args.continuity_window),
        continuity_max_jump=int(args.continuity_max_jump),
        include_per_column=bool(args.include_per_column),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(report)


if __name__ == "__main__":
    main()
