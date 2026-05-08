#!/usr/bin/env python3
"""raw->filtered->final 후보 축약 과정에서 GT-near 손실 분석."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column


def _norm(fc: Mapping[Any, Any]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for k, v in fc.items():
        try:
            kk = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[kk] = list(v)
    return out


def _nearest_stats(cands: Dict[int, List[dict]], gt: Dict[int, float], w: int) -> Dict[str, float]:
    dists: List[float] = []
    for col in range(w):
        g = gt.get(col)
        if g is None:
            continue
        lst = cands.get(col, [])
        if not lst:
            continue
        dists.append(min(abs(float(c.get("y", 0.0)) - float(g)) for c in lst))
    if not dists:
        return {
            "gt_near_recall_px3": float("nan"),
            "gt_near_recall_px5": float("nan"),
            "gt_near_recall_px10": float("nan"),
            "mean_nearest_candidate_gt_dist_px": float("nan"),
            "median_nearest_candidate_gt_dist_px": float("nan"),
            "p90_nearest_candidate_gt_dist_px": float("nan"),
        }
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "gt_near_recall_px3": float(np.mean(arr <= 3.0)),
        "gt_near_recall_px5": float(np.mean(arr <= 5.0)),
        "gt_near_recall_px10": float(np.mean(arr <= 10.0)),
        "mean_nearest_candidate_gt_dist_px": float(np.mean(arr)),
        "median_nearest_candidate_gt_dist_px": float(np.median(arr)),
        "p90_nearest_candidate_gt_dist_px": float(np.percentile(arr, 90)),
    }


def _stage_stats(name: str, cands: Dict[int, List[dict]], gt: Dict[int, float], w: int) -> Dict[str, float]:
    counts = [len(cands.get(col, [])) for col in range(w)]
    total = int(sum(counts))
    near = _nearest_stats(cands, gt, w)
    out: Dict[str, float] = {
        "total_columns": float(w),
        "total_candidates": float(total),
        "candidates_per_col_min": float(min(counts) if counts else 0),
        "candidates_per_col_mean": float(np.mean(counts) if counts else 0.0),
        "candidates_per_col_max": float(max(counts) if counts else 0),
    }
    out.update(near)
    print(f"\n[{name}]")
    for k in (
        "total_columns",
        "total_candidates",
        "candidates_per_col_min",
        "candidates_per_col_mean",
        "candidates_per_col_max",
        "gt_near_recall_px3",
        "gt_near_recall_px5",
        "gt_near_recall_px10",
        "mean_nearest_candidate_gt_dist_px",
        "median_nearest_candidate_gt_dist_px",
        "p90_nearest_candidate_gt_dist_px",
    ):
        print(f"{k}={out[k]}")
    return out


def _has_near(lst: List[dict], gty: float, thr: float) -> bool:
    return any(abs(float(c.get("y", 0.0)) - float(gty)) <= thr for c in lst)


def _best_near(lst: List[dict], gty: float, thr: float) -> dict | None:
    near = [c for c in lst if abs(float(c.get("y", 0.0)) - float(gty)) <= thr]
    if not near:
        return None
    return max(near, key=lambda c: float(c.get("confidence", 0.0)))


def _load_json(path: Path) -> Dict[int, List[dict]]:
    return _norm(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--gt-json", required=True, type=str)
    ap.add_argument("--near-px", default=5.0, type=float)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dbg = json.loads((run_dir / "debug.json").read_text(encoding="utf-8"))
    raw = _load_json(next(iter(sorted(run_dir.glob("*_raw_candidates.json")))))
    fil = _load_json(next(iter(sorted(run_dir.glob("*_filtered_candidates.json")))))
    fin = _load_json(next(iter(sorted(run_dir.glob("*_final_candidates.json")))))
    tr_path = ((dbg.get("trace") or {}).get("path") or [])
    plot_box = tuple(int(x) for x in dbg.get("plot_box", [0, 0, 0, 0]))
    roi_h = int(plot_box[3] - plot_box[1])
    roi_w = int(plot_box[2] - plot_box[0])
    gt = _load_gt_json(str(args.gt_json))
    gt_by_col, _ = build_gt_y_roi_per_column(gt, plot_box, roi_h, roi_w)

    _stage_stats("raw", raw, gt_by_col, roi_w)
    _stage_stats("filtered", fil, gt_by_col, roi_w)
    _stage_stats("final", fin, gt_by_col, roi_w)

    near_px = float(args.near_px)
    raw_to_fil_lost = 0
    fil_to_fin_lost = 0
    raw_to_fin_lost = 0
    rank_outside_topk = 0
    conf_below_keep = 0
    dedup_like_loss = 0
    topk_removed = 0
    final_upper_top1_cols = 0
    path_upper_pick = 0
    path_gt_near_pick = 0
    upper_bound = int(math.ceil(roi_h * 0.2))
    rows: List[Dict[str, Any]] = []
    for col in range(roi_w):
        g = gt_by_col.get(col)
        if g is None:
            continue
        rlst, flst, fnlst = raw.get(col, []), fil.get(col, []), fin.get(col, [])
        r_has = _has_near(rlst, g, near_px)
        f_has = _has_near(flst, g, near_px)
        n_has = _has_near(fnlst, g, near_px)
        if r_has and not f_has:
            raw_to_fil_lost += 1
        if f_has and not n_has:
            fil_to_fin_lost += 1
        if r_has and not n_has:
            raw_to_fin_lost += 1

        r_best = _best_near(rlst, g, near_px)
        rank = None
        source = ""
        raw_conf = None
        raw_y = None
        if r_best is not None:
            raw_y = int(r_best.get("y", -1))
            raw_conf = float(r_best.get("confidence", 0.0))
            source = str(r_best.get("source", ""))
            sorted_r = sorted(rlst, key=lambda c: -float(c.get("confidence", 0.0)))
            for i, c in enumerate(sorted_r, 1):
                if int(c.get("y", -9999)) == int(raw_y):
                    rank = i
                    break
            if rank is not None and rank > 8 and not n_has:
                rank_outside_topk += 1
            if raw_conf is not None and raw_conf < 0.25 and not f_has:
                conf_below_keep += 1
            if f_has and not n_has:
                grouped = {}
                for c in flst:
                    k = round(float(c.get("comp_score", 0.0)), 2)
                    grouped.setdefault(k, []).append(c)
                k0 = round(float(r_best.get("comp_score", 0.0)), 2)
                grp = grouped.get(k0, [])
                if grp:
                    best_g = max(grp, key=lambda c: float(c.get("confidence", 0.0)))
                    if int(best_g.get("y", -9999)) != int(raw_y):
                        dedup_like_loss += 1
                if rank is not None and rank <= 8:
                    topk_removed += 1

        top1 = max(fnlst, key=lambda c: float(c.get("confidence", 0.0))) if fnlst else None
        top1_y = int(top1.get("y", -1)) if top1 else None
        if top1_y is not None and top1_y < upper_bound:
            final_upper_top1_cols += 1

        py = tr_path[col] if col < len(tr_path) else None
        if py is not None:
            if int(py) < upper_bound:
                path_upper_pick += 1
            if abs(float(py) - float(g)) <= near_px:
                path_gt_near_pick += 1

        rows.append(
            {
                "col": col,
                "gt_y": round(float(g), 4),
                "raw_gt_near_y": raw_y,
                "raw_gt_near_conf": raw_conf,
                "raw_gt_near_rank_conf": rank,
                "raw_gt_near_source": source,
                "present_in_filtered": f_has,
                "present_in_final": n_has,
                "final_topk_y_list": [int(c.get("y", 0)) for c in sorted(fnlst, key=lambda x: -float(x.get("confidence", 0.0)))[:8]],
                "final_topk_conf_list": [round(float(c.get("confidence", 0.0)), 6) for c in sorted(fnlst, key=lambda x: -float(x.get("confidence", 0.0)))[:8]],
                "final_upper_band_y_conf": [
                    [int(c.get("y", 0)), round(float(c.get("confidence", 0.0)), 6)]
                    for c in fnlst if int(c.get("y", 0)) < upper_bound
                ],
            }
        )

    print("\n[transition_loss]")
    print(f"raw_to_filtered_gt_near_lost_columns_px5={raw_to_fil_lost}")
    print(f"filtered_to_final_gt_near_lost_columns_px5={fil_to_fin_lost}")
    print(f"raw_to_final_gt_near_lost_columns_px5={raw_to_fin_lost}")
    biggest = max(
        [
            ("raw->filtered", raw_to_fil_lost),
            ("filtered->final", fil_to_fin_lost),
            ("raw->final", raw_to_fin_lost),
        ],
        key=lambda x: x[1],
    )
    print(f"largest_loss_stage={biggest[0]} ({biggest[1]})")

    print("\n[inferred_reasons]")
    print(f"gt_near_removed_with_raw_conf_below_0.25={conf_below_keep}")
    print(f"gt_near_removed_rank_outside_top8={rank_outside_topk}")
    print(f"gt_near_removed_dedup_like_same_comp_score_bucket={dedup_like_loss}")
    print(f"gt_near_removed_even_rank<=8(topk/dp-rank-misaligned)_columns={topk_removed}")
    print(f"final_top1_upper_band_columns={final_upper_top1_cols}")
    print(f"final_path_upper_band_columns={path_upper_pick}")
    print(f"final_path_gt_near_columns_px5={path_gt_near_pick}")
    print("final_topk_sort_key_hint=confidence + skeleton_hint_y_proximity (trace/candidates.py::_dp_rank)")

    out_csv = run_dir / "candidate_pruning_loss_columns.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["col"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"columns_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

