"""GT y와 열별 후보 거리 기반 진단 메트릭 (gates 미사용)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np

from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column


def _col_bucket(fc: Mapping[Any, List[dict]], col: int) -> List[dict]:
    if col in fc:
        return list(fc[col])
    if str(col) in fc:
        return list(fc[str(col)])  # type: ignore[index]
    return []


def compute_candidate_gt_proximity_diag(
    final_cands: Mapping[Any, List[dict]],
    gt_json_path: str,
    plot_box_t: tuple,
    roi_h: int,
    roi_w: int,
) -> Dict[str, Any]:
    gt = _load_gt_json(gt_json_path)
    gt_by_col, meta = build_gt_y_roi_per_column(gt, plot_box_t, roi_h, roi_w)

    dists: List[float] = []
    for col in range(int(roi_w)):
        if col not in gt_by_col:
            continue
        gty = float(gt_by_col[col])
        cands = _col_bucket(final_cands, col)
        if not cands:
            continue
        best_d = min(abs(float(c["y"]) - gty) for c in cands)
        dists.append(best_d)

    if not dists:
        return {
            "candidate_gt_near_recall_px3": None,
            "candidate_gt_near_recall_px5": None,
            "candidate_gt_near_recall_px10": None,
            "mean_nearest_candidate_gt_dist_px": None,
            "median_nearest_candidate_gt_dist_px": None,
            "p90_nearest_candidate_gt_dist_px": None,
            "columns_evaluated": 0,
            "gt_columns_mapped_meta": int(meta.get("columns_mapped", 0)),
        }

    arr = np.asarray(dists, dtype=np.float64)
    n = int(arr.size)

    def recall_px(t: float) -> float:
        return float(np.mean(arr <= float(t)))

    return {
        "candidate_gt_near_recall_px3": round(recall_px(3.0), 6),
        "candidate_gt_near_recall_px5": round(recall_px(5.0), 6),
        "candidate_gt_near_recall_px10": round(recall_px(10.0), 6),
        "mean_nearest_candidate_gt_dist_px": round(float(np.mean(arr)), 6),
        "median_nearest_candidate_gt_dist_px": round(float(np.median(arr)), 6),
        "p90_nearest_candidate_gt_dist_px": round(float(np.percentile(arr, 90)), 6),
        "columns_evaluated": n,
        "gt_columns_mapped_meta": int(meta.get("columns_mapped", 0)),
    }
