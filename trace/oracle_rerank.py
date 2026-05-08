"""
GT oracle 후보 재랭크: 열별 후보 y 와 GT 곡선 보간 y 의 거리로 confidence 재설정 후 DP.

학습 모델 없이 \"후보만 잘 고르면 개선되는가\"의 상한을 본다.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trace.candidates import dp_transition_window_width
from trace.dp_trace import dp_trace, refine_dp_path_column_apex_pull


def _load_gt_json(gt_path: str) -> dict:
    p = Path(gt_path)
    if not p.is_file():
        raise FileNotFoundError(f"GT JSON not found: {gt_path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _points_from_gt(gt: dict) -> List[Tuple[float, float]]:
    by_x = gt.get("pixel_curve_by_x")
    if isinstance(by_x, dict) and by_x:
        pts = []
        for k, vy in by_x.items():
            try:
                pts.append((float(k), float(vy)))
            except (TypeError, ValueError):
                continue
        return pts

    raw = gt.get("pixel_curve_path") or []
    out: List[Tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def _collapse_sorted_curve(pts: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not pts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    xs_raw = np.array([p[0] for p in pts], dtype=np.float64)
    ys_raw = np.array([p[1] for p in pts], dtype=np.float64)
    order = np.argsort(xs_raw)
    xs_raw, ys_raw = xs_raw[order], ys_raw[order]
    xs_u, inv = np.unique(xs_raw, return_inverse=True)
    ys_sum = np.bincount(inv, weights=ys_raw)
    counts = np.bincount(inv)
    ys_u = ys_sum / np.maximum(counts, 1)
    return xs_u, ys_u


def interpolate_y_abs(xs: np.ndarray, ys: np.ndarray, x_query: float) -> Optional[float]:
    if xs.size == 0:
        return None
    if xs.size == 1:
        return float(ys[0])
    yq = float(np.interp(x_query, xs, ys, left=float(ys[0]), right=float(ys[-1])))
    return yq


def build_gt_y_roi_per_column(
    gt: dict,
    plot_box_t: Tuple[int, int, int, int],
    roi_h: int,
    roi_w: int,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """ROI 열 인덱스 -> GT 곡선의 기대 y (ROI 좌표)."""
    x0c, y0c, _, _ = plot_box_t
    pts = _points_from_gt(gt)
    xs_u, ys_u = _collapse_sorted_curve(pts)

    meta = {
        "num_gt_points_raw": len(pts),
        "num_gt_points_unique_x": int(xs_u.size),
        "plot_origin_xy": [int(x0c), int(y0c)],
    }

    out: Dict[int, float] = {}
    if xs_u.size == 0:
        return out, meta

    for col in range(int(roi_w)):
        x_abs = float(x0c + col)
        y_abs = interpolate_y_abs(xs_u, ys_u, x_abs)
        if y_abs is None:
            continue
        y_roi = float(y_abs - float(y0c))
        y_roi = float(np.clip(y_roi, 0.0, float(max(0, roi_h - 1))))
        out[int(col)] = y_roi

    meta["columns_mapped"] = len(out)
    return out, meta


def apply_oracle_scores_to_candidates(
    final_cands: Dict[int, List[dict]],
    gt_y_roi_by_col: Dict[int, float],
    sigma_px: float,
    *,
    sharpening: float = 1.0,
) -> Dict[str, float]:
    """후보 confidence 를 oracle 점수로 교체. 열별 정렬 유지."""
    sigma = max(float(sigma_px), 1e-6)
    sharp = max(float(sharpening), 1e-6)
    dists: List[float] = []
    touched_cols = 0
    touched_cands = 0

    for col, cands in final_cands.items():
        col_i = int(col)
        gt_y = gt_y_roi_by_col.get(col_i)
        if gt_y is None:
            continue
        if not cands:
            continue
        touched_cols += 1
        for c in cands:
            if "rule_confidence_before_oracle" not in c:
                c["rule_confidence_before_oracle"] = float(c.get("confidence", 0.0))
            cy = float(c.get("y", 0.0))
            d = abs(cy - gt_y)
            dists.append(d)
            c["oracle_dist_px"] = float(d)
            base_score = float(np.exp(-((d / sigma) ** 2)))
            c["confidence"] = float(base_score ** sharp)
            touched_cands += 1
        cands.sort(key=lambda cc: -float(cc.get("confidence", 0.0)))

    summary = {
        "sigma_px": float(sigma_px),
        "sharpening": float(sharp),
        "columns_with_oracle_score": float(touched_cols),
        "num_candidates_rescored": float(touched_cands),
        "mean_oracle_dist_px": float(np.mean(dists)) if dists else None,
        "median_oracle_dist_px": float(np.median(dists)) if dists else None,
    }
    return summary


def _valid_ratio(path: List[Optional[int]]) -> float:
    if not path:
        return 0.0
    return float(sum(1 for p in path if p is not None)) / float(len(path))


def _apex(path: List[Optional[int]], fc: Dict[int, List[dict]], roi_w: int, enabled: bool):
    if not enabled:
        return list(path)
    max_pull = max(120, 4 * int(dp_transition_window_width(roi_w)))
    return refine_dp_path_column_apex_pull(
        path,
        fc,
        conf_slack=0.22,
        max_upward_pull_px=max_pull,
    )


def run_dp_with_gt_oracle_rerank(
    final_cands_orig: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    comp_score_map: np.ndarray,
    gt_json_path: str,
    plot_box_t: Tuple[int, int, int, int],
    sigma_px: float,
    *,
    use_dp_column_apex_pull: bool,
    confidence_weight_multiplier: float = 1.0,
    transition_penalty_multiplier: float = 1.0,
    curvature_penalty_multiplier: float = 1.0,
    oracle_confidence_sharpening: float = 1.0,
) -> Tuple[Dict[int, List[dict]], dict, dict]:
    """
    규칙 confidence DP vs oracle confidence DP 비교 후, oracle 경로를 채택한다.

    Returns:
      final_cands_oracle — 후속 recovery 등에 사용
      trace_result — oracle DP + apex pull 결과
      oracle_meta — 결과 JSON 기록용
    """
    gt = _load_gt_json(gt_json_path)
    gt_y_roi_by_col, gt_curve_meta = build_gt_y_roi_per_column(gt, plot_box_t, roi_h, roi_w)

    fc_rule = copy.deepcopy(final_cands_orig)
    tr_rule = dp_trace(
        fc_rule,
        roi_w,
        roi_h,
        comp_score_map,
        confidence_weight_multiplier=confidence_weight_multiplier,
        transition_penalty_multiplier=transition_penalty_multiplier,
        curvature_penalty_multiplier=curvature_penalty_multiplier,
    )
    path_rule = _apex(tr_rule["path"], fc_rule, roi_w, use_dp_column_apex_pull)
    tr_rule = {**tr_rule, "path": path_rule}

    fc_oracle = copy.deepcopy(final_cands_orig)
    oracle_score_summary = apply_oracle_scores_to_candidates(
        fc_oracle,
        gt_y_roi_by_col,
        sigma_px,
        sharpening=oracle_confidence_sharpening,
    )
    tr_oracle = dp_trace(
        fc_oracle,
        roi_w,
        roi_h,
        comp_score_map,
        confidence_weight_multiplier=confidence_weight_multiplier,
        transition_penalty_multiplier=transition_penalty_multiplier,
        curvature_penalty_multiplier=curvature_penalty_multiplier,
    )
    path_oracle = _apex(tr_oracle["path"], fc_oracle, roi_w, use_dp_column_apex_pull)
    tr_oracle = {**tr_oracle, "path": path_oracle}

    vr_r = _valid_ratio(path_rule)
    vr_o = _valid_ratio(path_oracle)
    ts_r = float(tr_rule["trace_score"])
    ts_o = float(tr_oracle["trace_score"])

    oracle_meta: Dict[str, Any] = {
        "enabled": True,
        "gt_json_path": str(Path(gt_json_path).resolve()),
        "sigma_px": float(sigma_px),
        "oracle_confidence_sharpening": float(oracle_confidence_sharpening),
        "gt_curve": gt_curve_meta,
        "oracle_score_summary": oracle_score_summary,
        "dp_cost_params": tr_oracle.get("cost_params", {}),
        "trace_score_rule_dp": ts_r,
        "trace_score_oracle_dp": ts_o,
        "valid_ratio_rule_dp": vr_r,
        "valid_ratio_oracle_dp": vr_o,
        "dp_branch_committed": "oracle_rerank",
        "oracle_improves_dp_cost": bool(ts_o < ts_r),
        "oracle_non_worse_valid_ratio": bool(vr_o >= vr_r),
    }

    return fc_oracle, tr_oracle, oracle_meta
