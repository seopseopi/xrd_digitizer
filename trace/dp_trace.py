"""
$13: DP tracing - column-wise optimal path through candidate map.

Cost: J = a*|dy| + b*|d2y| + g*(1-conf) + d*comp_switch
미세 피크(3단계): b↓ 로 급변(뾰족한 피크) 경로 비용 완화.
Fixed: a=1.0, b=0.20, g=0.88, d=1.2, K=3, W=max(28, round(0.035*Pw))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ALPHA = 1.0
BETA = 0.20
GAMMA = 0.88
DELTA = 1.2
COMP_SWITCH_THRESH = 0.58
EPSILON = 3.0       # border proximity penalty weight
BORDER_RADIUS = 12   # penalty decays linearly to zero at this distance
K_TOP = 3
BLOCK_SIZE = 50


def refine_dp_path_column_apex_pull(
    path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
    *,
    conf_slack: float = 0.22,
    min_absolute_conf: float = 0.30,
    min_lift_px: int = 6,
    max_upward_pull_px: int = 320,
) -> List[Optional[int]]:
    """
    DP 전역 최소비용이 열마다 베이스라인 후보를 고르는 경우 완화.

    각 열에서 `confidence`가 열 최고치에 가깝거나(`conf_slack`),
    또는 절대값이 `min_absolute_conf` 이상인 후보 중 **더 높은 피크**(작은 y)가 있으면
    해당 열 경로를 위로 당긴다. 한 번에 당기는 세로 폭은 `max_upward_pull_px` 로 제한.
    """
    out: List[Optional[int]] = list(path)
    for col, py in enumerate(out):
        if py is None:
            continue
        cands = final_cands.get(col)
        if not cands:
            continue
        py_i = int(py)
        best_conf = max(float(c["confidence"]) for c in cands)

        tier_a = [
            int(c["y"]) for c in cands
            if float(c["confidence"]) >= best_conf - conf_slack and int(c["y"]) < py_i
        ]
        apex: Optional[int] = None
        if tier_a:
            apex = min(tier_a)

        if apex is None:
            tier_b = [
                int(c["y"]) for c in cands
                if float(c["confidence"]) >= min_absolute_conf
                and int(c["y"]) < py_i - min_lift_px
            ]
            if tier_b:
                apex = min(tier_b)

        if apex is None:
            continue
        if py_i - apex > max_upward_pull_px:
            continue
        out[col] = apex
    return out


def _compute_window(pw: int) -> int:
    """열 간 세로 점프 상한. 좁은 피크·근접 돌출을 더 따라가도록 약간 확대."""
    return max(28, round(0.035 * pw))


def _border_penalty(y: int, roi_height: int) -> float:
    """Penalize candidates near the ROI border to avoid axis-line following."""
    dist_top = y
    dist_bottom = roi_height - 1 - y
    dist = min(dist_top, dist_bottom)
    if dist >= BORDER_RADIUS:
        return 0.0
    return EPSILON * (1.0 - dist / BORDER_RADIUS)


def _transition_cost_terms(
    y_curr: int,
    y_prev: int,
    y_prev2: Optional[int],
    conf_curr: float,
    comp_score_curr: float,
    comp_score_prev: float,
    roi_height: int = 0,
    *,
    confidence_weight_multiplier: float = 1.0,
    transition_penalty_multiplier: float = 1.0,
    curvature_penalty_multiplier: float = 1.0,
) -> Dict[str, float]:
    dy = abs(y_curr - y_prev)
    d2y = abs((y_curr - y_prev) - (y_prev - y_prev2)) if y_prev2 is not None else 0.0
    conf_penalty = 1.0 - conf_curr
    comp_switch = 1.0 if abs(comp_score_curr - comp_score_prev) > COMP_SWITCH_THRESH else 0.0
    border_pen = _border_penalty(y_curr, roi_height) if roi_height > 0 else 0.0

    transition_cost = ALPHA * dy * float(transition_penalty_multiplier)
    curvature_cost = BETA * d2y * float(curvature_penalty_multiplier)
    confidence_cost = GAMMA * conf_penalty * float(confidence_weight_multiplier)
    component_switch_cost = DELTA * comp_switch
    total = transition_cost + curvature_cost + confidence_cost + component_switch_cost + border_pen
    return {
        "dy_abs": float(dy),
        "d2y_abs": float(d2y),
        "confidence_penalty_raw": float(conf_penalty),
        "confidence_cost": float(confidence_cost),
        "transition_cost": float(transition_cost),
        "curvature_cost": float(curvature_cost),
        "component_switch_cost": float(component_switch_cost),
        "border_cost": float(border_pen),
        "total": float(total),
    }


def _transition_cost(
    y_curr: int,
    y_prev: int,
    y_prev2: Optional[int],
    conf_curr: float,
    comp_score_curr: float,
    comp_score_prev: float,
    roi_height: int = 0,
    *,
    confidence_weight_multiplier: float = 1.0,
    transition_penalty_multiplier: float = 1.0,
    curvature_penalty_multiplier: float = 1.0,
) -> float:
    return _transition_cost_terms(
        y_curr,
        y_prev,
        y_prev2,
        conf_curr,
        comp_score_curr,
        comp_score_prev,
        roi_height,
        confidence_weight_multiplier=confidence_weight_multiplier,
        transition_penalty_multiplier=transition_penalty_multiplier,
        curvature_penalty_multiplier=curvature_penalty_multiplier,
    )["total"]


def dp_trace(
    final_candidates: Dict[int, List[dict]],
    roi_width: int,
    roi_height: int,
    comp_score_map: Optional[np.ndarray] = None,
    *,
    confidence_weight_multiplier: float = 1.0,
    transition_penalty_multiplier: float = 1.0,
    curvature_penalty_multiplier: float = 1.0,
) -> dict:
    """
    DP optimal path tracing through columns.
    Returns dict with: path, trace_score, valid_ratio, diagnostics, blockwise stats.
    """
    W = _compute_window(roi_width)
    columns = sorted(final_candidates.keys())
    if not columns:
        return _empty_result(roi_width)

    nonempty_cols = [c for c in columns if final_candidates[c]]
    if not nonempty_cols:
        return _empty_result(roi_width)

    first_col = nonempty_cols[0]
    last_col = nonempty_cols[-1]

    dp: Dict[int, Dict[int, dict]] = {}

    for ci, col in enumerate(columns):
        cands = final_candidates.get(col, [])
        dp[col] = {}

        if not cands:
            continue

        for c in cands:
            y = c["y"]
            conf = c["confidence"]
            cs = c.get("comp_score", 0.0)

            if ci == 0 or not dp.get(columns[ci - 1]):
                cost = (
                    GAMMA * float(confidence_weight_multiplier) * (1.0 - conf)
                    + _border_penalty(y, roi_height)
                )
                dp[col][y] = {
                    "cost": cost,
                    "prev_y": None,
                    "prev_col": None,
                    "prev2_y": None,
                    "conf": conf,
                    "comp_score": cs,
                }
                continue

            prev_col = columns[ci - 1]
            best_cost = float("inf")
            best_entry = None

            for py, pdata in dp[prev_col].items():
                if abs(y - py) > W:
                    continue
                prev2_y = pdata.get("prev_y")
                tc = _transition_cost(
                    y,
                    py,
                    prev2_y,
                    conf,
                    cs,
                    pdata["comp_score"],
                    roi_height,
                    confidence_weight_multiplier=confidence_weight_multiplier,
                    transition_penalty_multiplier=transition_penalty_multiplier,
                    curvature_penalty_multiplier=curvature_penalty_multiplier,
                )
                total = pdata["cost"] + tc

                if total < best_cost:
                    best_cost = total
                    best_entry = {
                        "cost": total,
                        "prev_y": py,
                        "prev_col": prev_col,
                        "prev2_y": pdata.get("prev_y"),
                        "conf": conf,
                        "comp_score": cs,
                    }

            if best_entry is None:
                nearest_py = min(dp[prev_col], key=lambda k: dp[prev_col][k]["cost"])
                pdata_nearest = dp[prev_col][nearest_py]
                cost = (
                    pdata_nearest["cost"]
                    + GAMMA * float(confidence_weight_multiplier) * (1.0 - conf)
                    + ALPHA * float(transition_penalty_multiplier) * abs(y - nearest_py)
                )
                best_entry = {
                    "cost": cost,
                    "prev_y": nearest_py,
                    "prev_col": prev_col,
                    "prev2_y": pdata_nearest.get("prev_y"),
                    "conf": conf,
                    "comp_score": cs,
                }

            dp[col][y] = best_entry

    path = _backtrack(dp, columns)
    diagnostics = _build_diagnostics(dp, columns, final_candidates, roi_width)
    blockwise = _build_blockwise_stats(dp, columns, path)

    valid_cols = sum(1 for p in path if p is not None)
    valid_ratio = float(valid_cols) / max(1, len(columns))
    total_cost = 0.0
    if path:
        last_valid = [(i, p) for i, p in enumerate(path) if p is not None]
        if last_valid:
            li, lp = last_valid[-1]
            col_idx = min(li, len(columns) - 1)
            if col_idx < len(columns) and columns[col_idx] in dp and lp in dp[columns[col_idx]]:
                total_cost = dp[columns[col_idx]][lp]["cost"]

    return {
        "path": path,
        "trace_score": float(total_cost),
        "valid_ratio": float(valid_ratio),
        "diagnostics": diagnostics,
        "blockwise": blockwise,
        "window_W": W,
        "cost_params": {
            "alpha_dy": float(ALPHA),
            "beta_curvature": float(BETA),
            "gamma_confidence": float(GAMMA),
            "delta_component_switch": float(DELTA),
            "epsilon_border": float(EPSILON),
            "confidence_weight_multiplier": float(confidence_weight_multiplier),
            "transition_penalty_multiplier": float(transition_penalty_multiplier),
            "curvature_penalty_multiplier": float(curvature_penalty_multiplier),
        },
    }


def _summary(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": round(float(np.mean(a)), 6),
        "median": round(float(np.median(a)), 6),
        "p10": round(float(np.percentile(a, 10)), 6),
        "p90": round(float(np.percentile(a, 90)), 6),
        "min": round(float(np.min(a)), 6),
        "max": round(float(np.max(a)), 6),
    }


def _run_lengths(mask: List[bool]) -> List[int]:
    out: List[int] = []
    cur = 0
    for b in mask:
        if b:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _find_candidate(cands: List[dict], y: Optional[int]) -> Optional[dict]:
    if y is None:
        return None
    yi = int(y)
    for c in cands:
        if int(c.get("y", -999999)) == yi:
            return c
    return None


def _rank_candidate(cands: List[dict], y: Optional[int]) -> Optional[int]:
    if y is None:
        return None
    yi = int(y)
    for i, c in enumerate(cands, 1):
        if int(c.get("y", -999999)) == yi:
            return i
    return None


def _best_gt_near_candidate(
    cands: List[dict],
    gt_y: Optional[float],
    near_px: float,
) -> tuple[Optional[dict], Optional[int], Optional[float]]:
    if gt_y is None:
        return None, None, None
    near: List[tuple[int, dict, float]] = []
    for i, c in enumerate(cands, 1):
        d = abs(float(c.get("y", 0.0)) - float(gt_y))
        if d <= float(near_px):
            near.append((i, c, d))
    if not near:
        return None, None, None
    rank, cand, dist = max(near, key=lambda x: float(x[1].get("confidence", 0.0)))
    return cand, rank, dist


def _candidate_terms_from_prev(
    cand: Optional[dict],
    prev_y: Optional[int],
    prev2_y: Optional[int],
    prev_comp_score: float,
    roi_height: int,
    *,
    confidence_weight_multiplier: float,
    transition_penalty_multiplier: float,
    curvature_penalty_multiplier: float,
) -> Dict[str, Optional[float]]:
    if cand is None:
        return {}
    y = int(cand.get("y", 0))
    conf = float(cand.get("confidence", 0.0))
    cs = float(cand.get("comp_score", 0.0))
    if prev_y is None:
        return {
            "dy_abs": None,
            "d2y_abs": None,
            "confidence_penalty_raw": round(float(1.0 - conf), 6),
            "confidence_cost": round(float(GAMMA * confidence_weight_multiplier * (1.0 - conf)), 6),
            "transition_cost": 0.0,
            "curvature_cost": 0.0,
            "component_switch_cost": 0.0,
            "border_cost": round(float(_border_penalty(y, roi_height)), 6),
            "total": round(float(GAMMA * confidence_weight_multiplier * (1.0 - conf) + _border_penalty(y, roi_height)), 6),
        }
    terms = _transition_cost_terms(
        y,
        int(prev_y),
        prev2_y,
        conf,
        cs,
        float(prev_comp_score),
        roi_height,
        confidence_weight_multiplier=confidence_weight_multiplier,
        transition_penalty_multiplier=transition_penalty_multiplier,
        curvature_penalty_multiplier=curvature_penalty_multiplier,
    )
    return {k: round(float(v), 6) for k, v in terms.items()}


def build_dp_cost_breakdown(
    final_candidates: Dict[int, List[dict]],
    path: List[Optional[int]],
    roi_width: int,
    roi_height: int,
    *,
    gt_y_by_col: Optional[Dict[int, float]] = None,
    gt_near_px: float = 5.0,
    upper_band_frac: float = 0.2,
    confidence_weight_multiplier: float = 1.0,
    transition_penalty_multiplier: float = 1.0,
    curvature_penalty_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Diagnostic-only local cost breakdown for the committed DP path.

    This does not alter DP state. It recomputes per-column local costs for the
    selected candidate and the best GT-near candidate, so a debug report can
    explain whether confidence terms are being overwhelmed by smoothness terms.
    """
    upper_y = float(roi_height) * float(upper_band_frac)
    selected_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []

    selected_conf: List[float] = []
    selected_transition: List[float] = []
    selected_curvature: List[float] = []
    selected_total: List[float] = []
    gt_conf: List[float] = []
    gt_transition_from_selected: List[float] = []
    gt_curvature_from_selected: List[float] = []
    conf_gaps: List[float] = []

    selected_upper_mask: List[bool] = []
    gt_exists_mask: List[bool] = []

    prev_sel_y: Optional[int] = None
    prev2_sel_y: Optional[int] = None
    prev_sel_cs = 0.0

    gt_greedy_path: List[Optional[int]] = []
    gt_greedy_conf: List[float] = []
    prev_gt_y: Optional[int] = None
    prev2_gt_y: Optional[int] = None
    prev_gt_cs = 0.0
    gt_greedy_jumps: List[float] = []
    gt_greedy_curv: List[float] = []
    gt_greedy_cost_total: List[float] = []

    for col in range(int(roi_width)):
        cands = final_candidates.get(col) or final_candidates.get(str(col)) or []
        selected_y = path[col] if col < len(path) else None
        selected = _find_candidate(cands, selected_y)
        gt_y = gt_y_by_col.get(col) if gt_y_by_col else None
        gt_cand, gt_rank, gt_dist = _best_gt_near_candidate(cands, gt_y, gt_near_px)

        selected_rank = _rank_candidate(cands, selected_y)
        is_upper = bool(selected is not None and float(selected.get("y", 0.0)) < upper_y)
        is_gt_near = bool(
            selected is not None
            and gt_y is not None
            and abs(float(selected.get("y", 0.0)) - float(gt_y)) <= float(gt_near_px)
        )
        selected_upper_mask.append(is_upper)
        gt_exists_mask.append(gt_cand is not None)

        sel_terms = _candidate_terms_from_prev(
            selected,
            prev_sel_y,
            prev2_sel_y,
            prev_sel_cs,
            roi_height,
            confidence_weight_multiplier=confidence_weight_multiplier,
            transition_penalty_multiplier=transition_penalty_multiplier,
            curvature_penalty_multiplier=curvature_penalty_multiplier,
        )
        gt_terms_vs_selected = _candidate_terms_from_prev(
            gt_cand,
            prev_sel_y,
            prev2_sel_y,
            prev_sel_cs,
            roi_height,
            confidence_weight_multiplier=confidence_weight_multiplier,
            transition_penalty_multiplier=transition_penalty_multiplier,
            curvature_penalty_multiplier=curvature_penalty_multiplier,
        )

        if selected is not None:
            selected_conf.append(float(selected.get("confidence", 0.0)))
            selected_transition.append(float(sel_terms.get("transition_cost") or 0.0))
            selected_curvature.append(float(sel_terms.get("curvature_cost") or 0.0))
            selected_total.append(float(sel_terms.get("total") or 0.0))
        if gt_cand is not None:
            gt_conf.append(float(gt_cand.get("confidence", 0.0)))
            gt_transition_from_selected.append(float(gt_terms_vs_selected.get("transition_cost") or 0.0))
            gt_curvature_from_selected.append(float(gt_terms_vs_selected.get("curvature_cost") or 0.0))
        if selected is not None and gt_cand is not None:
            conf_gaps.append(float(gt_cand.get("confidence", 0.0)) - float(selected.get("confidence", 0.0)))

        selected_rows.append({
            "col": int(col),
            "y": None if selected is None else int(selected.get("y", 0)),
            "confidence": None if selected is None else round(float(selected.get("confidence", 0.0)), 8),
            "source": None if selected is None else str(selected.get("source", "")),
            "rank": selected_rank,
            "is_upper_band": is_upper,
            "is_gt_near_px5": is_gt_near,
            "cost": sel_terms,
        })
        gt_rows.append({
            "col": int(col),
            "gt_y": None if gt_y is None else round(float(gt_y), 4),
            "y": None if gt_cand is None else int(gt_cand.get("y", 0)),
            "confidence": None if gt_cand is None else round(float(gt_cand.get("confidence", 0.0)), 8),
            "source": None if gt_cand is None else str(gt_cand.get("source", "")),
            "rank": gt_rank,
            "dist_px": None if gt_dist is None else round(float(gt_dist), 4),
            "cost_from_previous_selected": gt_terms_vs_selected,
        })

        # Greedy GT-near feasibility path: best GT-near candidate per column, reset across gaps.
        if gt_cand is None:
            gt_greedy_path.append(None)
            prev_gt_y = None
            prev2_gt_y = None
            prev_gt_cs = 0.0
        else:
            gy = int(gt_cand.get("y", 0))
            gt_greedy_path.append(gy)
            gt_greedy_conf.append(float(gt_cand.get("confidence", 0.0)))
            gt_terms = _candidate_terms_from_prev(
                gt_cand,
                prev_gt_y,
                prev2_gt_y,
                prev_gt_cs,
                roi_height,
                confidence_weight_multiplier=confidence_weight_multiplier,
                transition_penalty_multiplier=transition_penalty_multiplier,
                curvature_penalty_multiplier=curvature_penalty_multiplier,
            )
            gt_greedy_cost_total.append(float(gt_terms.get("total") or 0.0))
            if prev_gt_y is not None:
                gt_greedy_jumps.append(abs(float(gy - prev_gt_y)))
            if prev_gt_y is not None and prev2_gt_y is not None:
                gt_greedy_curv.append(abs(float((gy - prev_gt_y) - (prev_gt_y - prev2_gt_y))))
            prev2_gt_y = prev_gt_y
            prev_gt_y = gy
            prev_gt_cs = float(gt_cand.get("comp_score", 0.0))

        if selected is not None:
            prev2_sel_y = prev_sel_y
            prev_sel_y = int(selected.get("y", 0))
            prev_sel_cs = float(selected.get("comp_score", 0.0))

    return {
        "enabled": True,
        "gt_near_px": float(gt_near_px),
        "upper_band_frac": float(upper_band_frac),
        "upper_band_y_threshold": round(float(upper_y), 4),
        "cost_params": {
            "alpha_dy": float(ALPHA),
            "beta_curvature": float(BETA),
            "gamma_confidence": float(GAMMA),
            "delta_component_switch": float(DELTA),
            "confidence_weight_multiplier": float(confidence_weight_multiplier),
            "transition_penalty_multiplier": float(transition_penalty_multiplier),
            "curvature_penalty_multiplier": float(curvature_penalty_multiplier),
        },
        "confidence_cost_summary": _summary([r["cost"].get("confidence_cost", 0.0) for r in selected_rows if r["cost"]]),
        "transition_cost_summary": _summary(selected_transition),
        "curvature_cost_summary": _summary(selected_curvature),
        "selected_path_cost_summary": _summary(selected_total),
        "selected_path_confidence_summary": _summary(selected_conf),
        "gt_near_confidence_summary": _summary(gt_conf),
        "gt_near_minus_selected_confidence_gap_summary": _summary(conf_gaps),
        "alternative_gt_near_from_selected_transition_cost_summary": _summary(gt_transition_from_selected),
        "alternative_gt_near_from_selected_curvature_cost_summary": _summary(gt_curvature_from_selected),
        "alternative_gt_near_path_cost_summary": _summary(gt_greedy_cost_total),
        "gt_near_path_feasibility": {
            "columns_with_gt_near": int(sum(gt_exists_mask)),
            "gt_near_run_lengths": _run_lengths(gt_exists_mask),
            "gt_near_gap_lengths": _run_lengths([not b for b in gt_exists_mask]),
            "jump_abs_summary": _summary(gt_greedy_jumps),
            "curvature_abs_summary": _summary(gt_greedy_curv),
            "confidence_summary": _summary(gt_greedy_conf),
        },
        "upper_band_selected_run_lengths": _run_lengths(selected_upper_mask),
        "per_column_selected": selected_rows,
        "per_column_best_gt_near": gt_rows,
    }


def _backtrack(dp: Dict[int, Dict[int, dict]], columns: List[int]) -> List[Optional[int]]:
    """최적 경로를 역추적."""
    if not columns:
        return []

    last_col = None
    for c in reversed(columns):
        if dp.get(c):
            last_col = c
            break
    if last_col is None:
        return [None] * len(columns)

    best_y = min(dp[last_col], key=lambda y: dp[last_col][y]["cost"])
    path_map: Dict[int, Optional[int]] = {last_col: best_y}

    current_y = best_y
    current_col = last_col
    for ci in range(len(columns) - 1, -1, -1):
        col = columns[ci]
        if col == current_col:
            continue
        if current_col in dp and current_y in dp[current_col]:
            entry = dp[current_col][current_y]
            prev_y = entry.get("prev_y")
            prev_col = entry.get("prev_col")
            if prev_col is not None and prev_y is not None:
                path_map[prev_col] = prev_y
                current_y = prev_y
                current_col = prev_col

    path = []
    for col in columns:
        path.append(path_map.get(col))
    return path


def _build_diagnostics(
    dp: Dict[int, Dict[int, dict]],
    columns: List[int],
    candidates: Dict[int, List[dict]],
    roi_width: int,
) -> dict:
    """column별 진단 태그: ok / starvation / path_choice."""
    col_status: Dict[int, str] = {}
    for col in range(roi_width):
        cands = candidates.get(col, [])
        if not cands:
            col_status[col] = "starvation"
        elif col in dp and dp[col]:
            col_status[col] = "ok"
        else:
            col_status[col] = "path_choice"

    n_ok = sum(1 for v in col_status.values() if v == "ok")
    n_starve = sum(1 for v in col_status.values() if v == "starvation")
    n_path = sum(1 for v in col_status.values() if v == "path_choice")

    return {
        "ok_columns": n_ok,
        "starvation_columns": n_starve,
        "path_choice_columns": n_path,
    }


def _build_blockwise_stats(
    dp: Dict[int, Dict[int, dict]],
    columns: List[int],
    path: List[Optional[int]],
) -> List[dict]:
    """$13.7: block 단위 top-1/top-2 cost, margin."""
    blocks = []
    for bi in range(0, len(columns), BLOCK_SIZE):
        block_cols = columns[bi:bi + BLOCK_SIZE]
        if not block_cols:
            continue

        last_block_col = block_cols[-1]
        entries = dp.get(last_block_col, {})
        if not entries:
            blocks.append({
                "block_start": block_cols[0],
                "block_end": last_block_col,
                "top1_cost": None,
                "top2_cost": None,
                "margin": None,
            })
            continue

        sorted_costs = sorted(entries.values(), key=lambda e: e["cost"])
        top1 = sorted_costs[0]["cost"] if len(sorted_costs) >= 1 else None
        top2 = sorted_costs[1]["cost"] if len(sorted_costs) >= 2 else None
        margin = (top2 - top1) if top1 is not None and top2 is not None else None

        blocks.append({
            "block_start": block_cols[0],
            "block_end": last_block_col,
            "top1_cost": round(float(top1), 4) if top1 is not None else None,
            "top2_cost": round(float(top2), 4) if top2 is not None else None,
            "margin": round(float(margin), 4) if margin is not None else None,
        })

    return blocks


def _empty_result(roi_width: int) -> dict:
    return {
        "path": [None] * roi_width,
        "trace_score": float("inf"),
        "valid_ratio": 0.0,
        "diagnostics": {"ok_columns": 0, "starvation_columns": roi_width, "path_choice_columns": 0},
        "blockwise": [],
        "window_W": _compute_window(roi_width),
    }


def render_trace_path(
    roi: np.ndarray,
    path: List[Optional[int]],
    confidences: Optional[List[Optional[float]]] = None,
) -> np.ndarray:
    """trace_path.png: confidence별 색상 코딩 (green=high, red=low)."""
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    h, w = overlay.shape[:2]

    for col, y in enumerate(path):
        if y is None or col >= w:
            continue
        y = int(y)
        if y < 0 or y >= h:
            continue

        conf = 0.5
        if confidences and col < len(confidences) and confidences[col] is not None:
            conf = float(confidences[col])

        r = int(np.clip((1.0 - conf) * 255, 0, 255))
        g = int(np.clip(conf * 255, 0, 255))
        b = 30

        if 0 <= y < h:
            overlay[y, col] = [r, g, b]

    return overlay
