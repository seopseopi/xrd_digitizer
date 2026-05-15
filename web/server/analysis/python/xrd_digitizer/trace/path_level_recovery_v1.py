"""
Experimental path-level recovery v1: augment localized DP pools with filtered non-bottom candidates.

Default OFF. Uses only runtime-visible signals (no GT / source_numeric / expected_y).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from trace.dp_trace import ALPHA, BETA, DELTA, GAMMA, COMP_SWITCH_THRESH
from trace.dp_trace import _border_penalty
from trace.path_level_recovery_v0 import (
    _local_segment_dp,
    _path_to_array,
    plan_recovery_regions,
)
from trace.path_monitor import (
    compute_selected_path_monitor,
    selected_source_from_final_candidates,
    top1_y_from_final_candidates,
)


def _deep_copy_candidates(fc: Dict[int, List[dict]]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for k, v in fc.items():
        out[int(k)] = [dict(c) for c in v]
    return out


def _transition_terms_split(
    y_curr: int,
    y_prev: Optional[int],
    y_prev2: Optional[int],
    conf: float,
    cs: float,
    cs_prev: float,
    roi_h: int,
) -> Dict[str, float]:
    """Decompose _transition_terms into labeled components (for debug only)."""
    if y_prev is None:
        return {
            "confidence_penalty": float(GAMMA * (1.0 - conf)),
            "border_penalty": float(_border_penalty(y_curr, roi_h)),
            "dy_penalty": 0.0,
            "d2y_penalty": 0.0,
            "comp_switch_penalty": 0.0,
        }
    dy = abs(float(y_curr - y_prev))
    d2y = (
        abs(float((y_curr - y_prev) - (y_prev - y_prev2)))
        if y_prev2 is not None
        else 0.0
    )
    comp_sw = 1.0 if abs(float(cs - cs_prev)) > COMP_SWITCH_THRESH else 0.0
    d2y_pen = float(BETA * d2y)
    return {
        "confidence_penalty": float(GAMMA * (1.0 - conf)),
        "border_penalty": float(_border_penalty(y_curr, roi_h)),
        "dy_penalty": float(ALPHA * dy),
        "d2y_penalty": d2y_pen,
        "comp_switch_penalty": float(DELTA * comp_sw),
        "slope_jump_penalty": d2y_pen,
    }


def _anchor_predict_y(
    col: int,
    left_cols: List[int],
    right_cols: List[int],
    path: List[Optional[int]],
) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for c in left_cols + right_cols:
        if 0 <= c < len(path) and path[c] is not None:
            xs.append(float(c))
            ys.append(float(path[c]))
    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        return float(np.polyval(coef, float(col)))
    if ys:
        return float(np.median(np.asarray(ys, dtype=np.float64)))
    return float("nan")


def _segment_dp_cost_breakdown(
    seg_path: List[Optional[int]],
    merged_cands: Dict[int, List[dict]],
    anchor_path: List[Optional[int]],
    *,
    roi_w: int,
    roi_h: int,
    L: int,
    R: int,
    anchor_window: int,
    local_margin: int,
    anchor_weight: float,
    bottom_band_frac: float,
    bottom_weight: float,
    margin_extra_weight: float,
) -> Dict[str, Any]:
    """Recompute localized DP cost components along the chosen segment path."""
    cols = list(range(L, R + 1))
    left_anchor_cols = [c for c in range(max(0, L - anchor_window), L)]
    right_anchor_cols = [c for c in range(R + 1, min(roi_w, R + 1 + anchor_window))]
    thr_b = bottom_band_frac * float(roi_h)

    def margin_pen(py: int, y: int) -> float:
        dy = abs(float(y - py))
        if dy <= float(local_margin):
            return 0.0
        return margin_extra_weight * (dy - float(local_margin)) ** 2 / max(float(roi_h), 1.0)

    per_col: List[Dict[str, Any]] = []
    total = 0.0
    for idx, col in enumerate(cols):
        yv = seg_path[col]
        if yv is None:
            continue
        y = int(yv)
        cands = merged_cands.get(col) or []
        cand = None
        for c in cands:
            if int(c.get("y", -10**9)) == y:
                cand = c
                break
        conf = float(cand.get("confidence", 0.0)) if cand else 0.0
        cs = float(cand.get("comp_score", 0.0)) if cand else 0.0
        pred = _anchor_predict_y(col, left_anchor_cols, right_anchor_cols, anchor_path)
        anchor_pen = 0.0
        trend_dev_pen = 0.0
        if np.isfinite(pred):
            trend_dev_pen = anchor_weight * abs(float(y) - pred) / max(float(roi_h), 1.0)
            anchor_pen = trend_dev_pen
        bottom_pen = 0.0
        bottom_persist_pen = 0.0
        if float(y) >= thr_b:
            bottom_persist_pen = bottom_weight * (float(y) - thr_b) / max(float(roi_h), 1.0)
            bottom_pen = bottom_persist_pen

        y_prev = seg_path[col - 1] if col > 0 else None
        y_prev2 = seg_path[col - 2] if col > 1 else None
        if idx == 0:
            y_prev_out = anchor_path[col - 1] if col > 0 else None
            y_prev2_out = anchor_path[col - 2] if col > 1 else None
            prev_cs = 0.0
            trans = _transition_terms_split(
                y, y_prev_out, y_prev2_out, conf, cs, prev_cs, roi_h
            )
            mp = 0.0
            if y_prev_out is not None:
                mp = margin_pen(int(y_prev_out), y)
            trans_sum = sum(
                v for k, v in trans.items() if k != "slope_jump_penalty"
            )
            step_total = trans_sum + anchor_pen + bottom_pen + mp
        else:
            prev_col = cols[idx - 1]
            py = seg_path[prev_col]
            py_int = int(py) if py is not None else None
            prev_cs = 0.0
            for pc in merged_cands.get(prev_col) or []:
                if int(pc.get("y", -10**9)) == py_int:
                    prev_cs = float(pc.get("comp_score", 0.0))
                    break
            py2 = None
            if prev_col >= 2:
                pv2 = anchor_path[prev_col - 2]
                py2 = int(pv2) if pv2 is not None else None
            trans = _transition_terms_split(y, py_int, py2, conf, cs, prev_cs, roi_h)
            mp = margin_pen(int(py_int), y) if py_int is not None else 0.0
            trans_sum = sum(
                v for k, v in trans.items() if k != "slope_jump_penalty"
            )
            step_total = trans_sum + anchor_pen + bottom_pen + mp

        total += step_total
        per_col.append(
            {
                "col": int(col),
                "y": int(y),
                **trans,
                "local_trend_deviation_penalty": float(trend_dev_pen),
                "bottom_persistence_penalty": float(bottom_persist_pen),
                "anchor_penalty": float(anchor_pen),
                "bottom_band_penalty": float(bottom_pen),
                "margin_penalty": float(mp),
                "step_total": float(step_total),
            }
        )

    return {"terminal_cost_recomputed": float(total), "columns": per_col}


def _filter_regions_prelockin_v1(
    regions: List[Tuple[int, int]],
    monitor: Dict[str, Any],
    y_arr: np.ndarray,
    *,
    roi_h: int,
    bottom_band_ratio: float,
    guard_enabled: bool,
    min_bottom_branch_score: float,
    min_bottom_frac_in_region: float,
) -> Tuple[List[Tuple[int, int]], List[List[int]], List[str]]:
    skipped: List[List[int]] = []
    warns: List[str] = []
    if not guard_enabled:
        return list(regions), [], warns

    bb = monitor.get("bottom_branch_persistence_proxy") or {}
    score = float(bb.get("bottom_branch_score") or 0.0)
    if score < float(min_bottom_branch_score):
        skipped = [[int(a), int(b)] for a, b in regions]
        warns.append(f"prelockin_guard_global_low_bottom_branch_score:{score:.6f}")
        return [], skipped, warns

    thr_y = float(bottom_band_ratio) * float(roi_h)
    kept: List[Tuple[int, int]] = []
    for L, R in regions:
        vals: List[float] = []
        for c in range(int(L), int(R) + 1):
            if 0 <= c < len(y_arr) and np.isfinite(y_arr[c]):
                vals.append(float(y_arr[c]))
        if not vals:
            kept.append((int(L), int(R)))
            continue
        frac_b = sum(1 for v in vals if v >= thr_y) / float(len(vals))
        if frac_b < float(min_bottom_frac_in_region):
            skipped.append([int(L), int(R)])
            warns.append(
                f"prelockin_guard_region_not_bottom_dominated:[{int(L)},{int(R)}] frac_bottom={frac_b:.4f}"
            )
        else:
            kept.append((int(L), int(R)))
    return kept, skipped, warns


def _augment_with_filtered_non_bottom(
    merged: Dict[int, List[dict]],
    filtered_cands: Dict[int, List[dict]],
    columns: Set[int],
    *,
    roi_h: int,
    bottom_threshold_ratio: float,
    max_extra_non_bottom_per_col: int,
    filtered_pool_topk: int,
) -> Tuple[int, int]:
    """Append filtered non-bottom candidates (tagged) up to per-column caps."""
    thr_y = float(bottom_threshold_ratio) * float(roi_h)
    added_cands = 0
    added_cols = 0
    for col in columns:
        clist = merged.setdefault(col, [])
        existing_y = {int(c.get("y", -10**9)) for c in clist}
        filt = filtered_cands.get(col) or []
        filt_sorted = sorted(
            filt, key=lambda c: float(c.get("confidence", 0.0)), reverse=True
        )[: int(filtered_pool_topk)]
        max_conf = max((float(c.get("confidence", 0.0)) for c in filt_sorted), default=0.0)
        candidates_extra: List[dict] = []
        for c in filt_sorted:
            y = int(c.get("y", -10**9))
            if y >= thr_y:
                continue
            conf = float(c.get("confidence", 0.0))
            if conf < max(0.06, 0.20 * max_conf):
                continue
            candidates_extra.append(c)
        candidates_extra.sort(
            key=lambda c: float(c.get("confidence", 0.0)), reverse=True
        )
        taken = 0
        for c in candidates_extra:
            if taken >= int(max_extra_non_bottom_per_col):
                break
            y = int(c.get("y", -10**9))
            if y in existing_y:
                continue
            nc = dict(c)
            nc["source"] = "path_v1_filtered_non_bottom"
            nc["reason"] = "monitor_risk_region_non_bottom_pool"
            nc["path_v1_augmented"] = True
            clist.append(nc)
            existing_y.add(y)
            added_cands += 1
            taken += 1
        if taken > 0:
            added_cols += 1
    return added_cands, added_cols


def _count_selected_augmented_columns(
    trace_path: List[Optional[int]],
    merged: Dict[int, List[dict]],
    roi_w: int,
) -> int:
    n = 0
    for col in range(min(len(trace_path), int(roi_w))):
        yv = trace_path[col]
        if yv is None:
            continue
        y = int(yv)
        for c in merged.get(col) or []:
            if int(c.get("y", -10**9)) == y and c.get("path_v1_augmented"):
                n += 1
                break
    return int(n)


def apply_path_level_recovery_v1(
    trace_path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
    filtered_cands: Dict[int, List[dict]],
    *,
    roi_w: int,
    roi_h: int,
    monitor_window: int,
    monitor_min_run_for_detection: int,
    min_bottom_run_len: int,
    max_region_len: int,
    prelockin_guard_col: int,
    anchor_window: int,
    local_dp_margin: int,
    non_bottom_threshold_ratio: float = 0.84,
    max_extra_non_bottom_per_col: int = 4,
    filtered_pool_topk: int = 16,
    prelockin_guard: bool = True,
    prelockin_min_bottom_branch_score: float = 0.42,
    prelockin_min_bottom_frac_in_region: float = 0.50,
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """Apply optional localized recovery with augmented candidate pools."""
    y_arr = _path_to_array(trace_path, roi_w)
    top1 = top1_y_from_final_candidates(final_cands, roi_w)
    selected_src = selected_source_from_final_candidates(final_cands, y_arr)
    mon = compute_selected_path_monitor(
        y_arr,
        top1_y=top1,
        selected_source=selected_src,
        roi_h=int(roi_h),
        window=int(monitor_window),
        min_run=int(monitor_min_run_for_detection),
    )

    bb = mon.get("bottom_branch_persistence_proxy") or {}
    risk_run_ranges = [
        [int(a), int(b)] for a, b in (bb.get("bottom_branch_run_ranges") or [])
    ]

    regions = plan_recovery_regions(
        mon,
        roi_w=int(roi_w),
        min_bottom_run_len=int(min_bottom_run_len),
        max_region_len=int(max_region_len),
        prelockin_guard_col=int(prelockin_guard_col),
    )

    regions_f, skipped_guard, guard_warnings = _filter_regions_prelockin_v1(
        regions,
        mon,
        y_arr,
        roi_h=int(roi_h),
        bottom_band_ratio=float(non_bottom_threshold_ratio),
        guard_enabled=bool(prelockin_guard),
        min_bottom_branch_score=float(prelockin_min_bottom_branch_score),
        min_bottom_frac_in_region=float(prelockin_min_bottom_frac_in_region),
    )

    warnings: List[str] = list(guard_warnings)
    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "uses_source_numeric": False,
        "uses_gt": False,
        "risk_run_ranges": risk_run_ranges,
        "recovered_ranges": [],
        "added_non_bottom_candidates": 0,
        "added_non_bottom_columns": 0,
        "selected_added_non_bottom_columns": 0,
        "changed_columns": 0,
        "max_abs_delta_y": 0.0,
        "prelockin_guard_skipped_ranges": skipped_guard,
        "warnings": warnings,
        "monitor_snapshot": {
            "bottom_branch_score": bb.get("bottom_branch_score"),
            "longest_bottom_branch_run_len": bb.get("longest_bottom_branch_run_len"),
        },
        "dp_cost_breakdown": [],
        "lookup_candidates_for_trace": None,
        "original_selected_y_px": None,
        "recovered_selected_y_px": None,
    }

    if not regions_f:
        meta["warnings"].append("no_eligible_regions_after_prelockin_or_plan")
        if not regions:
            meta["warnings"].append("no_eligible_bottom_run_after_guards")
        return trace_path, meta

    cols_augment: Set[int] = set()
    for L, R in regions_f:
        for c in range(int(L), int(R) + 1):
            cols_augment.add(int(c))

    merged = _deep_copy_candidates(final_cands)
    added_n, added_cols = _augment_with_filtered_non_bottom(
        merged,
        filtered_cands,
        cols_augment,
        roi_h=int(roi_h),
        bottom_threshold_ratio=float(non_bottom_threshold_ratio),
        max_extra_non_bottom_per_col=int(max_extra_non_bottom_per_col),
        filtered_pool_topk=int(filtered_pool_topk),
    )
    meta["added_non_bottom_candidates"] = int(added_n)
    meta["added_non_bottom_columns"] = int(added_cols)

    cur = list(trace_path)
    before = list(trace_path)
    max_delta = 0.0
    n_changed = 0
    recovered: List[List[int]] = []
    dp_regions_out: List[Dict[str, Any]] = []

    bottom_frac = float(non_bottom_threshold_ratio)

    for L, R in regions_f:
        seg, dp_info = _local_segment_dp(
            cur,
            merged,
            int(roi_w),
            int(roi_h),
            int(L),
            int(R),
            anchor_window=int(anchor_window),
            local_margin=int(local_dp_margin),
            anchor_weight=2.6,
            bottom_band_frac=bottom_frac,
            bottom_weight=2.0,
            margin_extra_weight=0.06,
        )
        if seg is None:
            warnings.append(f"skip[{L},{R}]:{dp_info.get('skip_reason')}")
            continue
        bd = _segment_dp_cost_breakdown(
            seg,
            merged,
            cur,
            roi_w=int(roi_w),
            roi_h=int(roi_h),
            L=int(L),
            R=int(R),
            anchor_window=int(anchor_window),
            local_margin=int(local_dp_margin),
            anchor_weight=2.6,
            bottom_band_frac=bottom_frac,
            bottom_weight=2.0,
            margin_extra_weight=0.06,
        )
        bd["region"] = [int(L), int(R)]
        bd["segment_dp_info"] = dict(dp_info)
        dp_regions_out.append(bd)
        for col in range(int(L), int(R) + 1):
            ov = cur[col]
            nv = seg[col]
            if ov is not None and nv is not None and ov != nv:
                d = abs(float(int(nv) - int(ov)))
                max_delta = max(max_delta, d)
                n_changed += 1
            elif (ov is None) != (nv is None):
                n_changed += 1
        cur = seg
        recovered.append([int(L), int(R)])

    meta["dp_cost_breakdown"] = dp_regions_out
    meta["recovered_ranges"] = recovered
    meta["applied"] = bool(recovered)
    meta["max_abs_delta_y"] = float(max_delta)
    meta["changed_columns"] = int(n_changed)
    meta["lookup_candidates_for_trace"] = merged
    meta["selected_added_non_bottom_columns"] = int(
        _count_selected_augmented_columns(cur, merged, int(roi_w))
    )

    meta["original_selected_y_px"] = [
        float(before[i]) if i < len(before) and before[i] is not None else None
        for i in range(int(roi_w))
    ]
    meta["recovered_selected_y_px"] = [
        float(cur[i]) if i < len(cur) and cur[i] is not None else None
        for i in range(int(roi_w))
    ]
    meta["warnings"] = warnings
    if not meta["applied"]:
        meta["warnings"].append("no_segment_survived_dp")

    return cur, meta
