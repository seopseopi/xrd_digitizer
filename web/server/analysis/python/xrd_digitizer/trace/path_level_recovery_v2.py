"""
Experimental path-level recovery v2: limited two-state (bottom / escaped) branch escape DP.

Default OFF. Uses only runtime-visible signals (no GT / source_numeric / expected_y).
Does not store full candidate pools in meta (summary-only debug).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from trace.dp_trace import ALPHA, BETA, DELTA, GAMMA, COMP_SWITCH_THRESH
from trace.dp_trace import _border_penalty
from trace.path_level_recovery_v0 import _path_to_array, _transition_terms, plan_recovery_regions
from trace.path_level_recovery_v1 import _deep_copy_candidates, _filter_regions_prelockin_v1
from trace.path_level_recovery_v1 import _anchor_predict_y
from trace.path_monitor import (
    compute_selected_path_monitor,
    selected_source_from_final_candidates,
    top1_y_from_final_candidates,
)

INF = 1e30


def _augment_island_columns_only(
    merged: Dict[int, List[dict]],
    filtered_cands: Dict[int, List[dict]],
    island_cols: Set[int],
    *,
    roi_h: int,
    bottom_threshold_ratio: float,
    max_extra_non_bottom_per_col: int,
    filtered_pool_topk: int,
) -> Tuple[int, int]:
    """Append filtered non-bottom candidates only on validated island columns."""
    thr_y = float(bottom_threshold_ratio) * float(roi_h)
    added_cands = 0
    added_cols = 0
    for col in island_cols:
        clist = merged.setdefault(int(col), [])
        existing_y = {int(c.get("y", -10**9)) for c in clist}
        filt = filtered_cands.get(int(col)) or []
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
            nc["source"] = "path_v2_filtered_non_bottom"
            nc["reason"] = "island_escape_pool"
            nc["path_v2_island"] = True
            nc["path_v2_augmented"] = True
            clist.append(nc)
            existing_y.add(y)
            added_cands += 1
            taken += 1
        if taken > 0:
            added_cols += 1
    return added_cands, added_cols


def _detect_non_bottom_islands(
    filtered_cands: Dict[int, List[dict]],
    *,
    L: int,
    R: int,
    roi_h: int,
    thr_ratio: float,
    min_island_len: int,
    y_selected: np.ndarray,
    max_internal_y_jump: float = 32.0,
    min_y_separation_px: float = 48.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Find contiguous non-bottom islands inside [L,R] from filtered candidates."""
    thr_y = float(thr_ratio) * float(roi_h)
    cols = list(range(int(L), int(R) + 1))
    col_nb_y: List[Optional[float]] = []
    col_nb_conf: List[float] = []
    col_has_nb: List[bool] = []

    for col in cols:
        filt = filtered_cands.get(col) or []
        nb_cands = [c for c in filt if int(c.get("y", 10**9)) < thr_y]
        if not nb_cands:
            col_has_nb.append(False)
            col_nb_y.append(None)
            col_nb_conf.append(0.0)
            continue
        nb_cands.sort(key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
        top = nb_cands[: min(5, len(nb_cands))]
        ys = [float(int(c.get("y", 0))) for c in top]
        median_y = float(np.median(np.asarray(ys, dtype=np.float64)))
        max_cf = float(nb_cands[0].get("confidence", 0.0))
        col_has_nb.append(True)
        col_nb_y.append(median_y)
        col_nb_conf.append(max_cf)

    islands_raw: List[Tuple[int, int, List[int]]] = []
    i = 0
    n = len(cols)
    while i < n:
        if not col_has_nb[i]:
            i += 1
            continue
        j = i
        while j < n and col_has_nb[j]:
            j += 1
        run_len = j - i
        if run_len >= int(min_island_len):
            idxs = list(range(i, j))
            islands_raw.append((cols[i], cols[j - 1], idxs))
        i = j

    out: List[Dict[str, Any]] = []
    warns: List[str] = []

    for ia, ib, idxs in islands_raw:
        ys_run = [col_nb_y[k] for k in idxs if col_nb_y[k] is not None]
        if len(ys_run) < int(min_island_len):
            continue
        ok_jump = True
        for a in range(len(ys_run) - 1):
            if abs(ys_run[a + 1] - ys_run[a]) > float(max_internal_y_jump):
                ok_jump = False
                break
        if not ok_jump:
            warns.append(f"island[{ia},{ib}]:unstable_y_trajectory")
            continue

        sel_slice = y_selected[np.asarray(cols)[idxs]]
        sel_med = float(np.nanmedian(sel_slice[np.isfinite(sel_slice)]))
        isl_med = float(np.median(np.asarray(ys_run, dtype=np.float64)))
        sep = float(sel_med - isl_med)
        if sep < float(min_y_separation_px):
            warns.append(f"island[{ia},{ib}]:low_y_separation sep={sep:.1f}")
            continue

        confs = [col_nb_conf[k] for k in idxs]
        out.append(
            {
                "range": [int(ia), int(ib)],
                "island_len": int(ib - ia + 1),
                "island_y_median": round(isl_med, 4),
                "selected_y_median_overlap": round(sel_med, 4),
                "y_separation_px": round(sep, 4),
                "island_confidence_max": round(float(max(confs)), 6),
                "island_confidence_median": round(float(np.median(confs)), 6),
                "columns": [int(cols[k]) for k in idxs],
            }
        )

    return out, warns


def _instability_signal(mon: Dict[str, Any]) -> bool:
    dev = mon.get("local_trend_deviation_run") or {}
    sj = mon.get("slope_jump_cluster_score") or {}
    top1 = mon.get("top1_selected_divergence_rate") or {}
    if int(dev.get("longest_trend_deviation_run_len") or 0) >= 12:
        return True
    sj_sc = float(sj.get("slope_jump_cluster_score") or 0.0)
    if sj_sc > 0.002:
        return True
    dr = top1.get("divergence_rate_50px")
    if dr is not None and float(dr) > 0.05:
        return True
    return False


def _short_tail_like_risk_runs(
    risk_ranges: List[List[int]],
    *,
    prelockin_guard_col: int,
    min_span_px: int = 220,
) -> bool:
    """True if all bottom-branch risk runs look like narrow pre-lock-in tails only."""
    if not risk_ranges:
        return True
    spans = [int(b) - int(a) + 1 for a, b in risk_ranges]
    max_end = max(int(b) for a, b in risk_ranges)
    if max_end < int(prelockin_guard_col) + 120 and max(spans) < min_span_px:
        return True
    return False


def _two_state_segment_dp(
    path: List[Optional[int]],
    merged: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    L: int,
    R: int,
    island_cols: Set[int],
    thr_b_classify: float,
    *,
    anchor_window: int,
    local_margin: int,
    anchor_weight: float,
    bottom_band_frac: float,
    bottom_weight: float,
    margin_extra_weight: float,
    escape_entry_penalty: float,
    escape_max_jump_px: int,
    s1_to_s0_penalty_scale: float,
) -> Tuple[Optional[List[Optional[int]]], Dict[str, Any]]:
    """Two-state Viterbi: state0 bottom-follow, state1 escaped non-bottom island."""
    if L < 0 or R >= roi_w or L > R:
        return None, {"skip_reason": "bad_bounds"}

    cols = list(range(L, R + 1))
    left_anchor_cols = [c for c in range(max(0, L - anchor_window), L)]
    right_anchor_cols = [c for c in range(R + 1, min(roi_w, R + 1 + anchor_window))]

    def margin_pen(py: int, y: int) -> float:
        dy = abs(float(y - py))
        if dy <= float(local_margin):
            return 0.0
        return margin_extra_weight * (dy - float(local_margin)) ** 2 / max(float(roi_h), 1.0)

    s1_exit_pen = float(s1_to_s0_penalty_scale) * float(roi_h) / max(float(roi_h), 1.0)

    # dp0[col][y], dp1[col][y]
    dp0: Dict[int, Dict[int, float]] = {}
    dp1: Dict[int, Dict[int, float]] = {}
    b0: Dict[int, Dict[int, Tuple[Optional[int], int]]] = {}
    b1: Dict[int, Dict[int, Tuple[Optional[int], int]]] = {}
    # back stores (prev_y, prev_state) prev_state 0 or 1

    col0 = cols[0]
    c0 = merged.get(col0) or []
    if not c0:
        return None, {"skip_reason": "no_candidates_first_col"}

    y_prev_out = path[col0 - 1] if col0 > 0 else None
    y_prev2_out = path[col0 - 2] if col0 > 1 else None

    dp0[col0] = {}
    dp1[col0] = {}
    b0[col0] = {}
    b1[col0] = {}

    pred0 = _anchor_predict_y(col0, left_anchor_cols, right_anchor_cols, path)
    thr_b = float(bottom_band_frac) * float(roi_h)

    for cand in c0:
        y = int(cand["y"])
        conf = float(cand.get("confidence", 0.0))
        cs = float(cand.get("comp_score", 0.0))
        is_nb = y < thr_b_classify
        is_island = bool(cand.get("path_v2_island")) and int(col0) in island_cols

        anchor_pen = 0.0
        if np.isfinite(pred0):
            anchor_pen = anchor_weight * abs(float(y) - pred0) / max(float(roi_h), 1.0)
        bottom_pen = 0.0
        if float(y) >= thr_b:
            bottom_pen = bottom_weight * (float(y) - thr_b) / max(float(roi_h), 1.0)

        base0 = (
            _transition_terms(y, y_prev_out, y_prev2_out, conf, cs, 0.0, roi_h)
            + anchor_pen
            + bottom_pen
        )
        if y_prev_out is not None:
            base0 += margin_pen(int(y_prev_out), y)

        # state 0: bottom-like candidates only (or stay)
        if not is_nb:
            dp0[col0][y] = base0
            b0[col0][y] = (None, -1)

        # escape entry 0->1 at first col
        if is_nb and is_island:
            py = int(y_prev_out) if y_prev_out is not None else y
            dy_esc = abs(float(y - py))
            if y_prev_out is not None and dy_esc > float(escape_max_jump_px):
                esc_cost = base0 + float(escape_entry_penalty) + ALPHA * (
                    dy_esc - float(escape_max_jump_px)
                )
            else:
                esc_cost = base0 + float(escape_entry_penalty)
            if y not in dp1[col0] or esc_cost < dp1[col0][y]:
                dp1[col0][y] = esc_cost
                b1[col0][y] = (None, -1)

    if not dp0[col0] and not dp1[col0]:
        return None, {"skip_reason": "dp_empty_first_col"}

    for idx in range(1, len(cols)):
        col = cols[idx]
        prev_col = cols[idx - 1]
        cands = merged.get(col) or []
        if not cands:
            return None, {"skip_reason": f"no_candidates_col_{col}"}

        dp0[col] = {}
        dp1[col] = {}
        b0[col] = {}
        b1[col] = {}

        pred_col = _anchor_predict_y(col, left_anchor_cols, right_anchor_cols, path)

        for cand in cands:
            y = int(cand["y"])
            conf = float(cand.get("confidence", 0.0))
            cs = float(cand.get("comp_score", 0.0))
            is_nb = y < thr_b_classify
            is_island = bool(cand.get("path_v2_island")) and int(col) in island_cols

            anchor_pen = 0.0
            if np.isfinite(pred_col):
                anchor_pen = anchor_weight * abs(float(y) - pred_col) / max(float(roi_h), 1.0)
            bottom_pen = 0.0
            if float(y) >= thr_b:
                bottom_pen = bottom_weight * (float(y) - thr_b) / max(float(roi_h), 1.0)

            best0 = INF
            best01: Tuple[Optional[int], int] = (None, -1)
            best1 = INF
            best11: Tuple[Optional[int], int] = (None, -1)

            # --- incoming state 0 (bottom) ---
            for py in list(dp0.get(prev_col, {}).keys()):
                prev_cs = 0.0
                for pc in merged.get(prev_col) or []:
                    if int(pc.get("y", -99999)) == int(py):
                        prev_cs = float(pc.get("comp_score", 0.0))
                        break
                py2 = None
                if prev_col >= 2:
                    pv2 = path[prev_col - 2]
                    py2 = int(pv2) if pv2 is not None else None
                terms = _transition_terms(y, int(py), py2, conf, cs, prev_cs, roi_h)
                terms += margin_pen(int(py), y)
                step = dp0[prev_col][py] + terms + anchor_pen + bottom_pen

                if not is_nb:
                    if step < best0:
                        best0 = step
                        best01 = (py, 0)

                if is_nb and is_island:
                    dy_esc = abs(float(y - py))
                    extra = float(escape_entry_penalty)
                    if dy_esc > float(escape_max_jump_px):
                        extra += ALPHA * (dy_esc - float(escape_max_jump_px))
                    esc_tot = dp0[prev_col][py] + terms + anchor_pen + bottom_pen + extra
                    if esc_tot < best1:
                        best1 = esc_tot
                        best11 = (py, 0)

            # --- incoming state 1 (escaped) ---
            for py in list(dp1.get(prev_col, {}).keys()):
                prev_cs = 0.0
                for pc in merged.get(prev_col) or []:
                    if int(pc.get("y", -99999)) == int(py):
                        prev_cs = float(pc.get("comp_score", 0.0))
                        break
                py2 = None
                if prev_col >= 2:
                    pv2 = path[prev_col - 2]
                    py2 = int(pv2) if pv2 is not None else None
                terms = _transition_terms(y, int(py), py2, conf, cs, prev_cs, roi_h)
                terms += margin_pen(int(py), y)
                step = dp1[prev_col][py] + terms + anchor_pen + bottom_pen

                if is_nb:
                    if step < best1:
                        best1 = step
                        best11 = (py, 1)
                else:
                    step2 = step + s1_exit_pen
                    if step2 < best0:
                        best0 = step2
                        best01 = (py, 1)

            if not is_nb and best0 < INF:
                dp0[col][y] = best0
                b0[col][y] = best01
            elif is_nb:
                if best1 < INF:
                    dp1[col][y] = best1
                    b1[col][y] = best11

        if not dp0[col] and not dp1[col]:
            return None, {"skip_reason": f"dp_dead_col_{col}"}

    last = cols[-1]
    best_cost = INF
    best_y_last: Optional[int] = None
    best_st_last: Optional[int] = None
    for y, cst in dp0.get(last, {}).items():
        if cst < best_cost:
            best_cost = cst
            best_y_last, best_st_last = int(y), 0
    for y, cst in dp1.get(last, {}).items():
        if cst < best_cost:
            best_cost = cst
            best_y_last, best_st_last = int(y), 1

    if best_y_last is None or best_st_last is None or best_cost >= INF / 2:
        return None, {"skip_reason": "terminal_fail"}

    seg_path: Dict[int, int] = {}
    states_fwd: Dict[int, int] = {}
    cy = int(best_y_last)
    st_walk = int(best_st_last)
    for idx in range(len(cols) - 1, -1, -1):
        col = cols[idx]
        seg_path[col] = int(cy)
        states_fwd[col] = int(st_walk)
        if idx == 0:
            break
        if st_walk == 0:
            prev_y, prev_st = b0[col][cy]
        else:
            prev_y, prev_st = b1[col][cy]
        if prev_y is None:
            break
        cy = int(prev_y)
        st_walk = int(prev_st)
        if st_walk < 0:
            st_walk = 0

    escape_cols: List[int] = []
    for idx in range(1, len(cols)):
        c0 = cols[idx - 1]
        c1 = cols[idx]
        if states_fwd.get(c1, 0) == 1 and states_fwd.get(c0, 0) == 0:
            escape_cols.append(int(c1))

    out_path: List[Optional[int]] = list(path)
    for col in cols:
        out_path[col] = seg_path[col]

    meta_dp = {
        "terminal_cost": float(best_cost),
        "escape_transition_columns": escape_cols,
        "state_path_counts": {
            "bottom": int(sum(1 for c in cols if states_fwd.get(c, 0) == 0)),
            "escaped": int(sum(1 for c in cols if states_fwd.get(c, 0) == 1)),
        },
    }
    return out_path, meta_dp


def apply_path_level_recovery_v2(
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
    escape_entry_penalty: float = 0.35,
    escape_max_jump_px: int = 900,
    require_instability_signal: bool = True,
    min_y_separation_px: float = 48.0,
    min_island_len: int = 24,
    max_internal_y_jump: float = 34.0,
    island_confidence_floor: float = 0.08,
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """Apply optional two-state escape recovery (experimental)."""
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
    longest_bb = int(bb.get("longest_bottom_branch_run_len") or 0)

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

    thr_class = float(non_bottom_threshold_ratio) * float(roi_h)

    skipped_ranges: List[Dict[str, Any]] = []
    for pair in skipped_guard:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            skipped_ranges.append(
                {
                    "range": [int(pair[0]), int(pair[1])],
                    "reason": "skipped_prelockin_guard",
                }
            )

    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "max_internal_y_jump": float(max_internal_y_jump),
        "island_confidence_floor": float(island_confidence_floor),
        "uses_source_numeric": False,
        "uses_gt": False,
        "risk_run_ranges": risk_run_ranges,
        "triggered_ranges": [],
        "skipped_ranges": list(skipped_ranges),
        "non_bottom_island_ranges": [],
        "escape_transition_columns": [],
        "added_non_bottom_candidates": 0,
        "added_non_bottom_columns": 0,
        "selected_added_non_bottom_columns": 0,
        "changed_columns": 0,
        "max_abs_delta_y": 0.0,
        "state_path_counts": {"bottom": 0, "escaped": 0},
        "debug_size_guard": "summary_only_no_full_candidate_pool",
        "warnings": list(guard_warnings),
        "debug_notes": [],
    }

    if longest_bb < int(min_bottom_run_len):
        skipped_ranges.append(
            {"range": None, "reason": "skipped_no_long_bottom_run", "detail": longest_bb}
        )
        meta["skipped_ranges"] = skipped_ranges
        meta["warnings"].append("v2_skip_global_short_bottom_run")
        return trace_path, meta

    if bool(require_instability_signal) and not _instability_signal(mon):
        skipped_ranges.append({"range": None, "reason": "skipped_no_instability_signal"})
        meta["skipped_ranges"] = skipped_ranges
        meta["warnings"].append("v2_skip_no_instability")
        return trace_path, meta

    if _short_tail_like_risk_runs(risk_run_ranges, prelockin_guard_col=int(prelockin_guard_col)):
        skipped_ranges.append({"range": None, "reason": "skipped_short_tail_like_run"})
        meta["skipped_ranges"] = skipped_ranges
        meta["warnings"].append("v2_skip_short_tail_like")
        return trace_path, meta

    if not regions_f:
        meta["skipped_ranges"] = skipped_ranges
        meta["warnings"].append("no_eligible_regions_after_prelockin_or_plan")
        return trace_path, meta

    bottom_thr_frac = float(non_bottom_threshold_ratio)
    thr_y_band = bottom_thr_frac * float(roi_h)

    Island_cols_union: Set[int] = set()
    island_summaries: List[Dict[str, Any]] = []
    island_warns: List[str] = []

    for L, R in regions_f:
        cols_region = set(range(int(L), int(R) + 1))
        frac_bottom = 0.0
        vals = []
        for c in range(int(L), int(R) + 1):
            if 0 <= c < len(y_arr) and np.isfinite(y_arr[c]):
                vals.append(float(y_arr[c]))
        if vals:
            frac_bottom = sum(1 for v in vals if v >= thr_y_band) / float(len(vals))
        if frac_bottom < 0.52:
            skipped_ranges.append(
                {
                    "range": [int(L), int(R)],
                    "reason": "skipped_no_long_bottom_run",
                    "detail": "low_bottom_frac_selected",
                }
            )
            continue

        isl, iw = _detect_non_bottom_islands(
            filtered_cands,
            L=int(L),
            R=int(R),
            roi_h=int(roi_h),
            thr_ratio=bottom_thr_frac,
            min_island_len=int(min_island_len),
            y_selected=y_arr,
            max_internal_y_jump=float(max_internal_y_jump),
            min_y_separation_px=float(min_y_separation_px),
        )
        island_warns.extend(iw)
        if not isl:
            skipped_ranges.append(
                {"range": [int(L), int(R)], "reason": "skipped_no_non_bottom_island"}
            )
            continue

        # instability + island stability: reject if best island has very low confidence
        best_isl = max(isl, key=lambda d: int(d.get("island_len") or 0))
        if float(best_isl.get("island_confidence_max") or 0.0) < float(island_confidence_floor):
            skipped_ranges.append(
                {
                    "range": [int(L), int(R)],
                    "reason": "skipped_candidate_island_too_unstable",
                }
            )
            continue

        for isl_rec in isl:
            Island_cols_union.update(int(c) for c in isl_rec.get("columns") or [])
            island_summaries.append(dict(isl_rec))

    meta["non_bottom_island_ranges"] = island_summaries

    if not Island_cols_union:
        meta["skipped_ranges"] = skipped_ranges
        meta["warnings"].extend(island_warns)
        return trace_path, meta

    merged = _deep_copy_candidates(final_cands)
    added_n, added_cols = _augment_island_columns_only(
        merged,
        filtered_cands,
        Island_cols_union,
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
    all_escape_cols: List[int] = []
    total_bottom_st = 0
    total_esc_st = 0

    bottom_frac = float(non_bottom_threshold_ratio)

    for L, R in regions_f:
        # skip regions we marked skipped (no island) — re-check overlap with island cols
        cols_seg = [c for c in range(int(L), int(R) + 1) if c in Island_cols_union]
        if not cols_seg:
            continue

        isl_here = Island_cols_union & set(range(int(L), int(R) + 1))
        if not isl_here:
            skipped_ranges.append(
                {"range": [int(L), int(R)], "reason": "skipped_no_non_bottom_island"}
            )
            continue

        seg, dp_info = _two_state_segment_dp(
            cur,
            merged,
            int(roi_w),
            int(roi_h),
            int(L),
            int(R),
            isl_here,
            thr_class,
            anchor_window=int(anchor_window),
            local_margin=int(local_dp_margin),
            anchor_weight=2.6,
            bottom_band_frac=bottom_frac,
            bottom_weight=2.0,
            margin_extra_weight=0.06,
            escape_entry_penalty=float(escape_entry_penalty),
            escape_max_jump_px=int(escape_max_jump_px),
            s1_to_s0_penalty_scale=800.0,
        )
        if seg is None:
            meta["warnings"].append(f"skip[{L},{R}]:{dp_info.get('skip_reason')}")
            skipped_ranges.append(
                {"range": [int(L), int(R)], "reason": str(dp_info.get("skip_reason"))}
            )
            continue

        esc_cols = dp_info.get("escape_transition_columns") or []
        all_escape_cols.extend(int(x) for x in esc_cols)
        sc = dp_info.get("state_path_counts") or {}
        total_bottom_st += int(sc.get("bottom") or 0)
        total_esc_st += int(sc.get("escaped") or 0)

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

    meta["skipped_ranges"] = skipped_ranges
    meta["triggered_ranges"] = recovered
    meta["escape_transition_columns"] = sorted(set(all_escape_cols))
    meta["state_path_counts"] = {
        "bottom": int(total_bottom_st),
        "escaped": int(total_esc_st),
    }
    meta["escaped_state_columns"] = int(total_esc_st)
    meta["applied"] = bool(recovered)
    meta["max_abs_delta_y"] = float(max_delta)
    meta["changed_columns"] = int(n_changed)

    # confidence lookup for trace (full width list; not a candidate dict)
    trace_confidence_per_column: List[Optional[float]] = []
    for col in range(int(roi_w)):
        yv = cur[col]
        if yv is None:
            trace_confidence_per_column.append(None)
            continue
        yy = int(yv)
        conf_f = 0.5
        for c in merged.get(col) or []:
            if int(c.get("y", -10**9)) == yy:
                conf_f = float(c.get("confidence", 0.5))
                break
        trace_confidence_per_column.append(conf_f)

    meta["trace_confidence_per_column"] = trace_confidence_per_column

    sel_nb = 0
    for col in range(int(roi_w)):
        yv = cur[col]
        if yv is None:
            continue
        for c in merged.get(col) or []:
            if int(c.get("y", -10**9)) == int(yv) and c.get("path_v2_augmented"):
                sel_nb += 1
                break
    meta["selected_added_non_bottom_columns"] = int(sel_nb)

    meta["original_selected_y_px"] = [
        float(before[i]) if i < len(before) and before[i] is not None else None
        for i in range(int(roi_w))
    ]
    meta["recovered_selected_y_px"] = [
        float(cur[i]) if i < len(cur) and cur[i] is not None else None
        for i in range(int(roi_w))
    ]
    meta["warnings"].extend(island_warns)
    if not meta["applied"]:
        meta["warnings"].append("no_segment_survived_dp")

    return cur, meta
