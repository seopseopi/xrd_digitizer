"""
Experimental path-level recovery v0: localized re-selection inside monitor-flagged runs.

Default OFF. Uses only runtime-visible signals (candidates, selected_y, ROI geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trace.dp_trace import ALPHA, BETA, GAMMA, DELTA, COMP_SWITCH_THRESH
from trace.dp_trace import _border_penalty
from trace.path_monitor import compute_selected_path_monitor, top1_y_from_final_candidates


def _path_to_array(path: List[Optional[int]], roi_w: int) -> np.ndarray:
    y = np.full(int(roi_w), np.nan, dtype=np.float64)
    for col in range(min(len(path), int(roi_w))):
        v = path[col]
        if v is not None:
            y[col] = float(v)
    return y


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    rs = sorted(ranges, key=lambda x: x[0])
    out = [rs[0]]
    for a, b in rs[1:]:
        la, lb = out[-1]
        if a <= lb + 1:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _clip_region(a: int, b: int, max_len: int) -> List[Tuple[int, int]]:
    if b - a + 1 <= max_len:
        return [(a, b)]
    chunks = []
    cur = a
    while cur <= b:
        chunks.append((cur, min(b, cur + max_len - 1)))
        cur = chunks[-1][1] + 1
    return chunks


def plan_recovery_regions(
    monitor: Dict[str, Any],
    *,
    roi_w: int,
    min_bottom_run_len: int,
    max_region_len: int,
    prelockin_guard_col: int,
) -> List[Tuple[int, int]]:
    """Return inclusive [L,R] column ranges to attempt recovery (bottom-branch runs)."""
    bb = monitor.get("bottom_branch_persistence_proxy") or {}
    raw_ranges = bb.get("bottom_branch_run_ranges") or []
    out: List[Tuple[int, int]] = []
    for pair in raw_ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if b - a + 1 < int(min_bottom_run_len):
            continue
        if b < int(prelockin_guard_col):
            continue
        a = max(a, int(prelockin_guard_col))
        if a > b:
            continue
        for seg in _clip_region(a, b, int(max_region_len)):
            out.append(seg)
    return _merge_ranges(out)


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


def _transition_terms(
    y_curr: int,
    y_prev: Optional[int],
    y_prev2: Optional[int],
    conf: float,
    cs: float,
    cs_prev: float,
    roi_h: int,
) -> float:
    if y_prev is None:
        return float(GAMMA * (1.0 - conf) + _border_penalty(y_curr, roi_h))
    dy = abs(float(y_curr - y_prev))
    d2y = (
        abs(float((y_curr - y_prev) - (y_prev - y_prev2)))
        if y_prev2 is not None
        else 0.0
    )
    comp_sw = 1.0 if abs(float(cs - cs_prev)) > COMP_SWITCH_THRESH else 0.0
    return float(
        ALPHA * dy
        + BETA * d2y
        + GAMMA * (1.0 - conf)
        + DELTA * comp_sw
        + _border_penalty(y_curr, roi_h)
    )


def _local_segment_dp(
    path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
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
) -> Tuple[Optional[List[Optional[int]]], Dict[str, Any]]:
    """Viterbi-style DP on columns L..R inclusive. Returns full updated path copy."""
    if L < 0 or R >= roi_w or L > R:
        return None, {"skip_reason": "bad_bounds"}

    cols = list(range(L, R + 1))
    left_anchor_cols = [c for c in range(max(0, L - anchor_window), L)]
    right_anchor_cols = [c for c in range(R + 1, min(roi_w, R + 1 + anchor_window))]

    INF = 1e30
    dp_costs: Dict[int, Dict[int, float]] = {}
    back: Dict[int, Dict[int, Optional[int]]] = {}

    def margin_pen(py: int, y: int) -> float:
        dy = abs(float(y - py))
        if dy <= float(local_margin):
            return 0.0
        return margin_extra_weight * (dy - float(local_margin)) ** 2 / max(float(roi_h), 1.0)

    col0 = cols[0]
    c0 = final_cands.get(col0) or []
    if not c0:
        return None, {"skip_reason": "no_candidates_first_col"}

    y_prev_out = path[col0 - 1] if col0 > 0 else None
    y_prev2_out = path[col0 - 2] if col0 > 1 else None

    dp_costs[col0] = {}
    back[col0] = {}
    for cand in c0:
        y = int(cand["y"])
        conf = float(cand.get("confidence", 0.0))
        cs = float(cand.get("comp_score", 0.0))
        pred = _anchor_predict_y(col0, left_anchor_cols, right_anchor_cols, path)
        anchor_pen = 0.0
        if np.isfinite(pred):
            anchor_pen = anchor_weight * abs(float(y) - pred) / max(float(roi_h), 1.0)
        bottom_pen = 0.0
        thr_b = bottom_band_frac * float(roi_h)
        if float(y) >= thr_b:
            bottom_pen = bottom_weight * (float(y) - thr_b) / max(float(roi_h), 1.0)
        base = (
            _transition_terms(y, y_prev_out, y_prev2_out, conf, cs, 0.0, roi_h)
            + anchor_pen
            + bottom_pen
        )
        if y_prev_out is not None:
            base += margin_pen(int(y_prev_out), y)
        dp_costs[col0][y] = base
        back[col0][y] = None

    for idx in range(1, len(cols)):
        col = cols[idx]
        prev_col = cols[idx - 1]
        cands = final_cands.get(col) or []
        if not cands:
            return None, {"skip_reason": f"no_candidates_col_{col}"}
        dp_costs[col] = {}
        back[col] = {}
        pred_col = _anchor_predict_y(col, left_anchor_cols, right_anchor_cols, path)
        for cand in cands:
            y = int(cand["y"])
            conf = float(cand.get("confidence", 0.0))
            cs = float(cand.get("comp_score", 0.0))
            anchor_pen = 0.0
            if np.isfinite(pred_col):
                anchor_pen = anchor_weight * abs(float(y) - pred_col) / max(float(roi_h), 1.0)
            bottom_pen = 0.0
            thr_b = bottom_band_frac * float(roi_h)
            if float(y) >= thr_b:
                bottom_pen = bottom_weight * (float(y) - thr_b) / max(float(roi_h), 1.0)

            best_c = INF
            best_py: Optional[int] = None
            for py in dp_costs[prev_col].keys():
                prev_cs = 0.0
                for pc in final_cands.get(prev_col) or []:
                    if int(pc.get("y", -99999)) == int(py):
                        prev_cs = float(pc.get("comp_score", 0.0))
                        break
                py2 = None
                if prev_col >= 2:
                    pv2 = path[prev_col - 2]
                    py2 = int(pv2) if pv2 is not None else None
                terms = _transition_terms(y, int(py), py2, conf, cs, prev_cs, roi_h)
                terms += margin_pen(int(py), y)
                tot = dp_costs[prev_col][py] + terms + anchor_pen + bottom_pen
                if tot < best_c:
                    best_c = tot
                    best_py = int(py)
            if best_py is not None:
                dp_costs[col][y] = best_c
                back[col][y] = best_py

        if not dp_costs[col]:
            return None, {"skip_reason": f"dp_dead_col_{col}"}

    last = cols[-1]
    best_y = min(dp_costs[last], key=lambda yy: dp_costs[last][yy])
    seg_path: Dict[int, int] = {}
    cy: Optional[int] = best_y
    for col in reversed(cols):
        if cy is None or col not in dp_costs or cy not in dp_costs[col]:
            return None, {"skip_reason": "backtrack_fail"}
        seg_path[col] = int(cy)
        cy = back[col].get(int(cy)) if col != cols[0] else None

    out: List[Optional[int]] = list(path)
    for col in cols:
        out[col] = seg_path[col]

    return out, {"segment_dp_ok": True, "terminal_cost": float(dp_costs[last][best_y])}


def apply_path_level_recovery_v0(
    trace_path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
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
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """Apply optional localized recovery. Monitor uses monitor_min_run_for_detection."""
    y_arr = _path_to_array(trace_path, roi_w)
    top1 = top1_y_from_final_candidates(final_cands, roi_w)
    mon = compute_selected_path_monitor(
        y_arr,
        top1_y=top1,
        selected_source=None,
        roi_h=int(roi_h),
        window=int(monitor_window),
        min_run=int(monitor_min_run_for_detection),
    )

    regions = plan_recovery_regions(
        mon,
        roi_w=int(roi_w),
        min_bottom_run_len=int(min_bottom_run_len),
        max_region_len=int(max_region_len),
        prelockin_guard_col=int(prelockin_guard_col),
    )

    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "uses_source_numeric": False,
        "uses_gt": False,
        "risk_run_ranges": [[int(a), int(b)] for a, b in regions],
        "recovered_ranges": [],
        "anchor_windows": [],
        "max_abs_delta_y": 0.0,
        "changed_columns": 0,
        "warning": None,
        "monitor_snapshot": {
            "bottom_branch_score": (mon.get("bottom_branch_persistence_proxy") or {}).get(
                "bottom_branch_score"
            ),
            "longest_bottom_branch_run_len": (mon.get("bottom_branch_persistence_proxy") or {}).get(
                "longest_bottom_branch_run_len"
            ),
        },
    }

    if not regions:
        meta["warning"] = "no_eligible_bottom_run_after_guards"
        return trace_path, meta

    cur = list(trace_path)
    max_delta = 0.0
    n_changed = 0
    recovered_ranges: List[List[int]] = []

    for L, R in regions:
        seg, info = _local_segment_dp(
            cur,
            final_cands,
            roi_w,
            roi_h,
            L,
            R,
            anchor_window=int(anchor_window),
            local_margin=int(local_dp_margin),
            anchor_weight=2.6,
            bottom_band_frac=0.84,
            bottom_weight=2.0,
            margin_extra_weight=0.06,
        )
        if seg is None:
            meta["warning"] = (meta.get("warning") or "") + f";skip[{L},{R}]:{info.get('skip_reason')}"
            continue
        for col in range(L, R + 1):
            ov = cur[col]
            nv = seg[col]
            if ov is not None and nv is not None and ov != nv:
                d = abs(float(int(nv) - int(ov)))
                max_delta = max(max_delta, d)
                n_changed += 1
            elif (ov is None) != (nv is None):
                n_changed += 1
        cur = seg
        recovered_ranges.append([int(L), int(R)])
        meta["anchor_windows"].append(
            {
                "region": [int(L), int(R)],
                "left_anchor_cols": [c for c in range(max(0, L - anchor_window), L)],
                "right_anchor_cols": [c for c in range(R + 1, min(roi_w, R + 1 + anchor_window))],
            }
        )

    meta["applied"] = bool(recovered_ranges)
    meta["recovered_ranges"] = recovered_ranges
    meta["max_abs_delta_y"] = float(max_delta)
    meta["changed_columns"] = int(n_changed)
    if not meta["applied"]:
        meta["warning"] = (meta.get("warning") or "no_segment_survived_dp").strip(";")

    return cur, meta
