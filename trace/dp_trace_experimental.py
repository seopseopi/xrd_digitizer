"""§8: 실험용 양방향 DP 추적 + 불안정/모호 구간 통계."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from trace.dp_trace import _backtrack, _compute_window, BLOCK_SIZE


def _cont_hist(y_t: float, y_prev: Optional[float], y_prev2: Optional[float], s_h: float) -> float:
    if y_prev is None:
        return 0.0
    if y_prev2 is None:
        return abs(y_t - y_prev) / max(8.0 * s_h, 1e-9)
    y_hat = y_prev + float(np.clip(y_prev - y_prev2, -6 * s_h, 6 * s_h))
    return abs(y_t - y_hat) / max(7.0 * s_h, 1e-9)


def _ridge_mis(ridge_map: np.ndarray, y0: int, y1: int, c0: int, c1: int) -> float:
    n = max(abs(c1 - c0), 1)
    rs = []
    for t in range(n + 1):
        a = t / n
        yy = int(np.clip(round(y0 * (1 - a) + y1 * a), 0, ridge_map.shape[0] - 1))
        cc = int(np.clip(round(c0 * (1 - a) + c1 * a), 0, ridge_map.shape[1] - 1))
        rs.append(ridge_map[yy, cc])
    return float(1.0 - np.mean(rs))


def _tc_experimental(
    y_curr: int,
    y_prev: int,
    y_prev2: Optional[int],
    conf: float,
    grid_pen: float,
    ridge_map: np.ndarray,
    col_curr: int,
    col_prev: int,
    s_h: float,
    y_hist2: Optional[int],
    source_disagreement: float,
    future_gap_risk: float,
) -> float:
    dy = abs(y_curr - y_prev)
    d2y = abs((y_curr - y_prev) - (y_prev - y_prev2)) if y_prev2 is not None else 0.0
    rm = _ridge_mis(ridge_map, y_prev, y_curr, col_prev, col_curr)
    ch = _cont_hist(float(y_curr), float(y_prev), float(y_hist2) if y_hist2 is not None else None, s_h)
    return (
        0.9 * abs(dy)
        + 0.45 * abs(d2y)
        + 1.1 * (1.0 - conf)
        + 1.4 * grid_pen
        + 0.6 * source_disagreement
        + 0.5 * future_gap_risk
        + 0.6 * rm
        + 0.45 * ch
    )


def _source_disagreement(cand: dict) -> float:
    tags = cand.get("source_tags", [])
    if not tags:
        return 1.0
    return float(np.clip((3 - len(set(tags))) / 3.0, 0.0, 1.0))


def _future_gap_risk(final_candidates: Dict[int, List[dict]], col_curr: int, max_col: int, lookahead: int = 8) -> float:
    end = min(max_col, col_curr + lookahead)
    if end <= col_curr:
        return 0.0
    sparse = 0
    total = 0
    for c in range(col_curr + 1, end + 1):
        total += 1
        if len(final_candidates.get(c, [])) <= 1:
            sparse += 1
    return float(sparse / max(total, 1))


def _run_dp_single_pass(
    final_candidates: Dict[int, List[dict]],
    roi_width: int,
    roi_height: int,
    comp_score_map: Optional[np.ndarray],
    ridge_map: np.ndarray,
    grid_pen_map: np.ndarray,
    s_h: float,
    columns_fwd: bool,
) -> dict:
    W = _compute_window(roi_width)
    columns = sorted(final_candidates.keys())
    if not columns_fwd:
        columns = list(reversed(columns))

    nonempty = [c for c in columns if final_candidates.get(c)]
    if not nonempty:
        return {"path": [None] * roi_width, "trace_score": 0.0, "valid_ratio": 0.0, "dp": {}}

    dp: Dict[int, Dict[int, dict]] = {}

    for ci, col in enumerate(columns):
        cands = final_candidates.get(col, [])
        dp[col] = {}
        if not cands:
            continue

        for c in cands:
            y = int(c["y"])
            conf = float(c.get("confidence", 0.5))
            cs = float(c.get("comp_score", 0.0))
            gp = float(grid_pen_map[y, col]) if grid_pen_map is not None else 0.0
            sd = _source_disagreement(c)
            fgr = _future_gap_risk(final_candidates, col, columns[-1] if columns else col)

            if ci == 0 or not dp.get(columns[ci - 1]):
                dp[col][y] = {
                    "cost": 0.8 * (1.0 - conf) + 0.8 * gp,
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
                py = int(py)
                if abs(y - py) > W:
                    continue
                prev2_y = pdata.get("prev_y")
                prev2_y = int(prev2_y) if prev2_y is not None else None
                tc = _tc_experimental(
                    y, py, prev2_y, conf, gp, ridge_map, col, prev_col, s_h, pdata.get("prev2_y"), sd, fgr
                )
                sw = 1.0 if abs(cs - pdata.get("comp_score", 0)) > 0.5 else 0.0
                tc += 1.2 * sw
                tot = pdata["cost"] + tc
                if tot < best_cost:
                    best_cost = tot
                    best_entry = {
                        "cost": tot,
                        "prev_y": py,
                        "prev_col": prev_col,
                        "prev2_y": prev2_y,
                        "conf": conf,
                        "comp_score": cs,
                    }

            if best_entry is None and dp[prev_col]:
                nearest_py = min(dp[prev_col], key=lambda k: dp[prev_col][k]["cost"])
                pdata = dp[prev_col][nearest_py]
                tc = 0.8 * (1.0 - conf) + 1.0 * abs(y - int(nearest_py))
                best_entry = {
                    "cost": pdata["cost"] + tc,
                    "prev_y": int(nearest_py),
                    "prev_col": prev_col,
                    "prev2_y": pdata.get("prev_y"),
                    "conf": conf,
                    "comp_score": cs,
                }

            if best_entry is not None:
                dp[col][y] = best_entry

    path_ord = _backtrack(dp, columns)
    path_full = [None] * roi_width
    for i, col in enumerate(columns):
        if 0 <= col < roi_width and i < len(path_ord):
            path_full[col] = path_ord[i]

    valid = sum(1 for p in path_full if p is not None)
    vr = float(valid) / max(roi_width, 1)
    last_col = None
    for c in reversed(columns):
        if dp.get(c):
            last_col = c
            break
    if last_col is not None and dp[last_col]:
        best_y = min(dp[last_col], key=lambda yy: dp[last_col][yy]["cost"])
        tcost = float(dp[last_col][best_y]["cost"])
    else:
        tcost = 0.0
    return {"path": path_full, "trace_score": tcost, "valid_ratio": vr, "dp": dp}


def merge_lr_rl(
    path_lr: List[Optional[int]],
    path_rl: List[Optional[int]],
    final_candidates: Dict[int, List[dict]],
    roi_width: int,
    s_h: float,
) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for col in range(roi_width):
        a = path_lr[col] if col < len(path_lr) else None
        b = path_rl[col] if col < len(path_rl) else None
        if a is not None and b is not None:
            dis = abs(a - b)
            ca = next((c["confidence"] for c in final_candidates.get(col, []) if c["y"] == a), 0.5)
            cb = next((c["confidence"] for c in final_candidates.get(col, []) if c["y"] == b), 0.5)
            if dis <= 1.5 * s_h:
                ssum = ca + cb + 1e-9
                out.append(int(round((ca * a + cb * b) / ssum)))
            else:
                out.append(a if ca >= cb else b)
        elif a is not None:
            out.append(a)
        elif b is not None:
            out.append(b)
        else:
            out.append(None)
    return out


def _build_blockwise_dual_stats(
    path_lr: List[Optional[int]],
    path_rl: List[Optional[int]],
    final_candidates: Dict[int, List[dict]],
    roi_width: int,
    s_h: float,
) -> tuple[list[dict], list[dict]]:
    blockwise: list[dict] = []
    unstable_blocks: list[dict] = []
    for bs in range(0, roi_width, BLOCK_SIZE):
        be = min(roi_width - 1, bs + BLOCK_SIZE - 1)
        dis = []
        overlap = 0
        both_valid_cols = 0
        margins = []
        low_margin = 0
        for col in range(bs, be + 1):
            cands = final_candidates.get(col, [])
            if len(cands) >= 2:
                m = float(cands[0].get("confidence", 0.0) - cands[1].get("confidence", 0.0))
            elif len(cands) == 1:
                m = 1.0
            else:
                m = 0.0
            margins.append(m)
            if m < 0.10:
                low_margin += 1

            a = path_lr[col] if col < len(path_lr) else None
            b = path_rl[col] if col < len(path_rl) else None
            if a is not None or b is not None:
                overlap += 1
            if a is not None and b is not None:
                both_valid_cols += 1
                dis.append(abs(int(a) - int(b)))

        if dis:
            dis_mean = float(np.mean(dis))
            dis_p90 = float(np.percentile(dis, 90))
            dis_max = float(np.max(dis))
        else:
            dis_mean = 0.0
            dis_p90 = 0.0
            dis_max = 0.0

        overlap_ratio = float(both_valid_cols / max(overlap, 1))
        amb_block = 0.55 * (1.0 - float(np.clip((np.mean(margins) - 0.05) / 0.20, 0, 1))) + 0.25 * (
            low_margin / max(len(margins), 1)
        ) + 0.20 * float(np.clip(dis_mean / max(4.0 * s_h, 1e-9), 0, 1))

        unstable = bool(dis_mean > 2.5 * s_h or dis_p90 > 4.0 * s_h or overlap_ratio < 0.75)
        ambiguous = bool(amb_block > 0.60)

        row = {
            "block_start": bs,
            "block_end": be,
            "dis_mean": round(dis_mean, 4),
            "dis_p90": round(dis_p90, 4),
            "dis_max": round(dis_max, 4),
            "overlap_ratio": round(overlap_ratio, 4),
            "amb_block": round(float(amb_block), 4),
            "unstable": unstable,
            "ambiguous": ambiguous,
        }
        blockwise.append(row)
        if unstable:
            unstable_blocks.append(row)

    return blockwise, unstable_blocks


def dp_trace_bidirectional(
    final_candidates: Dict[int, List[dict]],
    roi_width: int,
    roi_height: int,
    ridge_map: np.ndarray,
    grid_pen_map: np.ndarray,
    s_h: float,
    comp_score_map: Optional[np.ndarray] = None,
) -> dict:
    r1 = _run_dp_single_pass(
        final_candidates, roi_width, roi_height, comp_score_map, ridge_map, grid_pen_map, s_h, True
    )
    r2 = _run_dp_single_pass(
        final_candidates, roi_width, roi_height, comp_score_map, ridge_map, grid_pen_map, s_h, False
    )
    path_lr = r1["path"]
    path_rl = r2["path"]
    m = max(len(path_lr), len(path_rl), roi_width)
    path_lr = (path_lr + [None] * m)[:roi_width]
    path_rl = (path_rl + [None] * m)[:roi_width]

    merged = merge_lr_rl(path_lr, path_rl, final_candidates, roi_width, s_h)
    dis = [
        abs(int(path_lr[i]) - int(path_rl[i]))
        for i in range(roi_width)
        if path_lr[i] is not None and path_rl[i] is not None
    ]
    dis_mean = float(np.mean(dis)) if dis else 0.0
    dis_p90 = float(np.percentile(dis, 90)) if dis else 0.0
    dis_max = float(np.max(dis)) if dis else 0.0
    valid = sum(1 for p in merged if p is not None)

    blockwise, unstable_blocks = _build_blockwise_dual_stats(path_lr, path_rl, final_candidates, roi_width, s_h)

    return {
        "path": merged,
        "path_lr": path_lr,
        "path_rl": path_rl,
        "trace_score": float(r1["trace_score"] + r2["trace_score"]),
        "valid_ratio": float(valid) / max(roi_width, 1),
        "diagnostics": {
            "dual_pass_dis_mean": dis_mean,
            "dual_pass_dis_p90": dis_p90,
            "dual_pass_dis_max": dis_max,
        },
        "blockwise": blockwise,
        "unstable_blocks": unstable_blocks,
        "window_W": _compute_window(roi_width),
    }
