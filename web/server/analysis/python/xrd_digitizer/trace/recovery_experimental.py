"""§8.5: unstable+ambiguous block local rescue."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _mean_block_cost(
    path: List[Optional[int]],
    final_candidates: Dict[int, List[dict]],
    col_start: int,
    col_end: int,
) -> float:
    vals = []
    for col in range(col_start, col_end + 1):
        y = path[col] if col < len(path) else None
        if y is None:
            vals.append(2.5)
            continue
        cands = final_candidates.get(col, [])
        cand = next((c for c in cands if int(c["y"]) == int(y)), None)
        conf = float(cand.get("confidence", 0.5)) if cand else 0.5
        vals.append(1.0 - conf)
    return float(np.mean(vals)) if vals else 9.9


def _anchors(
    block: dict,
    path_lr: List[Optional[int]],
    path_rl: List[Optional[int]],
    s_h: float,
    roi_w: int,
) -> tuple[Optional[int], Optional[int]]:
    bs = int(block["block_start"])
    be = int(block["block_end"])
    la = None
    ra = None
    for col in range(bs - 1, -1, -1):
        if path_lr[col] is not None and path_rl[col] is not None and abs(int(path_lr[col]) - int(path_rl[col])) <= 1.5 * s_h:
            la = col
            break
    for col in range(be + 1, roi_w):
        if path_lr[col] is not None and path_rl[col] is not None and abs(int(path_lr[col]) - int(path_rl[col])) <= 1.5 * s_h:
            ra = col
            break
    return la, ra


def _interior_with_boundary_check(
    incumbent: List[Optional[int]],
    rescue: List[Optional[int]],
    bs: int,
    be: int,
    left_anchor: Optional[int],
    right_anchor: Optional[int],
    s_h: float,
) -> tuple[List[Optional[int]], bool, float]:
    out = list(incumbent)
    # anchor mismatch
    bm = 0.0
    if left_anchor is not None and left_anchor + 1 < len(rescue):
        if incumbent[left_anchor] is not None and rescue[left_anchor + 1] is not None:
            bm = max(bm, abs(int(incumbent[left_anchor]) - int(rescue[left_anchor + 1])))
    if right_anchor is not None and right_anchor - 1 >= 0:
        if incumbent[right_anchor] is not None and rescue[right_anchor - 1] is not None:
            bm = max(bm, abs(int(incumbent[right_anchor]) - int(rescue[right_anchor - 1])))
    if bm > 2.0 * s_h:
        return incumbent, False, bm

    for col in range(bs, be + 1):
        out[col] = rescue[col] if col < len(rescue) else out[col]
    return out, True, bm


def _score_y(
    y: Optional[int],
    prev_y: Optional[int],
    cand: Optional[dict],
    lookahead_density: float,
) -> float:
    if y is None:
        return 3.0
    conf = float(cand.get("confidence", 0.4)) if cand else 0.4
    smooth = abs(int(y) - int(prev_y)) if prev_y is not None else 0.0
    src_tags = cand.get("source_tags", []) if cand else []
    src_div = min(len(set(src_tags)), 3) / 3.0 if src_tags else 0.0
    return 0.9 * smooth + 1.1 * (1.0 - conf) - 0.5 * src_div + 0.7 * (1.0 - lookahead_density)


def _beam_override_segment(
    path: List[Optional[int]],
    final_candidates: Dict[int, List[dict]],
    seg_start: int,
    seg_end: int,
    beam_width: int,
) -> Tuple[List[Optional[int]], dict]:
    beams: List[Tuple[float, List[Optional[int]]]] = [(0.0, [])]
    for col in range(seg_start, seg_end + 1):
        next_beams: List[Tuple[float, List[Optional[int]]]] = []
        cands = final_candidates.get(col, [])[:5]
        if not cands:
            cands = [{"y": None, "confidence": 0.0, "source_tags": []}]
        ahead_total = 0
        ahead_dense = 0
        for cc in range(col + 1, min(seg_end, col + 15) + 1):
            ahead_total += 1
            if len(final_candidates.get(cc, [])) >= 2:
                ahead_dense += 1
        lookahead_density = ahead_dense / max(ahead_total, 1)
        for score, seq in beams:
            prev_y = seq[-1] if seq else (path[col - 1] if col - 1 >= 0 else None)
            for cand in cands:
                y = cand.get("y")
                yv = int(y) if y is not None else None
                sc = _score_y(yv, prev_y, cand, lookahead_density)
                next_beams.append((score + sc, seq + [yv]))
        next_beams.sort(key=lambda x: x[0])
        beams = next_beams[:beam_width]
    best_score, best_seq = beams[0] if beams else (9999.0, [])
    updated = list(path)
    for i, col in enumerate(range(seg_start, seg_end + 1)):
        if i < len(best_seq):
            updated[col] = best_seq[i]
    return updated, {"score": float(best_score), "beam_width": beam_width, "seg_start": seg_start, "seg_end": seg_end}


def run_recovery_experimental(
    trace_result: dict,
    final_candidates: Dict[int, List[dict]],
    roi_w: int,
    s_h: float,
    params: Optional[dict] = None,
) -> dict:
    rp = (params or {})
    gain_min = float(rp.get("gain_min", 0.15))
    coverage_tol = float(rp.get("coverage_tol", 0.03))
    boundary_mul = float(rp.get("boundary_mul", 2.0))
    path = list(trace_result["path"])
    path_lr = trace_result.get("path_lr", path)
    path_rl = trace_result.get("path_rl", path)
    blockwise = trace_result.get("blockwise", [])
    logs = []
    accepted = 0

    for b in blockwise:
        if not (b.get("unstable") and b.get("ambiguous")):
            continue
        bs = int(b["block_start"])
        be = int(b["block_end"])
        seg_start = max(0, bs - 40)
        seg_end = be
        la, ra = _anchors(b, path_lr, path_rl, s_h, roi_w)
        if la is None and ra is None:
            logs.append({"block": [bs, be], "accepted": False, "reason": "no_anchor"})
            continue

        rescue, beam_info = _beam_override_segment(path, final_candidates, seg_start, seg_end, beam_width=3)

        inc_cost = _mean_block_cost(path, final_candidates, seg_start, seg_end)
        res_cost = _mean_block_cost(rescue, final_candidates, seg_start, seg_end)
        inc_cov = sum(1 for col in range(seg_start, seg_end + 1) if path[col] is not None) / max(seg_end - seg_start + 1, 1)
        res_cov = sum(1 for col in range(seg_start, seg_end + 1) if rescue[col] is not None) / max(seg_end - seg_start + 1, 1)
        gain = inc_cost - res_cost

        merged, ok_boundary, bm = _interior_with_boundary_check(path, rescue, seg_start, seg_end, la, ra, s_h * boundary_mul)
        accept = bool(gain >= max(gain_min, 0.08) and res_cov >= inc_cov - coverage_tol and ok_boundary)
        logs.append(
            {
                "block": [bs, be],
                "segment": [seg_start, seg_end],
                "left_anchor": la,
                "right_anchor": ra,
                "gain": round(float(gain), 4),
                "inc_cov": round(float(inc_cov), 4),
                "res_cov": round(float(res_cov), 4),
                "boundary_mismatch": round(float(bm), 4),
                "accepted": accept,
                "beam_info": beam_info,
            }
        )
        if accept:
            path = merged
            accepted += 1

    return {
        "recovery_triggered": bool(logs),
        "updated_path": path,
        "zones": logs,
        "accepted_count": accepted,
    }
