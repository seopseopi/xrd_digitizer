"""
$14: Recovery / Re-entry - DP 오접속 전파 방지.

Trigger -> candidate re-search -> re-score -> local re-trace ->
branch compare -> fail-fast -> user input request.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.contrast_aux_settings import ContrastAuxSettings
from preprocess.contrast_aux import blend_confidence_with_contrast_aux
from trace.candidates import (
    _candidate_confidence,
    filter_candidates,
    build_final_candidates,
    MIN_CONF_KEEP,
)
from trace.dp_trace import dp_trace, BLOCK_SIZE

MAX_RECOVERY_ATTEMPTS = 3
TRIGGER_WINDOW = 40
TRIGGER_VALID_RATIO_MIN = 0.85
TRIGGER_MARGIN_THRESHOLD = 0.15
TRIGGER_MARGIN_STREAK = 12
TRIGGER_SCORE_DROP_RATIO = 0.30
OVERLAP_BLEND = 5
THRESHOLD_RELAX_STEPS = [1.0, 1.25, 1.50]

FAILURE_TAXONOMY = {
    "candidate_starvation": "candidate recall too low in zone",
    "wrong_branch_lock_in": "path locked into wrong component",
    "legend_capture": "legend area pixels captured as curve",
    "grid_confusion": "grid lines confused with curve",
}


def _compute_max_consecutive_missing(pw: int) -> int:
    return max(8, round(0.015 * pw))


def detect_recovery_zones(
    trace_result: dict,
    final_candidates: Dict[int, List[dict]],
    roi_width: int,
) -> List[dict]:
    """
    $14.3: 4가지 trigger 조건을 검사하여 recovery가 필요한 구간 목록을 반환.
    """
    path = trace_result["path"]
    blockwise = trace_result["blockwise"]
    columns = sorted(final_candidates.keys())
    if not columns:
        return []

    max_consec = _compute_max_consecutive_missing(roi_width)
    zones: List[dict] = []

    col_to_path_idx = {c: i for i, c in enumerate(columns)}

    # Trigger 1: sliding window valid_ratio < 0.85
    for start in range(0, len(columns) - TRIGGER_WINDOW + 1, TRIGGER_WINDOW // 2):
        window = columns[start:start + TRIGGER_WINDOW]
        valid = sum(1 for c in window if col_to_path_idx.get(c) is not None
                    and path[col_to_path_idx[c]] is not None)
        local_vr = valid / len(window)
        if local_vr < TRIGGER_VALID_RATIO_MIN:
            zones.append({
                "trigger": "low_valid_ratio",
                "col_start": window[0],
                "col_end": window[-1],
                "metric": round(local_vr, 4),
            })

    # Trigger 2: consecutive missing > threshold
    consec = 0
    consec_start = 0
    for pi, y_val in enumerate(path):
        if y_val is None:
            if consec == 0:
                consec_start = pi
            consec += 1
            if consec > max_consec and pi < len(columns):
                c_start = columns[max(0, consec_start)]
                c_end = columns[min(pi, len(columns) - 1)]
                zones.append({
                    "trigger": "consecutive_missing",
                    "col_start": c_start,
                    "col_end": c_end,
                    "metric": consec,
                })
                consec = 0
        else:
            consec = 0

    # Trigger 3: local trace score 30% spike (global stats: mean + 2*std)
    WARMUP_BLOCKS = 3
    if len(blockwise) >= WARMUP_BLOCKS + 3:
        increments = []
        for bi in range(1, len(blockwise)):
            c = blockwise[bi].get("top1_cost")
            p = blockwise[bi - 1].get("top1_cost")
            if c is not None and p is not None:
                increments.append((bi, c - p))

        EDGE_SKIP = 2
        max_bi = len(blockwise) - 1
        stable = [(bi, v) for bi, v in increments
                  if bi >= WARMUP_BLOCKS and bi <= max_bi - EDGE_SKIP]
        if len(stable) >= 3:
            vals = [v for _, v in stable]
            mean_inc = sum(vals) / len(vals)
            std_inc = (sum((v - mean_inc) ** 2 for v in vals) / len(vals)) ** 0.5
            spike_threshold = mean_inc + max(2.5 * std_inc, mean_inc * TRIGGER_SCORE_DROP_RATIO)

            col_to_path_idx_local = {c: i for i, c in enumerate(columns)}
            for bi_idx, delta in stable:
                if delta > spike_threshold:
                    bs = blockwise[bi_idx]["block_start"]
                    be = blockwise[bi_idx]["block_end"]
                    zone_valid = sum(
                        1 for c in range(bs, be + 1)
                        if col_to_path_idx_local.get(c) is not None
                        and path[col_to_path_idx_local[c]] is not None
                    )
                    zone_cols = be - bs + 1
                    local_vr = zone_valid / max(zone_cols, 1)
                    if local_vr < 0.95:
                        zones.append({
                            "trigger": "score_spike",
                            "col_start": bs,
                            "col_end": be,
                            "metric": round(delta / max(mean_inc, 0.01), 4),
                        })

    # Trigger 4: margin < 0.15 for 12+ consecutive columns (min 2 blocks)
    MIN_MARGIN_BLOCKS = 2
    low_margin_streak = 0
    streak_start_block = 0
    for bi, bw in enumerate(blockwise):
        margin = bw.get("margin")
        if margin is not None and margin < TRIGGER_MARGIN_THRESHOLD:
            if low_margin_streak == 0:
                streak_start_block = bi
            low_margin_streak += 1
        else:
            if low_margin_streak >= MIN_MARGIN_BLOCKS:
                total_cols = sum(
                    blockwise[b]["block_end"] - blockwise[b]["block_start"] + 1
                    for b in range(streak_start_block, streak_start_block + low_margin_streak)
                )
                if total_cols >= TRIGGER_MARGIN_STREAK:
                    zones.append({
                        "trigger": "low_margin_streak",
                        "col_start": blockwise[streak_start_block]["block_start"],
                        "col_end": blockwise[min(bi - 1, len(blockwise) - 1)]["block_end"],
                        "metric": low_margin_streak,
                    })
            low_margin_streak = 0

    zones = _merge_overlapping_zones(zones)
    return zones


def _merge_overlapping_zones(zones: List[dict]) -> List[dict]:
    if not zones:
        return []
    unique: Dict[tuple, dict] = {}
    for z in zones:
        key = (z["col_start"], z["col_end"])
        if key not in unique:
            unique[key] = z
    zones = list(unique.values())
    zones.sort(key=lambda z: z["col_start"])
    merged = [zones[0]]
    for z in zones[1:]:
        if z["col_start"] <= merged[-1]["col_end"] + TRIGGER_WINDOW // 2:
            merged[-1]["col_end"] = max(merged[-1]["col_end"], z["col_end"])
            triggers = set(merged[-1]["trigger"].split("+"))
            triggers.add(z["trigger"])
            merged[-1]["trigger"] = "+".join(sorted(triggers))
        else:
            merged.append(z)
    return merged


def _research_candidates_in_zone(
    raw_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    color_threshold: float,
    col_start: int,
    col_end: int,
    relax_factor: float = 1.0,
    contrast_aux_map: Optional[np.ndarray] = None,
    contrast_aux_settings: Optional[ContrastAuxSettings] = None,
) -> Dict[int, List[dict]]:
    """
    $14.5 step 1: 구간 candidate 재탐색 (threshold 완화).
    """
    h, w = raw_mask.shape[:2]
    relaxed_threshold = color_threshold * relax_factor

    union = np.clip(raw_mask.astype(np.uint8) + skeleton_mask.astype(np.uint8), 0, 1)
    zone_cands: Dict[int, List[dict]] = {}
    prev_best_y: Optional[float] = None

    for col in range(max(0, col_start - OVERLAP_BLEND), min(w, col_end + OVERLAP_BLEND + 1)):
        ys_union = np.where(union[:, col] > 0)[0]
        ys_relaxed = np.where(color_dist_map[:, col] <= relaxed_threshold)[0]
        ys = np.unique(np.concatenate([ys_union, ys_relaxed]))

        if len(ys) == 0:
            zone_cands[col] = []
            continue

        cands = []
        for y_val in ys:
            yi = int(y_val)
            cd = float(color_dist_map[yi, col])
            cs = float(comp_score_map[yi, col])
            ad = float(axis_dist_map[yi, col])
            conf = _candidate_confidence(cd, float(yi), prev_best_y, cs, ad)
            if (
                contrast_aux_map is not None
                and contrast_aux_settings is not None
                and contrast_aux_settings.use_contrast_aux
            ):
                conf, _bon = blend_confidence_with_contrast_aux(
                    conf, col, yi, contrast_aux_map, contrast_aux_settings,
                )
            cands.append({
                "y": yi, "confidence": conf,
                "color_dist": cd, "comp_score": cs, "axis_dist": ad,
                "source": "recovery",
            })

        cands.sort(key=lambda c: -c["confidence"])
        zone_cands[col] = cands
        if cands:
            prev_best_y = float(cands[0]["y"])

    return zone_cands


def _local_retrace(
    zone_candidates: Dict[int, List[dict]],
    roi_width: int,
    roi_height: int,
    comp_score_map: Optional[np.ndarray],
) -> dict:
    """$14.5 step 3: local re-trace (zone DP)."""
    filtered = filter_candidates(zone_candidates)
    final, _ = build_final_candidates(
        filtered, comp_score_map, roi_width, roi_height=roi_height,
    )
    zone_only = {c: v for c, v in final.items() if c in zone_candidates}
    if not zone_only:
        return {"path": [], "trace_score": float("inf"), "valid_ratio": 0.0}
    zone_w = max(zone_only.keys()) - min(zone_only.keys()) + 1
    return dp_trace(zone_only, zone_w, roi_height, comp_score_map)


def _compare_branches(
    original_path: List[Optional[int]],
    recovery_path: List[Optional[int]],
    columns: List[int],
    zone_col_start: int,
    zone_col_end: int,
    final_candidates: Dict[int, List[dict]],
) -> dict:
    """$14.5 step 4: branch 비교."""
    col_to_idx = {c: i for i, c in enumerate(columns)}

    orig_valid = 0
    recov_valid = 0
    orig_conf_sum = 0.0
    recov_conf_sum = 0.0
    n = 0

    for col in range(zone_col_start, zone_col_end + 1):
        idx = col_to_idx.get(col)
        if idx is None:
            continue
        n += 1

        oy = original_path[idx] if idx < len(original_path) else None
        if oy is not None:
            orig_valid += 1
            match = [c for c in final_candidates.get(col, []) if c["y"] == oy]
            orig_conf_sum += match[0]["confidence"] if match else 0.0

        zone_offset = col - zone_col_start
        ry = recovery_path[zone_offset] if zone_offset < len(recovery_path) else None
        if ry is not None:
            recov_valid += 1
            recov_conf_sum += 0.5

    return {
        "original_valid": orig_valid,
        "recovery_valid": recov_valid,
        "original_avg_conf": round(orig_conf_sum / max(n, 1), 4),
        "recovery_avg_conf": round(recov_conf_sum / max(n, 1), 4),
        "prefer": "recovery" if recov_valid > orig_valid else "original",
    }


def _classify_failure(
    zone: dict,
    zone_candidates: Dict[int, List[dict]],
    final_candidates: Dict[int, List[dict]],
) -> str:
    """$14.7: failure taxonomy label."""
    nonempty = sum(1 for v in zone_candidates.values() if v)
    total = max(len(zone_candidates), 1)
    fill_ratio = nonempty / total

    if fill_ratio < 0.5:
        return "candidate_starvation"

    triggers = zone.get("trigger", "")
    if "low_margin" in triggers:
        return "wrong_branch_lock_in"
    if "score_spike" in triggers:
        return "grid_confusion"
    return "candidate_starvation"


def _blend_paths(
    original_path: List[Optional[int]],
    recovery_path: List[Optional[int]],
    columns: List[int],
    zone_col_start: int,
    zone_col_end: int,
) -> List[Optional[int]]:
    """Recovery 경로를 원래 경로에 blend."""
    result = list(original_path)
    col_to_idx = {c: i for i, c in enumerate(columns)}

    for col in range(zone_col_start, zone_col_end + 1):
        idx = col_to_idx.get(col)
        zone_offset = col - zone_col_start
        if idx is None or zone_offset >= len(recovery_path):
            continue

        ry = recovery_path[zone_offset]
        if ry is None:
            continue

        dist_to_start = col - zone_col_start
        dist_to_end = zone_col_end - col
        zone_len = zone_col_end - zone_col_start

        if dist_to_start < OVERLAP_BLEND and result[idx] is not None:
            alpha = dist_to_start / OVERLAP_BLEND
            result[idx] = int(round((1 - alpha) * result[idx] + alpha * ry))
        elif dist_to_end < OVERLAP_BLEND and result[idx] is not None:
            alpha = dist_to_end / OVERLAP_BLEND
            result[idx] = int(round((1 - alpha) * result[idx] + alpha * ry))
        else:
            result[idx] = ry

    return result


def run_recovery(
    trace_result: dict,
    final_candidates: Dict[int, List[dict]],
    raw_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    color_threshold: float,
    roi_width: int,
    roi_height: int,
    contrast_aux_map: Optional[np.ndarray] = None,
    contrast_aux_settings: Optional[ContrastAuxSettings] = None,
) -> dict:
    """
    $14 recovery main entry point.
    Returns: recovery_log, updated_path, before/after candidate maps.
    """
    columns = sorted(final_candidates.keys())
    zones = detect_recovery_zones(trace_result, final_candidates, roi_width)

    if not zones:
        return {
            "recovery_triggered": False,
            "zones": [],
            "updated_path": trace_result["path"],
            "failure_labels": [],
        }

    recovery_log: List[dict] = []
    failure_labels: List[dict] = []
    current_path = list(trace_result["path"])
    before_cands_all: Dict[int, List[dict]] = {}
    after_cands_all: Dict[int, List[dict]] = {}

    for zone in zones:
        cs = zone["col_start"]
        ce = zone["col_end"]

        for col in range(cs, ce + 1):
            before_cands_all[col] = final_candidates.get(col, [])

        zone_log = {
            "zone": zone,
            "attempts": [],
            "resolved": False,
            "failure_label": None,
        }

        resolved = False
        for attempt in range(MAX_RECOVERY_ATTEMPTS):
            relax = THRESHOLD_RELAX_STEPS[min(attempt, len(THRESHOLD_RELAX_STEPS) - 1)]

            # Step 1: candidate re-search
            zone_cands = _research_candidates_in_zone(
                raw_mask, skeleton_mask, color_dist_map,
                comp_score_map, axis_dist_map,
                color_threshold, cs, ce, relax_factor=relax,
                contrast_aux_map=contrast_aux_map,
                contrast_aux_settings=contrast_aux_settings,
            )

            new_count = sum(len(v) for v in zone_cands.values())

            # Step 2: re-score (already done in _research)
            # Step 3: local re-trace
            local_result = _local_retrace(
                zone_cands, roi_width, roi_height, comp_score_map,
            )

            local_path = local_result.get("path", [])
            local_vr = local_result.get("valid_ratio", 0.0)

            # Step 4: branch compare
            branch_cmp = _compare_branches(
                current_path, local_path, columns, cs, ce, final_candidates,
            )

            attempt_log = {
                "attempt": attempt,
                "relax_factor": relax,
                "new_candidates": new_count,
                "local_valid_ratio": round(local_vr, 4),
                "branch_comparison": branch_cmp,
            }
            zone_log["attempts"].append(attempt_log)

            if branch_cmp["prefer"] == "recovery" and local_vr > TRIGGER_VALID_RATIO_MIN:
                current_path = _blend_paths(current_path, local_path, columns, cs, ce)
                zone_log["resolved"] = True
                resolved = True

                for col in range(cs, ce + 1):
                    after_cands_all[col] = zone_cands.get(col, [])
                break

        # Step 5: fail-fast + taxonomy
        if not resolved:
            label = _classify_failure(zone, zone_cands, final_candidates)
            zone_log["failure_label"] = label
            failure_labels.append({"zone": [cs, ce], "label": label})

            for col in range(cs, ce + 1):
                after_cands_all[col] = zone_cands.get(col, [])

        recovery_log.append(zone_log)

    return {
        "recovery_triggered": True,
        "zones": recovery_log,
        "updated_path": current_path,
        "failure_labels": failure_labels,
        "before_candidates": before_cands_all,
        "after_candidates": after_cands_all,
    }


def render_branch_compare(
    roi: np.ndarray,
    original_path: List[Optional[int]],
    updated_path: List[Optional[int]],
    columns: List[int],
) -> np.ndarray:
    """branch_compare.png: original=blue, recovery=green, overlap=cyan."""
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    h, w = overlay.shape[:2]

    for pi, col in enumerate(columns):
        if col >= w:
            continue

        oy = original_path[pi] if pi < len(original_path) else None
        uy = updated_path[pi] if pi < len(updated_path) else None

        if oy is not None and 0 <= int(oy) < h:
            y = int(oy)
            for dy in range(-1, 2):
                yy = y + dy
                if 0 <= yy < h:
                    overlay[yy, col] = [255, 80, 80]

        if uy is not None and 0 <= int(uy) < h:
            y = int(uy)
            for dy in range(-1, 2):
                yy = y + dy
                if 0 <= yy < h:
                    if oy is not None and abs(int(oy) - y) <= 2:
                        overlay[yy, col] = [80, 255, 255]
                    else:
                        overlay[yy, col] = [80, 255, 80]

    return overlay


def render_candidates_overlay(
    roi: np.ndarray,
    candidates: Dict[int, List[dict]],
    label: str = "",
) -> np.ndarray:
    """candidate before/after overlay."""
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    h, w = overlay.shape[:2]

    for col, cands in candidates.items():
        if col >= w:
            continue
        for c in cands[:6]:
            y = int(c["y"])
            if 0 <= y < h:
                conf = c.get("confidence", 0.5)
                g = int(np.clip(conf * 255, 60, 255))
                overlay[y, col] = [0, g, 255]

    return overlay
