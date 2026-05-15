"""
$12.4-12.9: candidate generation pipeline.

3-stage: raw -> filtered -> final DP candidate.
- raw: recall 보존 (column별 top-K 제한 없음)
- filtered: confidence >= 0.25, column별 max 16 (미세 피크 4단계: 12→16)
- final: confidence 상위 8, component 중복 제거 (6→8)
- final 정렬: 스켈레톤 열별 y 힌트(창 중앙값 스무딩)로 DP 후보 순위 보조 (로드맵 2a)
- filtered 정렬: 굵은 선 열에서 열 최상단(y 최소) 근처 후보에 가산해 상위 N에 남기도록 보정
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# evidence-aware vertical preservation (GT-free): 인접 열·동일 열 y 밀도
_NEARBY_Y_SUPPORT_PX = 2
_NEIGHBOR_Y_MATCH_PX = 3
_UPPER_BAND_FRAC_FOR_CAP = 0.2
_LOCAL_EVIDENCE_WEIGHT_NEARBY = 0.25
_LOCAL_EVIDENCE_WEIGHT_NEIGHBOR = 0.25

MIN_CONF_KEEP = 0.25
MAX_CANDIDATES_AFTER_FILTER = 16
# 굵은 곡선(안티앨리어싱) 열: 신뢰도만으로 상위 N만 고르면 윤곽선 위쪽(y 최소)이 잘릴 수 있음
# 주피크가 세로로 크게 낮게 복원됨. 열 내 y 스팬이 클 때만 최상단 근처에 정렬 가산.
ENVELOPE_SORT_MIN_CANDIDATES = 8
ENVELOPE_SORT_MIN_SPAN_PX = 4
ENVELOPE_SORT_BONUS = 0.35
ENVELOPE_SORT_DECAY_PX = 6.0
# 이웃 열 최고 신뢰도 후보 y 중앙값과의 정렬 일치 가산(진단용 실험 플래그에서만 사용).
LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT = 0.09
LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT = 14.0
MAX_CANDIDATES_FOR_DP = 8
MAX_CANDIDATES_FOR_DP_EXPERIMENTAL = 7
NEIGHBOR_RADIUS = 3
# preprocess.ridge_map 기반 세로 능선 응답을 신뢰도에 더할 가중치.
# 0.05~0.08 은 일부 styled 샘플에서 주피크 미검출(999) 평균 왜곡이 커질 수 있어 0.10 권장.
# 끄려면 기본 실행; 켜려면 batch_run / run_local --use-ridge-candidates.
RIDGE_CONF_WEIGHT = 0.10


def _nearby_y_support_normalized(y: int, kept: List[dict]) -> float:
    """같은 열 kept 안에서 ±2px 이내 후보 비율 [0,1]."""
    if not kept:
        return 0.0
    cnt = sum(1 for c in kept if abs(int(c["y"]) - int(y)) <= _NEARBY_Y_SUPPORT_PX)
    return float(min(1.0, float(cnt) / float(len(kept))))


def _neighbor_column_support_normalized(
    col: int,
    y: int,
    raw_candidates: Dict[int, List[dict]],
    roi_w: int,
) -> float:
    """좌우 인접 열에 |Δy|<=3px 후보가 있으면 0.5씩 가산 → [0,1]."""
    hits = 0
    if col - 1 >= 0:
        lst = raw_candidates.get(col - 1, [])
        if any(abs(int(c["y"]) - int(y)) <= _NEIGHBOR_Y_MATCH_PX for c in lst):
            hits += 1
    if col + 1 < roi_w:
        lst = raw_candidates.get(col + 1, [])
        if any(abs(int(c["y"]) - int(y)) <= _NEIGHBOR_Y_MATCH_PX for c in lst):
            hits += 1
    return hits / 2.0


def candidate_local_evidence_score(
    c: dict,
    col: int,
    kept: List[dict],
    raw_candidates: Dict[int, List[dict]],
    roi_w: int,
) -> float:
    """GT 미사용: confidence + 근접 y 밀도 + 인접 열 지지."""
    y = int(c["y"])
    cf = float(c.get("confidence", 0.0))
    ny = _nearby_y_support_normalized(y, kept)
    nb = _neighbor_column_support_normalized(col, y, raw_candidates, roi_w)
    comp_term = 0.0
    if "comp_score" in c:
        try:
            comp_term = 0.05 * float(np.clip(float(c.get("comp_score", 0.0)) / 10.0, 0.0, 1.0))
        except (TypeError, ValueError):
            comp_term = 0.0
    return (
        cf
        + _LOCAL_EVIDENCE_WEIGHT_NEARBY * ny
        + _LOCAL_EVIDENCE_WEIGHT_NEIGHBOR * nb
        + comp_term
    )


def _evidence_aware_pick_filtered(
    *,
    col: int,
    kept: List[dict],
    rank_sorted: List[dict],
    raw_candidates: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    max_k: int,
    preserve_bins: int,
    preserve_per_bin: int,
    preserve_max_upper_frac: float,
    rank_score_fn: Callable[[dict], float],
) -> Tuple[List[dict], Dict[str, Any]]:
    """열 내 filtered 후보를 rank·수직 bin 보존·상단 비율 상한으로 선택."""
    bins_n = max(1, int(preserve_bins))
    per_bin = max(1, int(preserve_per_bin))
    cap_upper = float(min(1.0, max(0.0, preserve_max_upper_frac)))
    ub_y = max(1, int(math.ceil(float(roi_h) * _UPPER_BAND_FRAC_FOR_CAP)))

    y_lo = min(int(c["y"]) for c in kept)
    y_hi = max(int(c["y"]) for c in kept)
    span = max(y_hi - y_lo, 1)

    def _bin_idx(y: int) -> int:
        if bins_n <= 1:
            return 0
        b = int((int(y) - y_lo) * bins_n / float(span))
        return max(0, min(bins_n - 1, b))

    by_bin: Dict[int, List[dict]] = defaultdict(list)
    for c in kept:
        by_bin[_bin_idx(int(c["y"]))].append(c)

    preserved_order: List[dict] = []
    preserved_ids: set[int] = set()
    preserved_by_bin = [0] * bins_n
    for b in range(bins_n):
        bucket = list(by_bin.get(b, []))
        bucket.sort(
            key=lambda c: (
                -candidate_local_evidence_score(c, col, kept, raw_candidates, roi_w),
                -rank_score_fn(c),
            )
        )
        taken = 0
        for c in bucket:
            if taken >= per_bin:
                break
            cid = id(c)
            if cid in preserved_ids:
                continue
            preserved_order.append(c)
            preserved_ids.add(cid)
            preserved_by_bin[b] += 1
            taken += 1

    def _upper_frac(sel: List[dict]) -> float:
        if not sel:
            return 0.0
        nu = sum(1 for c in sel if int(c["y"]) < ub_y)
        return float(nu) / float(len(sel))

    selected: List[dict] = []
    sid: set[int] = set()
    for c in preserved_order:
        if len(selected) >= max_k:
            break
        if id(c) not in sid:
            selected.append(c)
            sid.add(id(c))
    for c in rank_sorted:
        if len(selected) >= max_k:
            break
        if id(c) not in sid:
            selected.append(c)
            sid.add(id(c))

    frac_before = _upper_frac(selected)
    work = list(selected)
    wid = set(id(c) for c in work)

    while _upper_frac(work) > cap_upper + 1e-12:
        ups = [c for c in work if int(c["y"]) < ub_y]
        if len(ups) <= 1:
            break
        u_rm = min(ups, key=lambda c: rank_score_fn(c))
        pool = [
            c
            for c in kept
            if int(c["y"]) >= ub_y and id(c) not in wid
        ]
        if not pool:
            break
        add_c = max(
            pool,
            key=lambda c: (
                candidate_local_evidence_score(c, col, kept, raw_candidates, roi_w),
                rank_score_fn(c),
            ),
        )
        work.remove(u_rm)
        work.append(add_c)
        wid.discard(id(u_rm))
        wid.add(id(add_c))

    frac_after = _upper_frac(work)
    ranked = sorted(work, key=lambda c: -rank_score_fn(c))
    meta = {
        "preserved_by_bin": preserved_by_bin,
        "upper_frac_before_cap": frac_before,
        "upper_frac_after_cap": frac_after,
        "evidence_preserved_in_selection": int(sum(1 for c in ranked if id(c) in preserved_ids)),
        "preserved_ids": preserved_ids,
    }
    return ranked, meta


def neighbor_conf_top1_y_median(
    raw_candidates: Dict[int, List[dict]], col: int, roi_w: int,
) -> Optional[float]:
    """좌우 인접 열 각각에서 confidence 최대 후보의 y를 모아 중앙값을 구한다. 후보 없으면 None."""
    ys: List[float] = []
    for ncol in (col - 1, col + 1):
        if ncol < 0 or ncol >= roi_w:
            continue
        cands = raw_candidates.get(ncol, [])
        if not cands:
            continue
        top = max(cands, key=lambda x: float(x.get("confidence", 0.0)))
        ys.append(float(top["y"]))
    if not ys:
        return None
    return float(np.median(np.asarray(ys, dtype=np.float64)))


def _color_consistency(color_dist: float) -> float:
    """$12.6(1): exp(-d / 20)"""
    return float(np.exp(-color_dist / 20.0))


def _local_continuity(y_curr: float, y_prev: Optional[float]) -> float:
    """$12.6(2): exp(-dy / 8), first column = 1.0"""
    if y_prev is None:
        return 1.0
    dy = abs(y_curr - y_prev)
    return float(np.exp(-dy / 8.0))


def _component_support(comp_score: float) -> float:
    """$12.6(3): 1 / (1 + exp(-0.8*(score - 2)))"""
    return float(1.0 / (1.0 + np.exp(-0.8 * (comp_score - 2.0))))


def _penalty(axis_dist: float) -> float:
    """$12.6(4): exp(-d_min / 6)"""
    return float(np.exp(-axis_dist / 6.0))


def _candidate_confidence(
    color_dist: float,
    y_curr: float,
    y_prev: Optional[float],
    comp_score: float,
    axis_dist: float,
    ridge_resp: Optional[float] = None,
) -> float:
    """$12.6 combined confidence in [0,1]. ridge_resp: [0,1] 세로 능선 응답(선택)."""
    cc = _color_consistency(color_dist)
    lc = _local_continuity(y_curr, y_prev)
    cs = _component_support(comp_score)
    pen = _penalty(axis_dist)
    conf = 0.35 * cc + 0.25 * lc + 0.20 * cs + 0.20 * (1.0 - pen)
    if ridge_resp is not None:
        conf += RIDGE_CONF_WEIGHT * float(np.clip(ridge_resp, 0.0, 1.0))
    return float(np.clip(conf, 0.0, 1.0))


def _candidate_confidence_batch(
    color_dists: np.ndarray,
    y_currs: np.ndarray,
    y_prev: Optional[float],
    comp_scores: np.ndarray,
    axis_dists: np.ndarray,
    ridge_resps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized equivalent of `_candidate_confidence` for one column."""
    color_dists = np.asarray(color_dists, dtype=np.float64)
    y_currs = np.asarray(y_currs, dtype=np.float64)
    comp_scores = np.asarray(comp_scores, dtype=np.float64)
    axis_dists = np.asarray(axis_dists, dtype=np.float64)

    cc = np.exp(-color_dists / 20.0)
    if y_prev is None:
        lc = np.ones_like(cc, dtype=np.float64)
    else:
        lc = np.exp(-np.abs(y_currs - float(y_prev)) / 8.0)
    cs = 1.0 / (1.0 + np.exp(-0.8 * (comp_scores - 2.0)))
    pen = np.exp(-axis_dists / 6.0)
    conf = 0.35 * cc + 0.25 * lc + 0.20 * cs + 0.20 * (1.0 - pen)
    if ridge_resps is not None:
        conf = conf + RIDGE_CONF_WEIGHT * np.clip(np.asarray(ridge_resps, dtype=np.float64), 0.0, 1.0)
    return np.clip(conf, 0.0, 1.0)


def decompose_candidate_confidence_terms(
    color_dist: float,
    y_curr: float,
    y_prev: Optional[float],
    comp_score: float,
    axis_dist: float,
    ridge_resp: Optional[float] = None,
) -> Dict[str, float]:
    """진단용: `_candidate_confidence`와 동일한 입력으로 가중 항목별 기여도 반환."""
    cc = _color_consistency(color_dist)
    lc = _local_continuity(y_curr, y_prev)
    cs_sig = _component_support(comp_score)
    pen = _penalty(axis_dist)
    w_cc = 0.35 * cc
    w_lc = 0.25 * lc
    w_cs = 0.20 * cs_sig
    w_ax = 0.20 * (1.0 - pen)
    ridge_add = 0.0
    if ridge_resp is not None:
        ridge_add = RIDGE_CONF_WEIGHT * float(np.clip(ridge_resp, 0.0, 1.0))
    return {
        "conf_weight_color": float(w_cc),
        "conf_weight_continuity": float(w_lc),
        "conf_weight_component": float(w_cs),
        "conf_weight_axis_penalty": float(w_ax),
        "conf_weight_ridge": float(ridge_add),
        "feat_color_consistency_unit": float(cc),
        "feat_local_continuity_unit": float(lc),
        "feat_component_support_unit": float(cs_sig),
        "feat_axis_penalty_unit": float(pen),
    }


def annotate_raw_candidates_confidence_dump(
    raw_candidates: Dict[int, List[dict]],
    *,
    raw_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    labeled: np.ndarray,
    ridge_map: Optional[np.ndarray] = None,
    roi_lab_l_channel: Optional[np.ndarray] = None,
    gt_y_by_col: Optional[Dict[int, float]] = None,
    roi_h: int,
    upper_band_frac: float = 0.2,
    gt_near_px: float = 5.0,
) -> None:
    """debug 전용: 후보 dict에 component/GT/분해 항목 등을 덮어쓴다. 무거운 경우만 호출."""
    h, w = raw_mask.shape[:2]
    union = np.clip(raw_mask.astype(np.uint8) + skeleton_mask.astype(np.uint8), 0, 1)
    if ridge_map is not None and ridge_map.shape[:2] != (h, w):
        ridge_map = None
    ub = max(1, int(math.ceil(float(roi_h) * float(upper_band_frac))))
    prev_best_y: Optional[float] = None

    for col in range(min(w, max(raw_candidates.keys(), default=-1) + 1)):
        ys = np.where(union[:, col] > 0)[0]
        by_y = {int(c["y"]): c for c in raw_candidates.get(col, [])}
        for y_val in ys:
            y_int = int(y_val)
            cand = by_y.get(y_int)
            if cand is None:
                continue
            cd = float(color_dist_map[y_int, col])
            cs = float(comp_score_map[y_int, col])
            ad = float(axis_dist_map[y_int, col])
            rr = float(ridge_map[y_int, col]) if ridge_map is not None else None
            terms = decompose_candidate_confidence_terms(cd, float(y_int), prev_best_y, cs, ad, ridge_resp=rr)
            for k, v in terms.items():
                cand[k] = float(v)
            cand["component_id"] = int(labeled[y_int, col])
            if roi_lab_l_channel is not None and roi_lab_l_channel.shape[:2] == (h, w):
                cand["local_lab_L"] = float(roi_lab_l_channel[y_int, col])
            cand["mask_union"] = 1.0
            cand["skeleton_hit"] = float(int(skeleton_mask[y_int, col] > 0))
            cand["raw_mask_hit"] = float(int(raw_mask[y_int, col] > 0))
            cand["distance_transform_axis_dist"] = float(ad)
            if rr is not None:
                cand["ridge_response"] = float(rr)
            cand["rank_score_raw_stage"] = float(cand.get("confidence", 0.0))
            cand["is_upper_band"] = bool(y_int < ub)
            if gt_y_by_col is not None:
                gy = gt_y_by_col.get(col)
                if gy is not None:
                    cand["distance_to_gt"] = float(abs(float(y_int) - float(gy)))
                    cand["is_gt_near_px5"] = bool(cand["distance_to_gt"] <= float(gt_near_px))
                else:
                    cand["distance_to_gt"] = None
                    cand["is_gt_near_px5"] = None
            else:
                cand["distance_to_gt"] = None
                cand["is_gt_near_px5"] = None
        cands_col = raw_candidates.get(col, [])
        if cands_col:
            prev_best_y = float(max(cands_col, key=lambda x: float(x["confidence"]))["y"])


def build_raw_candidates(
    raw_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    ridge_map: Optional[np.ndarray] = None,
) -> Dict[int, List[dict]]:
    """
    $12.4-12.5: column별 raw candidate 수집.
    raw_mask와 skeleton_mask 모두에서 후보 픽셀 수집 (합집합).
    column별 top-K 제한 없음.
    ridge_map: ROI와 동일 크기의 [0,1] 세로 능선 응답(없으면 기존과 동일).
    """
    h, w = raw_mask.shape[:2]
    union = np.clip(raw_mask.astype(np.uint8) + skeleton_mask.astype(np.uint8), 0, 1)
    if ridge_map is not None and ridge_map.shape[:2] != (h, w):
        ridge_map = None

    raw_candidates: Dict[int, List[dict]] = {}
    prev_best_y: Optional[float] = None

    for col in range(w):
        ys = np.where(union[:, col] > 0)[0]
        if len(ys) == 0:
            raw_candidates[col] = []
            continue

        ys_int = ys.astype(np.int64, copy=False)
        color_vals = color_dist_map[ys_int, col].astype(np.float64, copy=False)
        comp_vals = comp_score_map[ys_int, col].astype(np.float64, copy=False)
        axis_vals = axis_dist_map[ys_int, col].astype(np.float64, copy=False)
        ridge_vals = (
            ridge_map[ys_int, col].astype(np.float64, copy=False)
            if ridge_map is not None
            else None
        )
        conf_vals = _candidate_confidence_batch(
            color_vals,
            ys_int.astype(np.float64, copy=False),
            prev_best_y,
            comp_vals,
            axis_vals,
            ridge_resps=ridge_vals,
        )

        cands = []
        raw_hits = raw_mask[ys_int, col] > 0
        skel_hits = skeleton_mask[ys_int, col] > 0
        for i, y_int_np in enumerate(ys_int):
            y_int = int(y_int_np)
            cd = float(color_vals[i])
            cs = float(comp_vals[i])
            ad = float(axis_vals[i])
            conf = float(conf_vals[i])
            source = "both" if raw_hits[i] and skel_hits[i] else (
                "raw" if raw_hits[i] else "skeleton"
            )
            cands.append({
                "y": y_int,
                "confidence": conf,
                "color_dist": cd,
                "comp_score": cs,
                "axis_dist": ad,
                "source": source,
            })

        cands.sort(key=lambda c: -c["confidence"])
        raw_candidates[col] = cands

        if cands:
            prev_best_y = float(cands[0]["y"])

    return raw_candidates


def _rolling_nanmedian_1d(x: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    r = w // 2
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for i in range(int(x.size)):
        lo = max(0, i - r)
        hi = min(int(x.size), i + r + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def _fill_nan_linear_1d(x: np.ndarray) -> np.ndarray:
    idx = np.arange(int(x.size), dtype=np.float64)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float64)
    if int(np.sum(mask)) == 1:
        return np.full_like(x, float(x[mask][0]), dtype=np.float64)
    return np.interp(idx, idx[mask], x[mask]).astype(np.float64, copy=False)


def _band_midline_ranges(cols: List[int]) -> List[List[int]]:
    if not cols:
        return []
    cols_sorted = sorted(set(int(c) for c in cols))
    ranges: List[List[int]] = []
    start = prev = cols_sorted[0]
    for c in cols_sorted[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append([int(start), int(prev)])
        start = prev = c
    ranges.append([int(start), int(prev)])
    return ranges


def _peak_apex_double_tip_flags(
    y_top: np.ndarray,
    peak_mask: np.ndarray,
    thickness: np.ndarray,
    *,
    max_tip_distance: int,
) -> Dict[str, Any]:
    """Flag likely false double-tip windows without merging or using GT."""
    n = int(y_top.size)
    if n < 5:
        return {
            "false_double_tip_mask": np.zeros(n, dtype=bool),
            "true_doublet_possible_mask": np.zeros(n, dtype=bool),
            "false_double_tip_ranges": [],
            "true_doublet_possible_ranges": [],
            "pairs": [],
        }
    smooth = _fill_nan_linear_1d(y_top)
    local_min = np.zeros(n, dtype=bool)
    local_min[1:-1] = (
        np.isfinite(y_top[1:-1])
        & (smooth[1:-1] <= smooth[:-2])
        & (smooth[1:-1] <= smooth[2:])
        & peak_mask[1:-1]
    )
    mins = np.nonzero(local_min)[0].astype(int).tolist()
    false_mask = np.zeros(n, dtype=bool)
    true_mask = np.zeros(n, dtype=bool)
    pairs: List[Dict[str, Any]] = []
    max_dist = max(3, int(max_tip_distance))
    for i, left in enumerate(mins):
        for right in mins[i + 1 :]:
            dist = int(right - left)
            if dist <= 1:
                continue
            if dist > max_dist:
                break
            seg = smooth[left : right + 1]
            if seg.size == 0 or not np.any(np.isfinite(seg)):
                continue
            valley_y = float(np.nanmax(seg))
            tip_y = float(min(smooth[left], smooth[right]))
            valley_depth_px = valley_y - tip_y
            thick_seg = thickness[left : right + 1]
            thick_seg = thick_seg[np.isfinite(thick_seg)]
            thick_ref = float(np.median(thick_seg)) if thick_seg.size else 0.0
            shallow_thr = max(4.0, 0.18 * thick_ref)
            is_false_like = valley_depth_px <= shallow_thr
            if is_false_like:
                false_mask[left : right + 1] = True
            else:
                true_mask[left : right + 1] = True
            pairs.append(
                {
                    "left_col": int(left),
                    "right_col": int(right),
                    "tip_distance_cols": int(dist),
                    "valley_depth_px": float(valley_depth_px),
                    "thickness_ref_px": float(thick_ref),
                    "classification": (
                        "FALSE_DOUBLE_TIP_ARTIFACT_LIKELY"
                        if is_false_like
                        else "TRUE_DOUBLET_POSSIBLE"
                    ),
                }
            )
    return {
        "false_double_tip_mask": false_mask,
        "true_doublet_possible_mask": true_mask,
        "false_double_tip_ranges": _band_midline_ranges(np.nonzero(false_mask)[0].astype(int).tolist()),
        "true_doublet_possible_ranges": _band_midline_ranges(np.nonzero(true_mask)[0].astype(int).tolist()),
        "pairs": pairs[:256],
    }


def _peak_apex_evidence_features(
    raw_candidate_mask: np.ndarray,
    *,
    window: int,
    top_percentile: float,
    min_prominence: float,
    min_evidence_pixels: int,
    min_top_support: int,
) -> Dict[str, Any]:
    h, w = raw_candidate_mask.shape[:2]
    radius = max(0, int(window) // 2)
    top_pct = float(np.clip(float(top_percentile), 0.0, 50.0))
    y_top = np.full(int(w), np.nan, dtype=np.float64)
    y_bottom = np.full(int(w), np.nan, dtype=np.float64)
    thickness = np.full(int(w), np.nan, dtype=np.float64)
    evidence_counts = np.zeros(int(w), dtype=np.int64)
    top_support = np.zeros(int(w), dtype=np.int64)
    for col in range(int(w)):
        lo = max(0, col - radius)
        hi = min(int(w), col + radius + 1)
        ys = np.nonzero(raw_candidate_mask[:, lo:hi] > 0)[0]
        evidence_counts[col] = int(ys.size)
        if ys.size == 0:
            continue
        ys_f = ys.astype(np.float64, copy=False)
        yt = float(np.percentile(ys_f, top_pct))
        yb = float(np.percentile(ys_f, 80.0))
        y_top[col] = yt
        y_bottom[col] = yb
        thickness[col] = yb - yt
        top_support[col] = int(np.sum(ys_f <= yt + 2.0))
    smooth_top = _fill_nan_linear_1d(y_top)
    trend_top = _rolling_nanmedian_1d(smooth_top, max(31, int(window) * 9))
    prominence = trend_top - smooth_top
    slope = np.gradient(smooth_top) if int(w) > 1 else np.zeros_like(smooth_top)
    curvature = np.gradient(slope) if int(w) > 1 else np.zeros_like(smooth_top)
    peak_mask = (
        np.isfinite(y_top)
        & (evidence_counts >= int(min_evidence_pixels))
        & (top_support >= int(min_top_support))
        & (prominence >= float(min_prominence))
        & (thickness >= 2.0)
    )
    edge = max(3, int(window))
    if int(w) > 2 * edge:
        peak_mask[:edge] = False
        peak_mask[-edge:] = False
    return {
        "y_top": y_top,
        "y_bottom": y_bottom,
        "thickness": thickness,
        "evidence_counts": evidence_counts,
        "top_support": top_support,
        "smooth_top": smooth_top,
        "prominence": prominence,
        "slope": slope,
        "curvature": curvature,
        "peak_mask": peak_mask,
        "height": int(h),
        "width": int(w),
    }


def add_peak_apex_candidates_to_raw(
    raw_candidates: Dict[int, List[dict]],
    raw_candidate_mask: np.ndarray,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    *,
    window: int = 9,
    top_percentile: float = 10.0,
    min_prominence: float = 8.0,
    min_evidence_pixels: int = 5,
    min_top_support: int = 2,
    max_extra_per_column: int = 1,
    double_tip_guard: bool = True,
) -> Tuple[Dict[int, List[dict]], Dict[str, Any]]:
    """Add optional sharp-peak top-envelope candidates to raw candidates."""
    h, w = raw_candidate_mask.shape[:2]
    feat = _peak_apex_evidence_features(
        raw_candidate_mask,
        window=int(window),
        top_percentile=float(top_percentile),
        min_prominence=float(min_prominence),
        min_evidence_pixels=int(min_evidence_pixels),
        min_top_support=int(min_top_support),
    )
    double_tip = _peak_apex_double_tip_flags(
        feat["y_top"],
        feat["peak_mask"],
        feat["thickness"],
        max_tip_distance=max(6, int(window) * 4),
    )
    false_double_mask = double_tip["false_double_tip_mask"]
    reason_counts: Counter[str] = Counter()
    added_cols: List[int] = []
    peak_window_cols = np.nonzero(feat["peak_mask"])[0].astype(int).tolist()
    for col in range(int(w)):
        if not bool(feat["peak_mask"][col]):
            reason_counts["skipped_not_peak_window"] += 1
            continue
        if bool(double_tip_guard) and bool(false_double_mask[col]):
            reason_counts["skipped_double_tip_guard"] += 1
            continue
        if int(max_extra_per_column) <= 0:
            reason_counts["skipped_max_extra_zero"] += 1
            continue
        yt = feat["y_top"][col]
        if not np.isfinite(yt):
            reason_counts["skipped_invalid_top"] += 1
            continue
        y = int(round(float(yt)))
        if y < 0 or y >= int(h):
            reason_counts["skipped_out_of_bounds"] += 1
            continue
        current = raw_candidates.setdefault(int(col), [])
        if any(str(c.get("source", "")) == "peak_apex" for c in current):
            reason_counts["skipped_max_extra_per_column"] += 1
            continue
        if any(abs(int(c.get("y", -10**9)) - y) <= 1 for c in current):
            reason_counts["skipped_duplicate_y"] += 1
            continue
        nearest = min(current, key=lambda c: abs(int(c.get("y", -10**9)) - y)) if current else None
        conf = max(0.35, float(nearest.get("confidence", 0.0))) if nearest is not None else 0.35
        conf = float(np.clip(conf, 0.0, 0.95))
        cand = {
            "y": int(y),
            "confidence": conf,
            "color_dist": float(color_dist_map[y, col]),
            "comp_score": float(comp_score_map[y, col]),
            "axis_dist": float(axis_dist_map[y, col]),
            "source": "peak_apex",
            "reason": "peak_top_envelope_candidate",
            "peak_apex": True,
            "peak_apex_window": True,
            "peak_apex_false_double_tip_flag": bool(false_double_mask[col]),
            "peak_apex_true_doublet_possible_flag": bool(double_tip["true_doublet_possible_mask"][col]),
            "peak_apex_y_top_robust": float(feat["y_top"][col]),
            "peak_apex_y_bottom_robust": float(feat["y_bottom"][col]),
            "peak_apex_band_thickness": float(feat["thickness"][col]),
            "peak_apex_evidence_pixels": int(feat["evidence_counts"][col]),
            "peak_apex_top_support": int(feat["top_support"][col]),
            "peak_apex_prominence": float(feat["prominence"][col]),
            "peak_apex_slope": float(feat["slope"][col]),
            "peak_apex_curvature": float(feat["curvature"][col]),
        }
        current.append(cand)
        current.sort(key=lambda c: -float(c.get("confidence", 0.0)))
        reason_counts["added"] += 1
        added_cols.append(int(col))
    prom = feat["prominence"][np.isfinite(feat["prominence"])]
    return raw_candidates, {
        "enabled": True,
        "uses_gt": False,
        "uses_source_numeric": False,
        "source": "peak_apex",
        "reason": "peak_top_envelope_candidate",
        "params": {
            "window": int(window),
            "top_percentile": float(top_percentile),
            "min_prominence": float(min_prominence),
            "min_evidence_pixels": int(min_evidence_pixels),
            "min_top_support": int(min_top_support),
            "max_extra_per_column": int(max_extra_per_column),
            "double_tip_guard": bool(double_tip_guard),
        },
        "peak_window_count": int(len(peak_window_cols)),
        "peak_window_ranges": _band_midline_ranges(peak_window_cols),
        "added_candidate_count": int(reason_counts.get("added", 0)),
        "added_columns": int(len(added_cols)),
        "added_column_ranges": _band_midline_ranges(added_cols),
        "preserved_final_count": 0,
        "preserved_columns": 0,
        "preserved_column_ranges": [],
        "selected_columns": 0,
        "selected_column_ranges": [],
        "selected_source_distribution": {},
        "false_double_tip_flags": {
            "guard_enabled": bool(double_tip_guard),
            "false_double_tip_columns": int(np.sum(false_double_mask)),
            "false_double_tip_ranges": double_tip["false_double_tip_ranges"],
            "true_doublet_possible_columns": int(np.sum(double_tip["true_doublet_possible_mask"])),
            "true_doublet_possible_ranges": double_tip["true_doublet_possible_ranges"],
            "pairs": double_tip["pairs"],
        },
        "skip_reason_counts": dict(sorted(reason_counts.items())),
        "apex_prominence_summary": {
            "mean": float(np.mean(prom)) if prom.size else None,
            "median": float(np.median(prom)) if prom.size else None,
            "p90": float(np.percentile(prom, 90.0)) if prom.size else None,
        },
        "apex_reach_gap_before_summary": None,
        "apex_reach_gap_after_summary": None,
    }


def preserve_peak_apex_final_candidates(
    final_candidates: Dict[int, List[dict]],
    raw_candidates: Dict[int, List[dict]],
    *,
    max_extra_per_column: int = 1,
    double_tip_guard: bool = True,
) -> Tuple[Dict[int, List[dict]], Dict[str, Any]]:
    """Preserve at most one peak_apex candidate per final DP column."""
    reason_counts: Counter[str] = Counter()
    preserved_cols: List[int] = []
    for col, raw_col in raw_candidates.items():
        peak_raw = [c for c in raw_col if str(c.get("source", "")) == "peak_apex"]
        if not peak_raw:
            continue
        if bool(double_tip_guard) and any(bool(c.get("peak_apex_false_double_tip_flag")) for c in peak_raw):
            reason_counts["skipped_double_tip_guard"] += 1
            continue
        current = final_candidates.setdefault(int(col), [])
        if any(str(c.get("source", "")) == "peak_apex" for c in current):
            reason_counts["already_in_final"] += 1
            continue
        if int(max_extra_per_column) <= 0:
            reason_counts["skipped_max_extra_zero"] += 1
            continue
        best = max(
            peak_raw,
            key=lambda c: (
                float(c.get("confidence", 0.0)),
                float(c.get("peak_apex_prominence", 0.0)),
            ),
        )
        y = int(best.get("y", -10**9))
        if any(abs(int(c.get("y", -10**9)) - y) <= 1 for c in current):
            reason_counts["duplicate_y_in_final"] += 1
            continue
        current.append(dict(best))
        reason_counts["preserved"] += 1
        preserved_cols.append(int(col))
    return final_candidates, {
        "preserved_final_count": int(reason_counts.get("preserved", 0)),
        "preserved_columns": int(len(preserved_cols)),
        "preserved_column_ranges": _band_midline_ranges(preserved_cols),
        "preserve_reason_counts": dict(sorted(reason_counts.items())),
    }


def _band_midline_peak_guard_mask(
    y_top: np.ndarray,
    *,
    guard_enabled: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = int(y_top.size)
    if not guard_enabled or n == 0:
        return np.zeros(n, dtype=bool), {"enabled": float(bool(guard_enabled))}

    filled = _fill_nan_linear_1d(y_top)
    smooth = _rolling_nanmedian_1d(filled, 9)
    trend = _rolling_nanmedian_1d(smooth, 81)
    prominence = trend - smooth
    slope = np.gradient(smooth)
    curvature = np.gradient(slope)

    prom_vals = prominence[np.isfinite(prominence) & (prominence > 0)]
    slope_vals = np.abs(slope[np.isfinite(slope)])
    curv_vals = np.abs(curvature[np.isfinite(curvature)])

    prom_thr = max(
        12.0,
        float(np.percentile(prom_vals, 85.0)) if prom_vals.size else 12.0,
    )
    prom_strong_thr = max(
        24.0,
        float(np.percentile(prom_vals, 92.0)) if prom_vals.size else 24.0,
    )
    slope_low_thr = max(
        1.5,
        float(np.percentile(slope_vals, 55.0)) if slope_vals.size else 1.5,
    )
    curv_thr = max(
        1.5,
        float(np.percentile(curv_vals, 90.0)) if curv_vals.size else 1.5,
    )

    apex = (
        (prominence >= prom_thr) & (np.abs(slope) <= slope_low_thr)
    ) | (
        (prominence >= prom_strong_thr) & (np.abs(curvature) >= curv_thr)
    )
    apex = np.asarray(apex, dtype=bool)

    # Small dilation protects the immediate apex neighborhood without turning
    # broad peak slopes into a global midline ban.
    if n > 2 and np.any(apex):
        dil = apex.copy()
        dil[1:] |= apex[:-1]
        dil[:-1] |= apex[1:]
        apex = dil

    return apex, {
        "enabled": 1.0,
        "prominence_threshold_px": float(prom_thr),
        "strong_prominence_threshold_px": float(prom_strong_thr),
        "slope_low_threshold_px": float(slope_low_thr),
        "curvature_threshold_px": float(curv_thr),
        "guarded_columns": int(np.sum(apex)),
        "guarded_column_ranges": _band_midline_ranges(np.nonzero(apex)[0].astype(int).tolist()),
    }


def add_band_midline_candidates(
    final_candidates: Dict[int, List[dict]],
    raw_candidate_mask: np.ndarray,
    raw_candidates: Dict[int, List[dict]],
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    *,
    window: int = 3,
    top_percentile: float = 20.0,
    bottom_percentile: float = 80.0,
    min_thickness: float = 3.0,
    max_thickness: float = 120.0,
    max_extra_per_column: int = 1,
    peak_guard: bool = True,
    min_evidence_pixels: int = 3,
    duplicate_y_px: int = 1,
    strict_peak_guard: bool = False,
    peak_guard_curvature_thresh: Optional[float] = None,
    peak_guard_slope_thresh: Optional[float] = None,
    peak_guard_prominence_thresh: Optional[float] = None,
    low_priority_mode: bool = False,
    confidence_multiplier: float = 1.0,
    score_penalty: float = 0.0,
    flat_tail_only: bool = False,
    require_thick_band: bool = False,
    min_band_thickness_for_flat_tail: float = 8.0,
) -> Tuple[Dict[int, List[dict]], Dict[str, Any]]:
    """Add optional stroke-band midline candidates to final DP candidates.

    This is an experimental, GT-free candidate addition. Existing candidates are
    never removed. The source_numeric/GT path is intentionally not accepted here.
    """
    h, w = raw_candidate_mask.shape[:2]
    radius = max(0, int(window) // 2)
    top_pct = float(np.clip(float(top_percentile), 0.0, 100.0))
    bottom_pct = float(np.clip(float(bottom_percentile), 0.0, 100.0))
    if bottom_pct < top_pct:
        top_pct, bottom_pct = bottom_pct, top_pct

    y_top = np.full(int(w), np.nan, dtype=np.float64)
    y_bottom = np.full(int(w), np.nan, dtype=np.float64)
    y_mid = np.full(int(w), np.nan, dtype=np.float64)
    thickness = np.full(int(w), np.nan, dtype=np.float64)
    evidence_counts = np.zeros(int(w), dtype=np.int64)

    for col in range(int(w)):
        lo = max(0, col - radius)
        hi = min(int(w), col + radius + 1)
        ys = np.nonzero(raw_candidate_mask[:, lo:hi] > 0)[0]
        evidence_counts[col] = int(ys.size)
        if ys.size == 0:
            continue
        ys_f = ys.astype(np.float64, copy=False)
        yt = float(np.percentile(ys_f, top_pct))
        yb = float(np.percentile(ys_f, bottom_pct))
        y_top[col] = yt
        y_bottom[col] = yb
        y_mid[col] = 0.5 * (yt + yb)
        thickness[col] = yb - yt

    peak_mask, peak_meta = _band_midline_peak_guard_mask(
        y_top,
        guard_enabled=bool(peak_guard),
    )
    smooth_top = _fill_nan_linear_1d(y_top)
    trend_top = _rolling_nanmedian_1d(smooth_top, 61)
    prominence = trend_top - smooth_top
    slope = np.gradient(smooth_top) if int(w) > 1 else np.zeros_like(smooth_top)
    curvature = np.gradient(slope) if int(w) > 1 else np.zeros_like(smooth_top)
    finite_prom = prominence[np.isfinite(prominence)]
    finite_slope = np.abs(slope[np.isfinite(slope)])
    finite_curv = np.abs(curvature[np.isfinite(curvature)])
    prom_thr = (
        float(peak_guard_prominence_thresh)
        if peak_guard_prominence_thresh is not None
        else (float(np.percentile(finite_prom, 72.0)) if finite_prom.size else 12.0)
    )
    slope_thr = (
        float(peak_guard_slope_thresh)
        if peak_guard_slope_thresh is not None
        else (float(np.percentile(finite_slope, 62.0)) if finite_slope.size else 1.5)
    )
    curv_thr = (
        float(peak_guard_curvature_thresh)
        if peak_guard_curvature_thresh is not None
        else (float(np.percentile(finite_curv, 78.0)) if finite_curv.size else 1.5)
    )
    strict_mask = np.zeros(int(w), dtype=bool)
    if bool(strict_peak_guard):
        strict_mask = (
            peak_mask
            | ((prominence >= prom_thr) & (np.abs(slope) >= slope_thr))
            | ((prominence >= prom_thr) & (np.abs(curvature) >= curv_thr))
        )
        if np.any(strict_mask):
            dil = strict_mask.copy()
            for _ in range(3):
                dil[1:] |= strict_mask[:-1]
                dil[:-1] |= strict_mask[1:]
                strict_mask = dil.copy()
    effective_peak_mask = peak_mask | strict_mask

    flat_prom_thr = float(np.percentile(finite_prom, 58.0)) if finite_prom.size else 0.0
    flat_slope_thr = float(np.percentile(finite_slope, 55.0)) if finite_slope.size else 1.0
    flat_curv_thr = float(np.percentile(finite_curv, 62.0)) if finite_curv.size else 1.0
    flat_tail_mask = (
        np.isfinite(thickness)
        & (prominence <= flat_prom_thr)
        & (np.abs(slope) <= flat_slope_thr)
        & (np.abs(curvature) <= flat_curv_thr)
    )
    if bool(require_thick_band) or bool(flat_tail_only):
        flat_tail_mask &= thickness >= float(min_band_thickness_for_flat_tail)

    reason_counts: Counter[str] = Counter()
    added_columns: List[int] = []
    added_thicknesses: List[float] = []
    added_confidences: List[float] = []
    considered = 0

    for col in range(int(w)):
        considered += 1
        if int(max_extra_per_column) <= 0:
            reason_counts["skipped_max_extra_zero"] += 1
            continue
        if evidence_counts[col] < int(min_evidence_pixels):
            reason_counts["skipped_sparse_evidence"] += 1
            continue
        if not np.isfinite(y_mid[col]) or not np.isfinite(thickness[col]):
            reason_counts["skipped_invalid_band"] += 1
            continue
        bt = float(thickness[col])
        if bt < float(min_thickness):
            reason_counts["skipped_thin_band"] += 1
            continue
        if bt > float(max_thickness):
            reason_counts["skipped_too_thick_band"] += 1
            continue
        if bool(effective_peak_mask[col]):
            reason_counts["skipped_peak_guard"] += 1
            continue
        if bool(flat_tail_only) and not bool(flat_tail_mask[col]):
            reason_counts["skipped_flat_tail_guard"] += 1
            continue

        y = int(round(float(y_mid[col])))
        if y < 0 or y >= int(h):
            reason_counts["skipped_out_of_bounds"] += 1
            continue

        current = final_candidates.setdefault(int(col), [])
        if any(abs(int(c.get("y", -10**9)) - y) <= int(duplicate_y_px) for c in current):
            reason_counts["skipped_duplicate_y"] += 1
            continue

        existing_band = [
            c for c in current if str(c.get("source", "")) == "band_midline"
        ]
        if len(existing_band) >= int(max_extra_per_column):
            reason_counts["skipped_max_extra_per_column"] += 1
            continue

        raw_col = raw_candidates.get(int(col)) or []
        nearest = None
        if raw_col:
            nearest = min(raw_col, key=lambda c: abs(int(c.get("y", -10**9)) - y))

        if nearest is not None:
            conf = max(0.25, float(nearest.get("confidence", 0.0)))
        else:
            conf = 0.25
        conf_mult = float(confidence_multiplier)
        if bool(low_priority_mode):
            conf_mult = min(conf_mult, 0.55)
        conf *= conf_mult
        conf = float(np.clip(conf, 0.0, 0.98))
        comp_score = float(comp_score_map[y, col]) - float(score_penalty)

        cand = {
            "y": int(y),
            "confidence": conf,
            "color_dist": float(color_dist_map[y, col]),
            "comp_score": float(comp_score),
            "axis_dist": float(axis_dist_map[y, col]),
            "source": "band_midline",
            "reason": "stroke_centerline_candidate",
            "band_midline": True,
            "band_midline_mode": (
                "flat_tail_only"
                if bool(flat_tail_only)
                else "low_priority"
                if bool(low_priority_mode)
                else "strict_peak_guard"
                if bool(strict_peak_guard)
                else "baseline_v1"
            ),
            "band_midline_peak_region": bool(effective_peak_mask[col]),
            "band_midline_flat_tail_region": bool(flat_tail_mask[col]),
            "band_top_y": float(y_top[col]),
            "band_bottom_y": float(y_bottom[col]),
            "band_thickness": float(bt),
            "band_evidence_pixels": int(evidence_counts[col]),
            "band_local_slope": float(slope[col]),
            "band_local_curvature": float(curvature[col]),
            "band_local_prominence": float(prominence[col]),
        }
        current.append(cand)
        reason_counts["added"] += 1
        added_columns.append(int(col))
        added_thicknesses.append(float(bt))
        added_confidences.append(float(conf))

    valid_thickness = thickness[np.isfinite(thickness)]
    meta: Dict[str, Any] = {
        "enabled": True,
        "uses_source_numeric": False,
        "uses_gt": False,
        "source": "band_midline",
        "reason": "stroke_centerline_candidate",
        "mode": (
            "flat_tail_only"
            if bool(flat_tail_only)
            else "low_priority"
            if bool(low_priority_mode)
            else "strict_peak_guard"
            if bool(strict_peak_guard)
            else "baseline_v1"
        ),
        "params": {
            "window": int(window),
            "top_percentile": float(top_pct),
            "bottom_percentile": float(bottom_pct),
            "min_thickness": float(min_thickness),
            "max_thickness": float(max_thickness),
            "max_extra_per_column": int(max_extra_per_column),
            "peak_guard": bool(peak_guard),
            "min_evidence_pixels": int(min_evidence_pixels),
            "duplicate_y_px": int(duplicate_y_px),
            "strict_peak_guard": bool(strict_peak_guard),
            "peak_guard_curvature_thresh": peak_guard_curvature_thresh,
            "peak_guard_slope_thresh": peak_guard_slope_thresh,
            "peak_guard_prominence_thresh": peak_guard_prominence_thresh,
            "low_priority_mode": bool(low_priority_mode),
            "confidence_multiplier": float(confidence_multiplier),
            "score_penalty": float(score_penalty),
            "flat_tail_only": bool(flat_tail_only),
            "require_thick_band": bool(require_thick_band),
            "min_band_thickness_for_flat_tail": float(min_band_thickness_for_flat_tail),
        },
        "candidate_columns_considered": int(considered),
        "added_candidate_count": int(reason_counts.get("added", 0)),
        "added_columns": int(len(added_columns)),
        "added_column_ranges": _band_midline_ranges(added_columns),
        "skipped_by_peak_guard": int(reason_counts.get("skipped_peak_guard", 0)),
        "skipped_by_thin_band": int(reason_counts.get("skipped_thin_band", 0)),
        "skipped_by_sparse_evidence": int(reason_counts.get("skipped_sparse_evidence", 0)),
        "skipped_by_too_thick_band": int(reason_counts.get("skipped_too_thick_band", 0)),
        "skipped_by_duplicate_y": int(reason_counts.get("skipped_duplicate_y", 0)),
        "skipped_by_flat_tail_guard": int(reason_counts.get("skipped_flat_tail_guard", 0)),
        "skipped_by_low_priority": 0,
        "skip_reason_counts": dict(sorted(reason_counts.items())),
        "peak_guard_meta": {
            **peak_meta,
            "strict_peak_guard": bool(strict_peak_guard),
            "strict_guarded_columns": int(np.sum(strict_mask)),
            "effective_guarded_columns": int(np.sum(effective_peak_mask)),
            "effective_guarded_column_ranges": _band_midline_ranges(
                np.nonzero(effective_peak_mask)[0].astype(int).tolist()
            ),
            "strict_prominence_threshold_px": float(prom_thr),
            "strict_slope_threshold_px": float(slope_thr),
            "strict_curvature_threshold_px": float(curv_thr),
        },
        "flat_tail_guard_meta": {
            "enabled": bool(flat_tail_only),
            "require_thick_band": bool(require_thick_band),
            "flat_tail_columns": int(np.sum(flat_tail_mask)),
            "flat_tail_column_ranges": _band_midline_ranges(
                np.nonzero(flat_tail_mask)[0].astype(int).tolist()
            ),
            "prominence_threshold_px": float(flat_prom_thr),
            "slope_threshold_px": float(flat_slope_thr),
            "curvature_threshold_px": float(flat_curv_thr),
            "min_band_thickness_for_flat_tail": float(min_band_thickness_for_flat_tail),
        },
        "band_thickness_summary": {
            "valid_columns": int(valid_thickness.size),
            "mean": float(np.mean(valid_thickness)) if valid_thickness.size else None,
            "median": float(np.median(valid_thickness)) if valid_thickness.size else None,
            "p90": float(np.percentile(valid_thickness, 90.0)) if valid_thickness.size else None,
            "added_mean": float(np.mean(added_thicknesses)) if added_thicknesses else None,
            "added_median": float(np.median(added_thicknesses)) if added_thicknesses else None,
        },
        "added_confidence_summary": {
            "mean": float(np.mean(added_confidences)) if added_confidences else None,
            "median": float(np.median(added_confidences)) if added_confidences else None,
            "min": float(np.min(added_confidences)) if added_confidences else None,
            "max": float(np.max(added_confidences)) if added_confidences else None,
        },
        "selected_columns": 0,
        "selected_column_ranges": [],
        "selected_source_distribution": {},
    }
    return final_candidates, meta


def filter_candidates(
    raw_candidates: Dict[int, List[dict]],
    *,
    min_conf_keep: float = MIN_CONF_KEEP,
    max_candidates_after_filter: int = MAX_CANDIDATES_AFTER_FILTER,
    disable_envelope_bonus: bool = False,
    debug_gt_y_by_col: Optional[Dict[int, float]] = None,
    debug_gt_near_px: float = 5.0,
    debug_sink: Optional[Dict[str, object]] = None,
    debug_keep_gt_near: bool = False,
    enable_y_diversity: bool = False,
    y_diversity_bins: int = 8,
    debug_rank_breakdown: bool = False,
    debug_rank_breakdown_max_columns: int = 64,
    enable_source_balance: bool = False,
    source_balance_raw_quota: int = 2,
    enable_local_evidence_rank: bool = False,
    local_evidence_weight: float = LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT,
    local_evidence_tau_px: float = LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT,
    enable_column_rank_normalization: bool = False,
    enable_evidence_aware_preserve: bool = False,
    evidence_preserve_bins: int = 6,
    evidence_preserve_per_bin: int = 1,
    evidence_preserve_max_upper_frac: float = 0.5,
    filter_roi_h: Optional[int] = None,
) -> Dict[int, List[dict]]:
    """$12.9: confidence >= 0.25, column별 max MAX_CANDIDATES_AFTER_FILTER."""
    filtered: Dict[int, List[dict]] = {}
    reason_counts: Dict[str, int] = {}
    gt_reason_counts: Dict[str, int] = {}
    gt_near_removed_cols: List[int] = []
    gt_near_survived_cols: List[int] = []
    col_dbg: Dict[str, dict] = {}

    agg_ev: Optional[Dict[str, Any]] = None
    if debug_sink is not None and enable_evidence_aware_preserve:
        bn = max(1, int(evidence_preserve_bins))
        agg_ev = {
            "weighted_upper_before": 0.0,
            "weighted_upper_after": 0.0,
            "slot_total": 0,
            "preserved_by_bin_count": [0] * bn,
            "evidence_preserved_candidate_count": 0,
            "preserved_src": {},
        }

    def _inc(d: Dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    def _is_gt_near(col: int, cand: dict) -> bool:
        if debug_gt_y_by_col is None:
            return False
        g = debug_gt_y_by_col.get(col)
        if g is None:
            return False
        return abs(float(cand.get("y", 0.0)) - float(g)) <= float(debug_gt_near_px)

    for col, cands in raw_candidates.items():
        if not cands:
            filtered[col] = []
            continue
        valid: List[dict] = []
        for c in cands:
            try:
                _ = int(c["y"])
                cf = float(c["confidence"])
                if not np.isfinite(cf):
                    raise ValueError("nan confidence")
            except Exception:
                _inc(reason_counts, "invalid_candidate")
                if _is_gt_near(col, c):
                    _inc(gt_reason_counts, "invalid_candidate")
                continue
            valid.append(c)

        kept = [c for c in valid if float(c["confidence"]) >= float(min_conf_keep)]
        for c in valid:
            if float(c["confidence"]) < float(min_conf_keep):
                _inc(reason_counts, "below_min_conf_keep")
                if _is_gt_near(col, c):
                    _inc(gt_reason_counts, "below_min_conf_keep")
        if not kept:
            filtered[col] = []
            if debug_gt_y_by_col is not None:
                had_gt_near = any(_is_gt_near(col, c) for c in valid)
                if had_gt_near:
                    gt_near_removed_cols.append(int(col))
                    col_dbg[str(int(col))] = {
                        "top_raw_gt_near_rank_before_filter": None,
                        "top_raw_gt_near_confidence": None,
                        "top_raw_gt_near_score_key": None,
                        "filtered_y_list": [],
                    }
            continue
        if enable_column_rank_normalization:
            srn = sorted(kept, key=lambda c: -float(c["confidence"]))
            nk = len(srn)
            for i, c in enumerate(srn):
                rp = 1.0 if nk <= 1 else float(nk - 1 - i) / float(nk - 1)
                if "confidence_before_column_rank_norm" not in c:
                    c["confidence_before_column_rank_norm"] = float(c["confidence"])
                c["confidence"] = float(rp)
        n_raw = len(valid)
        y_top = min(int(c["y"]) for c in valid)
        y_bottom = max(int(c["y"]) for c in valid)
        span = y_bottom - y_top
        use_env = (
            (not disable_envelope_bonus)
            and n_raw >= ENVELOPE_SORT_MIN_CANDIDATES
            and span >= ENVELOPE_SORT_MIN_SPAN_PX
        )
        roi_w_fb = int(max(raw_candidates.keys(), default=-1)) + 1
        lev_ref: Optional[float] = None
        lev_w = float(local_evidence_weight) if enable_local_evidence_rank else 0.0
        lev_tau = float(local_evidence_tau_px)
        if enable_local_evidence_rank and lev_w > 0.0:
            lev_ref = neighbor_conf_top1_y_median(raw_candidates, col, roi_w_fb)

        def _rank_components(c: dict) -> Tuple[float, float, float, float]:
            conf_f = float(c["confidence"])
            env_b = 0.0
            if use_env:
                dy = int(c["y"]) - y_top
                env_b = ENVELOPE_SORT_BONUS * math.exp(-max(0, dy) / ENVELOPE_SORT_DECAY_PX)
            lev_b = 0.0
            if lev_ref is not None and lev_w > 0.0:
                lev_b = lev_w * math.exp(-abs(float(c["y"]) - lev_ref) / max(lev_tau, 1e-6))
            return conf_f, env_b, lev_b, conf_f + env_b + lev_b

        conf_sorted = sorted(kept, key=lambda c: -float(c["confidence"]))
        conf_top_set = {id(c) for c in conf_sorted[: int(max_candidates_after_filter)]}
        rank_sorted = sorted(kept, key=lambda c: -_rank_components(c)[3])

        use_evidence_preserve = bool(enable_evidence_aware_preserve) and int(evidence_preserve_bins) >= 1
        roi_h_eff = (
            int(filter_roi_h)
            if filter_roi_h is not None and int(filter_roi_h) > 0
            else max((int(c["y"]) for c in kept), default=0) + 1
        )
        roi_h_eff = max(roi_h_eff, 1)

        if use_evidence_preserve:
            ranked, ev_meta = _evidence_aware_pick_filtered(
                col=int(col),
                kept=kept,
                rank_sorted=rank_sorted,
                raw_candidates=raw_candidates,
                roi_w=int(roi_w_fb),
                roi_h=int(roi_h_eff),
                max_k=int(max_candidates_after_filter),
                preserve_bins=int(evidence_preserve_bins),
                preserve_per_bin=int(evidence_preserve_per_bin),
                preserve_max_upper_frac=float(evidence_preserve_max_upper_frac),
                rank_score_fn=lambda c: float(_rank_components(c)[3]),
            )
            if agg_ev is not None:
                rk = len(ranked)
                agg_ev["weighted_upper_before"] += float(ev_meta["upper_frac_before_cap"]) * rk
                agg_ev["weighted_upper_after"] += float(ev_meta["upper_frac_after_cap"]) * rk
                agg_ev["slot_total"] += rk
                agg_ev["evidence_preserved_candidate_count"] += int(
                    ev_meta["evidence_preserved_in_selection"]
                )
                pbb = ev_meta["preserved_by_bin"]
                for i, v in enumerate(pbb):
                    if i < len(agg_ev["preserved_by_bin_count"]):
                        agg_ev["preserved_by_bin_count"][i] += int(v)
                pid = ev_meta["preserved_ids"]
                for c in ranked:
                    if id(c) in pid:
                        sk = str(c.get("source", ""))
                        agg_ev["preserved_src"][sk] = agg_ev["preserved_src"].get(sk, 0) + 1
        else:
            kept.sort(key=lambda c: -_rank_components(c)[3])
            ranked = kept[: int(max_candidates_after_filter)]
        if enable_source_balance and ranked:
            quota = max(0, int(source_balance_raw_quota))
            if quota > 0:
                raw_pool = [c for c in kept if str(c.get("source", "")) == "raw"]
                raw_ranked = [c for c in ranked if str(c.get("source", "")) == "raw"]
                need = max(0, quota - len(raw_ranked))
                if need > 0 and raw_pool:
                    add_raw = [c for c in raw_pool if id(c) not in {id(x) for x in ranked}]
                    add_raw = sorted(add_raw, key=lambda c: -float(c.get("confidence", 0.0)))
                    for rc in add_raw[:need]:
                        ranked.append(rc)
                    if len(ranked) > int(max_candidates_after_filter):
                        # raw quota를 보존하면서 non-raw 낮은 점수를 제거
                        fixed_raw_ids = {id(c) for c in ranked if str(c.get("source", "")) == "raw"}  
                        non_raw = [c for c in ranked if id(c) not in fixed_raw_ids]
                        non_raw = sorted(non_raw, key=lambda c: float(c.get("confidence", 0.0)))
                        while len(ranked) > int(max_candidates_after_filter) and non_raw:
                            rm = non_raw.pop(0)
                            ranked.remove(rm)
                    if len(ranked) > int(max_candidates_after_filter):
                        ranked = sorted(ranked, key=lambda c: -float(c.get("confidence", 0.0)))[: int(max_candidates_after_filter)]
        if enable_y_diversity and kept:
            bins = max(2, int(y_diversity_bins))
            y_lo = min(int(c["y"]) for c in kept)
            y_hi = max(int(c["y"]) for c in kept)
            if y_hi > y_lo:
                # 1차: bin별 최고 점수 1개씩 보존
                winners: Dict[int, dict] = {}
                for c in kept:
                    yy = int(c["y"])
                    bi = int((yy - y_lo) * bins / max(1, (y_hi - y_lo + 1)))
                    bi = max(0, min(bins - 1, bi))
                    prev = winners.get(bi)
                    if prev is None:
                        winners[bi] = c
                    else:
                        _, _, _, psc = _rank_components(prev)
                        _, _, _, csc = _rank_components(c)
                        if csc > psc:
                            winners[bi] = c
                diverse = list(winners.values())
                diverse_ids = {id(c) for c in diverse}
                fill = [c for c in kept if id(c) not in diverse_ids]
                fill = sorted(fill, key=lambda c: -float(c["confidence"]))
                merged = diverse + fill
                merged.sort(key=lambda c: -_rank_components(c)[3])
                ranked = merged[: int(max_candidates_after_filter)]
        ranked_set = {id(c) for c in ranked}
        for c in kept[int(max_candidates_after_filter):]:
            _inc(reason_counts, "outside_top16_after_rank")
            if _is_gt_near(col, c):
                _inc(gt_reason_counts, "outside_top16_after_rank")
            if use_env and id(c) in conf_top_set and id(c) not in ranked_set:
                _inc(reason_counts, "envelope_bonus_rank_loss")
                if _is_gt_near(col, c):
                    _inc(gt_reason_counts, "envelope_bonus_rank_loss")

        # 진단 옵션: GT-near 후보를 filtered 단계에 강제 보존
        if debug_keep_gt_near and debug_gt_y_by_col is not None:
            gt_nears = [c for c in conf_sorted if _is_gt_near(col, c)]
            if gt_nears:
                best_gt = gt_nears[0]
                if id(best_gt) not in ranked_set:
                    ranked.append(best_gt)
                    non_gt = [c for c in ranked if not _is_gt_near(col, c)]
                    if len(ranked) > int(max_candidates_after_filter) and non_gt:
                        worst_non_gt = min(non_gt, key=lambda c: float(c["confidence"]))
                        ranked.remove(worst_non_gt)
                    ranked = sorted(ranked, key=lambda c: -float(c["confidence"]))[: int(max_candidates_after_filter)]
                    ranked_set = {id(c) for c in ranked}

        filtered[col] = ranked

        if debug_gt_y_by_col is not None:
            gt_raw = [c for c in conf_sorted if _is_gt_near(col, c)]
            had_gt_near = bool(gt_raw)
            survived_gt_near = any(_is_gt_near(col, c) for c in ranked)
            if had_gt_near and survived_gt_near:
                gt_near_survived_cols.append(int(col))
            elif had_gt_near and not survived_gt_near:
                gt_near_removed_cols.append(int(col))
            if had_gt_near:
                top_raw = gt_raw[0]
                top_rank = next((i + 1 for i, c in enumerate(conf_sorted) if id(c) == id(top_raw)), None)
                bonus_key = float(_rank_components(top_raw)[3])
                col_dbg[str(int(col))] = {
                    "top_raw_gt_near_rank_before_filter": top_rank,
                    "top_raw_gt_near_confidence": float(top_raw["confidence"]),
                    "top_raw_gt_near_score_key": float(bonus_key),
                    "filtered_y_list": [int(x["y"]) for x in ranked],
                    "filtered_conf_list": [float(x["confidence"]) for x in ranked],
                }
            if debug_rank_breakdown and len(col_dbg) <= int(debug_rank_breakdown_max_columns):
                breakdown = []
                for i, c in enumerate(kept, 1):
                    bc, env_bonus, lev_bonus, total_score = _rank_components(c)
                    breakdown.append(
                        {
                            "rank_after_sort": int(i),
                            "y": int(c["y"]),
                            "base_confidence": float(bc),
                            "envelope_bonus": float(env_bonus),
                            "local_evidence_bonus": float(lev_bonus),
                            "final_filter_rank_score": float(total_score),
                            "source": str(c.get("source", "")),
                            "comp_score": float(c.get("comp_score", 0.0)),
                            "is_gt_near": bool(_is_gt_near(col, c)),
                            "survived_filtered": bool(id(c) in ranked_set),
                        }
                    )
                d = col_dbg.get(str(int(col)), {})
                d["rank_breakdown"] = breakdown
                col_dbg[str(int(col))] = d

    # placeholder reason keys for unified downstream schema
    for k in ("source_priority_loss", "y_boundary_filter", "component_filter", "duplicate_or_bucket_removed", "unknown_removed"):
        reason_counts.setdefault(k, 0)
        gt_reason_counts.setdefault(k, 0)
    if debug_sink is not None:
        debug_sink["filter_removal_reason_counts"] = dict(sorted(reason_counts.items()))
        debug_sink["gt_near_filter_removal_reason_counts"] = dict(sorted(gt_reason_counts.items()))
        debug_sink["gt_near_removed_columns"] = sorted(gt_near_removed_cols)
        debug_sink["gt_near_survived_columns"] = sorted(gt_near_survived_cols)
        debug_sink["gt_near_removed_columns_count"] = int(len(gt_near_removed_cols))
        debug_sink["gt_near_survived_columns_count"] = int(len(gt_near_survived_cols))
        if col_dbg:
            debug_sink["gt_near_column_debug"] = col_dbg
        src_all: Dict[str, int] = {}
        src_top1: Dict[str, int] = {}
        src_filtered: Dict[str, int] = {}
        for col, cands in raw_candidates.items():
            for c in cands:
                k = str(c.get("source", ""))
                src_all[k] = src_all.get(k, 0) + 1
            if cands:
                top = max(cands, key=lambda x: float(x.get("confidence", 0.0)))
                tk = str(top.get("source", ""))
                src_top1[tk] = src_top1.get(tk, 0) + 1
        for cands in filtered.values():
            for c in cands:
                k = str(c.get("source", ""))
                src_filtered[k] = src_filtered.get(k, 0) + 1
        debug_sink["raw_source_distribution"] = dict(sorted(src_all.items()))
        debug_sink["raw_top1_source_distribution"] = dict(sorted(src_top1.items()))
        debug_sink["filtered_source_distribution"] = dict(sorted(src_filtered.items()))

        debug_sink["evidence_preserve_enabled"] = bool(enable_evidence_aware_preserve)
        if enable_evidence_aware_preserve and agg_ev is not None:
            st = float(max(int(agg_ev["slot_total"]), 1))
            debug_sink["preserve_bins"] = int(evidence_preserve_bins)
            debug_sink["preserve_per_bin"] = int(evidence_preserve_per_bin)
            debug_sink["preserve_max_upper_frac"] = float(evidence_preserve_max_upper_frac)
            debug_sink["preserved_by_bin_count"] = list(agg_ev["preserved_by_bin_count"])
            debug_sink["selected_upper_fraction_before_cap"] = float(
                agg_ev["weighted_upper_before"] / st
            )
            debug_sink["selected_upper_fraction_after_cap"] = float(
                agg_ev["weighted_upper_after"] / st
            )
            debug_sink["evidence_preserved_candidate_count"] = int(
                agg_ev["evidence_preserved_candidate_count"]
            )
            debug_sink["evidence_preserved_source_distribution"] = dict(
                sorted(agg_ev["preserved_src"].items())
            )

        if debug_gt_y_by_col is not None:
            roi_w_tot = int(max(filtered.keys(), default=-1)) + 1
            dists_f: List[float] = []
            for ci in range(roi_w_tot):
                g = debug_gt_y_by_col.get(ci)
                if g is None:
                    continue
                lst = filtered.get(ci, [])
                if not lst:
                    continue
                dists_f.append(
                    min(abs(float(c["y"]) - float(g)) for c in lst)
                )
            if dists_f:
                arr = np.asarray(dists_f, dtype=np.float64)
                debug_sink["filtered_gt_near_recall_px3"] = float(np.mean(arr <= 3.0))
                debug_sink["filtered_gt_near_recall_px5"] = float(np.mean(arr <= 5.0))
                debug_sink["filtered_gt_near_recall_px10"] = float(np.mean(arr <= 10.0))
    return filtered


def _column_upper_y_from_filtered(cands: List[dict]) -> Optional[int]:
    """열 내 후보 중 가장 위쪽 픽셀(y 최소). 이웃 보간 시 [0] 신뢰도 1순위 대신 윤곽 기준."""
    if not cands:
        return None
    return min(int(c["y"]) for c in cands)


def _interpolate_from_neighbors(
    filtered: Dict[int, List[dict]],
    col: int,
    w: int,
    roi_height: int = 0,
) -> Optional[dict]:
    """$12.7: +/-3 column에서 보조 후보 linear interpolation.

    마스크 없는 가장자리 열: 이웃 열의 윤곽선 y(열 내 y 최소)로 선형 외삽해
    신뢰도 1순위 y만 복사할 때 생기는 수평 구간을 줄인다.
    """
    left_y: Optional[float] = None
    right_y: Optional[float] = None
    left_col: Optional[int] = None
    right_col: Optional[int] = None

    for offset in range(1, NEIGHBOR_RADIUS + 1):
        lc = col - offset
        if lc >= 0 and filtered.get(lc) and left_y is None:
            uy = _column_upper_y_from_filtered(filtered[lc])
            if uy is not None:
                left_y = float(uy)
                left_col = lc
        rc = col + offset
        if rc < w and filtered.get(rc) and right_y is None:
            uy = _column_upper_y_from_filtered(filtered[rc])
            if uy is not None:
                right_y = float(uy)
                right_col = rc

    if left_y is not None and right_y is not None and left_col is not None and right_col is not None:
        t = float(col - left_col) / float(right_col - left_col)
        interp_y = left_y + t * (right_y - left_y)
        iy = int(round(interp_y))
        if roi_height > 0:
            iy = int(np.clip(iy, 0, roi_height - 1))
        return {"y": iy, "confidence": 0.3, "source": "interpolated",
                "color_dist": 0.0, "comp_score": 0.0, "axis_dist": 99.0}
    if left_y is not None and left_col is not None:
        lc2 = left_col - 1
        if (
            roi_height > 0
            and lc2 >= 0
            and filtered.get(lc2)
            and left_col > lc2
        ):
            uy2 = _column_upper_y_from_filtered(filtered[lc2])
            if uy2 is not None:
                slope = (left_y - float(uy2)) / float(left_col - lc2)
                iy = int(round(left_y + slope * float(col - left_col)))
                iy = int(np.clip(iy, 0, roi_height - 1))
                return {"y": iy, "confidence": 0.25, "source": "neighbor_left_extrap",
                        "color_dist": 0.0, "comp_score": 0.0, "axis_dist": 99.0}
        return {"y": int(round(left_y)), "confidence": 0.25, "source": "neighbor_left",
                "color_dist": 0.0, "comp_score": 0.0, "axis_dist": 99.0}
    if right_y is not None and right_col is not None:
        rc2 = right_col + 1
        if (
            roi_height > 0
            and rc2 < w
            and filtered.get(rc2)
            and rc2 > right_col
        ):
            uy2 = _column_upper_y_from_filtered(filtered[rc2])
            if uy2 is not None:
                slope = (float(uy2) - right_y) / float(rc2 - right_col)
                iy = int(round(right_y - slope * float(right_col - col)))
                iy = int(np.clip(iy, 0, roi_height - 1))
                return {"y": iy, "confidence": 0.25, "source": "neighbor_right_extrap",
                        "color_dist": 0.0, "comp_score": 0.0, "axis_dist": 99.0}
        return {"y": int(round(right_y)), "confidence": 0.25, "source": "neighbor_right",
                "color_dist": 0.0, "comp_score": 0.0, "axis_dist": 99.0}
    return None


def skeleton_column_hint_y(skeleton: np.ndarray) -> np.ndarray:
    """열당 스켈레톤 픽셀 y의 중앙값(없으면 nan). DP 최종 후보 정렬 보조."""
    h, w = skeleton.shape[:2]
    out = np.full(w, np.nan, dtype=np.float64)
    for col in range(w):
        ys = np.where(skeleton[:, col] > 0)[0]
        if ys.size:
            out[col] = float(np.median(ys.astype(np.float64)))
    return out


def dp_transition_window_width(plot_width: int) -> int:
    """`dp_trace` 열 간 허용 |dy| 상한과 동일 — 브리지 후보 간격에 사용."""
    return max(28, round(0.035 * plot_width))


def _dedupe_candidates_max_conf(cands: List[dict]) -> List[dict]:
    best: Dict[int, dict] = {}
    for c in cands:
        y = int(c["y"])
        prev = best.get(y)
        if prev is None or float(c["confidence"]) > float(prev["confidence"]):
            best[y] = c
    return list(best.values())


def _trim_candidates_after_bridge(
    cands: List[dict],
    col: int,
    max_for_dp: int,
    skeleton_hint_y: Optional[np.ndarray],
    expanded_neighbors: Dict[int, List[dict]],
    roi_w: int,
    W: int,
    max_dp_bridge_frac: Optional[float] = None,
    expanded_neighbor_measured_y: Optional[Dict[int, np.ndarray]] = None,
) -> List[dict]:
    """브리지 행은 신뢰도만으로 잘리지 않도록 일부 슬롯 확보."""
    uniq = _dedupe_candidates_max_conf(cands)
    measured = [c for c in uniq if c.get("source") != "dp_bridge"]
    bridges = [c for c in uniq if c.get("source") == "dp_bridge"]

    def rank_meas(c: dict) -> float:
        r = float(c["confidence"])
        if skeleton_hint_y is not None and col < skeleton_hint_y.shape[0]:
            gy = float(skeleton_hint_y[col])
            if not np.isnan(gy):
                r += 0.16 * float(math.exp(-abs(float(c["y"]) - gy) / 11.0))
        return r

    measured.sort(key=lambda c: -rank_meas(c))

    def bridge_bonus(y: int) -> float:
        b = 0.0
        for dc in (-1, 1):
            nc = col + dc
            if nc < 0 or nc >= roi_w:
                continue
            if expanded_neighbor_measured_y is not None:
                ys = expanded_neighbor_measured_y.get(nc)
                if ys is not None and ys.size:
                    left = int(np.searchsorted(ys, int(y) - int(W), side="left"))
                    if left < ys.size and int(ys[left]) <= int(y) + int(W):
                        b = max(b, 0.25)
                continue
            for oc in expanded_neighbors.get(nc, []):
                if oc.get("source") == "dp_bridge":
                    continue
                if abs(int(oc["y"]) - y) <= W:
                    b = max(b, 0.25)
        return b

    bridges.sort(key=lambda c: -(float(c["confidence"]) + bridge_bonus(int(c["y"]))))

    if not measured:
        return bridges[:max_for_dp]
    if not bridges:
        return measured[:max_for_dp]

    n_bridge_cap = min(4, max_for_dp // 2)
    if max_dp_bridge_frac is not None:
        mf = float(min(1.0, max(0.0, max_dp_bridge_frac)))
        n_bridge_cap = min(n_bridge_cap, int(math.ceil(float(max_for_dp) * mf)))
    n_meas_target = max(1, max_for_dp - n_bridge_cap)
    out = measured[: min(len(measured), n_meas_target)]
    ys_have = {int(c["y"]) for c in out}
    for b in bridges:
        if len(out) >= max_for_dp:
            break
        yi = int(b["y"])
        if yi in ys_have:
            continue
        out.append(b)
        ys_have.add(yi)
    return out


def _bridge_confidence_batch(
    color_dists: np.ndarray,
    comp_scores: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of the dp_bridge synthetic confidence formula."""
    color_dists = np.asarray(color_dists, dtype=np.float64)
    comp_scores = np.asarray(comp_scores, dtype=np.float64)
    cc = np.exp(-color_dists / 20.0)
    csu = 1.0 / (1.0 + np.exp(-0.8 * (comp_scores - 2.0)))
    return np.clip(0.26 + 0.42 * cc * csu, 0.22, 0.52)


def _synth_bridge_candidates_batch(
    col: int,
    ys: List[int],
    roi_w: int,
    roi_h: int,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
) -> List[dict]:
    """Create dp_bridge candidates in the caller-provided y order."""
    if not ys:
        return []
    ym = max(0, int(roi_h) - 1)
    xi = int(np.clip(col, 0, int(roi_w) - 1))
    ys_arr = np.clip(np.asarray(ys, dtype=np.int64), 0, ym)
    color_vals = color_dist_map[ys_arr, xi].astype(np.float64, copy=False)
    comp_vals = comp_score_map[ys_arr, xi].astype(np.float64, copy=False)
    axis_vals = axis_dist_map[ys_arr, xi].astype(np.float64, copy=False)
    conf_vals = _bridge_confidence_batch(color_vals, comp_vals)
    return [
        {
            "y": int(ys_arr[i]),
            "confidence": float(conf_vals[i]),
            "color_dist": float(color_vals[i]),
            "comp_score": float(comp_vals[i]),
            "axis_dist": float(axis_vals[i]),
            "source": "dp_bridge",
        }
        for i in range(int(ys_arr.shape[0]))
    ]


def _collect_missing_bridge_ys_in_order(
    source_candidates: List[dict],
    existing_candidates: List[dict],
    W: int,
    ym: int,
    max_out: Optional[int] = None,
) -> List[int]:
    """Collect bridge y values in the same order as the original nested loops.

    max_out caps total output to avoid O(roi_h) blow-up per column at high
    upscale factors; source candidates are visited in priority order so the
    most relevant y values are collected first.
    """
    missing = np.ones(int(ym) + 1, dtype=bool)
    for c in existing_candidates:
        yi = int(c["y"])
        if 0 <= yi <= ym:
            missing[yi] = False

    out: List[int] = []
    for c in source_candidates:
        yc = int(c["y"])
        lo = max(0, yc - W)
        hi = min(ym, yc + W)
        idx = np.flatnonzero(missing[lo : hi + 1])
        if idx.size:
            vals = idx + lo
            if max_out is not None:
                remaining = max_out - len(out)
                if vals.size > remaining:
                    # Evenly-spaced subsample to preserve coverage across the window
                    step = max(1, vals.size // remaining)
                    vals = vals[::step][:remaining]
            out.extend(int(v) for v in vals)
            missing[vals] = False
        if max_out is not None and len(out) >= max_out:
            break
    return out


def bridge_final_candidates_for_dp(
    final: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    color_dist_map: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    *,
    skeleton_hint_y: Optional[np.ndarray] = None,
    max_for_dp: int = MAX_CANDIDATES_FOR_DP,
    max_dp_bridge_frac: Optional[float] = None,
) -> Dict[int, List[dict]]:
    """
    DP 전에 이웃 열 후보의 y±W 세로 구간을 합집합으로 채워 |dy|≤W 단계 연결을 가능하게 한다.

    마스크상 떨어진 위·아래 덩어리 사이에 사다리 후보(source=dp_bridge)를 넣고,
    트림 시 일부 슬롯을 브리지에 남긴다.
    """
    ym = max(0, roi_h - 1)
    W = dp_transition_window_width(roi_w)

    # trim 후 최대 4개만 남으므로 열당 브리지 후보 생성을 24개로 제한.
    # 고해상도(roi-upscale 2×)에서 전체 y 범위를 채우는 O(roi_h) 폭발을 방지.
    _MAX_BRIDGE_PER_COL = max(max_for_dp * 3, 24)

    out: Dict[int, List[dict]] = {}
    for col in range(roi_w):
        out[col] = [{**c} for c in final.get(col, [])]

    # Forward: col-1 후보에서 한 스텝 도달 가능한 y 전부
    for col in range(1, roi_w):
        prev_list = out[col - 1]
        if not prev_list:
            continue
        cur_list = out[col]
        new_ys = _collect_missing_bridge_ys_in_order(prev_list, cur_list, W, ym, max_out=_MAX_BRIDGE_PER_COL)
        if new_ys:
            cur_list.extend(
                _synth_bridge_candidates_batch(
                    col,
                    new_ys,
                    roi_w,
                    roi_h,
                    color_dist_map,
                    comp_score_map,
                    axis_dist_map,
                )
            )

    # Backward: col+1 과 연결
    for col in range(roi_w - 2, -1, -1):
        nxt_list = out[col + 1]
        if not nxt_list:
            continue
        cur_list = out[col]
        new_ys = _collect_missing_bridge_ys_in_order(nxt_list, cur_list, W, ym, max_out=_MAX_BRIDGE_PER_COL)
        if new_ys:
            cur_list.extend(
                _synth_bridge_candidates_batch(
                    col,
                    new_ys,
                    roi_w,
                    roi_h,
                    color_dist_map,
                    comp_score_map,
                    axis_dist_map,
                )
            )

    expanded = {k: tuple(v) for k, v in out.items()}
    expanded_measured_y = {
        k: np.asarray(
            sorted(int(c["y"]) for c in v if c.get("source") != "dp_bridge"),
            dtype=np.int32,
        )
        for k, v in expanded.items()
    }
    for col in range(roi_w):
        out[col] = _trim_candidates_after_bridge(
            out[col],
            col,
            max_for_dp,
            skeleton_hint_y,
            expanded,
            roi_w,
            W,
            max_dp_bridge_frac=max_dp_bridge_frac,
            expanded_neighbor_measured_y=expanded_measured_y,
        )

    return out


def smooth_hint_y_column(hints: np.ndarray, half: int = 5) -> np.ndarray:
    """nan 무시·창 내 중앙값으로 열별 힌트 단발 튐 완화."""
    n = int(hints.shape[0])
    out = np.copy(hints)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = hints[lo:hi]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = float(np.median(chunk))
    return out


def _final_evidence_quota_score(c: dict, col_cands: List[dict], ub_y: int) -> float:
    """GT 없음: non-upper(y>=ub)에 소량 보너스 + 동일 열 근접 밀도."""
    cf = float(c.get("confidence", 0.0))
    ny = _nearby_y_support_normalized(int(c["y"]), col_cands)
    bonus = 0.12 if int(c["y"]) >= ub_y else 0.0
    return cf + 0.25 * ny + bonus


def _merge_final_evidence_quota(
    deduped_in_sort_order: List[dict],
    col_cands: List[dict],
    max_for_dp: int,
    quota_slots: int,
    ub_y: int,
) -> List[dict]:
    """non-upper·local evidence 상위 quota_slots개를 최대한 유지한 채 나머지를 dedup 순으로 채운다."""
    qs = max(0, int(quota_slots))
    if qs <= 0 or not deduped_in_sort_order:
        return deduped_in_sort_order[:max_for_dp]
    non_upper = [c for c in col_cands if int(c["y"]) >= ub_y]
    non_upper.sort(
        key=lambda c: (
            -_final_evidence_quota_score(c, col_cands, ub_y),
            -float(c.get("confidence", 0.0)),
        )
    )
    out: List[dict] = []
    seen_out: Set[int] = set()
    for c in non_upper:
        if len(out) >= qs:
            break
        cid = id(c)
        if cid not in seen_out:
            out.append(c)
            seen_out.add(cid)
    for c in deduped_in_sort_order:
        if len(out) >= max_for_dp:
            break
        if id(c) not in seen_out:
            out.append(c)
            seen_out.add(id(c))
    return out[:max_for_dp]


def _continuity_candidate_score(
    c: dict,
    col: int,
    filtered: Dict[int, List[dict]],
    roi_w: int,
    *,
    window: int,
    max_jump: int,
) -> Tuple[float, Dict[str, Any]]:
    """GT-free continuity score from candidate confidence and nearby-column support."""
    y = int(c.get("y", 0))
    cf = float(c.get("confidence", 0.0))
    win = max(1, int(window))
    mj = max(1, int(max_jump))
    support = 0.0
    weighted_jump = 0.0
    hits = 0
    best_prev_jump: Optional[float] = None
    best_next_jump: Optional[float] = None
    for dcol in range(1, win + 1):
        decay = 1.0 / float(dcol)
        for ncol, side in ((col - dcol, "prev"), (col + dcol, "next")):
            if ncol < 0 or ncol >= roi_w:
                continue
            near = [
                abs(float(nc.get("y", 0.0)) - float(y))
                for nc in filtered.get(ncol, [])
                if abs(float(nc.get("y", 0.0)) - float(y)) <= float(mj)
            ]
            if not near:
                continue
            bj = min(near)
            hits += 1
            support += decay * (1.0 - min(bj / float(mj), 1.0))
            weighted_jump += decay * bj
            if side == "prev":
                best_prev_jump = bj if best_prev_jump is None else min(best_prev_jump, bj)
            else:
                best_next_jump = bj if best_next_jump is None else min(best_next_jump, bj)
    local_density = _nearby_y_support_normalized(y, filtered.get(col, []))
    jump_penalty = weighted_jump / float(max(hits, 1)) / float(mj)
    score = cf + 0.34 * support + 0.12 * local_density - 0.18 * jump_penalty
    meta = {
        "continuity_support_hits": int(hits),
        "continuity_support_score": float(support),
        "continuity_local_density": float(local_density),
        "continuity_jump_penalty": float(jump_penalty),
        "continuity_best_prev_jump": best_prev_jump,
        "continuity_best_next_jump": best_next_jump,
        "continuity_score": float(score),
    }
    return float(score), meta


def _build_continuity_preserve_by_col(
    filtered: Dict[int, List[dict]],
    w: int,
    *,
    slots: int,
    window: int,
    max_jump: int,
) -> Tuple[Dict[int, List[dict]], Dict[str, Any]]:
    """Pick GT-free y-continuous branches to preserve in final DP candidates."""
    sl = max(0, int(slots))
    if sl <= 0:
        return {}, {
            "enabled": False,
            "continuity_slots": int(slots),
            "continuity_window": int(window),
            "continuity_max_jump": int(max_jump),
        }
    win = max(1, int(window))
    mj = max(1, int(max_jump))
    scored_by_col: Dict[int, List[Tuple[dict, float, Dict[str, Any]]]] = {}
    score_vals_all: List[float] = []
    hit_vals_all: List[int] = []
    for col in range(w):
        rows: List[Tuple[dict, float, Dict[str, Any]]] = []
        for c in filtered.get(col, []):
            sc, meta = _continuity_candidate_score(
                c,
                col,
                filtered,
                int(w),
                window=win,
                max_jump=mj,
            )
            rows.append((c, sc, meta))
            score_vals_all.append(float(sc))
            hit_vals_all.append(int(meta.get("continuity_support_hits", 0)))
        rows.sort(key=lambda row: (-row[1], -float(row[0].get("confidence", 0.0))))
        scored_by_col[col] = rows

    dp: Dict[Tuple[int, int], Tuple[float, int, Optional[Tuple[int, int]]]] = {}
    for col in range(w):
        rows = scored_by_col.get(col, [])
        if not rows:
            continue
        for idx, (c, unary, _) in enumerate(rows):
            y = float(c.get("y", 0.0))
            best_score = float(unary)
            best_len = 1
            best_prev: Optional[Tuple[int, int]] = None
            for dcol in range(1, win + 1):
                pcol = col - dcol
                if pcol < 0:
                    break
                prev_rows = scored_by_col.get(pcol, [])
                if not prev_rows:
                    continue
                allowed = float(mj * dcol)
                for pidx, (pc, _, _) in enumerate(prev_rows):
                    py = float(pc.get("y", 0.0))
                    jump = abs(y - py)
                    if jump > allowed:
                        continue
                    pkey = (pcol, pidx)
                    if pkey not in dp:
                        continue
                    pscore, plen, _ = dp[pkey]
                    transition_bonus = (
                        0.18
                        - 0.16 * min(jump / max(allowed, 1e-6), 1.0)
                        - 0.08 * float(dcol - 1)
                    )
                    cand_score = float(pscore) + float(unary) + float(transition_bonus)
                    cand_len = int(plen) + 1
                    if (cand_score + 0.03 * cand_len) > (best_score + 0.03 * best_len):
                        best_score = cand_score
                        best_len = cand_len
                        best_prev = pkey
            dp[(col, idx)] = (best_score, best_len, best_prev)

    def _trace_path(end_key: Tuple[int, int]) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = end_key
        seen: Set[Tuple[int, int]] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = dp.get(cur, (0.0, 0, None))[2]
        path.reverse()
        return path

    endpoints = sorted(
        dp.keys(),
        key=lambda key: (dp[key][0] + 0.08 * dp[key][1], dp[key][1]),
        reverse=True,
    )
    branch_paths: List[List[Tuple[int, int]]] = []
    branch_target = max(1, sl)
    for end_key in endpoints:
        path = _trace_path(end_key)
        if not path:
            continue
        path_ids = {
            id(scored_by_col[col][idx][0])
            for col, idx in path
            if col in scored_by_col and idx < len(scored_by_col[col])
        }
        if not path_ids:
            continue
        too_similar = False
        for prev_path in branch_paths:
            prev_ids = {
                id(scored_by_col[col][idx][0])
                for col, idx in prev_path
                if col in scored_by_col and idx < len(scored_by_col[col])
            }
            overlap = len(path_ids & prev_ids) / float(max(min(len(path_ids), len(prev_ids)), 1))
            if overlap >= 0.80:
                too_similar = True
                break
        if too_similar:
            continue
        branch_paths.append(path)
        if len(branch_paths) >= branch_target:
            break

    selected: Dict[int, List[dict]] = {}
    score_vals: List[float] = []
    hit_vals: List[int] = []
    src_ct: Counter[str] = Counter()
    branch_lengths: List[int] = []
    branch_scores: List[float] = []
    for branch_idx, path in enumerate(branch_paths):
        branch_lengths.append(len(path))
        if path:
            end_key = path[-1]
            branch_scores.append(float(dp[end_key][0]))
        for col, idx in path:
            rows = scored_by_col.get(col, [])
            if idx >= len(rows):
                continue
            c, sc, meta = rows[idx]
            yy = int(c.get("y", 0))
            picks = selected.setdefault(col, [])
            if any(int(pc.get("y", 0)) == yy for pc in picks):
                continue
            if len(picks) >= sl:
                continue
            c["debug_continuity_preserve"] = True
            c["continuity_branch_id"] = int(branch_idx)
            c["continuity_branch_length"] = int(len(path))
            c["continuity_preserve_score"] = float(sc)
            c["continuity_preserve_meta"] = meta
            picks.append(c)
            score_vals.append(float(sc))
            hit_vals.append(int(meta.get("continuity_support_hits", 0)))
            src_ct[str(c.get("source", ""))] += 1
    arr = np.asarray(score_vals, dtype=np.float64) if score_vals else np.asarray([], dtype=np.float64)
    meta_out: Dict[str, Any] = {
        "enabled": True,
        "continuity_slots": int(sl),
        "continuity_window": int(win),
        "continuity_max_jump": int(mj),
        "continuity_branch_count": int(len(branch_paths)),
        "continuity_branch_lengths": [int(v) for v in branch_lengths],
        "continuity_branch_scores": [float(v) for v in branch_scores],
        "columns_with_continuity_candidates": int(len(selected)),
        "continuity_candidate_count": int(sum(len(v) for v in selected.values())),
        "continuity_source_distribution": dict(sorted(src_ct.items())),
    }
    arr_all = (
        np.asarray(score_vals_all, dtype=np.float64)
        if score_vals_all
        else np.asarray([], dtype=np.float64)
    )
    if arr_all.size:
        meta_out["continuity_all_score_distribution"] = {
            "mean": float(np.mean(arr_all)),
            "median": float(np.median(arr_all)),
            "p90": float(np.percentile(arr_all, 90)),
            "min": float(np.min(arr_all)),
            "max": float(np.max(arr_all)),
        }
        harr_all = np.asarray(hit_vals_all, dtype=np.float64)
        meta_out["continuity_all_support_hits_distribution"] = {
            "mean": float(np.mean(harr_all)),
            "median": float(np.median(harr_all)),
            "p90": float(np.percentile(harr_all, 90)),
            "min": float(np.min(harr_all)),
            "max": float(np.max(harr_all)),
        }
    if arr.size:
        meta_out["continuity_score_distribution"] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        harr = np.asarray(hit_vals, dtype=np.float64)
        meta_out["continuity_support_hits_distribution"] = {
            "mean": float(np.mean(harr)),
            "median": float(np.median(harr)),
            "p90": float(np.percentile(harr, 90)),
            "min": float(np.min(harr)),
            "max": float(np.max(harr)),
        }
    return selected, meta_out


def _merge_final_continuity_quota(
    current: List[dict],
    continuity_picks: List[dict],
    max_for_dp: int,
    slots: int,
) -> List[dict]:
    """Preserve continuity picks without changing existing default behavior when disabled."""
    sl = max(0, int(slots))
    if sl <= 0 or not continuity_picks:
        return current[: int(max_for_dp)]
    out: List[dict] = []
    seen: Set[int] = set()
    for c in continuity_picks:
        if len(out) >= sl:
            break
        if id(c) in seen:
            continue
        out.append(c)
        seen.add(id(c))
    for c in current:
        if len(out) >= int(max_for_dp):
            break
        if id(c) not in seen:
            out.append(c)
            seen.add(id(c))
    return out[: int(max_for_dp)]


def attach_candidate_final_bridge_debug(
    candidate_final_debug: Dict[str, Any],
    *,
    final_pre_bridge: Dict[int, List[dict]],
    final_post_bridge: Dict[int, List[dict]],
    filtered: Dict[int, List[dict]],
    debug_gt_y_by_col: Dict[int, float],
    roi_h: int,
    w: int,
    gt_near_px: float = 5.0,
    debug_columns: bool = False,
) -> None:
    """브리지 후 후보·소스 분포·상단 밴드 비율·GT-near 최종 집계."""
    ub_y = max(1, int(math.ceil(float(roi_h) * _UPPER_BAND_FRAC_FOR_CAP)))
    loss_bridge = Counter()
    src_ct: Counter[str] = Counter()
    src_gt_near: Counter[str] = Counter()
    upper_ct = 0
    total_slots = 0
    dp_b = neigh = interp = 0
    n_fil_gt = 0
    n_fin_gt = 0
    lost_ft = 0

    per_bridge: Dict[str, Any] = {}

    for col in range(w):
        lst = final_post_bridge.get(col, [])
        for c in lst:
            total_slots += 1
            sk = str(c.get("source", ""))
            src_ct[sk] += 1
            if int(c["y"]) < ub_y:
                upper_ct += 1
            if sk == "dp_bridge":
                dp_b += 1
            elif sk.startswith("neighbor"):
                neigh += 1
            elif sk == "interpolated":
                interp += 1

        gy = debug_gt_y_by_col.get(col)
        if gy is None:
            continue
        fl = filtered.get(col, [])
        pre_b = final_pre_bridge.get(col, [])
        post_b = final_post_bridge.get(col, [])

        def _has_near(xs: List[dict]) -> bool:
            return any(abs(float(c["y"]) - float(gy)) <= gt_near_px for c in xs)

        if _has_near(fl):
            n_fil_gt += 1
        if _has_near(post_b):
            n_fin_gt += 1
            for c in post_b:
                if abs(float(c["y"]) - float(gy)) <= gt_near_px:
                    src_gt_near[str(c.get("source", ""))] += 1
                    break
        elif _has_near(fl):
            lost_ft += 1
            survived_pre = _has_near(pre_b)
            key = (
                "bridge_trim_lost_gt_near"
                if survived_pre
                else "filtered_gt_near_lost_before_bridge_stage"
            )
            loss_bridge[key] += 1
            if debug_columns:
                per_bridge[str(col)] = {
                    "had_gt_near_pre_bridge": survived_pre,
                    "post_bridge_gt_near": False,
                }

    cdb = candidate_final_debug.pop("_column_loss_build", None)
    if debug_columns:
        details_merged: Dict[str, Any] = {}
        for k, v in per_bridge.items():
            details_merged.setdefault(k, {}).update(v)
        if cdb:
            for k, v in cdb.items():
                details_merged.setdefault(k, {})["build_final_stage"] = v
        if details_merged:
            candidate_final_debug["column_details"] = details_merged
    else:
        candidate_final_debug.pop("column_details", None)

    candidate_final_debug["final_source_distribution"] = dict(sorted(src_ct.items()))
    candidate_final_debug["final_gt_near_source_distribution"] = dict(sorted(src_gt_near.items()))
    candidate_final_debug["final_upper_band_count"] = int(upper_ct)
    candidate_final_debug["final_upper_band_fraction"] = (
        float(upper_ct) / float(max(total_slots, 1))
    )
    candidate_final_debug["dp_bridge_count"] = int(dp_b)
    candidate_final_debug["neighbor_candidate_count"] = int(neigh)
    candidate_final_debug["interpolated_candidate_count"] = int(interp)
    candidate_final_debug["gt_near_filtered_columns_px5"] = int(n_fil_gt)
    candidate_final_debug["gt_near_final_columns_px5"] = int(n_fin_gt)
    candidate_final_debug["gt_near_filtered_to_final_lost_columns_px5"] = int(lost_ft)
    if loss_bridge:
        existing = Counter(candidate_final_debug.get("gt_near_final_loss_reason_counts") or {})
        existing.update(loss_bridge)
        candidate_final_debug["gt_near_final_loss_reason_counts"] = dict(
            sorted(existing.items())
        )


def build_final_candidates(
    filtered: Dict[int, List[dict]],
    comp_score_map: np.ndarray,
    w: int,
    max_for_dp: int = MAX_CANDIDATES_FOR_DP,
    skeleton_hint_y: Optional[np.ndarray] = None,
    roi_height: int = 0,
    *,
    debug_final_sink: Optional[Dict[str, Any]] = None,
    debug_gt_y_by_col: Optional[Dict[int, float]] = None,
    debug_gt_near_px: float = 5.0,
    debug_final_selection_reasons: bool = False,
    candidate_final_enable_evidence_aware_preserve: bool = False,
    candidate_final_evidence_preserve_slots: int = 2,
    candidate_final_disable_score_bucket_dedupe: bool = False,
    candidate_final_dedupe_score_decimals: Optional[int] = None,
    candidate_final_enable_continuity_preserve: bool = False,
    candidate_final_continuity_slots: int = 2,
    candidate_final_continuity_window: int = 3,
    candidate_final_continuity_max_jump: int = 8,
) -> Tuple[Dict[int, List[dict]], List[int]]:
    """
    $12.7 + $12.9: empty column interpolation + final DP candidate (top K, dedup).
    skeleton_hint_y: 길이 w, 열별 스켈레톤 y 힌트(nan 가능) — 신뢰도 동률·격자 혼동 시 정렬 보조.
    roi_height: 0이면 이웃 외삽 y를 ROI 높이로 클립하지 않음(하위 호환).
    Returns: (final_candidates, missing_columns)
    """
    final: Dict[int, List[dict]] = {}
    missing_cols: List[int] = []

    loss_counts: Counter[str] = Counter()
    col_loss_build: Dict[str, str] = {}
    rank_samples: List[int] = []
    dedupe_decimals = (
        2 if candidate_final_dedupe_score_decimals is None else int(candidate_final_dedupe_score_decimals)
    )

    ub_y = max(0, int(math.ceil(float(roi_height) * _UPPER_BAND_FRAC_FOR_CAP))) if roi_height > 0 else 0
    continuity_by_col: Dict[int, List[dict]] = {}
    continuity_meta: Dict[str, Any] = {
        "enabled": False,
        "continuity_slots": int(candidate_final_continuity_slots),
        "continuity_window": int(candidate_final_continuity_window),
        "continuity_max_jump": int(candidate_final_continuity_max_jump),
    }
    if candidate_final_enable_continuity_preserve:
        continuity_by_col, continuity_meta = _build_continuity_preserve_by_col(
            filtered,
            int(w),
            slots=int(candidate_final_continuity_slots),
            window=int(candidate_final_continuity_window),
            max_jump=int(candidate_final_continuity_max_jump),
        )

    for col in range(w):
        cands = filtered.get(col, [])

        if not cands:
            interp = _interpolate_from_neighbors(filtered, col, w, roi_height=roi_height)
            if interp is not None:
                cands = [interp]
            else:
                missing_cols.append(col)
                final[col] = []
                continue

        def _dp_rank(c: dict) -> float:
            r = float(c["confidence"])
            if skeleton_hint_y is not None and col < skeleton_hint_y.shape[0]:
                gy = float(skeleton_hint_y[col])
                if not np.isnan(gy):
                    r += 0.16 * float(np.exp(-abs(float(c["y"]) - gy) / 11.0))
            return r

        sorted_cands = sorted(cands, key=lambda c: -_dp_rank(c))

        if candidate_final_disable_score_bucket_dedupe:
            deduped = list(sorted_cands)
        else:
            seen_scores = set()
            deduped = []
            for c in sorted_cands:
                score_key = round(float(c["comp_score"]), dedupe_decimals)
                if score_key in seen_scores:
                    continue
                seen_scores.add(score_key)
                deduped.append(c)

        merged = deduped
        if candidate_final_enable_evidence_aware_preserve and roi_height > 0:
            merged = _merge_final_evidence_quota(
                deduped,
                cands,
                int(max_for_dp),
                int(candidate_final_evidence_preserve_slots),
                ub_y,
            )
        else:
            merged = deduped[: int(max_for_dp)]

        if candidate_final_enable_continuity_preserve:
            merged = _merge_final_continuity_quota(
                merged,
                continuity_by_col.get(col, []),
                int(max_for_dp),
                int(candidate_final_continuity_slots),
            )

        final[col] = merged

        if debug_gt_y_by_col is not None:
            gy = debug_gt_y_by_col.get(col)
            if gy is not None:
                gt_near_f = [
                    c for c in cands if abs(float(c["y"]) - float(gy)) <= float(debug_gt_near_px)
                ]
                if gt_near_f:
                    best_gt = min(gt_near_f, key=lambda c: abs(float(c["y"]) - float(gy)))
                    try:
                        i_sort = next(
                            i for i, c in enumerate(sorted_cands) if id(c) == id(best_gt)
                        )
                    except StopIteration:
                        i_sort = 999999
                    in_dedup = any(id(c) == id(best_gt) for c in deduped)
                    try:
                        i_d = next(i for i, c in enumerate(deduped) if id(c) == id(best_gt))
                    except StopIteration:
                        i_d = -1

                    survived = any(
                        abs(float(c["y"]) - float(gy)) <= float(debug_gt_near_px) for c in merged
                    )
                    if not survived:
                        if not candidate_final_disable_score_bucket_dedupe and not in_dedup:
                            reason = "score_bucket_dedupe"
                        elif i_d >= int(max_for_dp):
                            reason = "outside_topk_after_dedupe"
                        elif i_sort >= int(max_for_dp):
                            reason = "outside_dp_rank_topk"
                        elif candidate_final_enable_evidence_aware_preserve:
                            reason = "final_evidence_quota_displaced"
                        else:
                            reason = "outside_topk_after_dedupe"
                        loss_counts[reason] += 1
                        if debug_final_selection_reasons:
                            col_loss_build[str(col)] = reason
                    else:
                        try:
                            ir = next(
                                i
                                for i, c in enumerate(merged)
                                if abs(float(c["y"]) - float(gy)) <= float(debug_gt_near_px)
                            )
                        except StopIteration:
                            ir = -1
                        if ir >= 0:
                            rank_samples.append(ir + 1)
                        if debug_final_selection_reasons:
                            col_loss_build[str(col)] = "survived_build_final"

    if debug_final_sink is not None:
        dbg = {
            "final_topk": int(max_for_dp),
            "final_dedupe_enabled": not bool(candidate_final_disable_score_bucket_dedupe),
            "final_rank_key": "confidence + 0.16*exp(-|y - skeleton_hint_y[col]| / 11), descending",
            "final_dedupe_key": (
                "disabled"
                if candidate_final_disable_score_bucket_dedupe
                else f"round(comp_score, {dedupe_decimals})"
            ),
            "candidate_final_evidence_preserve_enabled": bool(
                candidate_final_enable_evidence_aware_preserve
            ),
            "candidate_final_evidence_preserve_slots": int(candidate_final_evidence_preserve_slots),
            "candidate_final_continuity_preserve_enabled": bool(
                candidate_final_enable_continuity_preserve
            ),
            "candidate_final_continuity_preserve": continuity_meta,
            "gt_near_final_loss_reason_counts": dict(sorted(loss_counts.items())),
            "_column_loss_build": dict(col_loss_build),
        }
        if rank_samples:
            arr = np.asarray(rank_samples, dtype=np.float64)
            dbg["filtered_gt_near_rank_in_final_topk_distribution"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "n": int(len(arr)),
            }
        debug_final_sink["candidate_final_debug"] = dbg

    return final, missing_cols


def candidates_to_map(
    candidates: Dict[int, List[dict]],
    shape: Tuple[int, int],
    top_n: Optional[int] = None,
) -> np.ndarray:
    """candidate dict -> visualization image (grayscale, brighter = higher confidence)."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for col, cands in candidates.items():
        subset = cands[:top_n] if top_n else cands
        for c in subset:
            y = int(c["y"])
            if 0 <= y < h and 0 <= col < w:
                val = int(np.clip(c["confidence"] * 255, 30, 255))
                out[y, col] = max(out[y, col], val)
    return out


def compute_candidate_stats(
    raw: Dict[int, List[dict]],
    filtered: Dict[int, List[dict]],
    final: Dict[int, List[dict]],
    missing_cols: List[int],
    w: int,
) -> dict:
    """debug.json용 통계."""
    raw_total = sum(len(v) for v in raw.values())
    filt_total = sum(len(v) for v in filtered.values())
    final_total = sum(len(v) for v in final.values())
    raw_nonempty = sum(1 for v in raw.values() if v)
    filt_nonempty = sum(1 for v in filtered.values() if v)
    final_nonempty = sum(1 for v in final.values() if v)

    return {
        "raw_candidates_total": raw_total,
        "filtered_candidates_total": filt_total,
        "final_candidates_total": final_total,
        "raw_nonempty_columns": raw_nonempty,
        "filtered_nonempty_columns": filt_nonempty,
        "final_nonempty_columns": final_nonempty,
        "missing_columns": len(missing_cols),
        "missing_column_ratio": round(float(len(missing_cols)) / max(1, w), 4),
        "total_columns": w,
    }


def _f_local_column(ridge_score: np.ndarray, col: int, y: int) -> float:
    col = max(0, min(ridge_score.shape[1] - 1, col))
    prof = ridge_score[:, col]
    p95 = float(np.percentile(prof, 95)) + 1e-9
    return float(np.clip(prof[y] / p95, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _merge_by_y_distance(cands: List[dict], y_merge_dist: int) -> List[dict]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: int(c["y"]))
    groups: List[List[dict]] = [[cands[0]]]
    for c in cands[1:]:
        if abs(int(c["y"]) - int(groups[-1][-1]["y"])) <= y_merge_dist:
            groups[-1].append(c)
        else:
            groups.append([c])
    merged: List[dict] = []
    for g in groups:
        best = max(g, key=lambda x: x["confidence"])
        src_tags = sorted({s for x in g for s in x.get("source_tags", [])})
        merged.append(
            {
                **best,
                "source_tags": src_tags,
                "source": "+".join(src_tags) if src_tags else best.get("source", "exp"),
            }
        )
    return merged


def build_raw_candidates_experimental(
    raw_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    color_score: np.ndarray,
    ridge_score: np.ndarray,
    grid_penalty: np.ndarray,
    edge_support: np.ndarray,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    params: Optional[dict] = None,
) -> Dict[int, List[dict]]:
    """§7.3 V2 confidence from color/ridge/grid/edge/component."""
    h, w = raw_mask.shape[:2]
    raw_candidates: Dict[int, List[dict]] = {}
    prev_best_y: Optional[float] = None

    cp = (params or {})
    ridge_fallback_thr = float(cp.get("ridge_fallback_thr", 0.18))
    merge_dist_px = int(cp.get("source_merge_dist_px", 3))
    for col in range(w):
        ys_raw = set(np.where(raw_mask[:, col] > 0)[0].tolist())
        ys_thin = set(np.where(skeleton_mask[:, col] > 0)[0].tolist())
        ys_ridge: List[int] = []
        prof = ridge_score[:, col]
        if prof.size:
            peaks = []
            for y in range(1, h - 1):
                if prof[y] >= prof[y - 1] and prof[y] >= prof[y + 1] and prof[y] >= ridge_fallback_thr:
                    peaks.append((float(prof[y]), y))
            peaks.sort(key=lambda x: -x[0])
            if not ys_raw and not ys_thin:
                ys_ridge = [int(p[1]) for p in peaks[:2]]
            elif peaks and float(peaks[0][0]) >= max(0.35, ridge_fallback_thr + 0.1):
                ys_ridge = [int(peaks[0][1])]
        ys_all = sorted(set(ys_raw) | set(ys_thin) | set(ys_ridge))
        if len(ys_all) == 0:
            raw_candidates[col] = []
            continue

        cands = []
        for y_val in ys_all:
            y_int = int(y_val)
            fc = float(color_score[y_int, col])
            fr = float(ridge_score[y_int, col])
            fe = float(edge_support[y_int, col])
            fg = float(grid_penalty[y_int, col])
            raw_comp = float(comp_score_map[y_int, col])
            fcomp = float(1.0 / (1.0 + np.exp(-0.8 * (raw_comp - 2.0))))
            fl = _f_local_column(ridge_score, col, y_int)
            local_cont = _local_continuity(float(y_int), prev_best_y)
            src_tags: List[str] = []
            if y_int in ys_raw:
                src_tags.append("raw")
            if y_int in ys_thin:
                src_tags.append("thin")
            if y_int in ys_ridge:
                src_tags.append("ridge")

            source_score = 0.5 * fc + 0.35 * fr + 0.15 * fe
            conf_pre = 0.35 * source_score + 0.25 * local_cont + 0.20 * fcomp + 0.20 * fl
            adj = 0.0
            if "thin" in src_tags:
                adj += 0.20
            if src_tags == ["raw"]:
                conf_pre = max(conf_pre, 0.45)
            if src_tags == ["ridge"]:
                conf_pre = max(conf_pre, 0.50)
            if len(src_tags) >= 2:
                adj += 0.15
            if float(axis_dist_map[y_int, col]) < 2.5:
                adj -= 0.20
            if raw_comp < 0.8:
                adj -= 0.25
            if local_cont >= 0.85:
                adj += 0.10

            conf = _sigmoid(4.0 * (conf_pre + adj - 0.18 * fg - 0.5))
            conf = float(np.clip(conf, 0.0, 1.0))
            cands.append({
                "y": y_int,
                "confidence": conf,
                "color_dist": 1.0 - fc,
                "comp_score": raw_comp,
                "axis_dist": float(axis_dist_map[y_int, col]),
                "source": "exp",
                "source_tags": src_tags,
                "feature_terms": {
                    "source_score": round(float(source_score), 6),
                    "local_continuity": round(float(local_cont), 6),
                    "component_score": round(float(fcomp), 6),
                    "ridge_strength": round(float(fl), 6),
                    "grid_penalty": round(float(fg), 6),
                    "conf_pre": round(float(conf_pre), 6),
                    "adjustment": round(float(adj), 6),
                },
            })

        merged = _merge_by_y_distance(cands, merge_dist_px)
        merged.sort(key=lambda c: -c["confidence"])
        raw_candidates[col] = merged
        if merged:
            prev_best_y = float(merged[0]["y"])

    return raw_candidates


def filter_candidates_experimental(
    raw_candidates: Dict[int, List[dict]],
    ridge_score: np.ndarray,
    grid_penalty: np.ndarray,
    plot_h: int,
    params: Optional[dict] = None,
) -> Dict[int, List[dict]]:
    """§7.4"""
    cp = (params or {})
    merge_dist = max(2, int(round(float(cp.get("merge_dist_scale", 0.004)) * plot_h)))
    conf_min = float(cp.get("conf_min", 0.12))
    comp_support_min = float(cp.get("comp_support_min", 0.10))
    grid_penalty_high = float(cp.get("grid_penalty_high", 0.55))
    grid_ridge_low = float(cp.get("grid_ridge_low", 0.15))
    filtered: Dict[int, List[dict]] = {}
    for col, cands in raw_candidates.items():
        kept = []
        for c in cands:
            y = c["y"]
            fr = float(ridge_score[y, col])
            fg = float(grid_penalty[y, col])
            raw_cs = float(c.get("comp_score", 0))
            comp_sup = float(1.0 / (1.0 + np.exp(-0.8 * (raw_cs - 2.0))))
            if c["confidence"] < conf_min:
                continue
            if comp_sup < comp_support_min:
                continue
            if fg > grid_penalty_high and fr < grid_ridge_low:
                continue
            kept.append(c)
        kept.sort(key=lambda c: -c["confidence"])
        merged = []
        for c in kept:
            if not merged:
                merged.append(c)
                continue
            if abs(c["y"] - merged[-1]["y"]) <= merge_dist:
                if c["confidence"] > merged[-1]["confidence"]:
                    merged[-1] = c
            else:
                merged.append(c)
        filtered[col] = merged[:12]
    return filtered


def compute_margin_histogram(final: Dict[int, List[dict]]) -> dict:
    margins: List[float] = []
    for cands in final.values():
        if len(cands) >= 2:
            margins.append(float(cands[0]["confidence"] - cands[1]["confidence"]))
        elif len(cands) == 1:
            margins.append(float(cands[0]["confidence"]))
    if not margins:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(margins, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
    }
