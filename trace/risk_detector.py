"""
Rule-based risk detector: column-level features + risk flags + dilated segments.

CNN 없음. 실패 시 일부 feature는 None/NaN으로 두고 계속 진행한다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.selective_oracle_settings import SelectiveOracleSettings
from trace.oracle_rerank import _load_gt_json


def _entropy_normalized(confs: List[float]) -> float:
    if not confs:
        return float("nan")
    arr = np.clip(np.asarray(confs, dtype=np.float64), 1e-12, 1.0)
    s = float(arr.sum())
    if s <= 0:
        return float("nan")
    p = arr / s
    h = -float(np.sum(p * np.log(p)))
    return h


def _peak_roi_columns(gt: Optional[dict], plot_box_t: Tuple[int, int, int, int], roi_w: int) -> Set[int]:
    cols: Set[int] = set()
    if not gt:
        return cols
    x0 = int(plot_box_t[0])
    pts = gt.get("peak_pixel_points") or []
    for item in pts:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            try:
                xa = float(item[0])
                c = int(round(xa - float(x0)))
                if 0 <= c < int(roi_w):
                    cols.add(c)
            except (TypeError, ValueError):
                continue
    return cols


def _blockwise_margin_for_column(col: int, tr_rule: Optional[dict]) -> Optional[float]:
    if not tr_rule:
        return None
    bw = tr_rule.get("blockwise") or []
    if not bw:
        return None
    for blk in bw:
        try:
            bs = int(blk.get("block_start", -1))
            be = int(blk.get("block_end", -1))
        except (TypeError, ValueError):
            continue
        if bs <= col <= be:
            m = blk.get("margin")
            if m is None:
                return None
            return float(m)
    return None


def extract_column_risk_features(
    final_cands: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    plot_box_t: Tuple[int, int, int, int],
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    ridge_map: Optional[np.ndarray],
    gt: Optional[dict],
    style_group: str,
    taxonomy_prior: Optional[str],
    tr_rule: Optional[dict],
    *,
    peak_window_radius: int = 12,
    local_window: int = 2,
) -> List[Dict[str, Any]]:
    """열별 feature 행 리스트. 예외는 행 단위로 삼키고 빈 리스트는 거의 없음."""
    rows: List[Dict[str, Any]] = []
    peak_cols = _peak_roi_columns(gt, plot_box_t, roi_w)
    tax_set = set()
    if taxonomy_prior:
        tax_set = {t.strip() for t in str(taxonomy_prior).split(";") if t.strip()}

    columns = sorted(int(k) for k in final_cands.keys())

    def neighbors_empty(ci: int) -> int:
        n = 0
        for d in (-1, 1):
            j = ci + d
            if j < 0 or j >= roi_w:
                continue
            if not final_cands.get(j):
                n += 1
        return n

    for col in columns:
        try:
            cands = sorted(final_cands.get(col, []), key=lambda c: -float(c.get("confidence", 0.0)))
            n_c = len(cands)
            confs = [float(c.get("confidence", 0.0)) for c in cands]
            top1_conf = float(confs[0]) if confs else float("nan")
            top2_conf = float(confs[1]) if len(confs) > 1 else top1_conf
            conf_margin = float(top1_conf - top2_conf) if confs else float("nan")
            ent = _entropy_normalized(confs) if n_c > 0 else float("nan")

            top1 = cands[0] if cands else {}
            top2 = cands[1] if len(cands) > 1 else top1
            y1 = int(top1.get("y", 0)) if top1 else 0
            y2 = int(top2.get("y", 0)) if top2 else y1
            top1_axis_proximity = float(top1.get("axis_dist", float("nan"))) if top1 else float("nan")
            top1_comp = float(top1.get("comp_score", float("nan"))) if top1 else float("nan")
            rr = None
            if ridge_map is not None and ridge_map.shape[:2] == (roi_h, roi_w):
                try:
                    rr = float(ridge_map[y1, col])
                except Exception:
                    rr = None

            y_gap = float(abs(y1 - y2))

            near_peak = any(abs(col - pc) <= peak_window_radius for pc in peak_cols)

            ys_local: List[float] = []
            confs_local: List[float] = []
            for d in range(-local_window, local_window + 1):
                j = col + d
                if j < 0 or j >= roi_w:
                    continue
                for c in final_cands.get(j, []):
                    ys_local.append(float(c.get("y", 0)))
                    confs_local.append(float(c.get("confidence", 0.0)))
            loc_y_std = float(np.std(ys_local)) if len(ys_local) > 1 else 0.0
            loc_c_std = float(np.std(confs_local)) if len(confs_local) > 1 else 0.0

            dp_margin = _blockwise_margin_for_column(col, tr_rule)
            dp_margin_low = bool(dp_margin is not None and not math.isnan(dp_margin) and dp_margin < 0.12)

            row: Dict[str, Any] = {
                "x": int(col),
                "candidate_count": int(n_c),
                "top1_conf": top1_conf,
                "top2_conf": top2_conf,
                "conf_margin": conf_margin,
                "conf_entropy": ent,
                "top1_axis_proximity": top1_axis_proximity,
                "top1_ridge_response": rr,
                "top1_component_score": top1_comp,
                "top1_y": float(y1),
                "top2_y": float(y2),
                "top1_top2_y_gap": y_gap,
                "empty_neighbor_count": int(neighbors_empty(col)),
                "is_near_peak_window": bool(near_peak),
                "local_candidate_y_std": loc_y_std,
                "local_conf_std": loc_c_std,
                "style_group": str(style_group),
                "taxonomy_prior": taxonomy_prior or "",
                "dp_local_cost_spike": None,
                "dp_margin_low": dp_margin_low,
                "dp_block_margin": dp_margin,
                "recovery_trigger_nearby": None,
                "grid_like_proxy": None,
                "axis_boundary_nearby": bool(
                    y1 <= 2 or y1 >= roi_h - 3 or (not math.isnan(top1_axis_proximity) and top1_axis_proximity < 1.2)
                ),
            }
            rows.append(row)
        except Exception:
            rows.append(
                {
                    "x": int(col),
                    "candidate_count": 0,
                    "top1_conf": float("nan"),
                    "top2_conf": float("nan"),
                    "conf_margin": float("nan"),
                    "conf_entropy": float("nan"),
                    "top1_axis_proximity": float("nan"),
                    "top1_ridge_response": None,
                    "top1_component_score": float("nan"),
                    "top1_y": float("nan"),
                    "top2_y": float("nan"),
                    "top1_top2_y_gap": float("nan"),
                    "empty_neighbor_count": 0,
                    "is_near_peak_window": False,
                    "local_candidate_y_std": float("nan"),
                    "local_conf_std": float("nan"),
                    "style_group": str(style_group),
                    "taxonomy_prior": taxonomy_prior or "",
                    "dp_local_cost_spike": None,
                    "dp_margin_low": None,
                    "dp_block_margin": None,
                    "recovery_trigger_nearby": None,
                    "grid_like_proxy": None,
                    "axis_boundary_nearby": False,
                }
            )

    # sample-level grid proxy into every row if provided via gt meta — caller sets row externally; skip

    return rows


def _column_risk_flags(row: Dict[str, Any], st: SelectiveOracleSettings, tax_set: Set[str]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    cm = row.get("conf_margin")
    n_c = int(row.get("candidate_count", 0))
    ent = row.get("conf_entropy")
    ad = row.get("top1_axis_proximity")
    y_gap = row.get("top1_top2_y_gap")
    near_peak = bool(row.get("is_near_peak_window"))
    dp_ml = row.get("dp_margin_low")

    if (not st.disable_low_conf_margin_risk) and isinstance(cm, (int, float)) and not math.isnan(float(cm)) and float(cm) < st.conf_margin_thr:
        reasons.append("low_conf_margin")

    high_entropy_hit = (
        (not st.disable_high_entropy_risk)
        and n_c >= st.candidate_count_high_thr
        and isinstance(ent, (int, float))
        and not math.isnan(float(ent))
        and float(ent) > st.entropy_high_thr
    )
    if high_entropy_hit:
        if st.high_entropy_requires_low_margin:
            if isinstance(cm, (int, float)) and not math.isnan(float(cm)) and float(cm) < float(st.conf_margin_thr):
                reasons.append("high_entropy_many_cands")
        else:
            reasons.append("high_entropy_many_cands")

    if (not st.disable_axis_proximity_risk) and isinstance(ad, (int, float)) and not math.isnan(float(ad)) and float(ad) < st.axis_dist_risk_thr:
        reasons.append("axis_proximity_risk")

    if (
        (not st.disable_large_y_gap_risk)
        and isinstance(y_gap, (int, float))
        and not math.isnan(float(y_gap))
        and float(y_gap) >= st.y_gap_thr
        and isinstance(cm, (int, float))
        and not math.isnan(float(cm))
        and float(cm) < st.y_gap_margin_thr
    ):
        reasons.append("large_y_gap_low_margin")

    if (not st.disable_peak_window_risk) and near_peak and isinstance(cm, (int, float)) and not math.isnan(float(cm)) and float(cm) < st.peak_margin_thr:
        reasons.append("peak_window_low_margin")

    if (not st.disable_taxonomy_prior_for_risk) and tax_set & {"grid_confusion", "peak_miss_after_smoothing"}:
        if st.taxonomy_prior_requires_margin:
            margin_low = (
                isinstance(cm, (int, float))
                and not math.isnan(float(cm))
                and float(cm) < float(st.conf_margin_thr)
            )
            high_ent_gate = (
                n_c >= st.candidate_count_high_thr
                and isinstance(ent, (int, float))
                and not math.isnan(float(ent))
                and float(ent) > st.entropy_high_thr
            )
            if margin_low or high_ent_gate:
                reasons.append("taxonomy_prior")
        else:
            cmv = float(cm) if isinstance(cm, (int, float)) and not math.isnan(float(cm)) else 999.0
            if cmv < 0.25 or n_c >= 4:
                reasons.append("taxonomy_prior")

    if (not st.disable_dp_margin_low_risk) and dp_ml is True:
        reasons.append("dp_margin_low")

    return bool(reasons), reasons


def compute_risk_columns(
    feature_rows: List[Dict[str, Any]],
    st: SelectiveOracleSettings,
    taxonomy_prior: Optional[str],
) -> Tuple[Set[int], Dict[int, List[str]]]:
    tax_set = set()
    if taxonomy_prior:
        tax_set = {t.strip() for t in str(taxonomy_prior).split(";") if t.strip()}
    risk: Set[int] = set()
    reasons_by_col: Dict[int, List[str]] = {}
    for row in feature_rows:
        col = int(row["x"])
        hit, rs = _column_risk_flags(row, st, tax_set)
        if hit:
            risk.add(col)
            reasons_by_col[col] = rs
    return risk, reasons_by_col


def dilate_risk_columns(
    risk: Set[int],
    reasons_by_col: Dict[int, List[str]],
    roi_w: int,
    radius: int,
) -> Set[int]:
    if radius <= 0:
        return set(risk)
    out: Set[int] = set()
    for c in risk:
        for d in range(-radius, radius + 1):
            j = c + d
            if 0 <= j < roi_w:
                out.add(j)
                if j != c and j not in reasons_by_col and c in reasons_by_col:
                    reasons_by_col[j] = list(dict.fromkeys(reasons_by_col.get(c, []) + ["dilated_neighbor"]))
    return out


def merge_risk_segments(
    risk: Set[int],
    reasons_by_col: Dict[int, List[str]],
    roi_w: int,
    merge_gap: int,
    min_len: int,
) -> List[Dict[str, Any]]:
    if not risk:
        return []
    cols = sorted(risk)
    segs: List[Tuple[int, int, List[str]]] = []
    start = cols[0]
    end = cols[0]
    rs_acc: Set[str] = set(reasons_by_col.get(start, []))
    for c in cols[1:]:
        if c <= end + merge_gap + 1:
            end = c
            rs_acc.update(reasons_by_col.get(c, []))
        else:
            segs.append((start, end, sorted(rs_acc)))
            start = end = c
            rs_acc = set(reasons_by_col.get(c, []))
    segs.append((start, end, sorted(rs_acc)))

    out: List[Dict[str, Any]] = []
    for a, b, rs in segs:
        ln = b - a + 1
        if ln < min_len:
            continue
        out.append(
            {
                "segment_start_x": int(a),
                "segment_end_x": int(b),
                "segment_len": int(ln),
                "risk_reasons": ";".join(rs),
            }
        )
    return out


def build_risk_context(
    final_cands: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    plot_box_t: Tuple[int, int, int, int],
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    ridge_map: Optional[np.ndarray],
    gt_json_path: str,
    st: SelectiveOracleSettings,
    tr_rule: Optional[dict],
    grid_like_proxy: Optional[float],
) -> Tuple[List[Dict[str, Any]], Set[int], List[Dict[str, Any]], Dict[str, Any]]:
    """feature_rows, dilated_risk_columns, segments, summary."""
    gt = None
    try:
        gt = _load_gt_json(gt_json_path)
    except Exception:
        gt = None

    feature_rows = extract_column_risk_features(
        final_cands,
        roi_w,
        roi_h,
        plot_box_t,
        comp_score_map,
        axis_dist_map,
        ridge_map,
        gt,
        st.run_domain,
        st.taxonomy_prior,
        tr_rule,
    )
    if grid_like_proxy is not None:
        for r in feature_rows:
            r["grid_like_proxy"] = float(grid_like_proxy)

    taxonomy_for_risk = None if st.disable_taxonomy_prior_for_risk else st.taxonomy_prior

    # stage 1) raw heuristic-only risk (taxonomy prior 제외)
    risk_raw, reasons_raw = compute_risk_columns(feature_rows, st, None)
    # stage 2) taxonomy prior 반영 후 risk
    risk_tax, reasons_tax = compute_risk_columns(feature_rows, st, taxonomy_for_risk)
    # stage 3) dilation
    risk_dil = dilate_risk_columns(risk_tax, reasons_tax, roi_w, st.risk_dilate_radius_columns)
    # stage 4) merge (segments only; 현재 최종 risk columns는 dilated set 유지)
    segments = merge_risk_segments(risk_dil, reasons_tax, roi_w, st.merge_gap_columns, st.min_segment_columns)
    merged_cols: Set[int] = set()
    for sg in segments:
        a = int(sg.get("segment_start_x", 0))
        b = int(sg.get("segment_end_x", -1))
        if b >= a:
            merged_cols.update(range(a, b + 1))

    summary = {
        "risk_column_count_raw": len(risk_raw),
        "risk_column_count_after_taxonomy_prior": len(risk_tax),
        "risk_column_count_after_dilation": len(risk_dil),
        "risk_column_count_after_merge": len(merged_cols),
        "risk_column_count_dilated": len(risk_dil),
        "total_column_count": int(roi_w),
        "risk_ratio_raw": float(len(risk_raw)) / float(max(1, roi_w)),
        "risk_ratio_after_taxonomy_prior": float(len(risk_tax)) / float(max(1, roi_w)),
        "risk_ratio_after_dilation": float(len(risk_dil)) / float(max(1, roi_w)),
        "risk_ratio_after_merge": float(len(merged_cols)) / float(max(1, roi_w)),
        "risk_ratio_final": float(len(risk_dil)) / float(max(1, roi_w)),
        "risk_ratio": float(len(risk_dil)) / float(max(1, roi_w)),
        "risk_segments": len(segments),
        "dilation_radius_columns": int(st.risk_dilate_radius_columns),
        "merge_gap_columns": int(st.merge_gap_columns),
        "min_segment_columns": int(st.min_segment_columns),
        "taxonomy_prior": st.taxonomy_prior or "",
        "taxonomy_prior_labels": (
            [t.strip() for t in str(st.taxonomy_prior).split(";") if t.strip()]
            if st.taxonomy_prior
            else []
        ),
        "taxonomy_prior_disabled_for_risk": bool(st.disable_taxonomy_prior_for_risk),
        "risk_disable_flags": {
            "disable_low_conf_margin_risk": bool(st.disable_low_conf_margin_risk),
            "disable_high_entropy_risk": bool(st.disable_high_entropy_risk),
            "disable_axis_proximity_risk": bool(st.disable_axis_proximity_risk),
            "disable_large_y_gap_risk": bool(st.disable_large_y_gap_risk),
            "disable_peak_window_risk": bool(st.disable_peak_window_risk),
            "disable_dp_margin_low_risk": bool(st.disable_dp_margin_low_risk),
            "disable_taxonomy_prior_for_risk": bool(st.disable_taxonomy_prior_for_risk),
            "taxonomy_prior_requires_margin": bool(st.taxonomy_prior_requires_margin),
            "high_entropy_requires_low_margin": bool(st.high_entropy_requires_low_margin),
        },
    }

    reason_counts: Dict[str, int] = {}
    reason_examples: Dict[str, List[int]] = {}
    for col, rs in reasons_raw.items():
        for r in rs:
            reason_counts[r] = reason_counts.get(r, 0) + 1
            if r not in reason_examples:
                reason_examples[r] = []
            if len(reason_examples[r]) < 10:
                reason_examples[r].append(int(col))
    summary["raw_risk_reason_counts"] = dict(sorted(reason_counts.items()))
    summary["raw_risk_reason_ratios"] = {
        k: float(v) / float(max(1, roi_w)) for k, v in sorted(reason_counts.items())
    }
    summary["raw_risk_reason_examples"] = {k: reason_examples[k] for k in sorted(reason_examples)}
    if st.risk_debug_include_columns:
        summary["raw_risk_by_column"] = {
            str(int(col)): list(dict.fromkeys(rs)) for col, rs in sorted(reasons_raw.items(), key=lambda it: int(it[0]))
        }
    return feature_rows, risk_dil, segments, summary


def append_risk_features_csv(
    path: str,
    sample_id: str,
    domain: str,
    rows: List[Dict[str, Any]],
) -> None:
    from pathlib import Path
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "domain",
        "x",
        "candidate_count",
        "top1_conf",
        "top2_conf",
        "conf_margin",
        "conf_entropy",
        "top1_axis_proximity",
        "top1_ridge_response",
        "top1_component_score",
        "top1_y",
        "top2_y",
        "top1_top2_y_gap",
        "empty_neighbor_count",
        "is_near_peak_window",
        "local_candidate_y_std",
        "local_conf_std",
        "style_group",
        "taxonomy_prior",
        "dp_local_cost_spike",
        "dp_margin_low",
        "dp_block_margin",
        "recovery_trigger_nearby",
        "grid_like_proxy",
        "axis_boundary_nearby",
    ]
    write_header = not p.is_file()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in fieldnames if k not in ("sample_id", "domain")}
            out["sample_id"] = sample_id
            out["domain"] = domain
            w.writerow(out)
