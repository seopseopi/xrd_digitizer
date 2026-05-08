"""
Selective GT oracle: 위험 열에서만 oracle confidence, 나머지는 rule confidence 유지.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.selective_oracle_settings import SelectiveOracleSettings
from trace.dp_trace import dp_trace, refine_dp_path_column_apex_pull
from trace.candidates import dp_transition_window_width
from trace.oracle_rerank import apply_oracle_scores_to_candidates, build_gt_y_roi_per_column
from trace.risk_detector import build_risk_context, extract_column_risk_features


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


def _valid_ratio(path: List[Optional[int]]) -> float:
    if not path:
        return 0.0
    return float(sum(1 for p in path if p is not None)) / float(len(path))


def _confidence_map(fc: Dict[int, List[dict]]) -> Dict[Tuple[int, int], float]:
    m: Dict[Tuple[int, int], float] = {}
    for col, cands in fc.items():
        ci = int(col)
        for c in cands:
            m[(ci, int(c["y"]))] = float(c.get("confidence", 0.0))
    return m


def _confidence_sha256(fc: Dict[int, List[dict]]) -> str:
    vec: List[Tuple[int, int, float]] = []
    for col in sorted(fc.keys(), key=lambda x: int(x)):
        for c in fc[int(col)]:
            vec.append((int(col), int(c["y"]), float(c.get("confidence", 0.0))))
    raw = json.dumps(vec, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _candidate_y_keys_sha256(fc: Dict[int, List[dict]]) -> str:
    keys: List[Tuple[int, int]] = []
    for col in sorted(fc.keys(), key=lambda x: int(x)):
        ys = sorted({int(c["y"]) for c in fc[int(col)]})
        for y in ys:
            keys.append((int(col), y))
    raw = json.dumps(keys, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _candidate_field_summary(fc: Dict[int, List[dict]]) -> Dict[str, Any]:
    cols = sorted(fc.keys(), key=lambda x: int(x))
    ys: List[int] = []
    confs: List[float] = []
    n_cand = 0
    for col in cols:
        for c in fc[int(col)]:
            n_cand += 1
            ys.append(int(c["y"]))
            confs.append(float(c.get("confidence", 0.0)))
    first5 = {}
    for col in cols[:5]:
        cc = fc[int(col)]
        first5[str(col)] = len(cc)
    return {
        "num_columns": len(cols),
        "candidates_per_first_5_columns": first5,
        "total_candidates": n_cand,
        "y_min": int(min(ys)) if ys else None,
        "y_max": int(max(ys)) if ys else None,
        "confidence_min": float(min(confs)) if confs else None,
        "confidence_max": float(max(confs)) if confs else None,
        "confidence_mean": float(np.mean(confs)) if confs else None,
    }


def _diff_conf_maps(
    a: Dict[Tuple[int, int], float],
    b: Dict[Tuple[int, int], float],
) -> Tuple[int, float, float, float]:
    """confidence가 실제로 다른 (col,y) 셀만 집계한다. 한쪽만 존재하면 구조적 차이로 1.0 페널티."""
    keys = set(a) | set(b)
    abs_diffs: List[float] = []
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            if va is None and vb is None:
                continue
            abs_diffs.append(1.0)
            continue
        d = abs(va - vb)
        if d > 1e-15:
            abs_diffs.append(float(d))
    n_diff = len(abs_diffs)
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    mean_abs = float(sum(abs_diffs) / n_diff) if n_diff else 0.0
    mean_sq = float(sum(d * d for d in abs_diffs) / n_diff) if n_diff else 0.0
    return n_diff, max_abs, mean_abs, mean_sq


def _first_col_where_maps_differ(
    ma: Dict[Tuple[int, int], float],
    mb: Dict[Tuple[int, int], float],
) -> Optional[int]:
    cols_a = {k[0] for k in ma}
    cols_b = {k[0] for k in mb}
    for col in sorted(cols_a | cols_b):
        keys = [k for k in ma if k[0] == col] + [k for k in mb if k[0] == col]
        ys = sorted({k[1] for k in keys})
        for y in ys:
            va = ma.get((col, y))
            vb = mb.get((col, y))
            if va is None and vb is None:
                continue
            if va is None or vb is None or abs(va - vb) > 1e-12:
                return int(col)
    return None


def _column_triplet_lists(
    fc_r: Dict[int, List[dict]],
    fc_s: Dict[int, List[dict]],
    fc_g: Dict[int, List[dict]],
    col: int,
) -> Dict[str, Any]:
    def _cands(fc: Dict[int, List[dict]], ci: int) -> List[dict]:
        if ci in fc:
            return fc[ci]
        if str(ci) in fc:  # pragma: no cover
            return fc[str(ci)]  # type: ignore[index]
        for k in fc:
            if int(k) == ci:
                return fc[k]
        return []

    cr, cs, cg = _cands(fc_r, col), _cands(fc_s, col), _cands(fc_g, col)
    ys = sorted({int(c["y"]) for c in cr + cs + cg})

    def tops(cc: List[dict]) -> Optional[int]:
        if not cc:
            return None
        best = max(cc, key=lambda z: float(z.get("confidence", 0.0)))
        return int(best["y"])

    rule_conf = [next((float(c.get("confidence", 0.0)) for c in cr if int(c["y"]) == y), None) for y in ys]
    sel_conf = [next((float(c.get("confidence", 0.0)) for c in cs if int(c["y"]) == y), None) for y in ys]
    glo_conf = [next((float(c.get("confidence", 0.0)) for c in cg if int(c["y"]) == y), None) for y in ys]

    return {
        "column": col,
        "y_list": ys,
        "rule_confidence_list": rule_conf,
        "selective_confidence_list": sel_conf,
        "global_confidence_list": glo_conf,
        "rule_top_y": tops(cr),
        "selective_top_y": tops(cs),
        "global_top_y": tops(cg),
    }


def _path_sha256(path: List[Optional[int]]) -> str:
    raw = json.dumps(path, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _maybe_dump_selective_candidate_compare(
    dump_path: str,
    *,
    fc_rule: Dict[int, List[dict]],
    fc_global: Dict[int, List[dict]],
    fc_sel: Dict[int, List[dict]],
    tr_rule_dp: Dict[str, Any],
    tr_global_dp: Dict[str, Any],
    tr_sel_dp: Dict[str, Any],
    path_rule_pre: List[Optional[int]],
    path_global_pre: List[Optional[int]],
    path_sel_pre: List[Optional[int]],
    path_rule_final: List[Optional[int]],
    path_global_final: List[Optional[int]],
    path_sel_final: List[Optional[int]],
    sel_summary: Dict[str, Any],
) -> None:
    """환경변수 XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON 경로가 있을 때만 덤프."""
    p = dump_path.strip()
    if not p:
        return
    out = Path(p)
    out.parent.mkdir(parents=True, exist_ok=True)

    mr, ms, mg = _confidence_map(fc_rule), _confidence_map(fc_sel), _confidence_map(fc_global)
    hr, hs, hg = _confidence_sha256(fc_rule), _confidence_sha256(fc_sel), _confidence_sha256(fc_global)
    ykr, yks, ykg = (
        _candidate_y_keys_sha256(fc_rule),
        _candidate_y_keys_sha256(fc_sel),
        _candidate_y_keys_sha256(fc_global),
    )

    n_rs, max_rs, mean_abs_rs, mean_rs = _diff_conf_maps(mr, ms)
    n_gs, max_gs, mean_abs_gs, mean_gs = _diff_conf_maps(mg, ms)
    n_gr, max_gr, mean_abs_gr, mean_gr = _diff_conf_maps(mr, mg)

    first_sg = _first_col_where_maps_differ(ms, mg)
    detail_col: Optional[Dict[str, Any]] = None
    if first_sg is not None:
        detail_col = _column_triplet_lists(fc_rule, fc_sel, fc_global, first_sg)

    payload = {
        "dump_reason": "dp_trace 직입력 fc_rule/fc_global/fc_sel confidence 비교",
        "note_apply_selective_return": (
            "apply_selective_oracle_scores_to_candidates는 (fc, summary)를 반환하며 "
            "dp_trace에는 반환된 fc만 사용한다."
        ),
        "fc_rule_summary": _candidate_field_summary(fc_rule),
        "fc_sel_summary": _candidate_field_summary(fc_sel),
        "fc_global_summary": _candidate_field_summary(fc_global),
        "confidence_sha256": {"fc_rule": hr, "fc_sel": hs, "fc_global": hg},
        "candidate_y_keys_sha256": {"fc_rule": ykr, "fc_sel": yks, "fc_global": ykg},
        "fc_sel_equals_fc_rule_y_keys_sha": ykr == yks,
        "fc_sel_equals_fc_global_y_keys_sha": yks == ykg,
        "fc_sel_equals_fc_rule_sha": hr == hs,
        "fc_sel_equals_fc_global_sha": hs == hg,
        "confidence_diff_counts": {
            "rule_vs_selective_diff_cells": n_rs,
            "global_vs_selective_diff_cells": n_gs,
            "rule_vs_global_diff_cells": n_gr,
        },
        "confidence_diff_abs_max": {
            "rule_vs_selective": max_rs,
            "global_vs_selective": max_gs,
            "rule_vs_global": max_gr,
        },
        "confidence_diff_mean_abs": {
            "rule_vs_selective": mean_abs_rs,
            "global_vs_selective": mean_abs_gs,
            "rule_vs_global": mean_abs_gr,
        },
        "confidence_diff_mean_sq": {
            "rule_vs_selective": mean_rs,
            "global_vs_selective": mean_gs,
            "rule_vs_global": mean_gr,
        },
        "first_col_where_selective_differs_from_global": first_sg,
        "first_diff_column_detail": detail_col,
        "dp_trace_input": {
            "fc_sel_python_id": id(fc_sel),
            "fc_sel_type": type(fc_sel).__name__,
            "fc_sel_num_columns": len(fc_sel),
            "fc_sel_confidence_sha256_pre_dp_trace": hs,
        },
        "dp_trace_pre_apex": {
            "rule_trace_score": float(tr_rule_dp.get("trace_score", float("nan"))),
            "global_trace_score": float(tr_global_dp.get("trace_score", float("nan"))),
            "selective_trace_score": float(tr_sel_dp.get("trace_score", float("nan"))),
            "rule_path_sha256": _path_sha256(path_rule_pre),
            "global_path_sha256": _path_sha256(path_global_pre),
            "selective_path_sha256": _path_sha256(path_sel_pre),
            "selective_path_equals_rule": path_sel_pre == path_rule_pre,
            "selective_path_equals_global": path_sel_pre == path_global_pre,
        },
        "dp_trace_post_apex": {
            "rule_final_y_sha256": _path_sha256(path_rule_final),
            "global_final_y_sha256": _path_sha256(path_global_final),
            "selective_final_y_sha256": _path_sha256(path_sel_final),
            "selective_final_y_equals_rule": path_sel_final == path_rule_final,
            "selective_final_y_equals_global": path_sel_final == path_global_final,
        },
        "selective_apply_summary_returned": sel_summary,
    }

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_selective_oracle_scores_to_candidates(
    final_cands: Dict[int, List[dict]],
    gt_y_roi_by_col: Dict[int, float],
    sigma_px: float,
    risk_columns: Set[int],
) -> Tuple[Dict[int, List[dict]], Dict[str, Any]]:
    """risk_columns 안에서만 oracle confidence; 밖에서는 rule confidence 유지.

    ``final_cands``는 변경하지 않으며, deepcopy한 dict ``fc``를 수정해 (fc, summary)로 반환한다.
    """
    sigma = max(float(sigma_px), 1e-6)
    applied = 0
    preserved = 0
    dists: List[float] = []

    fc = copy.deepcopy(final_cands)
    for col, cands in fc.items():
        col_i = int(col)
        gt_y = gt_y_roi_by_col.get(col_i)
        use = col_i in risk_columns and gt_y is not None and bool(cands)
        for c in cands:
            if "rule_confidence_before_oracle" not in c:
                c["rule_confidence_before_oracle"] = float(c.get("confidence", 0.0))
            if use:
                cy = float(c.get("y", 0.0))
                d = abs(cy - float(gt_y))
                dists.append(d)
                c["oracle_dist_px"] = float(d)
                oc = float(np.exp(-((d / sigma) ** 2)))
                c["oracle_confidence"] = oc
                c["confidence"] = oc
                c["selective_oracle_applied"] = True
                c["risk_reason"] = "in_risk_segment"
                applied += 1
            else:
                c["oracle_confidence"] = None
                c["oracle_dist_px"] = None
                c["confidence"] = float(c["rule_confidence_before_oracle"])
                c["selective_oracle_applied"] = False
                c["risk_reason"] = None
                preserved += 1
        if use:
            cands.sort(key=lambda cc: -float(cc.get("confidence", 0.0)))

    summary = {
        "sigma_px": float(sigma_px),
        "applied_candidate_count": float(applied),
        "preserved_rule_candidate_count": float(preserved),
        "mean_oracle_dist_px": float(np.mean(dists)) if dists else None,
        "risk_columns_used": int(len(risk_columns)),
    }
    return fc, summary


def run_dp_with_gt_selective_oracle_rerank(
    final_cands_orig: Dict[int, List[dict]],
    roi_w: int,
    roi_h: int,
    comp_score_map: np.ndarray,
    axis_dist_map: np.ndarray,
    ridge_map: Optional[np.ndarray],
    gt_json_path: str,
    plot_box_t: Tuple[int, int, int, int],
    st: SelectiveOracleSettings,
    *,
    use_dp_column_apex_pull: bool,
) -> Tuple[Dict[int, List[dict]], dict, dict, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      final_cands_selective, trace_selective, meta_bundle,
      feature_rows, segments, risk_summary
    meta_bundle includes keys for debug.json model_assist selective branch.
    """
    from trace.oracle_rerank import _load_gt_json

    gt = _load_gt_json(gt_json_path)
    gt_y_roi_by_col, gt_curve_meta = build_gt_y_roi_per_column(gt, plot_box_t, roi_h, roi_w)

    fc_rule = copy.deepcopy(final_cands_orig)
    tr_rule_dp = dp_trace(fc_rule, roi_w, roi_h, comp_score_map)
    path_rule_pre = list(tr_rule_dp["path"])
    path_rule_final = _apex(tr_rule_dp["path"], fc_rule, roi_w, use_dp_column_apex_pull)
    tr_rule = {**tr_rule_dp, "path": path_rule_final}
    ts_r = float(tr_rule["trace_score"])
    vr_r = _valid_ratio(path_rule_final)

    # global oracle path (full replace) for comparison meta
    fc_global = copy.deepcopy(final_cands_orig)
    global_summary = apply_oracle_scores_to_candidates(fc_global, gt_y_roi_by_col, st.sigma_px)
    tr_global_dp = dp_trace(fc_global, roi_w, roi_h, comp_score_map)
    path_global_pre = list(tr_global_dp["path"])
    path_global_final = _apex(tr_global_dp["path"], fc_global, roi_w, use_dp_column_apex_pull)
    tr_global = {**tr_global_dp, "path": path_global_final}
    ts_g = float(tr_global["trace_score"])
    vr_g = _valid_ratio(path_global_final)

    force_clear = (
        st.styled_real_default_off
        and str(st.run_domain).lower() in ("styled", "real_like")
        and not st.allow_styled_real_selective
    )
    domain_ok = str(st.run_domain).lower() in tuple(x.lower() for x in st.apply_to_styles)
    style_policy_meta = {
        "run_domain": str(st.run_domain),
        "allow_styled_real_selective": bool(st.allow_styled_real_selective),
        "apply_to_styles": list(st.apply_to_styles),
        "domain_ok": bool(domain_ok),
        "force_clear": bool(force_clear),
    }
    if force_clear or not domain_ok:
        from trace.oracle_rerank import _load_gt_json

        gt_feat = None
        try:
            gt_feat = _load_gt_json(gt_json_path)
        except Exception:
            gt_feat = None
        feature_rows = extract_column_risk_features(
            final_cands_orig,
            roi_w,
            roi_h,
            plot_box_t,
            comp_score_map,
            axis_dist_map,
            ridge_map,
            gt_feat,
            st.run_domain,
            st.taxonomy_prior,
            tr_rule,
        )
        if st.grid_like_proxy is not None:
            for r in feature_rows:
                r["grid_like_proxy"] = float(st.grid_like_proxy)
        risk_cols = set()
        segments = []
        risk_summary = {
            "risk_column_count_raw": 0,
            "risk_column_count_dilated": 0,
            "total_column_count": int(roi_w),
            "risk_ratio": 0.0,
            "risk_segments": 0,
            "style_policy_skip": True,
            **style_policy_meta,
        }
    else:
        feature_rows, risk_cols, segments, risk_summary = build_risk_context(
            final_cands_orig,
            roi_w,
            roi_h,
            plot_box_t,
            comp_score_map,
            axis_dist_map,
            ridge_map,
            gt_json_path,
            st,
            tr_rule,
            st.grid_like_proxy,
        )
        risk_summary = {**risk_summary, "style_policy_skip": False, **style_policy_meta}

    fc_sel, sel_summary = apply_selective_oracle_scores_to_candidates(
        final_cands_orig,
        gt_y_roi_by_col,
        st.sigma_px,
        risk_cols,
    )
    tr_sel_dp = dp_trace(fc_sel, roi_w, roi_h, comp_score_map)
    path_sel_pre = list(tr_sel_dp["path"])
    path_sel_final = _apex(tr_sel_dp["path"], fc_sel, roi_w, use_dp_column_apex_pull)
    tr_sel = {**tr_sel_dp, "path": path_sel_final}

    dump_json = os.environ.get("XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON", "").strip()
    if dump_json:
        _maybe_dump_selective_candidate_compare(
            dump_json,
            fc_rule=fc_rule,
            fc_global=fc_global,
            fc_sel=fc_sel,
            tr_rule_dp=tr_rule_dp,
            tr_global_dp=tr_global_dp,
            tr_sel_dp=tr_sel_dp,
            path_rule_pre=path_rule_pre,
            path_global_pre=path_global_pre,
            path_sel_pre=path_sel_pre,
            path_rule_final=path_rule_final,
            path_global_final=path_global_final,
            path_sel_final=path_sel_final,
            sel_summary=sel_summary,
        )
    ts_s = float(tr_sel["trace_score"])
    vr_s = _valid_ratio(path_sel_final)

    meta = {
        "enabled": True,
        "gt_json_path": str(Path(gt_json_path).resolve()),
        "sigma_px": float(st.sigma_px),
        "gt_curve": gt_curve_meta,
        "trace_score_rule_dp": ts_r,
        "trace_score_global_oracle_dp": ts_g,
        "trace_score_selective_oracle_dp": ts_s,
        "valid_ratio_rule_dp": vr_r,
        "valid_ratio_global_oracle_dp": vr_g,
        "valid_ratio_selective_oracle_dp": vr_s,
        "global_oracle_score_summary": global_summary,
        "selective_oracle_score_summary": sel_summary,
        "risk_summary": risk_summary,
        "risk_segments_detail": segments,
        "dp_branch_committed": "selective_oracle_rerank",
    }

    return fc_sel, tr_sel, meta, feature_rows, segments, risk_summary
