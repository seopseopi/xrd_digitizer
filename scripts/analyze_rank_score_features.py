#!/usr/bin/env python3
"""filtered 단계 rank score 특성 분석: raw 재구성 + rank_breakdown(있으면) + 가상 ablation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace.candidates import (
    ENVELOPE_SORT_BONUS,
    ENVELOPE_SORT_DECAY_PX,
    ENVELOPE_SORT_MIN_CANDIDATES,
    ENVELOPE_SORT_MIN_SPAN_PX,
    MIN_CONF_KEEP,
    LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT,
    LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT,
    neighbor_conf_top1_y_median,
)
from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column

_SK_WEIGHT = 0.16
_SK_TAU = 11.0


def _norm(fc: Mapping[Any, Any]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for k, v in fc.items():
        try:
            kk = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[kk] = list(v)
    return out


def _glob_one(run_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        raise FileNotFoundError(f"ambiguous {pattern!r}: {[str(h) for h in hits]}")
    return hits[0]


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _summ_gaps(name: str, gaps: List[float]) -> None:
    if not gaps:
        print(f"{name}: (없음)")
        return
    a = np.asarray(gaps, dtype=np.float64)
    print(
        f"{name}: mean={float(np.mean(a)):.6f} median={float(np.median(a)):.6f} p90={_pct(list(gaps), 90):.6f}"
    )


def _median_smooth_nan(hints: np.ndarray, half: int = 5) -> np.ndarray:
    n = int(hints.shape[0])
    out = np.copy(hints)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = hints[lo:hi]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = float(np.median(chunk))
    return out


def _envelope_bonus(y: int, y_top: int, use_env: bool) -> float:
    if not use_env:
        return 0.0
    dy = int(y) - y_top
    return ENVELOPE_SORT_BONUS * math.exp(-max(0, dy) / ENVELOPE_SORT_DECAY_PX)


def _local_evidence_bonus(y: float, lev_ref: Optional[float], lev_w: float, lev_tau: float) -> float:
    if lev_ref is None or lev_w <= 0.0:
        return 0.0
    return lev_w * math.exp(-abs(float(y) - lev_ref) / max(lev_tau, 1e-6))


def _pseudo_skeleton_hint(raw: Dict[int, List[dict]], roi_w: int) -> np.ndarray:
    """열 최소 y를 스무딩해 DP skeleton_hint 프록시."""
    ys = np.full(roi_w, np.nan, dtype=np.float64)
    for c in range(roi_w):
        lst = raw.get(c, [])
        if lst:
            ys[c] = float(min(int(x["y"]) for x in lst))
    return _median_smooth_nan(ys, half=5)


def _sk_bonus_proxy(y: float, gy: float) -> float:
    return _SK_WEIGHT * math.exp(-abs(float(y) - float(gy)) / _SK_TAU)


def _build_kept_and_meta(
    raw: Dict[int, List[dict]],
    col: int,
    roi_w: int,
    min_conf: float,
    disable_envelope: bool,
) -> Tuple[List[dict], int, int, bool, float]:
    valid: List[dict] = []
    cands = raw.get(col, [])
    for c in cands:
        try:
            cf = float(c["confidence"])
            _ = int(c["y"])
            if not math.isfinite(cf):
                continue
        except Exception:
            continue
        valid.append(c)
    kept = [c for c in valid if float(c["confidence"]) >= float(min_conf)]
    if not kept:
        return [], 0, 0, False, float("nan")
    n_raw = len(valid)
    y_top = min(int(c["y"]) for c in valid)
    y_bottom = max(int(c["y"]) for c in valid)
    span = y_bottom - y_top
    use_env = (
        (not disable_envelope)
        and n_raw >= ENVELOPE_SORT_MIN_CANDIDATES
        and span >= ENVELOPE_SORT_MIN_SPAN_PX
    )
    return kept, y_top, n_raw, use_env, float(span)


def _score_full(
    c: dict,
    *,
    y_top: int,
    use_env: bool,
    lev_ref: Optional[float],
    lev_w: float,
    lev_tau: float,
) -> float:
    return (
        float(c["confidence"])
        + _envelope_bonus(int(c["y"]), y_top, use_env)
        + _local_evidence_bonus(float(c["y"]), lev_ref, lev_w, lev_tau)
    )


def _virtual_metrics_one_column(
    kept: List[dict],
    gt_y: float,
    near_px: float,
    topk: int,
    ub_bound: int,
    score_fn: Callable[[dict], float],
) -> Tuple[bool, float, bool, bool]:
    kept_near = [c for c in kept if abs(float(c["y"]) - gt_y) <= near_px]
    has_near = bool(kept_near)
    scored = sorted(kept, key=lambda c: -score_fn(c))
    best_rank = float("inf")
    if kept_near:
        near_best = max(kept_near, key=lambda c: score_fn(c))
        bid = id(near_best)
        for i, c in enumerate(scored, 1):
            if id(c) == bid:
                best_rank = float(i)
                break
    top1 = scored[0]
    top1_upper = int(top1["y"]) < ub_bound
    tier = scored[: int(topk)]
    survives = any(abs(float(c["y"]) - gt_y) <= near_px for c in tier)
    return has_near, best_rank, top1_upper, survives


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank score feature attribution + virtual ablations")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--near-px", type=float, default=5.0)
    ap.add_argument("--upper-frac", type=float, default=0.2)
    ap.add_argument("--min-conf", type=float, default=float(MIN_CONF_KEEP))
    ap.add_argument("--disable-envelope", action="store_true", help="재구성 시 envelope 비활성화")
    ap.add_argument("--topk-filter", type=int, default=16)
    ap.add_argument(
        "--local-evidence-weight",
        type=float,
        default=float(LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT),
    )
    ap.add_argument(
        "--local-evidence-tau-px",
        type=float,
        default=float(LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dbg_path = run_dir / "debug.json"
    dbg = json.loads(dbg_path.read_text(encoding="utf-8")) if dbg_path.is_file() else {}
    plot_box = tuple(int(x) for x in dbg.get("plot_box", [0, 0, 0, 0]))
    roi_h = int(plot_box[3] - plot_box[1])
    roi_w = int(plot_box[2] - plot_box[0])
    ub_bound = max(1, int(math.ceil(roi_h * float(args.upper_frac))))

    raw_p = _glob_one(run_dir, "*_raw_candidates.json")
    if raw_p is None:
        print("raw_candidates.json 없음", file=sys.stderr)
        return 2
    raw = _norm(json.loads(raw_p.read_text(encoding="utf-8")))
    gt = _load_gt_json(str(args.gt_json))
    gt_by_col, _ = build_gt_y_roi_per_column(gt, plot_box, roi_h, roi_w)

    sk_hint = _pseudo_skeleton_hint(raw, roi_w)

    filt_dbg = dbg.get("candidate_filter_debug") if isinstance(dbg.get("candidate_filter_debug"), dict) else {}
    rb_cols = (filt_dbg.get("gt_near_column_debug") or {}) if filt_dbg else {}

    disable_env = bool(args.disable_envelope)

    total_cols = roi_w
    gt_mapped_kept_nonempty = 0
    top1_upper_among_gt_kept_nonempty = 0
    gt_near_cols = 0
    top1_upper_among_gt_near_kept = 0

    gap_bc: List[float] = []
    gap_total: List[float] = []
    gap_env: List[float] = []
    gap_lev: List[float] = []
    gap_sk: List[float] = []
    gap_comp: List[float] = []
    faux_src_gap: List[float] = []

    top1_y_hist: Counter[int] = Counter()
    gtn_y_hist: Counter[int] = Counter()
    top1_src: Counter[str] = Counter()
    gtn_src: Counter[str] = Counter()

    rb_checked = 0
    rb_total_score_mismatch = 0

    threshold_list = [16, 32, 64, 128]

    def empty_ablation() -> Dict[str, Any]:
        return {
            "denom_gt_near_kept": 0,
            "recall_topk": 0.0,
            "top1_upper_cols": 0,
            "rank_le_t": {t: 0 for t in threshold_list},
        }

    ablation_names = [
        "baseline_full_conf_plus_env",
        "confidence_only",
        "rank_without_envelope",
        "rank_plus_local_evidence_virtual",
        "rank_demote_non_raw_source",
        "rank_minus_comp_proxy",
        "rank_without_source_bonus_hypothesis_identical_to_baseline",
    ]
    ablations = {k: empty_ablation() for k in ablation_names}

    for col in range(roi_w):
        gty = gt_by_col.get(col)
        if gty is None:
            continue

        kept, y_top, n_raw, use_env, _span = _build_kept_and_meta(
            raw, col, roi_w, float(args.min_conf), disable_env
        )
        if not kept:
            continue
        gt_mapped_kept_nonempty += 1

        lev_ref = neighbor_conf_top1_y_median(raw, col, roi_w)
        lev_w = float(args.local_evidence_weight)
        lev_tau = float(args.local_evidence_tau_px)

        use_env_eff = use_env and not disable_env

        def sf_full(c: dict) -> float:
            return _score_full(c, y_top=y_top, use_env=use_env_eff, lev_ref=None, lev_w=0.0, lev_tau=lev_tau)

        def sf_conf(c: dict) -> float:
            return float(c["confidence"])

        def sf_env_off(c: dict) -> float:
            return float(c["confidence"])

        def sf_local_virt(c: dict) -> float:
            return _score_full(
                c, y_top=y_top, use_env=use_env_eff, lev_ref=lev_ref, lev_w=lev_w, lev_tau=lev_tau
            )

        def sf_demote_non_raw(c: dict) -> float:
            pen = 0.018 if str(c.get("source", "")) != "raw" else 0.0
            return sf_full(c) - pen

        def sf_minus_comp(c: dict) -> float:
            cs = float(c.get("comp_score", 0.0))
            return sf_full(c) - 0.028 * math.tanh(cs / 7.0)

        score_specs = {
            "baseline_full_conf_plus_env": sf_full,
            "confidence_only": sf_conf,
            "rank_without_envelope": sf_env_off,
            "rank_plus_local_evidence_virtual": sf_local_virt,
            "rank_demote_non_raw_source": sf_demote_non_raw,
            "rank_minus_comp_proxy": sf_minus_comp,
            "rank_without_source_bonus_hypothesis_identical_to_baseline": sf_full,
        }

        scored_main = sorted(kept, key=lambda c: -sf_full(c))
        top1 = scored_main[0]
        if int(top1["y"]) < ub_bound:
            top1_upper_among_gt_kept_nonempty += 1

        near_kept = [c for c in kept if abs(float(c["y"]) - float(gty)) <= float(args.near_px)]
        if not near_kept:
            continue

        gt_near_cols += 1
        if int(top1["y"]) < ub_bound:
            top1_upper_among_gt_near_kept += 1

        best_near = max(near_kept, key=lambda c: sf_full(c))

        bc_t = float(top1["confidence"])
        bc_g = float(best_near["confidence"])
        env_t = _envelope_bonus(int(top1["y"]), y_top, use_env_eff)
        env_g = _envelope_bonus(int(best_near["y"]), y_top, use_env_eff)
        lev_t = _local_evidence_bonus(float(top1["y"]), lev_ref, lev_w, lev_tau)
        lev_g = _local_evidence_bonus(float(best_near["y"]), lev_ref, lev_w, lev_tau)
        total_t = bc_t + env_t + 0.0
        total_g = bc_g + env_g + 0.0

        gap_bc.append(bc_t - bc_g)
        gap_total.append(total_t - total_g)
        gap_env.append(env_t - env_g)
        gap_lev.append(lev_t - lev_g)

        gy_col = float(sk_hint[col]) if not math.isnan(sk_hint[col]) else float("nan")
        if math.isfinite(gy_col):
            gap_sk.append(_sk_bonus_proxy(float(top1["y"]), gy_col) - _sk_bonus_proxy(float(best_near["y"]), gy_col))
        gap_comp.append(float(top1.get("comp_score", 0.0)) - float(best_near.get("comp_score", 0.0)))

        faux_top = 0.022 if str(top1.get("source", "")) != "raw" else 0.0
        faux_g = 0.022 if str(best_near.get("source", "")) != "raw" else 0.0
        faux_src_gap.append(faux_top - faux_g)

        top1_y_hist[int(top1["y"])] += 1
        gtn_y_hist[int(best_near["y"])] += 1
        top1_src[str(top1.get("source", ""))] += 1
        gtn_src[str(best_near.get("source", ""))] += 1

        ce = rb_cols.get(str(int(col))) if isinstance(rb_cols, dict) else None
        if isinstance(ce, dict) and ce.get("rank_breakdown"):
            for row in ce["rank_breakdown"][:5]:
                rb_checked += 1
                ry = int(row["y"])
                mm = [
                    m
                    for m in kept
                    if int(m["y"]) == ry and abs(float(m["confidence"]) - float(row["base_confidence"])) < 1e-5
                ]
                if not mm:
                    continue
                m0 = mm[0]
                lev_x = float(row.get("local_evidence_bonus", 0.0))
                rec_total = float(row["final_filter_rank_score"])
                exp_total = (
                    float(m0["confidence"]) + _envelope_bonus(int(m0["y"]), y_top, use_env_eff) + lev_x
                )
                if abs(rec_total - exp_total) > 0.02:
                    rb_total_score_mismatch += 1

        for ab_name, fn in score_specs.items():
            bucket = ablations[ab_name]
            has_near, best_rank, t1_upper, survives = _virtual_metrics_one_column(
                kept,
                float(gty),
                float(args.near_px),
                int(args.topk_filter),
                ub_bound,
                fn,
            )
            if not has_near:
                continue
            bucket["denom_gt_near_kept"] += 1
            if survives:
                bucket["recall_topk"] += 1.0
            if t1_upper:
                bucket["top1_upper_cols"] += 1
            for t in threshold_list:
                if best_rank <= float(t):
                    bucket["rank_le_t"][t] += 1

    print("=== cohort ===")
    print(f"total_roi_columns={total_cols}")
    print(f"gt_mapped_columns_nonempty_kept={gt_mapped_kept_nonempty}")
    print(f"top1_upper_band_columns(among gt_mapped_nonempty_kept)={top1_upper_among_gt_kept_nonempty}")
    print(f"gt_near_present_columns_px{args.near_px:g}={gt_near_cols}")
    print(f"top1_upper_band_columns(among gt_near_present)={top1_upper_among_gt_near_kept}")

    print("\n=== top1 vs best GT-near (동일 열, kept 내, baseline 정렬점수 최고 GT-near) gap ===")
    _summ_gaps("gap_base_confidence (top1 - GTnear)", gap_bc)
    _summ_gaps("gap_final_filter_rank_score (conf+env, 로컬증거 미포함)", gap_total)
    _summ_gaps("gap_envelope_bonus", gap_env)
    _summ_gaps(
        f"gap_local_evidence_bonus (가상 w={args.local_evidence_weight}, tau={args.local_evidence_tau_px})",
        gap_lev,
    )
    _summ_gaps("gap_skeleton_hint_bonus_proxy_DP단계참고", gap_sk)
    _summ_gaps("gap_source_implicit_hypothesis_non_raw_bonus022", faux_src_gap)
    _summ_gaps("gap_comp_score (explicit 정렬 미사용, 신뢰도 내재만)", gap_comp)

    print("\n=== component/source bonus gap ===")
    print(
        "filter 단계 명시적 source/component 보너스 없음 → rank_without_source_bonus는 baseline과 동일 순서(가설 tilt는 위 faux 행 참고)."
    )

    print("\n=== y 분포 (상위 12 bins, GT-near 열만) ===")
    print("top1_y:", top1_y_hist.most_common(12))
    print("GTnear_best_y:", gtn_y_hist.most_common(12))

    print("\n=== source 분포 ===")
    print("top1_source:", dict(top1_src))
    print("GTnear_best_source:", dict(gtn_src))

    print("\n=== rank_gap 설명 후보 (평균 절대 갭 기준 상위) ===")
    feats = [
        ("|gap_base_confidence|", float(np.mean(np.abs(np.asarray(gap_bc)))) if gap_bc else 0.0),
        ("|gap_envelope_bonus|", float(np.mean(np.abs(np.asarray(gap_env)))) if gap_env else 0.0),
        ("|gap_final_score_conf_plus_env|", float(np.mean(np.abs(np.asarray(gap_total)))) if gap_total else 0.0),
        ("|gap_local_evidence_virt|", float(np.mean(np.abs(np.asarray(gap_lev)))) if gap_lev else 0.0),
        ("|gap_skeleton_hint_proxy|", float(np.mean(np.abs(np.asarray(gap_sk)))) if gap_sk else 0.0),
        ("|gap_comp_score|", float(np.mean(np.abs(np.asarray(gap_comp)))) if gap_comp else 0.0),
    ]
    for name, val in sorted(feats, key=lambda x: -x[1]):
        print(f"  {name}: {val:.6f}")

    print("\n=== 가상 ablation (각 열별 score_fn 정렬 후 filtered topK 내 GT-near 생존) ===")
    for name in ablation_names:
        b = ablations[name]
        d = max(int(b["denom_gt_near_kept"]), 1)
        recall = float(b["recall_topk"]) / float(d)
        print(f"\n[{name}]")
        print(f"  denom_columns_with_gt_near_in_kept={b['denom_gt_near_kept']}")
        print(f"  expected_filtered_gt_near_recall_px{args.near_px:g}={recall:.6f}")
        print(f"  top1_upper_band_columns_under_this_ranking={b['top1_upper_cols']}")
        for t in threshold_list:
            print(f"  columns_best_gt_near_rank_le_{t}={b['rank_le_t'][t]}")

    ident_recall = abs(
        ablations["confidence_only"]["recall_topk"] - ablations["rank_without_envelope"]["recall_topk"]
    ) < 1e-9
    print(
        f"\nconfidence_only 와 rank_without_envelope 동일 recall 일치: {ident_recall} "
        f"(envelope가 유일한 비-confidence 단조 가산이면 순서 동일)"
    )

    print("\n=== rank_breakdown 교차검증 ===")
    print(f"sampled_rows_checked={rb_checked} rough_total_mismatch={rb_total_score_mismatch}")

    print("\n=== 재설계 후보 선택 (이 스크립트 출력 기준 가이드) ===")
    print(
        "가상 ablation에서 rank_plus_local_evidence_virtual 가 baseline_full 대비 recall↑·upper-top1↓이면 "
        "옵션 B(local curve evidence) 우선. "
        "confidence_only 만 크게 나아가면 A(confidence normalization). "
        "변화 미미하면 E(메트릭 유지) 또는 후속 SCORE_MAP 단계."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
