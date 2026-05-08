#!/usr/bin/env python3
"""raw 후보 confidence·특성 비교 및 정규화 가상 실험(filter 단계 재현)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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
)
from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column

CONF_PART_KEYS = [
    "conf_weight_color",
    "conf_weight_continuity",
    "conf_weight_component",
    "conf_weight_axis_penalty",
]
RAW_FEAT_KEYS = ["color_dist", "axis_dist", "comp_score"]


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


def _glob_one(run_dir: Path, pattern: str) -> Path:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    if len(hits) > 1:
        raise FileNotFoundError(f"ambiguous {pattern}: {hits}")
    return hits[0]


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _summ(xs: List[float], label: str) -> None:
    if not xs:
        print(f"{label}: (없음)")
        return
    print(
        f"{label}: mean={float(np.mean(xs)):.6f} median={float(np.median(xs)):.6f} "
        f"p90={_pct(xs, 90):.6f} n={len(xs)}"
    )


def _env_bonus(y: int, y_top: int, use_env: bool) -> float:
    if not use_env:
        return 0.0
    dy = int(y) - y_top
    return ENVELOPE_SORT_BONUS * math.exp(-max(0, dy) / ENVELOPE_SORT_DECAY_PX)


def _column_kept(
    raw: Dict[int, List[dict]],
    col: int,
    min_conf: float,
) -> Tuple[List[dict], List[dict], int, bool]:
    """valid, kept, y_top_from_valid, use_env."""
    cands = raw.get(col, [])
    valid: List[dict] = []
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
        return valid, kept, 0, False
    y_top = min(int(c["y"]) for c in valid)
    y_bottom = max(int(c["y"]) for c in valid)
    span = y_bottom - y_top
    n_raw = len(valid)
    use_env = n_raw >= ENVELOPE_SORT_MIN_CANDIDATES and span >= ENVELOPE_SORT_MIN_SPAN_PX
    return valid, kept, y_top, use_env


def _sort_total(
    kept: List[dict],
    col: int,
    use_env: bool,
    y_top: int,
    virt_map: Dict[Tuple[int, int], float],
) -> List[Tuple[float, dict]]:
    scored = []
    for c in kept:
        vc = float(virt_map[(col, int(c["y"]))])
        tot = vc + _env_bonus(int(c["y"]), y_top, use_env)
        scored.append((tot, c))
    scored.sort(key=lambda t: -t[0])
    return scored


def _metrics_virtual(
    scored: List[Tuple[float, dict]],
    gt_y: float,
    near_px: float,
    topk: int,
    ub_bound: int,
) -> Tuple[float, bool, bool]:
    order = [c for _, c in scored]
    best_rank = float("inf")
    for i, c in enumerate(order, 1):
        if abs(float(c["y"]) - gt_y) <= near_px:
            best_rank = min(best_rank, float(i))
    tier = order[:topk]
    survives = any(abs(float(c["y"]) - gt_y) <= near_px for c in tier)
    top1_u = bool(order and int(order[0]["y"]) < ub_bound)
    return best_rank, top1_u, survives


def _softmax_scores(confs: np.ndarray, T: float) -> np.ndarray:
    T = max(float(T), 1e-9)
    z = np.asarray(confs, dtype=np.float64) / T
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def _virtual_aggregate(
    raw: Dict[int, List[dict]],
    gt_by_col: Dict[int, float],
    roi_w: int,
    min_conf: float,
    near_px: float,
    topk: int,
    ub: int,
    virt_builder: Callable[[int, List[dict]], Dict[Tuple[int, int], float]],
) -> Dict[str, Any]:
    thresh = [16, 32, 64, 128]
    le_t = {t: 0 for t in thresh}
    denom = 0
    recall_hit = 0
    top1_ub_n = 0
    path_ub = 0
    for col in range(roi_w):
        gty = gt_by_col.get(col)
        if gty is None:
            continue
        _, kept, y_top, use_env = _column_kept(raw, col, min_conf)
        if not kept:
            continue
        if not any(abs(float(c["y"]) - float(gty)) <= near_px for c in kept):
            continue
        denom += 1
        vmap = virt_builder(col, kept)
        scored = _sort_total(kept, col, use_env, y_top, vmap)
        br, t1u, surv = _metrics_virtual(scored, float(gty), near_px, topk, ub)
        if surv:
            recall_hit += 1
        if t1u:
            top1_ub_n += 1
        if scored and int(scored[0][1]["y"]) < ub:
            path_ub += 1
        for t in thresh:
            if br <= float(t):
                le_t[t] += 1
    return {
        "denom_gt_near_kept": denom,
        "expected_filtered_gt_near_recall": recall_hit / max(denom, 1),
        "top1_upper_band_columns": top1_ub_n,
        "path_upper_proxy_fraction": path_ub / max(denom, 1),
        "rank_le": dict(le_t),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--near-px", type=float, default=5.0)
    ap.add_argument("--upper-frac", type=float, default=0.2)
    ap.add_argument("--min-conf", type=float, default=float(MIN_CONF_KEEP))
    ap.add_argument("--topk-filter", type=int, default=16)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dbg = json.loads((run_dir / "debug.json").read_text(encoding="utf-8"))
    plot_box = tuple(int(x) for x in dbg["plot_box"])
    roi_h = int(plot_box[3] - plot_box[1])
    roi_w = int(plot_box[2] - plot_box[0])
    ub = max(1, int(math.ceil(roi_h * float(args.upper_frac))))
    topk = int(args.topk_filter)
    min_conf = float(args.min_conf)
    near_px = float(args.near_px)

    raw = _norm(json.loads(_glob_one(run_dir, "*_raw_candidates.json").read_text(encoding="utf-8")))
    gt = _load_gt_json(str(args.gt_json))
    gt_by_col, _ = build_gt_y_roi_per_column(gt, plot_box, roi_h, roi_w)

    first_nonempty = next((raw[k][0] for k in sorted(raw.keys()) if raw[k]), {})
    enrich = bool(first_nonempty.get("conf_weight_color") is not None)

    ub_confs: List[float] = []
    gn_confs: List[float] = []
    ub_parts = {k: [] for k in CONF_PART_KEYS}
    gn_parts = {k: [] for k in CONF_PART_KEYS}
    ub_rawf = {k: [] for k in RAW_FEAT_KEYS}
    gn_rawf = {k: [] for k in RAW_FEAT_KEYS}

    cols_gt_near = 0
    cols_top1_ub = 0
    src_ub = Counter()
    src_gn = Counter()
    src_tot = Counter()
    top1_conf_best_y: Counter[int] = Counter()

    for col in range(roi_w):
        gty = gt_by_col.get(col)
        if gty is None:
            continue
        _, kept, y_top, use_env = _column_kept(raw, col, min_conf)
        if not kept:
            continue
        has_near = any(abs(float(c["y"]) - float(gty)) <= near_px for c in kept)
        if has_near:
            cols_gt_near += 1

        vmap_ident = {(col, int(c["y"])): float(c["confidence"]) for c in kept}
        scored = _sort_total(kept, col, use_env, y_top, vmap_ident)
        order = [c for _, c in scored]
        top1 = order[0]
        top1_conf_best_y[int(top1["y"])] += 1
        if int(top1["y"]) < ub:
            cols_top1_ub += 1

        for c in kept:
            iy = int(c["y"])
            src = str(c.get("source", ""))
            src_tot[src] += 1
            is_ub = iy < ub
            is_gn = abs(float(iy) - float(gty)) <= near_px
            cf = float(c["confidence"])
            if is_ub:
                ub_confs.append(cf)
                src_ub[src] += 1
                if enrich:
                    for k in CONF_PART_KEYS:
                        if k in c:
                            ub_parts[k].append(float(c[k]))
                    for k in RAW_FEAT_KEYS:
                        if k in c:
                            ub_rawf[k].append(float(c[k]))
            if is_gn:
                gn_confs.append(cf)
                src_gn[src] += 1
                if enrich:
                    for k in CONF_PART_KEYS:
                        if k in c:
                            gn_parts[k].append(float(c[k]))
                    for k in RAW_FEAT_KEYS:
                        if k in c:
                            gn_rawf[k].append(float(c[k]))

    print("=== 코호트 ===")
    print(f"columns_with_gt_near_raw_kept_px{near_px:g}={cols_gt_near}")
    print(f"columns_where_top1_rank_is_upper_band={cols_top1_ub}")
    print(f"enriched_dump_present(conf_weight_*)={enrich}")

    print("\n=== confidence 분포 ===")
    _summ(ub_confs, "upper_band_candidate_confidence")
    _summ(gn_confs, "GT_near_candidate_confidence")

    if enrich:
        print("\n=== feature mean gap (upper_band - GT_near) ===")
        for k in CONF_PART_KEYS:
            mu_u = float(np.mean(ub_parts[k])) if ub_parts[k] else float("nan")
            mu_g = float(np.mean(gn_parts[k])) if gn_parts[k] else float("nan")
            print(f"  {k}: gap={mu_u - mu_g:.6f}")
        for k in RAW_FEAT_KEYS:
            mu_u = float(np.mean(ub_rawf[k])) if ub_rawf[k] else float("nan")
            mu_g = float(np.mean(gn_rawf[k])) if gn_rawf[k] else float("nan")
            print(f"  {k}: gap={mu_u - mu_g:.6f}")
    else:
        print("\n(conf_weight_* 없음 → `--debug-dump-raw-confidence-features` 로 재덤프 가능)")

    print("\n=== source별 confidence 분포 ===")
    by_src_conf: Dict[str, List[float]] = defaultdict(list)
    for col in range(roi_w):
        if gt_by_col.get(col) is None:
            continue
        _, kept, _, _ = _column_kept(raw, col, min_conf)
        for c in kept:
            by_src_conf[str(c.get("source", ""))].append(float(c["confidence"]))
    for s in sorted(by_src_conf.keys()):
        _summ(by_src_conf[s], f"source={s!r}")

    print("\n=== source별 GT-near / upper-band 비율 (후보 픽셀) ===")
    for s in sorted(src_tot.keys()):
        t = max(src_tot[s], 1)
        print(
            f"  {s}: gt_near_frac={src_gn[s] / t:.4f} upper_band_frac={src_ub[s] / t:.4f} n={src_tot[s]}"
        )

    print("\n=== 열별 top1(conf+env) y 분포 상위 15 (GT 존재 열) ===")
    for y, cnt in top1_conf_best_y.most_common(15):
        print(f"  y={y}: {cnt}")

    pool_ub: List[dict] = []
    pool_gn: List[dict] = []
    for col in range(roi_w):
        gty = gt_by_col.get(col)
        if gty is None:
            continue
        _, kept, _, _ = _column_kept(raw, col, min_conf)
        for c in kept:
            row = {"col": col, "y": c.get("y"), "confidence": c.get("confidence"), "source": c.get("source")}
            if int(c["y"]) < ub:
                pool_ub.append(row)
            if abs(float(c["y"]) - float(gty)) <= near_px:
                pool_gn.append(row)
    pool_ub.sort(key=lambda x: -float(x["confidence"]))
    pool_gn.sort(key=lambda x: -float(x["confidence"]))
    print("\n=== GT-near 후보 confidence 상위 10 ===")
    for row in pool_gn[:10]:
        print(row)
    print("\n=== upper-band 후보 confidence 상위 10 ===")
    for row in pool_ub[:10]:
        print(row)

    # 전역 source percentile
    src_lists: Dict[str, List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
    for col in range(roi_w):
        _, kept, _, _ = _column_kept(raw, col, min_conf)
        for c in kept:
            s = str(c.get("source", ""))
            src_lists[s].append(((col, int(c["y"])), float(c["confidence"])))
    src_norm_map: Dict[Tuple[int, int], float] = {}
    for _s, lst in src_lists.items():
        lst.sort(key=lambda t: t[1])
        n_s = len(lst)
        for ri, (ky, _) in enumerate(lst):
            src_norm_map[ky] = ri / max(n_s - 1, 1)

    def print_block(title: str, agg: Dict[str, Any]) -> None:
        print(f"\n[{title}]")
        print(f"  denom_gt_near_kept={agg['denom_gt_near_kept']}")
        print(f"  expected_filtered_gt_near_recall_px{near_px:g}={agg['expected_filtered_gt_near_recall']:.5f}")
        print(f"  top1_upper_band_columns={agg['top1_upper_band_columns']}")
        print(f"  path_upper_proxy_fraction={agg['path_upper_proxy_fraction']:.5f}")
        for t in [16, 32, 64, 128]:
            print(f"  columns_best_gt_near_rank_le_{t}={agg['rank_le'][t]}")

    baseline = _virtual_aggregate(
        raw,
        gt_by_col,
        roi_w,
        min_conf,
        near_px,
        topk,
        ub,
        lambda col, kep: {(col, int(x["y"])): float(x["confidence"]) for x in kep},
    )
    print_block("가상: baseline (confidence 원본)", baseline)

    def builder_col_rank(col: int, kep: List[dict]) -> Dict[Tuple[int, int], float]:
        srt = sorted(kep, key=lambda z: -float(z["confidence"]))
        nk = len(srt)
        return {(col, int(srt[i]["y"])): float(nk - 1 - i) / float(max(nk - 1, 1)) for i in range(nk)}

    col_rank = _virtual_aggregate(raw, gt_by_col, roi_w, min_conf, near_px, topk, ub, builder_col_rank)
    print_block("가상: 열내 순위 백분위(컬럼 rank normalization 아이디어)", col_rank)

    def builder_zscore(col: int, kep: List[dict]) -> Dict[Tuple[int, int], float]:
        arr = np.asarray([float(x["confidence"]) for x in kep], dtype=np.float64)
        mu = float(np.mean(arr))
        sig = float(np.std(arr)) + 1e-9
        return {(col, int(x["y"])): (float(x["confidence"]) - mu) / sig for x in kep}

    z_agg = _virtual_aggregate(raw, gt_by_col, roi_w, min_conf, near_px, topk, ub, builder_zscore)
    print_block("가상: 열별 z-score(conf)", z_agg)

    def builder_src_pct(col: int, kep: List[dict]) -> Dict[Tuple[int, int], float]:
        return {(col, int(x["y"])): float(src_norm_map.get((col, int(x["y"])), 0.5)) for x in kep}

    src_agg = _virtual_aggregate(raw, gt_by_col, roi_w, min_conf, near_px, topk, ub, builder_src_pct)
    print_block("가상: 전역 source별 percentile 정규화", src_agg)

    softmax_aggs: Dict[float, Dict[str, Any]] = {}
    for T in [0.5, 1.0, 2.0, 4.0]:

        def make_softmax_builder(Tt: float):
            def builder_softmax(col: int, kep: List[dict]) -> Dict[Tuple[int, int], float]:
                confs = np.array([float(x["confidence"]) for x in kep], dtype=np.float64)
                probs = _softmax_scores(confs, Tt)
                return {(col, int(kep[i]["y"])): float(probs[i]) for i in range(len(kep))}

            return builder_softmax

        sm_agg = _virtual_aggregate(
            raw, gt_by_col, roi_w, min_conf, near_px, topk, ub, make_softmax_builder(float(T))
        )
        softmax_aggs[float(T)] = sm_agg
        print_block(f"가상: 열별 softmax(conf/T), T={T}", sm_agg)

    print("\n[가상 실험 D] 상단 밴드 후보 confidence 질량 비율")
    mass_ub = 0.0
    mass_all = 0.0
    cols_dominated = 0
    cols_any = 0
    for col in range(roi_w):
        _, kept, _, _ = _column_kept(raw, col, min_conf)
        if not kept:
            continue
        cols_any += 1
        su = sum(float(c["confidence"]) for c in kept if int(c["y"]) < ub)
        sa = sum(float(c["confidence"]) for c in kept)
        mass_ub += su
        mass_all += sa
        if su >= 0.5 * max(sa, 1e-9):
            cols_dominated += 1
    print(f"  global_mass_share_upper_band={mass_ub / max(mass_all, 1e-9):.5f}")
    print(f"  columns_upper_band_ge_half_mass={cols_dominated}/{cols_any}")

    candidates_r = [
        ("baseline", baseline["expected_filtered_gt_near_recall"]),
        ("column_rank_pct", col_rank["expected_filtered_gt_near_recall"]),
        ("zscore_col", z_agg["expected_filtered_gt_near_recall"]),
        ("source_pct_global", src_agg["expected_filtered_gt_near_recall"]),
    ] + [(f"softmax_T{T}", softmax_aggs[T]["expected_filtered_gt_near_recall"]) for T in softmax_aggs]

    best_name, best_v = max(candidates_r, key=lambda x: x[1])
    print(f"\n=== 가상 실험 요약: filtered_gt_near_recall 최대 === {best_name} = {best_v:.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
