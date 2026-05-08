#!/usr/bin/env python3
"""filtered → final 후보 단계 GT-near 소실과 원인 요약(debug.json + 후보 JSON)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column


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


def _load_cands(run_dir: Path, suffix: str) -> Dict[int, List[dict]]:
    hits = sorted(run_dir.glob(f"*_{suffix}_candidates.json"))
    if not hits:
        raise FileNotFoundError(f"no *_{suffix}_candidates.json under {run_dir}")
    return _norm(json.loads(hits[0].read_text(encoding="utf-8")))


def _nearest_stats(cands: Dict[int, List[dict]], gt: Dict[int, float], w: int) -> float:
    dists: List[float] = []
    for col in range(w):
        g = gt.get(col)
        if g is None:
            continue
        lst = cands.get(col, [])
        if not lst:
            continue
        dists.append(min(abs(float(c.get("y", 0.0)) - float(g)) for c in lst))
    if not dists:
        return float("nan")
    return float(np.mean(np.asarray(dists) <= 5.0))


def _simulate_final_stage_column(
    cands: List[dict],
    *,
    gy: float,
    near_px: float,
    skeleton_y: Optional[float],
    max_k: int,
    dedupe_disabled: bool,
    dedupe_decimals: int,
) -> Dict[str, Any]:
    """저장된 skeleton 없을 때 단열 근사 시뮬레이션(confidence + 선택 skeleton)."""

    def dp_rank(c: dict) -> float:
        r = float(c["confidence"])
        if skeleton_y is not None and math.isfinite(float(skeleton_y)):
            r += 0.16 * math.exp(-abs(float(c["y"]) - float(skeleton_y)) / 11.0)
        return r

    sorted_c = sorted(cands, key=lambda c: -dp_rank(c))
    gt_near = [c for c in cands if abs(float(c["y"]) - float(gy)) <= near_px]
    if not gt_near:
        return {"has_filtered_gt_near": False}
    best = min(gt_near, key=lambda c: abs(float(c["y"]) - float(gy)))
    try:
        rank_sort = next(i + 1 for i, c in enumerate(sorted_c) if id(c) == id(best))
    except StopIteration:
        rank_sort = None

    if dedupe_disabled:
        deduped = list(sorted_c)
    else:
        seen = set()
        deduped = []
        for c in sorted_c:
            key = round(float(c["comp_score"]), dedupe_decimals)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
    try:
        rank_dedupe = next(i + 1 for i, c in enumerate(deduped) if id(c) == id(best))
    except StopIteration:
        rank_dedupe = None

    topk = deduped[:max_k]
    in_topk = next((True for c in topk if id(c) == id(best)), False)

    return {
        "has_filtered_gt_near": True,
        "rank_after_dp_sort": rank_sort,
        "rank_after_dedupe": rank_dedupe,
        "in_pre_bridge_topk": bool(in_topk),
        "dedupe_survived": rank_dedupe is not None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--near-px", type=float, default=5.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dbg_path = run_dir / "debug.json"
    dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
    plot_box = tuple(int(x) for x in dbg["plot_box"])
    roi_h = plot_box[3] - plot_box[1]
    roi_w = plot_box[2] - plot_box[0]
    ub = int(math.ceil(float(roi_h) * 0.2))

    gt = _load_gt_json(str(args.gt_json))
    gt_map, _ = build_gt_y_roi_per_column(gt, plot_box, roi_h, roi_w)

    filt = _load_cands(run_dir, "filtered")
    fin = _load_cands(run_dir, "final")

    fil_recall = _nearest_stats(filt, gt_map, roi_w)
    fin_recall = _nearest_stats(fin, gt_map, roi_w)

    near_px = float(args.near_px)
    loss_cols: List[int] = []
    rank_after_sort: List[int] = []
    in_topk_flags: List[bool] = []
    dedupe_lost = 0

    for col in range(roi_w):
        gy = gt_map.get(col)
        if gy is None:
            continue
        fl = filt.get(col, [])
        fn = fin.get(col, [])
        if not any(abs(float(c["y"]) - float(gy)) <= near_px for c in fl):
            continue
        if any(abs(float(c["y"]) - float(gy)) <= near_px for c in fn):
            continue
        loss_cols.append(col)
        sim = _simulate_final_stage_column(
            fl,
            gy=float(gy),
            near_px=near_px,
            skeleton_y=None,
            max_k=8,
            dedupe_disabled=False,
            dedupe_decimals=2,
        )
        if sim.get("rank_after_dp_sort") is not None:
            rank_after_sort.append(int(sim["rank_after_dp_sort"]))
        in_topk_flags.append(bool(sim.get("in_pre_bridge_topk")))
        if sim.get("dedupe_survived") is False:
            dedupe_lost += 1

    print("[filtered gt_near_recall_px5]", round(fil_recall, 6))
    print("[final gt_near_recall_px5]", round(fin_recall, 6))
    print("[filtered→final loss columns]", len(loss_cols))

    cfd = dbg.get("candidate_final_debug")
    if isinstance(cfd, dict):
        print("\n[candidate_final_debug 요약]")
        for k in (
            "final_topk",
            "final_dedupe_enabled",
            "final_rank_key",
            "final_dedupe_key",
            "gt_near_filtered_columns_px5",
            "gt_near_final_columns_px5",
            "gt_near_filtered_to_final_lost_columns_px5",
            "gt_near_final_loss_reason_counts",
            "final_source_distribution",
            "dp_bridge_count",
        ):
            if k in cfd:
                print(f"  {k}: {cfd[k]}")
    else:
        print("\n(candidate_final_debug 없음 — --debug-final-selection-reasons 실행 필요)")

    if rank_after_sort:
        arr = np.asarray(rank_after_sort, dtype=np.float64)
        print("\n[근사 시뮬레이션] skeleton 힌트 없이 confidence만 정렬한 순위(참고용)")
        print(f"  mean={np.mean(arr):.2f} median={np.median(arr):.2f} p90={np.percentile(arr,90):.2f} n={len(arr)}")
        print(f"  pre_bridge_top8 안에 들어갈 수 있었던 비율(근사): {float(np.mean(in_topk_flags)):.4f}")
        print(f"  dedupe에서 best_gt 제거 추정 열 수: {dedupe_lost}")

    src_ct: Counter[str] = Counter()
    top1_ub = 0
    n_gt = 0
    for col in range(roi_w):
        gy = gt_map.get(col)
        if gy is None:
            continue
        n_gt += 1
        lst = fin.get(col, [])
        if lst:
            top = max(lst, key=lambda c: float(c.get("confidence", 0.0)))
            if int(top.get("y", 0)) < ub:
                top1_ub += 1
        for c in lst:
            src_ct[str(c.get("source", ""))] += 1

    print("\n[final source distribution]", dict(sorted(src_ct.items())))
    print(f"[final_top1_upper_band_columns] {top1_ub} / gt-columns {n_gt}")

    cpc = [len(fin.get(c, [])) for c in range(roi_w)]
    print(f"[final candidates per column] min={min(cpc)} max={max(cpc)} mean={float(np.mean(cpc)):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
