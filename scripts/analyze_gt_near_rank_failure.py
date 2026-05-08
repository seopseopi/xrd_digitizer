#!/usr/bin/env python3
"""raw 후보에서 GT-near가 왜 낮은 rank인지 분포/비교 분석."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column
from trace.candidates import (
    ENVELOPE_SORT_BONUS,
    ENVELOPE_SORT_DECAY_PX,
    ENVELOPE_SORT_MIN_CANDIDATES,
    ENVELOPE_SORT_MIN_SPAN_PX,
)


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


def _summ(arr: List[float], pfx: str) -> None:
    if not arr:
        print(f"{pfx}=None")
        return
    a = np.asarray(arr, dtype=np.float64)
    print(f"{pfx}.min={float(np.min(a))}")
    print(f"{pfx}.mean={float(np.mean(a))}")
    print(f"{pfx}.median={float(np.median(a))}")
    print(f"{pfx}.p90={float(np.percentile(a,90))}")
    print(f"{pfx}.p95={float(np.percentile(a,95))}")
    print(f"{pfx}.p99={float(np.percentile(a,99))}")
    print(f"{pfx}.max={float(np.max(a))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--upper-frac", type=float, default=0.2)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dbg = json.loads((run_dir / "debug.json").read_text(encoding="utf-8"))
    raw = _norm(json.loads(next(iter(sorted(run_dir.glob("*_raw_candidates.json")))).read_text(encoding="utf-8")))
    plot_box = tuple(int(x) for x in dbg.get("plot_box", [0, 0, 0, 0]))
    roi_h = int(plot_box[3] - plot_box[1])
    roi_w = int(plot_box[2] - plot_box[0])
    ub = max(1, int(math.ceil(roi_h * float(args.upper_frac))))
    gt = _load_gt_json(str(args.gt_json))
    gt_by_col, _ = build_gt_y_roi_per_column(gt, plot_box, roi_h, roi_w)

    near_cols = {3.0: 0, 5.0: 0, 10.0: 0}
    best_rank: List[float] = []
    best_conf: List[float] = []
    top1_conf: List[float] = []
    conf_gap: List[float] = []
    score_gap: List[float] = []
    rank_gt64 = 0
    rank_gt128 = 0
    top1_upper = 0
    gtnear_and_top1_upper = 0
    top1_src: Dict[str, int] = {}
    gt_src: Dict[str, int] = {}
    top1_comp: List[float] = []
    gt_comp: List[float] = []

    for col in range(roi_w):
        g = gt_by_col.get(col)
        if g is None:
            continue
        cands = raw.get(col, [])
        if not cands:
            continue

        # filter 랭킹키 재구성
        y_top = min(int(c["y"]) for c in cands)
        y_bottom = max(int(c["y"]) for c in cands)
        use_env = len(cands) >= ENVELOPE_SORT_MIN_CANDIDATES and (y_bottom - y_top) >= ENVELOPE_SORT_MIN_SPAN_PX
        scored = []
        for c in cands:
            env = 0.0
            if use_env:
                dy = int(c["y"]) - y_top
                env = ENVELOPE_SORT_BONUS * math.exp(-max(0, dy) / ENVELOPE_SORT_DECAY_PX)
            sc = float(c.get("confidence", 0.0)) + env
            scored.append((c, env, sc))
        scored.sort(key=lambda t: -float(t[2]))

        top1 = scored[0][0]
        top1c = float(top1.get("confidence", 0.0))
        top1_conf.append(top1c)
        if int(top1.get("y", 0)) < ub:
            top1_upper += 1

        for th in (3.0, 5.0, 10.0):
            if any(abs(float(c.get("y", 0.0)) - float(g)) <= th for c, _, _ in scored):
                near_cols[th] += 1

        near_scored = [(c, env, sc) for (c, env, sc) in scored if abs(float(c.get("y", 0.0)) - float(g)) <= 5.0]
        if not near_scored:
            continue
        # GT-near 중 score 최고 후보
        best = max(near_scored, key=lambda t: float(t[2]))
        best_c, _, best_sc = best
        rank = next((i + 1 for i, t in enumerate(scored) if id(t[0]) == id(best_c)), None)
        if rank is None:
            continue
        best_rank.append(float(rank))
        bconf = float(best_c.get("confidence", 0.0))
        best_conf.append(bconf)
        conf_gap.append(top1c - bconf)
        score_gap.append(float(scored[0][2]) - float(best_sc))
        if rank > 64:
            rank_gt64 += 1
        if rank > 128:
            rank_gt128 += 1
        if int(top1.get("y", 0)) < ub:
            gtnear_and_top1_upper += 1

        s1 = str(top1.get("source", ""))
        sg = str(best_c.get("source", ""))
        top1_src[s1] = top1_src.get(s1, 0) + 1
        gt_src[sg] = gt_src.get(sg, 0) + 1
        top1_comp.append(float(top1.get("comp_score", 0.0)))
        gt_comp.append(float(best_c.get("comp_score", 0.0)))

    print(f"total_columns={roi_w}")
    print(f"columns_with_gt_near_raw_px3={near_cols[3.0]}")
    print(f"columns_with_gt_near_raw_px5={near_cols[5.0]}")
    print(f"columns_with_gt_near_raw_px10={near_cols[10.0]}")
    _summ(best_rank, "gt_near_best_rank")
    n = max(1, len(best_rank))
    for thr in (16, 32, 64, 128, 256):
        cnt = int(sum(1 for r in best_rank if r <= thr))
        print(f"gt_near_best_rank_le_{thr}_columns={cnt}")
    _summ(best_conf, "gt_near_best_confidence")
    _summ(top1_conf, "top1_confidence")
    _summ(conf_gap, "top1_minus_gt_near_confidence_gap")
    _summ(score_gap, "top1_minus_gt_near_rank_key_gap")
    print(f"gt_near_best_rank_gt64_ratio={float(rank_gt64)/float(n)}")
    print(f"gt_near_best_rank_gt128_ratio={float(rank_gt128)/float(n)}")
    print(f"top1_upper_band_columns={top1_upper}")
    print(f"gt_near_exists_and_top1_upper_band_columns={gtnear_and_top1_upper}")
    print(f"top1_source_distribution={json.dumps(top1_src, ensure_ascii=False, sort_keys=True)}")
    print(f"gt_near_source_distribution={json.dumps(gt_src, ensure_ascii=False, sort_keys=True)}")
    if top1_comp and gt_comp:
        print(f"top1_comp_score_mean={float(np.mean(np.asarray(top1_comp)))}")
        print(f"gt_near_comp_score_mean={float(np.mean(np.asarray(gt_comp)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

