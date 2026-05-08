#!/usr/bin/env python3
"""고엔트로피 위험 규칙(high_entropy_many_cands) 진단 — trace/risk_detector.py 와 동일 조건."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selective_oracle_settings import SelectiveOracleSettings


def _norm_sid(sample_id: str) -> Tuple[str, str]:
    s = sample_id.strip()
    base = s[: -len("_result")] if s.endswith("_result") else s
    return s, base


def _pctiles(vals: np.ndarray, qs: Iterable[float]) -> List[float]:
    if vals.size == 0:
        return [float("nan")] * len(list(qs))
    out = np.quantile(vals, list(qs))
    return [float(x) for x in np.atleast_1d(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--sample-id", required=True, dest="sample_id")
    ap.add_argument("--domain", required=True)
    ap.add_argument(
        "--risk-features-csv",
        default=None,
        help="기본: <root>/risk_features_columns.csv",
    )
    ap.add_argument("--candidate-count-high-thr", type=int, default=None)
    ap.add_argument("--entropy-high-thr", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    csv_path = Path(args.risk_features_csv) if args.risk_features_csv else root / "risk_features_columns.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    st = SelectiveOracleSettings()
    cand_thr = int(args.candidate_count_high_thr if args.candidate_count_high_thr is not None else st.candidate_count_high_thr)
    ent_thr = float(args.entropy_high_thr if args.entropy_high_thr is not None else st.entropy_high_thr)

    _, sid_base = _norm_sid(args.sample_id)
    df = pd.read_csv(csv_path)
    df["sample_key"] = df["sample_id"].astype(str).apply(lambda x: _norm_sid(x)[1])

    sub = df[(df["sample_key"] == sid_base) & (df["domain"].astype(str) == str(args.domain))].copy()
    if sub.empty:
        raise SystemExit(f"No rows for sample_id={args.sample_id!r} (base={sid_base!r}), domain={args.domain!r}")

    n_c = pd.to_numeric(sub["candidate_count"], errors="coerce").to_numpy(dtype=np.float64)
    ent = pd.to_numeric(sub["conf_entropy"], errors="coerce").to_numpy(dtype=np.float64)
    t1 = pd.to_numeric(sub["top1_conf"], errors="coerce").to_numpy(dtype=np.float64)
    t2 = pd.to_numeric(sub["top2_conf"], errors="coerce").to_numpy(dtype=np.float64)
    margin = pd.to_numeric(sub["conf_margin"], errors="coerce").to_numpy(dtype=np.float64)

    valid_ent = np.isfinite(ent) & (n_c > 0)
    hit = (
        valid_ent
        & (n_c >= float(cand_thr))
        & (ent > float(ent_thr))
    )
    hit_count = int(hit.sum())

    xs = sub["x"].to_numpy()

    def stats_line(name: str, arr: np.ndarray) -> None:
        v = arr[np.isfinite(arr)]
        print(f"\n{name} (finite n={len(v)}):")
        if len(v) == 0:
            print("  (no finite values)")
            return
        p50, p90, p95, p99 = _pctiles(v, [0.5, 0.9, 0.95, 0.99])
        print(
            f"  min={float(np.min(v)):.6g} mean={float(np.mean(v)):.6g} max={float(np.max(v)):.6g} "
            f"p50={p50:.6g} p90={p90:.6g} p95={p95:.6g} p99={p99:.6g}"
        )

    print(
        "high_entropy_many_cands 판정(trace/risk_detector._column_risk_flags와 동일):\n"
        "  candidate_count >= candidate_count_high_thr\n"
        "  AND conf_entropy > entropy_high_thr\n"
        "(disable_high_entropy_risk 가 True면 규칙 자체가 꺼짐 — 본 스크립트는 CSV feature만 검사)"
    )
    print(f"\nroot={root}")
    print(f"csv={csv_path}")
    print(f"rows matched={len(sub)}")
    print(f"total columns (rows)={len(sub)}")
    print(f"high_entropy_many_cands hit_count={hit_count}")
    print("\nthresholds used:")
    print(f"  candidate_count_high_thr={cand_thr}")
    print(f"  entropy_high_thr={ent_thr}")

    stats_line("candidate_count", n_c)
    stats_line("conf_entropy ( Shannon H = -sum p log p on normalized confidences, 미정규화 )", ent)
    stats_line("top1_conf", t1)
    stats_line("top2_conf", t2)
    stats_line("conf_margin (top1_conf - top2_conf)", margin)

    hi_idx = np.where(hit)[0]
    lo_idx = np.where(~hit)[0]

    def examples(idxs: np.ndarray, label: str, k: int = 10) -> None:
        pick = idxs[:k]
        print(f"\n{label} (up to {k}):")
        if pick.size == 0:
            print("  (none)")
            return
        for i in pick:
            xi = int(xs[i])
            print(
                f"  x={xi} candidate_count={int(n_c[i]) if np.isfinite(n_c[i]) else 'nan'} "
                f"entropy={ent[i]:.6g} top1={t1[i]:.6g} top2={t2[i]:.6g} margin={margin[i]:.6g}"
            )

    examples(hi_idx, "high_entropy_many_cands == True columns")
    examples(lo_idx, "high_entropy_many_cands == False columns")


if __name__ == "__main__":
    main()
