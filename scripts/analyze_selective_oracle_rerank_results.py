#!/usr/bin/env python3
"""Selective oracle rerank 결과 CSV 정량 분석 유틸."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _is_mae_like(metric_name: str) -> bool:
    n = metric_name.lower()
    return ("mae" in n) or ("error" in n) or ("gap" in n)


def _is_higher_better(metric_name: str) -> bool:
    n = metric_name.lower()
    if "pass" in n:
        return True
    if "f1" in n or "recall" in n or "precision" in n or "iou" in n:
        return True
    return False


def _win_tie_loss(series: pd.Series, higher_better: bool) -> Dict[str, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if higher_better:
        win = int((s > 0).sum())
        loss = int((s < 0).sum())
    else:
        win = int((s < 0).sum())
        loss = int((s > 0).sum())
    tie = int((s == 0).sum())
    return {"win": win, "tie": tie, "loss": loss, "n": int(len(s))}


def _collect_metric_pairs(delta_cols: List[str]) -> Dict[str, Dict[str, str]]:
    pairs: Dict[str, Dict[str, str]] = {}
    for col in delta_cols:
        if not col.startswith("delta_"):
            continue
        if col.endswith("_selective_minus_rule"):
            metric = col[len("delta_") : -len("_selective_minus_rule")]
            pairs.setdefault(metric, {})["rule"] = col
        elif col.endswith("_selective_minus_global"):
            metric = col[len("delta_") : -len("_selective_minus_global")]
            pairs.setdefault(metric, {})["global"] = col
    return pairs


def _bucket_label(x: float) -> str:
    if pd.isna(x):
        return "nan"
    if 0.0 <= x < 0.25:
        return "[0.00,0.25)"
    if 0.25 <= x < 0.5:
        return "[0.25,0.50)"
    if 0.5 <= x < 0.75:
        return "[0.50,0.75)"
    if 0.75 <= x <= 1.0:
        return "[0.75,1.00]"
    return "out_of_range"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze selective oracle rerank CSV outputs")
    ap.add_argument("--root", type=str, default="outputs/_sel_study_full")
    args = ap.parse_args()

    root = Path(args.root)
    results_p = root / "selective_oracle_rerank_results.csv"
    summary_p = root / "selective_oracle_rerank_summary.csv"
    delta_p = root / "selective_oracle_rerank_delta.csv"

    for p in (results_p, summary_p, delta_p):
        if not p.exists():
            raise FileNotFoundError(f"missing file: {p}")

    results = pd.read_csv(results_p)
    summary = pd.read_csv(summary_p)
    delta = pd.read_csv(delta_p)

    _print_section("A. Summary by arm")
    wanted_summary_cols = [
        "domain",
        "taxonomy_prior",
        "arm",
        "n",
        "strict_pass_mean",
        "mean_curve_y_mae_px",
        "mean_trace_valid_ratio",
        "mean_peak_f1",
        "mean_peak_recall",
    ]
    missing_summary = [c for c in wanted_summary_cols if c not in summary.columns]
    if missing_summary:
        print("[INFO] summary 생략 컬럼:", ", ".join(missing_summary))
    use_summary = [c for c in wanted_summary_cols if c in summary.columns]
    print(summary[use_summary].to_string(index=False))

    metric_pairs = _collect_metric_pairs(list(delta.columns))

    requested_metrics = ["strict_pass", "curve_y_mae_px", "strict_curve_y_mae_px", "peak_f1", "peak_recall"]
    missing_requested = []
    analysis_rows = []

    _print_section("B. Selective vs rule: win/tie/loss counts")
    for m in requested_metrics:
        rule_col = metric_pairs.get(m, {}).get("rule")
        if rule_col is None:
            missing_requested.append(m)
            continue
        higher_better = _is_higher_better(m)
        if _is_mae_like(m):
            higher_better = False
        wtl = _win_tie_loss(delta[rule_col], higher_better=higher_better)
        print(f"{m}: win={wtl['win']} tie={wtl['tie']} loss={wtl['loss']} (n={wtl['n']})")
        analysis_rows.append(
            {
                "metric": m,
                "compare": "selective_minus_rule",
                "win": wtl["win"],
                "tie": wtl["tie"],
                "loss": wtl["loss"],
                "n": wtl["n"],
                "mean_delta": float(pd.to_numeric(delta[rule_col], errors="coerce").dropna().mean()),
            }
        )

    if missing_requested:
        print("[INFO] 요청 metric 중 미존재:", ", ".join(missing_requested))

    _print_section("C. Selective vs rule: mean deltas")
    for m, cols in sorted(metric_pairs.items()):
        c = cols.get("rule")
        if c is None:
            continue
        s = pd.to_numeric(delta[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        print(f"{m}: mean={s.mean():.6f}")

    _print_section("D. Selective vs global: mean deltas")
    for m, cols in sorted(metric_pairs.items()):
        c = cols.get("global")
        if c is None:
            continue
        s = pd.to_numeric(delta[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        print(f"{m}: mean={s.mean():.6f}")

    _print_section("E. trace_valid_ratio bucket analysis")
    sel = results[results["arm"] == "selective_oracle"].copy()
    if "trace_valid_ratio" not in sel.columns:
        print("[WARN] trace_valid_ratio 없음 - bucket 분석 생략")
    else:
        sel["vr_bucket"] = sel["trace_valid_ratio"].apply(_bucket_label)
        join_cols = ["sample_id", "domain", "taxonomy_prior"]
        dcol = metric_pairs.get("curve_y_mae_px", {}).get("rule")
        if dcol is None:
            print("[WARN] curve_y_mae_px delta 없음 - bucket 분석 생략")
        else:
            bdf = sel.merge(delta[join_cols + [dcol]], on=join_cols, how="left")
            out = (
                bdf.groupby("vr_bucket", dropna=False)
                .agg(
                    n=("sample_id", "count"),
                    mean_trace_valid_ratio=("trace_valid_ratio", "mean"),
                    mean_delta_curve_y_mae_px_selective_minus_rule=(dcol, "mean"),
                )
                .reset_index()
            )
            print(out.to_string(index=False))

    _print_section("F. Worst 5 samples where selective worsened vs rule")
    worst_metric = metric_pairs.get("curve_y_mae_px", {}).get("rule")
    if worst_metric is None:
        print("[WARN] curve_y_mae_px delta 없음 - worst/best 생략")
        worst_df = pd.DataFrame(columns=["sample_id", "domain", "taxonomy_prior"])
        best_df = worst_df.copy()
    else:
        base_cols = ["sample_id", "domain", "taxonomy_prior", worst_metric]
        w = delta.sort_values(worst_metric, ascending=False).head(5)[base_cols].copy()
        w = w.rename(columns={worst_metric: "delta_curve_y_mae_px_selective_minus_rule"})
        print(w.to_string(index=False))

        _print_section("G. Best 5 samples where selective improved vs rule")
        b = delta.sort_values(worst_metric, ascending=True).head(5)[base_cols].copy()
        b = b.rename(columns={worst_metric: "delta_curve_y_mae_px_selective_minus_rule"})
        print(b.to_string(index=False))

        worst_df = w
        best_df = b

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_out = root / "selective_oracle_rerank_analysis.csv"
    worst_out = root / "selective_oracle_rerank_worst.csv"
    best_out = root / "selective_oracle_rerank_best.csv"

    analysis_df.to_csv(analysis_out, index=False)
    worst_df.to_csv(worst_out, index=False)
    best_df.to_csv(best_out, index=False)

    print("\n[SAVED]")
    print(analysis_out)
    print(worst_out)
    print(best_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
