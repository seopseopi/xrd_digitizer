#!/usr/bin/env python3
"""
outputs/runs/<tag>/report_*.json 의 aggregate.main_metrics_stats 를 표로 묶는다.

예:
  python3 scripts/summarize_eval_reports.py --repo-root . \\
    --tags eval_ablation_01_default eval_ablation_02_ridge \\
    --out experiments/ablation_eval_summary.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

GATES = [
    ("clean", "report_clean.json"),
    ("styled", "report_styled.json"),
    ("real_like", "report_real_like.json"),
]

METRICS = [
    "curve_y_mae_px",
    "major_peak_x_error",
    "peak_recall",
    "max_gap_px",
]


def _mean_main(report_path: Path) -> Dict[str, float]:
    if not report_path.is_file():
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    agg = data.get("aggregate") or {}
    stats = agg.get("main_metrics_stats") or {}
    out: Dict[str, float] = {}
    for m in METRICS:
        block = stats.get(m) or {}
        if "mean" in block:
            out[m] = float(block["mean"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None, help="TSV 경로 (태그×게이트 행)")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    runs = root / "outputs" / "runs"

    rows_out: List[Dict[str, Any]] = []
    header = ["tag", "gate"] + METRICS

    for tag in args.tags:
        base = runs / tag
        for gate_name, report_name in GATES:
            means = _mean_main(base / report_name)
            row = {"tag": tag, "gate": gate_name}
            for m in METRICS:
                row[m] = means.get(m, "")
            rows_out.append(row)

    # stdout human table
    w = max(len(t) for t in args.tags) + 2
    for gate_name, _ in GATES:
        print(f"\n=== gate: {gate_name} ===")
        print(f"{'tag':<{w}} " + "  ".join(f"{m:>22}" for m in METRICS))
        for tag in args.tags:
            r = next(x for x in rows_out if x["tag"] == tag and x["gate"] == gate_name)
            vals = "  ".join(
                f"{r[m]:>22.4f}" if isinstance(r[m], float) else f"{'':>22}"
                for m in METRICS
            )
            print(f"{tag:<{w}} {vals}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=header)
            wr.writeheader()
            for row in rows_out:
                wr.writerow({k: row[k] for k in header})
        print(f"\n[OK] wrote {args.out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
