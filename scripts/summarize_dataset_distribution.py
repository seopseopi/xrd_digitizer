"""
Step 4 helper: metadata + dev_subset 분포 요약 리포트 생성.

xrd_digitizer_v1_master_spec.md §23.8 준수.
- peak_count, tail_energy, dynamic_range 분포 요약
- family 분포 요약
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def summarize_metadata(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("== METADATA SUMMARY ==")
    lines.append(f"rows: {len(df)}")
    lines.append(f"columns: {len(df.columns)}")

    if "is_valid" in df.columns:
        s = df["is_valid"]
        if s.dtype == object:
            s = s.astype(str).str.lower().isin(["true", "1", "yes"])
        lines.append(f"valid_rows: {int(s.sum())}")
        lines.append(f"invalid_rows: {int((~s).sum())}")

    for col in ("peak_count_est", "tail_energy_ratio", "dynamic_range_log"):
        if col in df.columns:
            desc = pd.to_numeric(df[col], errors="coerce").describe(percentiles=[0.1, 0.5, 0.9])
            lines.append(f"\n-- {col} describe --")
            lines.append(desc.to_string())

    if {"bin_peak", "bin_tail", "bin_dr"}.issubset(df.columns):
        lines.append("\n== STRAT BIN COUNTS ==")
        g = df.groupby(["bin_peak", "bin_tail", "bin_dr"], dropna=False).size().reset_index(name="n")
        lines.append(g.sort_values("n", ascending=False).head(30).to_string(index=False))

    return "\n".join(lines) + "\n"


def summarize_dev_subset(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("== DEV_SUBSET SUMMARY ==")
    lines.append(f"rows: {len(df)}")
    if "debug_split" in df.columns:
        lines.append("\n-- debug_split counts --")
        lines.append(df["debug_split"].value_counts().to_string())

    for col in ("peak_count_est", "tail_energy_ratio", "dynamic_range_log"):
        if col in df.columns:
            desc = pd.to_numeric(df[col], errors="coerce").describe(percentiles=[0.1, 0.5, 0.9])
            lines.append(f"\n-- {col} describe --")
            lines.append(desc.to_string())

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dataset distributions")
    parser.add_argument("--input_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\all_samples.csv")
    parser.add_argument("--subset_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\dev_subset.csv")
    parser.add_argument("--output_txt", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\dataset_distribution_summary.txt")
    args = parser.parse_args()

    report = ""

    if Path(args.input_csv).exists():
        meta = _safe_read_csv(args.input_csv)
        report += summarize_metadata(meta) + "\n"

    if Path(args.subset_csv).exists():
        subset = _safe_read_csv(args.subset_csv)
        report += summarize_dev_subset(subset) + "\n"

    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_txt).write_text(report, encoding="utf-8")

    print(f"[DONE] Wrote summary: {args.output_txt}")
    print(report)


if __name__ == "__main__":
    main()
