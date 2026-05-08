from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


def _failure_type(rule_mae: float, global_mae: float) -> str:
    if global_mae > rule_mae + 1.0:
        return "GLOBAL_WORSE_THAN_RULE"
    if global_mae >= 30.0:
        return "GLOBAL_STILL_BAD"
    if abs(global_mae - rule_mae) <= 1.0:
        return "GLOBAL_NO_IMPROVEMENT"
    if (rule_mae - global_mae) >= 3.0:
        return "RERANK_POTENTIAL"
    if rule_mae < 10.0 and global_mae < 10.0:
        return "ALREADY_GOOD"
    return "GLOBAL_NO_IMPROVEMENT"


def _failure_rank(ft: str) -> int:
    order = {
        "GLOBAL_WORSE_THAN_RULE": 0,
        "GLOBAL_STILL_BAD": 1,
        "GLOBAL_NO_IMPROVEMENT": 2,
        "RERANK_POTENTIAL": 3,
        "ALREADY_GOOD": 4,
    }
    return int(order.get(ft, 99))


def _run_key(sample_id: str, domain: str) -> str:
    return f"{domain}_{sample_id}"


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="study output root")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    results_csv = root / "selective_oracle_rerank_results.csv"
    delta_csv = root / "selective_oracle_rerank_delta.csv"
    out_csv = root / "global_oracle_failure_candidates.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(results_csv)
    if not delta_csv.is_file():
        raise FileNotFoundError(delta_csv)

    rdf = pd.read_csv(results_csv)
    ddf = pd.read_csv(delta_csv)

    rdf = _to_num(
        rdf,
        ["curve_y_mae_px", "peak_f1", "candidate_recall_per_column", "empty_column_rate", "trace_valid_ratio"],
    )
    ddf = _to_num(ddf, ["delta_curve_y_mae_px_selective_minus_rule"])

    arm_map: Dict[str, str] = {"rule": "rule", "global_oracle": "global", "selective_oracle": "selective"}
    wide: Dict[str, pd.DataFrame] = {}
    for arm, key in arm_map.items():
        sub = rdf[rdf["arm"] == arm].copy()
        sub = sub.rename(
            columns={
                "curve_y_mae_px": f"{key}_curve_y_mae_px",
                "peak_f1": f"peak_f1_{key}",
                "candidate_recall_per_column": f"candidate_recall_{key}",
                "empty_column_rate": f"empty_column_rate_{key}",
                "trace_valid_ratio": f"trace_valid_ratio_{key}",
            }
        )
        wide[key] = sub[
            [
                "sample_id",
                "domain",
                "taxonomy_prior",
                f"{key}_curve_y_mae_px",
                f"peak_f1_{key}",
                f"candidate_recall_{key}",
                f"empty_column_rate_{key}",
                f"trace_valid_ratio_{key}",
            ]
        ]

    merged = wide["rule"].merge(wide["global"], on=["sample_id", "domain", "taxonomy_prior"], how="inner")
    merged = merged.merge(wide["selective"], on=["sample_id", "domain", "taxonomy_prior"], how="inner")
    merged = merged.merge(ddf, on=["sample_id", "domain", "taxonomy_prior"], how="left")
    merged["delta_global_minus_rule"] = merged["global_curve_y_mae_px"] - merged["rule_curve_y_mae_px"]
    merged["delta_selective_minus_rule"] = merged["delta_curve_y_mae_px_selective_minus_rule"]
    merged["failure_type"] = merged.apply(
        lambda r: _failure_type(float(r["rule_curve_y_mae_px"]), float(r["global_curve_y_mae_px"])),
        axis=1,
    )
    merged["_failure_rank"] = merged["failure_type"].map(_failure_rank)

    def _debug_path(row: pd.Series, arm_dir: str) -> str:
        run_key = _run_key(str(row["sample_id"]), str(row["domain"]))
        suffix = "global" if arm_dir == "global" else arm_dir
        return str(root / "runs" / run_key / arm_dir / f"debug_{row['sample_id']}_{suffix}" / "debug.json")

    def _overlay_path(row: pd.Series, arm_dir: str) -> str:
        run_key = _run_key(str(row["sample_id"]), str(row["domain"]))
        suffix = "global" if arm_dir == "global" else arm_dir
        return str(root / "runs" / run_key / arm_dir / f"debug_{row['sample_id']}_{suffix}" / "14_peaks_overlay.png")

    merged["rule_debug_json"] = merged.apply(lambda r: _debug_path(r, "rule"), axis=1)
    merged["global_debug_json"] = merged.apply(lambda r: _debug_path(r, "global"), axis=1)
    merged["selective_debug_json"] = merged.apply(lambda r: _debug_path(r, "selective"), axis=1)
    merged["rule_overlay_path"] = merged.apply(lambda r: _overlay_path(r, "rule"), axis=1)
    merged["global_overlay_path"] = merged.apply(lambda r: _overlay_path(r, "global"), axis=1)
    merged["selective_overlay_path"] = merged.apply(lambda r: _overlay_path(r, "selective"), axis=1)

    out = merged.sort_values(by=["_failure_rank", "sample_id"], ascending=[True, True]).copy()
    out = out[
        [
            "sample_id",
            "domain",
            "taxonomy_prior",
            "rule_curve_y_mae_px",
            "global_curve_y_mae_px",
            "selective_curve_y_mae_px",
            "delta_global_minus_rule",
            "delta_selective_minus_rule",
            "peak_f1_rule",
            "peak_f1_global",
            "candidate_recall_rule",
            "empty_column_rate_rule",
            "trace_valid_ratio_rule",
            "trace_valid_ratio_global",
            "failure_type",
            "rule_debug_json",
            "global_debug_json",
            "selective_debug_json",
            "rule_overlay_path",
            "global_overlay_path",
            "selective_overlay_path",
        ]
    ]
    out.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()

