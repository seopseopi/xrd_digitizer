#!/usr/bin/env python3
"""Analyze baseline/filter_preserve/final_preserve candidate patch panel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.report import evaluate_single  # noqa: E402

VARIANTS = ("baseline", "filter_preserve", "final_preserve")


def _resolve_path(value: str) -> str:
    s = str(value).strip()
    s = s.replace("\\", "/")
    marker = "xrd_digitizer_v1/"
    if marker in s:
        s = s.split(marker, 1)[1]
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _path_upper_fraction(debug: Dict[str, Any]) -> Optional[float]:
    plot_box = debug.get("plot_box")
    path = debug.get("trace", {}).get("path")
    if not isinstance(plot_box, list) or len(plot_box) != 4 or not isinstance(path, list):
        return None
    roi_h = int(plot_box[3]) - int(plot_box[1])
    if roi_h <= 0:
        return None
    upper_y = float(roi_h) * 0.2
    valid = []
    for y in path:
        if y is None:
            continue
        fy = _safe_float(y)
        if fy is not None:
            valid.append(fy)
    if not valid:
        return None
    return float(sum(1 for y in valid if y < upper_y) / len(valid))


def _final_gt_near_recall(debug: Dict[str, Any]) -> Optional[float]:
    cfd = debug.get("candidate_final_debug") or {}
    total = (
        debug.get("candidate_stats", {}).get("total_columns")
        or debug.get("candidate_gt_proximity", {}).get("columns_evaluated")
    )
    n = cfd.get("gt_near_final_columns_px5")
    if n is None or total in (None, 0):
        prox = debug.get("candidate_gt_proximity") or {}
        return _safe_float(prox.get("candidate_gt_near_recall_px5"))
    return float(n) / float(total)


def _extract_row(
    *,
    panel_row: pd.Series,
    root: Path,
    variant: str,
) -> Dict[str, Any]:
    sid = str(panel_row["sample_id"])
    domain = str(panel_row["domain"])
    run_key = f"{domain}_{sid}"
    variant_root = root / run_key / variant
    result_json = variant_root / f"{sid}_result.json"
    debug_json = variant_root / f"debug_{sid}_global" / "debug.json"
    gt_col = "gt_json" if "gt_json" in panel_row.index else "gt_path"
    gt_path = _resolve_path(str(panel_row[gt_col]))
    if not result_json.is_file() or not debug_json.is_file():
        raise FileNotFoundError(f"missing run output: {run_key}/{variant}")
    ev = evaluate_single(
        str(result_json),
        str(debug_json),
        gt_path,
        gate_type=domain if domain in ("clean", "styled") else "clean",
        gate_level="development",
    )
    debug = _read_json(debug_json)
    main = ev["metrics"]["main"]
    dbg_metrics = ev["metrics"].get("debug", {})
    prox = debug.get("candidate_gt_proximity") or {}
    filt = debug.get("candidate_filter_debug") or {}
    cfd = debug.get("candidate_final_debug") or {}
    oracle = (
        debug.get("model_assist", {})
        .get("oracle_rerank", {})
        .get("oracle_score_summary", {})
    )
    return {
        "sample_id": sid,
        "domain": domain,
        "failure_type": str(panel_row.get("failure_type", "")),
        "taxonomy_prior": str(panel_row.get("taxonomy_prior", "")),
        "variant": variant,
        "curve_y_mae_px": _safe_float(main.get("curve_y_mae_px")),
        "numeric_y_mae_norm": _safe_float(main.get("numeric_y_mae_norm")),
        "peak_f1": _safe_float(dbg_metrics.get("peak_f1")),
        "peak_recall": _safe_float(main.get("peak_recall")),
        "candidate_gt_near_recall_px3": _safe_float(
            prox.get("candidate_gt_near_recall_px3")
        ),
        "candidate_gt_near_recall_px5": _safe_float(
            prox.get("candidate_gt_near_recall_px5")
        ),
        "candidate_gt_near_recall_px10": _safe_float(
            prox.get("candidate_gt_near_recall_px10")
        ),
        "mean_nearest_candidate_gt_dist_px": _safe_float(
            prox.get("mean_nearest_candidate_gt_dist_px")
        ),
        "final_path_upper_band_fraction": _path_upper_fraction(debug),
        "mean_oracle_dist_px": _safe_float(oracle.get("mean_oracle_dist_px")),
        "filtered_gt_near_recall_px5": _safe_float(
            filt.get("filtered_gt_near_recall_px5")
        ),
        "final_gt_near_recall_px5": _final_gt_near_recall(debug),
        "candidate_final_upper_band_fraction": _safe_float(
            cfd.get("final_upper_band_fraction")
        ),
    }


def _print_summary(comp: pd.DataFrame, variants: List[str]) -> Dict[str, Any]:
    pivot_curve = comp.pivot_table(
        index=["sample_id", "domain", "failure_type"],
        columns="variant",
        values="curve_y_mae_px",
        aggfunc="first",
    ).reset_index()
    pivot_peak = comp.pivot_table(
        index=["sample_id", "domain", "failure_type"],
        columns="variant",
        values="peak_f1",
        aggfunc="first",
    ).reset_index()
    pivot_cand = comp.pivot_table(
        index=["sample_id", "domain", "failure_type"],
        columns="variant",
        values="candidate_gt_near_recall_px5",
        aggfunc="first",
    ).reset_index()
    merged = pivot_curve.copy()
    if "baseline" not in merged.columns:
        raise ValueError("baseline variant is required for delta comparisons")
    has_final = "final_preserve" in merged.columns
    has_filter = "filter_preserve" in merged.columns
    if has_final:
        merged["delta_final_minus_baseline"] = merged["final_preserve"] - merged["baseline"]
    else:
        merged["delta_final_minus_baseline"] = np.nan
    if has_filter:
        merged["delta_filter_minus_baseline"] = merged["filter_preserve"] - merged["baseline"]
    else:
        merged["delta_filter_minus_baseline"] = np.nan

    variant_mean = comp.groupby("variant")["curve_y_mae_px"].mean().to_dict()
    variant_median = comp.groupby("variant")["curve_y_mae_px"].median().to_dict()
    by_failure = (
        merged.groupby("failure_type")["delta_final_minus_baseline"]
        .mean()
        .to_dict()
    )
    improved = int((merged["delta_final_minus_baseline"] < -1e-9).sum())
    worsened = int((merged["delta_final_minus_baseline"] > 1e-9).sum())
    good = merged[merged["failure_type"].astype(str).eq("ALREADY_GOOD")]
    already_good_worse = int((good["delta_final_minus_baseline"] > 5.0).sum())

    peak_join = pivot_peak.copy()
    if has_final and "baseline" in peak_join.columns and "final_preserve" in peak_join.columns:
        peak_join["delta_final_minus_baseline"] = peak_join["final_preserve"] - peak_join["baseline"]
    else:
        peak_join["delta_final_minus_baseline"] = np.nan
    peak_worse = int((peak_join["delta_final_minus_baseline"] < -1e-9).sum())
    peak_mean_delta = float(peak_join["delta_final_minus_baseline"].mean())

    cand_join = pivot_cand.copy()
    if has_final and "baseline" in cand_join.columns and "final_preserve" in cand_join.columns:
        cand_join["delta_final_minus_baseline"] = cand_join["final_preserve"] - cand_join["baseline"]
    else:
        cand_join["delta_final_minus_baseline"] = np.nan
    cand_mean_delta = float(cand_join["delta_final_minus_baseline"].mean())

    near_join = comp.pivot_table(
        index=["sample_id", "domain", "failure_type"],
        columns="variant",
        values="mean_nearest_candidate_gt_dist_px",
        aggfunc="first",
    ).reset_index()
    if has_final and "baseline" in near_join.columns and "final_preserve" in near_join.columns:
        near_join["delta_final_minus_baseline"] = near_join["final_preserve"] - near_join["baseline"]
    else:
        near_join["delta_final_minus_baseline"] = np.nan
    near_mean_delta = float(near_join["delta_final_minus_baseline"].mean())

    recall_join = comp.pivot_table(
        index=["sample_id", "domain", "failure_type"],
        columns="variant",
        values="peak_recall",
        aggfunc="first",
    ).reset_index()
    if has_final and "baseline" in recall_join.columns and "final_preserve" in recall_join.columns:
        recall_join["delta_final_minus_baseline"] = recall_join["final_preserve"] - recall_join["baseline"]
    else:
        recall_join["delta_final_minus_baseline"] = np.nan
    peak_recall_worse = int((recall_join["delta_final_minus_baseline"] < -1e-9).sum())
    peak_recall_mean_delta = float(recall_join["delta_final_minus_baseline"].mean())

    delta_series = merged["delta_final_minus_baseline"]
    worsen_gt1 = int((delta_series > 1.0).sum())
    worsen_gt5 = int((delta_series > 5.0).sum())
    by_domain = merged.groupby("domain")["delta_final_minus_baseline"].agg(
        improved=lambda s: int((s < -1e-9).sum()),
        worsened=lambda s: int((s > 1e-9).sum()),
        mean_delta=lambda s: float(s.mean()),
    )

    global_bad = merged[
        merged["failure_type"].astype(str).isin(
            ["GLOBAL_STILL_BAD", "GLOBAL_WORSE_THAN_RULE"]
        )
    ]
    rerank = merged[merged["failure_type"].astype(str).eq("RERANK_POTENTIAL")]
    curve_mean_delta = float(merged["delta_final_minus_baseline"].mean())

    if (
        len(global_bad)
        and float(global_bad["delta_final_minus_baseline"].mean()) < 0
        and (not len(rerank) or float(rerank["delta_final_minus_baseline"].max()) <= 5.0)
        and already_good_worse == 0
        and peak_mean_delta >= -1e-9
        and cand_mean_delta > 0
    ):
        classification = "PATCH_GENERALIZES"
    elif len(global_bad) and float(global_bad["delta_final_minus_baseline"].mean()) < 0:
        classification = "PATCH_HELPS_GLOBAL_FAILURE_ONLY"
    elif improved <= 1 and any(
        (merged["sample_id"].eq("pattern_72296"))
        & (merged["domain"].eq("styled"))
        & (merged["delta_final_minus_baseline"] < 0)
    ):
        classification = "PATCH_OVERFITS_72296"
    elif already_good_worse > 0 or peak_mean_delta < -1e-9:
        classification = "PATCH_HURTS_GOOD_SAMPLES"
    else:
        classification = "PATCH_NOT_READY"

    summary = {
        "variants": variants,
        "classification": classification,
        "variant_mean_curve_y_mae_px": {
            k: round(float(v), 6) for k, v in variant_mean.items()
        },
        "variant_median_curve_y_mae_px": {
            k: round(float(v), 6) for k, v in variant_median.items()
        },
        "failure_type_mean_delta_final_minus_baseline": {
            k: round(float(v), 6) for k, v in by_failure.items()
        },
        "final_preserve_better_than_baseline_count": improved,
        "final_preserve_worse_than_baseline_count": worsened,
        "already_good_worse_over_5px_count": already_good_worse,
        "peak_f1_worse_count": peak_worse,
        "peak_f1_mean_delta_final_minus_baseline": round(peak_mean_delta, 6),
        "peak_recall_worse_count": peak_recall_worse,
        "peak_recall_mean_delta_final_minus_baseline": round(peak_recall_mean_delta, 6),
        "candidate_gt_near_recall_px5_mean_delta_final_minus_baseline": round(
            cand_mean_delta,
            6,
        ),
        "mean_nearest_candidate_gt_dist_px_mean_delta_final_minus_baseline": round(
            near_mean_delta,
            6,
        ),
        "curve_y_mae_px_mean_delta_final_minus_baseline": round(curve_mean_delta, 6),
        "curve_y_mae_worse_gt_1px_count": worsen_gt1,
        "curve_y_mae_worse_gt_5px_count": worsen_gt5,
        "domain_delta_final_minus_baseline": {
            str(idx): {
                "improved": int(row["improved"]),
                "worsened": int(row["worsened"]),
                "mean_delta": round(float(row["mean_delta"]), 6),
            }
            for idx, row in by_domain.iterrows()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--panel-csv", required=True, type=Path)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "outputs" / "_candidate_patch_panel" / "panel_comparison.csv",
    )
    ap.add_argument(
        "--variants",
        default="baseline,filter_preserve,final_preserve",
        help="comma-separated variants to analyze",
    )
    args = ap.parse_args()

    panel = pd.read_csv(args.panel_csv)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; allowed={VARIANTS}")
    rows: List[Dict[str, Any]] = []
    for _, row in panel.iterrows():
        for variant in variants:
            rows.append(_extract_row(panel_row=row, root=args.root, variant=variant))
    comp = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(args.out_csv, index=False)
    summary = _print_summary(comp, variants)
    summary_path = args.out_csv.with_name("panel_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
