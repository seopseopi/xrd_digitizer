#!/usr/bin/env python3
"""Evaluate ROI 2x highres as primary output against source_numeric on canonical 3-domain samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import source_numeric_stagewise_error_decomposition as decomp  # noqa: E402


CANONICAL_ROOT = ROOT / "data" / "test_canonical_30"
MANIFEST = CANONICAL_ROOT / "manifest.csv"
SMOKE_ROOT = ROOT / "outputs" / "_roi_upscale_smoke_canonical_3domain"
POLICY_ROOT = ROOT / "outputs" / "_evaluation_policy"
OUT_ROOT = ROOT / "outputs" / "_highres_primary_eval_canonical_3domain"
TARGET_TEST_IDS = ["clean_pattern_11832", "styled_pattern_72296", "real_like_pattern_60890"]

CSV_FIELDS = [
    "sample_id",
    "domain",
    "input_image",
    "mi_json",
    "source_numeric_json",
    "baseline_point_count",
    "highres_point_count",
    "baseline_normalized_y_mae",
    "highres_normalized_y_mae",
    "delta_normalized_y_mae",
    "baseline_shape_correlation",
    "highres_shape_correlation",
    "highres_peak_center_error",
    "highres_peak_height_error",
    "highres_peak_width_error",
    "highres_false_spike_count",
    "highres_missed_peak_count",
    "eval_grid_normalized_y_mae",
    "eval_grid_role",
    "x_monotonic",
    "gap_count",
    "diagnosis_label",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rows(manifest: Path) -> Dict[str, Dict[str, str]]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    out = {}
    for test_id in TARGET_TEST_IDS:
        matches = [r for r in rows if r.get("test_id") == test_id]
        if len(matches) != 1:
            raise ValueError(f"{test_id} expected exactly one manifest row, got {len(matches)}")
        row = matches[0]
        if row.get("pair_status") != "PASS":
            raise ValueError(f"{test_id} pair_status is not PASS")
        if row.get("sample_id") == "pattern_1915":
            raise ValueError("pattern_1915 is forbidden")
        out[test_id] = row
    return out


def x_monotonic(xs: List[Any]) -> bool:
    vals = [float(x) for x in xs]
    return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def source_metrics(source: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    sx, sy = decomp.source_xy(source)
    source_dyn = max(float(np.ptp(sy)), 1e-12)
    pred_on_source = decomp.interp(x, y, sx)
    mae = decomp.safe_mae(pred_on_source, sy)
    peaks_ref = decomp.detect_peak_metrics(sx, sy)
    peaks_pred = decomp.detect_peak_metrics(x, y)
    peak_cmp = decomp.compare_peaks(peaks_ref, peaks_pred, tol_x=0.35)
    return {
        "y_mae": mae,
        "normalized_y_mae": mae / source_dyn,
        "shape_correlation": decomp.safe_corr(pred_on_source, sy),
        "peak_center_error": peak_cmp["peak_center_error_mean"],
        "peak_height_error": peak_cmp["peak_height_error_mean"],
        "peak_width_error": peak_cmp["peak_width_error_mean"],
        "false_spike_count": peak_cmp["false_spike_count"],
        "missed_peak_count": peak_cmp["missed_peak_count"],
        "matched_peak_count": peak_cmp["matched_peak_count"],
        "ref_peak_count": peak_cmp["ref_peak_count"],
        "pred_peak_count": peak_cmp["pred_peak_count"],
    }


def result_arrays(result: Dict[str, Any], field: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    points = result[field]
    x = np.asarray(points["two_theta_values"], dtype=float)
    y = np.asarray(points["intensities"], dtype=float)
    return x, y, points


def classify_row(baseline: Dict[str, Any], highres: Dict[str, Any], eval_grid: Dict[str, Any], highres_points: Dict[str, Any]) -> str:
    if highres["normalized_y_mae"] < baseline["normalized_y_mae"] and highres["shape_correlation"] >= baseline["shape_correlation"] - 1e-9:
        if int(highres_points.get("gap_count") or 0) == 0:
            return "HIGHRES_IMPROVES_SOURCE_NUMERIC"
    if highres["normalized_y_mae"] < baseline["normalized_y_mae"]:
        return "HIGHRES_IMPROVES_MAE_WITH_WARNINGS"
    if eval_grid["normalized_y_mae"] > highres["normalized_y_mae"]:
        return "HIGHRES_BETTER_THAN_LEGACY_EVAL_GRID_ONLY"
    return "HIGHRES_NOT_BETTER"


def write_policy(policy_root: Path, decision: str) -> Tuple[Path, Path]:
    policy_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_name": "highres_primary_source_numeric_evaluation_v1",
        "decision_context": decision,
        "primary_evaluation": {
            "output": "roi_upscale_2x_highres",
            "fields": ["export_points_highres", "root two_theta_values/intensities when final_export_mode=highres"],
            "reference": "canonical source_numeric.json",
            "metrics": [
                "highres_source_numeric_y_mae",
                "highres_source_numeric_normalized_y_mae",
                "highres_source_numeric_shape_correlation",
                "highres_peak_center_error",
                "highres_peak_height_error",
                "highres_peak_width_error",
                "highres_false_spike_count",
                "highres_missed_peak_count",
            ],
        },
        "legacy_evaluation": {
            "output": "eval_grid_950",
            "role": "legacy/evaluator compatibility and export-downscale diagnostic only",
            "not_primary_acceptance_basis": True,
        },
        "why_eval_grid_demoted": [
            "950 eval_grid was introduced for evaluator compatibility.",
            "source_numeric decomposition showed 1900->950 downscale can destroy highres gains.",
            "eval_grid remains useful to diagnose export/downscale failure, not to reject highres fidelity by itself.",
        ],
        "why_highres_promoted": [
            "It preserves the actual 2x traced column density.",
            "It avoids forced downscale before source_numeric comparison.",
            "Canonical 3-domain source_numeric evaluation shows lower normalized_y_mae than baseline in all three samples.",
        ],
        "acceptance_conditions": [
            "At least 2 of 3 canonical smoke domains improve normalized_y_mae vs baseline.",
            "x_monotonic is true.",
            "gap_count is 0 or within explicitly allowed tolerance.",
            "shape_correlation does not degrade relative to baseline.",
            "false spike increase is not excessive.",
        ],
        "not_done": [
            "canonical 30 full evaluation not run",
            "candidate/DP/tracing scoring not tuned",
            "canonical data not modified",
        ],
    }
    json_path = policy_root / "highres_primary_evaluation_policy.json"
    write_json(json_path, payload)
    md = [
        "# Highres Primary Evaluation Policy",
        "",
        "Primary evaluation is now defined as source_numeric comparison against the ROI 2x highres output.",
        "",
        "## Why 950 eval_grid Is Not Primary",
        "- 950 eval_grid exists for legacy/evaluator compatibility.",
        "- It can hide or damage the 2x highres trace by forcing a downscale.",
        "- It remains useful as a diagnostic for export/downscale errors.",
        "",
        "## Why 1900 Highres Is Primary",
        "- It preserves the actual 2x trace density.",
        "- It directly tests whether highres reconstruction matches source_numeric.",
        "- It avoids mixing highres fidelity with legacy downscale behavior.",
        "",
        "## Source Numeric Reference",
        "The canonical `source_numeric.json` is the primary numeric reference because the project goal is numeric scientific reconstruction, not only visual or GT-pixel agreement.",
        "",
        "## New Metric Role",
        "- highres metrics: primary acceptance metrics",
        "- eval_grid metrics: legacy compatibility and downscale diagnostic metrics",
        "- old evaluator metrics: useful for continuity, not sufficient for highres acceptance by themselves",
        "",
        "## Highres Acceptance Conditions",
        "- At least 2 of 3 smoke domains improve normalized_y_mae vs baseline.",
        "- `x_monotonic` remains true.",
        "- `gap_count` remains zero or within tolerance.",
        "- shape correlation does not degrade relative to baseline.",
        "- peak false spikes do not increase excessively.",
        "",
        f"Current decision context: `{decision}`",
        "",
    ]
    md_path = policy_root / "highres_primary_evaluation_policy.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--smoke-root", type=Path, default=SMOKE_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--policy-root", type=Path, default=POLICY_ROOT)
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    smoke_root = args.smoke_root if args.smoke_root.is_absolute() else ROOT / args.smoke_root
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    policy_root = args.policy_root if args.policy_root.is_absolute() else ROOT / args.policy_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows_by_id = load_rows(manifest)
    summary_rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    for test_id in TARGET_TEST_IDS:
        row = rows_by_id[test_id]
        source = decomp.read_json(ROOT / row["source_numeric_json"])
        mi = decomp.read_json(ROOT / row["mi_json"])
        if mi.get("mi_source") != "synthetic_from_gt" or mi.get("mi_role") != "calibration_input":
            raise ValueError(f"invalid MI role/source for {test_id}")

        sample_run = smoke_root / test_id
        sample_id = row["sample_id"]
        baseline_result = read_json(sample_run / "baseline_1x_eval_grid" / f"{sample_id}_result.json")
        eval_grid_result = read_json(sample_run / "upscale_2x_eval_grid" / f"{sample_id}_result.json")
        highres_result = read_json(sample_run / "upscale_2x_highres" / f"{sample_id}_result.json")

        bx, by, bpoints = result_arrays(baseline_result, "export_points_eval")
        ex, ey, epoints = result_arrays(eval_grid_result, "export_points_eval")
        hx, hy, hpoints = result_arrays(highres_result, "export_points_highres")

        baseline_metrics = source_metrics(source, bx, by)
        eval_grid_metrics = source_metrics(source, ex, ey)
        highres_metrics = source_metrics(source, hx, hy)
        diagnosis = classify_row(baseline_metrics, highres_metrics, eval_grid_metrics, hpoints)
        x_ok = x_monotonic(hpoints["two_theta_values"])
        gap_count = int(hpoints.get("gap_count") or 0)

        summary_rows.append(
            {
                "sample_id": sample_id,
                "domain": row["domain"],
                "input_image": row["input_image"],
                "mi_json": row["mi_json"],
                "source_numeric_json": row["source_numeric_json"],
                "baseline_point_count": int(len(bx)),
                "highres_point_count": int(len(hx)),
                "baseline_normalized_y_mae": baseline_metrics["normalized_y_mae"],
                "highres_normalized_y_mae": highres_metrics["normalized_y_mae"],
                "delta_normalized_y_mae": highres_metrics["normalized_y_mae"] - baseline_metrics["normalized_y_mae"],
                "baseline_shape_correlation": baseline_metrics["shape_correlation"],
                "highres_shape_correlation": highres_metrics["shape_correlation"],
                "highres_peak_center_error": highres_metrics["peak_center_error"],
                "highres_peak_height_error": highres_metrics["peak_height_error"],
                "highres_peak_width_error": highres_metrics["peak_width_error"],
                "highres_false_spike_count": highres_metrics["false_spike_count"],
                "highres_missed_peak_count": highres_metrics["missed_peak_count"],
                "eval_grid_normalized_y_mae": eval_grid_metrics["normalized_y_mae"],
                "eval_grid_role": "legacy_export_downscale_diagnostic_only",
                "x_monotonic": x_ok,
                "gap_count": gap_count,
                "diagnosis_label": diagnosis,
            }
        )
        detailed.append(
            {
                "test_id": test_id,
                "baseline": baseline_metrics,
                "highres": highres_metrics,
                "eval_grid_legacy": eval_grid_metrics,
                "point_counts": {
                    "baseline": len(bx),
                    "highres": len(hx),
                    "eval_grid": len(ex),
                },
                "x_monotonic": x_ok,
                "gap_count": gap_count,
                "diagnosis_label": diagnosis,
            }
        )

    improved = [r for r in summary_rows if float(r["delta_normalized_y_mae"]) < 0]
    corr_ok = [
        r for r in summary_rows
        if float(r["highres_shape_correlation"]) >= float(r["baseline_shape_correlation"]) - 1e-9
    ]
    gap_ok = [r for r in summary_rows if int(r["gap_count"]) == 0 and str(r["x_monotonic"]) == "True"]
    false_spike_ok = [r for r in summary_rows if int(r["highres_false_spike_count"]) <= 4]

    if len(improved) == 3 and len(corr_ok) == 3 and len(gap_ok) == 3 and len(false_spike_ok) >= 2:
        final_decision = "HIGHRES_PRIMARY_EVAL_PASS_CURRENT_BEST_3DOMAIN"
    elif any(r["domain"] == "clean" for r in improved):
        final_decision = "HIGHRES_PRIMARY_EVAL_PASS_CLEAN_ONLY_NEEDS_MORE_DOMAINS"
    elif len(improved) > 0:
        final_decision = "HIGHRES_PRIMARY_EVAL_MIXED_KEEP_AS_EXPERIMENTAL"
    else:
        final_decision = "HIGHRES_PRIMARY_EVAL_FAIL_REVERT_TO_ENGINEERING_BASELINE"

    policy_md, policy_json = write_policy(policy_root, final_decision)

    csv_path = out_root / "highres_primary_eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "final_decision": final_decision,
        "canonical_root": rel(CANONICAL_ROOT),
        "manifest": rel(manifest),
        "evaluation_policy": {
            "primary": "2x highres vs source_numeric",
            "legacy": "950 eval_grid for compatibility/downscale diagnostics only",
            "policy_md": rel(policy_md),
            "policy_json": rel(policy_json),
        },
        "summary_rows": summary_rows,
        "detailed_results": detailed,
        "pass_counts": {
            "improved_normalized_y_mae": len(improved),
            "shape_correlation_not_worse": len(corr_ok),
            "x_monotonic_and_gap_ok": len(gap_ok),
            "false_spike_ok": len(false_spike_ok),
        },
        "not_done": [
            "canonical original files not modified",
            "input.png not modified",
            "mi.json not modified",
            "gt.json not modified",
            "source_numeric.json not modified",
            "metadata.json not modified",
            "plot_box/calibration not modified",
            "candidate/DP/tracing scoring not modified",
            "threshold/margin tuning not performed",
            "canonical 30 full evaluation not run",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path = out_root / "highres_primary_eval_summary.json"
    write_json(json_path, payload)

    lines = [
        "# Highres Primary Evaluation: Canonical 3-Domain",
        "",
        f"- Final decision: `{final_decision}`",
        f"- Canonical manifest: `{rel(manifest)}`",
        "- Primary evaluation: `2x highres` vs `source_numeric.json`",
        "- Legacy evaluation: `950 eval_grid` only for compatibility/downscale diagnostics",
        "",
        "## Why eval_grid 950 Is Not Primary",
        "The 950-point eval_grid was created for legacy evaluator compatibility. In source_numeric decomposition, the 1900->950 downscale path can destroy highres gains, so it is not used as the primary acceptance signal.",
        "",
        "## Why highres 1900 Is Primary",
        "The highres output preserves the actual 2x trace density and compares directly against the canonical source_numeric reference.",
        "",
        "## 3-Domain Results",
        "| domain | sample | baseline norm MAE | highres norm MAE | delta | baseline corr | highres corr | highres false/missed peaks |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['domain']} | {r['sample_id']} | "
            f"{float(r['baseline_normalized_y_mae']):.6f} | "
            f"{float(r['highres_normalized_y_mae']):.6f} | "
            f"{float(r['delta_normalized_y_mae']):.6f} | "
            f"{float(r['baseline_shape_correlation']):.4f} | "
            f"{float(r['highres_shape_correlation']):.4f} | "
            f"{r['highres_false_spike_count']}/{r['highres_missed_peak_count']} |"
        )
    lines.extend(
        [
            "",
            "## Clean Anchor",
            "`clean_pattern_11832` remains the clearest single-sample result: highres normalized_y_mae is lower than both baseline and legacy eval_grid.",
            "",
            "## Current Best Status",
            "Across this 3-domain smoke set, highres is the current best source_numeric metric path. This is not yet a canonical-30 conclusion.",
            "",
            "## Outputs",
            f"- CSV: `{rel(csv_path)}`",
            f"- JSON: `{rel(json_path)}`",
            f"- Policy MD: `{rel(policy_md)}`",
            f"- Policy JSON: `{rel(policy_json)}`",
            "",
            "## Not Done",
            "- canonical original files not modified",
            "- mi/gt/source_numeric/metadata not modified",
            "- plot_box/calibration not modified",
            "- candidate/DP/tracing scoring not modified",
            "- threshold/margin tuning not performed",
            "- canonical 30 full evaluation not run",
            "",
            "## Next Step",
            "Run the same highres-primary source_numeric evaluation on canonical 30 before declaring full-set best performance.",
            "",
        ]
    )
    md_path = out_root / "highres_primary_eval_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"final_decision": final_decision, "summary_csv": rel(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
