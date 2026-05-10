#!/usr/bin/env python3
"""Run and evaluate highres-primary source_numeric metrics on canonical 30.

Official performance baseline: upscale_2x_highres (ROI 2×, --final-export-mode highres).
That track is the best-performing product output; new work should be judged mainly on
highres_* metrics vs source_numeric.json.

The script also runs baseline_1x_eval_grid (1×, eval_grid) as a legacy reference arm
to detect regression vs the old eval grid — not as the primary baseline.

Reads paths only from data/test_canonical_30/manifest.csv.
Optional: legacy 950 eval_grid 2× run (--run-legacy-eval-grid); not a primary signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_highres_primary_canonical_3domain import (  # noqa: E402
    source_metrics,
    result_arrays,
    x_monotonic,
)


CANONICAL_ROOT = ROOT / "data" / "test_canonical_30"
MANIFEST = CANONICAL_ROOT / "manifest.csv"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "_highres_primary_eval_canonical_30"

CSV_FIELDS = [
    "sample_key",
    "sample_id",
    "domain",
    "input_image",
    "mi_json",
    "source_numeric_json",
    "baseline_point_count",
    "highres_point_count",
    "eval_grid_point_count",
    "baseline_normalized_y_mae",
    "highres_normalized_y_mae",
    "delta_normalized_y_mae",
    "baseline_shape_correlation",
    "highres_shape_correlation",
    "delta_shape_correlation",
    "highres_peak_center_error",
    "highres_peak_height_error",
    "highres_peak_width_error",
    "highres_false_spike_count",
    "highres_missed_peak_count",
    "x_monotonic",
    "gap_count",
    "improved",
    "eval_grid_normalized_y_mae",
    "eval_grid_shape_correlation",
    "eval_grid_role",
    "diagnosis_label",
    "failure_reason",
    "run_dir",
]

DOMAIN_FIELDS = [
    "domain",
    "sample_count",
    "improved_count",
    "worsened_count",
    "unchanged_count",
    "mean_baseline_normalized_y_mae",
    "mean_highres_normalized_y_mae",
    "mean_delta_normalized_y_mae",
    "median_delta_normalized_y_mae",
    "mean_baseline_shape_correlation",
    "mean_highres_shape_correlation",
    "mean_delta_shape_correlation",
    "x_monotonic_fail_count",
    "gap_fail_count",
    "excessive_false_spike_count",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_and_validate_manifest(path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, int]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    failures: List[Dict[str, str]] = []
    required = {
        "test_id",
        "sample_id",
        "domain",
        "input_image",
        "mi_json",
        "gt_json",
        "source_numeric_json",
        "metadata_json",
        "pair_status",
        "mi_source",
        "mi_role",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    seen: set[Tuple[str, str]] = set()
    domain_counts: Dict[str, int] = {}
    valid: List[Dict[str, str]] = []
    for row in rows:
        reasons: List[str] = []
        key = (row["sample_id"], row["domain"])
        if key in seen:
            reasons.append("duplicate_sample_domain")
        seen.add(key)
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
        if row["sample_id"] == "pattern_1915":
            reasons.append("pattern_1915_forbidden")
        if row.get("pair_status") != "PASS":
            reasons.append("pair_status_not_PASS")

        for col in ["input_image", "mi_json", "gt_json", "source_numeric_json", "metadata_json"]:
            if not resolve(row[col]).is_file():
                reasons.append(f"{col}_missing")

        if resolve(row["mi_json"]).is_file():
            mi = read_json(resolve(row["mi_json"]))
            if mi.get("mi_source") != "synthetic_from_gt":
                reasons.append("mi_source_not_synthetic_from_gt")
            if mi.get("mi_role") != "calibration_input":
                reasons.append("mi_role_not_calibration_input")
            if not isinstance(mi.get("plot_box"), list) or len(mi.get("plot_box", [])) != 4:
                reasons.append("mi_plot_box_missing")
            for k in ("x_axis_values", "y_axis_values", "x_axis_points", "y_axis_points"):
                if k not in mi:
                    reasons.append(f"mi_calibration_missing_{k}")

        if resolve(row["source_numeric_json"]).is_file():
            src = read_json(resolve(row["source_numeric_json"]))
            if "two_theta_values" not in src or "intensities" not in src:
                reasons.append("source_numeric_arrays_missing")

        if reasons:
            bad = dict(row)
            bad["failure_reason"] = ";".join(reasons)
            failures.append(bad)
        else:
            valid.append(row)

    expected_counts = {"clean": 20, "styled": 5, "real_like": 5}
    if len(rows) != 30:
        failures.append({"sample_id": "__manifest__", "domain": "", "failure_reason": f"row_count_expected_30_got_{len(rows)}"})
    for dom, n in expected_counts.items():
        if domain_counts.get(dom, 0) != n:
            failures.append({"sample_id": "__manifest__", "domain": dom, "failure_reason": f"domain_count_expected_{n}_got_{domain_counts.get(dom, 0)}"})
    if not any(r.get("test_id") == "clean_pattern_11832" for r in rows):
        failures.append({"sample_id": "__manifest__", "domain": "clean", "failure_reason": "clean_pattern_11832_missing"})
    return valid, failures, domain_counts


def make_out_root(base: Path, overwrite: bool, resume: bool) -> Path:
    if resume:
        base.mkdir(parents=True, exist_ok=True)
        return base
    if not base.exists():
        base.mkdir(parents=True)
        return base
    if overwrite:
        shutil.rmtree(base)
        base.mkdir(parents=True)
        return base
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = base / stamp
    out.mkdir(parents=True)
    return out


def run_local_cmd(
    *,
    image_path: Path,
    mi_json: Path,
    result_json: Path,
    debug_dir: Path,
    pipeline: str,
    roi_upscale_factor: int,
    final_export_mode: str,
    axis_mask_margin: int = 15,
    mask_b_mag_percentile: float = 50.0,
    mask_b_thr_clip_lo: float = 10.0,
    mask_b_thr_clip_hi: float = 40.0,
) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "runner" / "run_local.py"),
        "--image_path",
        str(image_path),
        "--manual_inputs_path",
        str(mi_json),
        "--output_json_path",
        str(result_json),
        "--debug_dir",
        str(debug_dir),
        "--pipeline",
        pipeline,
        "--roi-upscale-factor",
        str(int(roi_upscale_factor)),
        "--roi-upscale-method",
        "lanczos",
        "--final-export-mode",
        final_export_mode,
        "--axis-mask-margin",
        str(int(axis_mask_margin)),
        "--mask-b-mag-percentile",
        str(float(mask_b_mag_percentile)),
        "--mask-b-thr-clip-lo",
        str(float(mask_b_thr_clip_lo)),
        "--mask-b-thr-clip-hi",
        str(float(mask_b_thr_clip_hi)),
    ]


def execute(cmd: List[str], run_dir: Path, dry_run: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    if dry_run:
        return
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)


def extract_metrics(source: Dict[str, Any], result: Dict[str, Any], field: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    x, y, points = result_arrays(result, field)
    return source_metrics(source, x, y), points


def diagnosis_label(row: Dict[str, Any]) -> str:
    if row["failure_reason"]:
        return "RUN_OR_METRIC_FAILURE"
    if not bool(row["x_monotonic"]):
        return "HIGHRES_X_MONOTONIC_FAIL"
    if int(row["gap_count"]) > 0:
        return "HIGHRES_GAP_WARNING"
    if row["improved"]:
        return "HIGHRES_IMPROVES_SOURCE_NUMERIC"
    return "HIGHRES_WORSE_SOURCE_NUMERIC"


def process_sample(
    row: Dict[str, str],
    out_root: Path,
    pipeline: str,
    dry_run: bool,
    run_legacy_eval_grid: bool,
    *,
    axis_mask_margin: int = 15,
    mask_b_mag_percentile: float = 50.0,
    mask_b_thr_clip_lo: float = 10.0,
    mask_b_thr_clip_hi: float = 40.0,
) -> Optional[Dict[str, Any]]:
    sample_key = f"{row['domain']}_{row['sample_id']}"
    sample_dir = out_root / sample_key
    paths = {
        "input_image": resolve(row["input_image"]),
        "mi_json": resolve(row["mi_json"]),
        "source_numeric_json": resolve(row["source_numeric_json"]),
    }
    source = read_json(paths["source_numeric_json"])

    planned = [
        ("baseline_1x_eval_grid", 1, "eval_grid", "export_points_eval"),
        ("upscale_2x_highres", 2, "highres", "export_points_highres"),
    ]
    if run_legacy_eval_grid:
        planned.append(("upscale_2x_eval_grid", 2, "eval_grid", "export_points_eval"))

    results: Dict[str, Dict[str, Any]] = {}
    for mode, factor, final_mode, _field in planned:
        run_dir = sample_dir / mode
        result_json = run_dir / f"{row['sample_id']}_result.json"
        debug_dir = run_dir / f"debug_{row['sample_id']}_global"
        cmd = run_local_cmd(
            image_path=paths["input_image"],
            mi_json=paths["mi_json"],
            result_json=result_json,
            debug_dir=debug_dir,
            pipeline=pipeline,
            roi_upscale_factor=factor,
            final_export_mode=final_mode,
            axis_mask_margin=int(axis_mask_margin),
            mask_b_mag_percentile=float(mask_b_mag_percentile),
            mask_b_thr_clip_lo=float(mask_b_thr_clip_lo),
            mask_b_thr_clip_hi=float(mask_b_thr_clip_hi),
        )
        if result_json.is_file():
            print(f"[resume] {sample_key}/{mode}")
            if not dry_run:
                results[mode] = read_json(result_json)
            continue
        print(f"[run] {sample_key}/{mode}")
        print(f"  input_image={row['input_image']}")
        print(f"  mi_json={row['mi_json']}")
        print(f"  source_numeric_json={row['source_numeric_json']}")
        execute(cmd, run_dir, dry_run=dry_run)
        if not dry_run:
            results[mode] = read_json(result_json)

    if dry_run:
        return None

    failure_reason = ""
    try:
        baseline_metrics, baseline_points = extract_metrics(source, results["baseline_1x_eval_grid"], "export_points_eval")
        highres_metrics, highres_points = extract_metrics(source, results["upscale_2x_highres"], "export_points_highres")
    except Exception as exc:
        baseline_metrics, highres_metrics = {}, {}
        baseline_points, highres_points = {}, {}
        failure_reason = f"metric_error:{exc}"

    eval_grid_metrics: Dict[str, Any] = {}
    eval_grid_points: Dict[str, Any] = {}
    eval_grid_role = "legacy_downscale_diagnostic_skipped_cost"
    if run_legacy_eval_grid and not failure_reason:
        try:
            eval_grid_metrics, eval_grid_points = extract_metrics(source, results["upscale_2x_eval_grid"], "export_points_eval")
            eval_grid_role = "legacy_downscale_diagnostic"
        except Exception as exc:
            eval_grid_role = f"legacy_downscale_diagnostic_metric_error:{exc}"

    delta_norm = highres_metrics.get("normalized_y_mae", float("nan")) - baseline_metrics.get("normalized_y_mae", float("nan"))
    delta_corr = highres_metrics.get("shape_correlation", float("nan")) - baseline_metrics.get("shape_correlation", float("nan"))
    out: Dict[str, Any] = {
        "sample_key": sample_key,
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "input_image": row["input_image"],
        "mi_json": row["mi_json"],
        "source_numeric_json": row["source_numeric_json"],
        "baseline_point_count": int(baseline_points.get("point_count") or len(baseline_points.get("two_theta_values", []))),
        "highres_point_count": int(highres_points.get("point_count") or len(highres_points.get("two_theta_values", []))),
        "eval_grid_point_count": "" if not eval_grid_points else int(eval_grid_points.get("point_count") or len(eval_grid_points.get("two_theta_values", []))),
        "baseline_normalized_y_mae": baseline_metrics.get("normalized_y_mae", ""),
        "highres_normalized_y_mae": highres_metrics.get("normalized_y_mae", ""),
        "delta_normalized_y_mae": delta_norm,
        "baseline_shape_correlation": baseline_metrics.get("shape_correlation", ""),
        "highres_shape_correlation": highres_metrics.get("shape_correlation", ""),
        "delta_shape_correlation": delta_corr,
        "highres_peak_center_error": highres_metrics.get("peak_center_error", ""),
        "highres_peak_height_error": highres_metrics.get("peak_height_error", ""),
        "highres_peak_width_error": highres_metrics.get("peak_width_error", ""),
        "highres_false_spike_count": int(highres_metrics.get("false_spike_count") or 0),
        "highres_missed_peak_count": int(highres_metrics.get("missed_peak_count") or 0),
        "x_monotonic": x_monotonic(highres_points.get("two_theta_values", [])) if highres_points else False,
        "gap_count": int(highres_points.get("gap_count") or 0),
        "improved": bool(delta_norm < -1e-12),
        "eval_grid_normalized_y_mae": eval_grid_metrics.get("normalized_y_mae", ""),
        "eval_grid_shape_correlation": eval_grid_metrics.get("shape_correlation", ""),
        "eval_grid_role": eval_grid_role,
        "failure_reason": failure_reason,
        "run_dir": rel(sample_dir),
    }
    out["diagnosis_label"] = diagnosis_label(out)
    return out


def to_float_list(rows: List[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except Exception:
            pass
    return vals


def aggregate_domain(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for domain in ["clean", "styled", "real_like"]:
        grp = [r for r in rows if r["domain"] == domain]
        deltas = to_float_list(grp, "delta_normalized_y_mae")
        bmae = to_float_list(grp, "baseline_normalized_y_mae")
        hmae = to_float_list(grp, "highres_normalized_y_mae")
        bcorr = to_float_list(grp, "baseline_shape_correlation")
        hcorr = to_float_list(grp, "highres_shape_correlation")
        dcorr = to_float_list(grp, "delta_shape_correlation")
        out.append(
            {
                "domain": domain,
                "sample_count": len(grp),
                "improved_count": sum(1 for r in grp if r.get("improved") is True),
                "worsened_count": sum(1 for r in grp if r.get("improved") is False and not r.get("failure_reason")),
                "unchanged_count": sum(1 for d in deltas if abs(d) <= 1e-12),
                "mean_baseline_normalized_y_mae": float(np.mean(bmae)) if bmae else "",
                "mean_highres_normalized_y_mae": float(np.mean(hmae)) if hmae else "",
                "mean_delta_normalized_y_mae": float(np.mean(deltas)) if deltas else "",
                "median_delta_normalized_y_mae": float(np.median(deltas)) if deltas else "",
                "mean_baseline_shape_correlation": float(np.mean(bcorr)) if bcorr else "",
                "mean_highres_shape_correlation": float(np.mean(hcorr)) if hcorr else "",
                "mean_delta_shape_correlation": float(np.mean(dcorr)) if dcorr else "",
                "x_monotonic_fail_count": sum(1 for r in grp if not bool(r.get("x_monotonic"))),
                "gap_fail_count": sum(1 for r in grp if int(r.get("gap_count") or 0) > 0),
                "excessive_false_spike_count": sum(1 for r in grp if int(r.get("highres_false_spike_count") or 0) > 4),
            }
        )
    return out


def decide(rows: List[Dict[str, Any]], domain_rows: List[Dict[str, Any]], valid_count: int) -> Tuple[str, Dict[str, Any]]:
    executed = [r for r in rows if not r.get("failure_reason")]
    improved = [r for r in executed if r.get("improved") is True]
    worsened = [r for r in executed if r.get("improved") is False]
    deltas = to_float_list(executed, "delta_normalized_y_mae")
    dcorr = to_float_list(executed, "delta_shape_correlation")
    x_fail = sum(1 for r in executed if not bool(r.get("x_monotonic")))
    gap_fail = sum(1 for r in executed if int(r.get("gap_count") or 0) > 0)
    excessive_false = sum(1 for r in executed if int(r.get("highres_false_spike_count") or 0) > 4)
    improvement_rate = len(improved) / max(len(executed), 1)
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    median_delta = float(np.median(deltas)) if deltas else float("nan")
    mean_delta_corr = float(np.mean(dcorr)) if dcorr else float("nan")

    if (
        len(executed) >= 30
        and improvement_rate >= 0.70
        and mean_delta < 0
        and median_delta < 0
        and mean_delta_corr >= -1e-9
        and x_fail == 0
        and gap_fail == 0
        and excessive_false / max(len(executed), 1) <= 0.25
    ):
        decision = "HIGHRES_PRIMARY_EVAL_PASS_CURRENT_BEST_CANONICAL_30"
    elif (
        len(executed) >= max(1, valid_count)
        and improvement_rate >= 0.60
        and mean_delta < 0
        and x_fail == 0
        and gap_fail == 0
    ):
        decision = "HIGHRES_PRIMARY_EVAL_CONDITIONAL_PASS_DOMAIN_LIMITED"
    elif improvement_rate > 0:
        decision = "HIGHRES_PRIMARY_EVAL_MIXED_NEEDS_FAILURE_ANALYSIS"
    else:
        decision = "HIGHRES_PRIMARY_EVAL_FAIL_NOT_CURRENT_BEST"

    return decision, {
        "valid_executed_sample_count": len(executed),
        "total_improved_count": len(improved),
        "total_worsened_count": len(worsened),
        "improvement_rate": improvement_rate,
        "mean_delta_normalized_y_mae": mean_delta,
        "median_delta_normalized_y_mae": median_delta,
        "mean_delta_shape_correlation": mean_delta_corr,
        "x_monotonic_fail_count": x_fail,
        "gap_fail_count": gap_fail,
        "excessive_false_spike_count": excessive_false,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--pipeline", default="v1_2")
    ap.add_argument("--run-legacy-eval-grid", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Use existing output root and skip completed result JSONs")
    ap.add_argument("--workers", type=int, default=1, help="Number of sample-level workers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--axis-mask-margin", type=int, default=15, help="Forwarded to run_local (mask_axis_lines px before upscale scale)")
    ap.add_argument("--mask-b-mag-percentile", type=float, default=50.0)
    ap.add_argument("--mask-b-thr-clip-lo", type=float, default=10.0)
    ap.add_argument("--mask-b-thr-clip-hi", type=float, default=40.0)
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    out_root = make_out_root(
        args.out_root if args.out_root.is_absolute() else ROOT / args.out_root,
        args.overwrite,
        args.resume,
    )
    valid_rows, manifest_failures, domain_counts = load_and_validate_manifest(manifest)
    failures_path = out_root / "highres_primary_eval_canonical_30_failures.csv"

    if manifest_failures:
        write_csv(failures_path, manifest_failures, sorted(set().union(*(r.keys() for r in manifest_failures))))
        print(f"[manifest validation failed] wrote {rel(failures_path)}", file=sys.stderr)
        raise SystemExit(1)

    results: List[Dict[str, Any]] = []

    def _one(row: Dict[str, str]) -> Dict[str, Any]:
        try:
            result = process_sample(
                row,
                out_root,
                pipeline=str(args.pipeline),
                dry_run=bool(args.dry_run),
                run_legacy_eval_grid=bool(args.run_legacy_eval_grid),
                axis_mask_margin=int(args.axis_mask_margin),
                mask_b_mag_percentile=float(args.mask_b_mag_percentile),
                mask_b_thr_clip_lo=float(args.mask_b_thr_clip_lo),
                mask_b_thr_clip_hi=float(args.mask_b_thr_clip_hi),
            )
            if result is not None:
                return result
            return {}
        except Exception as exc:
            fail = {
                "sample_key": f"{row.get('domain')}_{row.get('sample_id')}",
                "sample_id": row.get("sample_id", ""),
                "domain": row.get("domain", ""),
                "failure_reason": str(exc),
                "run_dir": rel(out_root / f"{row.get('domain')}_{row.get('sample_id')}"),
            }
            return {**{k: "" for k in CSV_FIELDS}, **fail, "improved": False, "x_monotonic": False, "gap_count": ""}

    if int(args.workers) > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(_one, row) for row in valid_rows]
            for fut in as_completed(futures):
                item = fut.result()
                if item:
                    results.append(item)
    else:
        for row in valid_rows:
            item = _one(row)
            if item:
                results.append(item)

    if args.dry_run:
        print(f"[dry-run] planned_samples={len(valid_rows)} run_legacy_eval_grid={args.run_legacy_eval_grid} out_root={rel(out_root)}")
        return

    domain_summary = aggregate_domain(results)
    decision, totals = decide(results, domain_summary, len(valid_rows))

    summary_csv = out_root / "highres_primary_eval_canonical_30_summary.csv"
    domain_csv = out_root / "highres_primary_eval_canonical_30_domain_summary.csv"
    write_csv(summary_csv, results, CSV_FIELDS)
    write_csv(domain_csv, domain_summary, DOMAIN_FIELDS)
    failure_rows = [r for r in results if r.get("failure_reason") or r.get("improved") is False]
    write_csv(failures_path, failure_rows, CSV_FIELDS)

    payload = {
        "final_decision": decision,
        "canonical_root": rel(CANONICAL_ROOT),
        "manifest": rel(manifest),
        "manifest_validation": {
            "row_count": len(valid_rows),
            "domain_counts": domain_counts,
            "pattern_1915_present": False,
            "clean_pattern_11832_present": True,
            "failure_count": len(manifest_failures),
        },
        "execution": {
            "run_legacy_eval_grid": bool(args.run_legacy_eval_grid),
            "legacy_eval_grid_role": (
                "legacy_downscale_diagnostic"
                if args.run_legacy_eval_grid
                else "legacy_downscale_diagnostic_skipped_cost"
            ),
            "valid_executed_sample_count": totals["valid_executed_sample_count"],
            "axis_mask_margin": int(args.axis_mask_margin),
            "mask_b_mag_percentile": float(args.mask_b_mag_percentile),
            "mask_b_thr_clip_lo": float(args.mask_b_thr_clip_lo),
            "mask_b_thr_clip_hi": float(args.mask_b_thr_clip_hi),
        },
        "totals": totals,
        "domain_summary": domain_summary,
        "summary_rows": results,
        "failures": failure_rows,
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
            "950 eval_grid not used as primary failure basis",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_json = out_root / "highres_primary_eval_canonical_30_summary.json"
    write_json(summary_json, payload)

    improved_rows = [r for r in results if r.get("improved") is True]
    worsened_rows = [r for r in results if r.get("improved") is False and not r.get("failure_reason")]
    md = [
        "# Highres Primary Evaluation: Canonical 30",
        "",
        f"- Final decision: `{decision}`",
        f"- Canonical manifest: `{rel(manifest)}`",
        f"- Executed samples: {totals['valid_executed_sample_count']}",
        f"- Domain counts: {domain_counts}",
        "- Primary evaluation: `roi_upscale_2x_highres` vs `source_numeric.json`",
        "- 950 eval_grid role: legacy/downscale diagnostic only",
        "",
        "## Overall",
        f"- improved_count: {totals['total_improved_count']}",
        f"- worsened_count: {totals['total_worsened_count']}",
        f"- improvement_rate: {totals['improvement_rate']:.4f}",
        f"- mean_delta_normalized_y_mae: {totals['mean_delta_normalized_y_mae']:.6f}",
        f"- median_delta_normalized_y_mae: {totals['median_delta_normalized_y_mae']:.6f}",
        f"- mean_delta_shape_correlation: {totals['mean_delta_shape_correlation']:.6f}",
        f"- x_monotonic_fail_count: {totals['x_monotonic_fail_count']}",
        f"- gap_fail_count: {totals['gap_fail_count']}",
        f"- excessive_false_spike_count: {totals['excessive_false_spike_count']}",
        "",
        "## Domain Summary",
        "| domain | n | improved | worsened | mean baseline MAE | mean highres MAE | mean delta | mean delta corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in domain_summary:
        md.append(
            f"| {r['domain']} | {r['sample_count']} | {r['improved_count']} | {r['worsened_count']} | "
            f"{float(r['mean_baseline_normalized_y_mae']):.6f} | "
            f"{float(r['mean_highres_normalized_y_mae']):.6f} | "
            f"{float(r['mean_delta_normalized_y_mae']):.6f} | "
            f"{float(r['mean_delta_shape_correlation']):.6f} |"
        )
    md.extend(["", "## Worsened / Failure Samples"])
    if failure_rows:
        for r in failure_rows:
            md.append(f"- `{r.get('sample_key')}`: improved={r.get('improved')} reason={r.get('failure_reason') or 'highres_norm_mae_not_better'} delta={r.get('delta_normalized_y_mae')}")
    else:
        md.append("- none")
    md.extend(
        [
            "",
            "## Current Best Status",
            "Primary baseline = 2× highres vs source_numeric (highres_* columns).",
            "The `improved` flag is still defined as highres beating the 1× eval_grid arm (legacy reference), not as the definition of baseline.",
            "",
            "## Outputs",
            f"- Summary CSV: `{rel(summary_csv)}`",
            f"- Summary JSON: `{rel(summary_json)}`",
            f"- Domain summary CSV: `{rel(domain_csv)}`",
            f"- Failures CSV: `{rel(failures_path)}`",
            "",
            "## Not Done",
            "- canonical original files not modified",
            "- input/mi/gt/source_numeric/metadata not modified",
            "- plot_box/calibration not modified",
            "- candidate/DP/tracing scoring not modified",
            "- threshold/margin tuning not performed",
            "- 950 eval_grid not used as primary failure basis",
            "",
        ]
    )
    summary_md = out_root / "highres_primary_eval_canonical_30_summary.md"
    summary_md.write_text("\n".join(md), encoding="utf-8")

    print(
        json.dumps(
            {
                "final_decision": decision,
                "executed": totals["valid_executed_sample_count"],
                "improved": totals["total_improved_count"],
                "worsened": totals["total_worsened_count"],
                "summary_csv": rel(summary_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
