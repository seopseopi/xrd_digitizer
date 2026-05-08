#!/usr/bin/env python3
"""Run canonical-manifest ROI upscale 2x smoke test for 3 domains.

This script uses only data/test_canonical_30/manifest.csv paths. It does not
infer image/MI/GT/source paths from sample names, and it does not modify
canonical files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANONICAL_ROOT = ROOT / "data" / "test_canonical_30"
MANIFEST_PATH = CANONICAL_ROOT / "manifest.csv"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "_roi_upscale_smoke_canonical_3domain"

RUN_MODES = [
    ("baseline_1x_eval_grid", 1, "eval_grid"),
    ("upscale_2x_eval_grid", 2, "eval_grid"),
    ("upscale_2x_highres", 2, "highres"),
]

SUMMARY_FIELDS = [
    "canonical_root",
    "manifest_path",
    "sample_id",
    "domain",
    "run_mode",
    "final_export_mode",
    "upscale_factor",
    "input_image",
    "mi_json",
    "gt_json",
    "source_numeric_json",
    "mi_source",
    "mi_role",
    "pair_status",
    "fallback_used",
    "roi_width_original",
    "roi_width_after_upscale",
    "raw_trace_point_count",
    "eval_export_point_count",
    "highres_export_point_count",
    "final_export_point_count",
    "two_theta_len",
    "intensity_len",
    "highres_available",
    "gap_count",
    "valid_point_count",
    "x_monotonic",
    "strict_curve_y_mae_px",
    "curve_y_mae_px",
    "failure_reason",
    "run_dir",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_manifest_path(value: str) -> Path:
    p = Path(str(value).strip())
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"canonical manifest missing columns: {sorted(missing)}")
    if (df["sample_id"].astype(str) == "pattern_1915").any():
        raise ValueError("pattern_1915 is forbidden in canonical smoke tests")
    return df


def select_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    def require_one(mask: pd.Series, label: str) -> Dict[str, Any]:
        sel = df[mask]
        if len(sel) != 1:
            raise ValueError(f"{label} expected exactly 1 row, got {len(sel)}")
        return sel.iloc[0].to_dict()

    clean = require_one(df["test_id"].astype(str) == "clean_pattern_11832", "clean_pattern_11832")
    styled_sel = df[df["test_id"].astype(str) == "styled_pattern_72296"]
    if len(styled_sel) == 1:
        styled = styled_sel.iloc[0].to_dict()
    else:
        styled = df[(df["domain"].astype(str) == "styled") & (df["pair_status"].astype(str) == "PASS")].iloc[0].to_dict()
    real = df[(df["domain"].astype(str) == "real_like") & (df["pair_status"].astype(str) == "PASS")].iloc[0].to_dict()
    return [clean, styled, real]


def validate_row(row: Dict[str, Any], all_rows: pd.DataFrame) -> Dict[str, Any]:
    reasons: List[str] = []
    sample_id = str(row["sample_id"])
    domain = str(row["domain"])
    test_id = str(row["test_id"])

    dup = all_rows[(all_rows["sample_id"].astype(str) == sample_id) & (all_rows["domain"].astype(str) == domain)]
    if len(dup) != 1:
        reasons.append(f"duplicate_sample_domain_count={len(dup)}")
    if sample_id == "pattern_1915":
        reasons.append("pattern_1915_forbidden")
    if str(row.get("pair_status", "")) != "PASS":
        reasons.append("manifest_pair_status_not_PASS")

    paths = {
        "input_image": resolve_manifest_path(row["input_image"]),
        "mi_json": resolve_manifest_path(row["mi_json"]),
        "gt_json": resolve_manifest_path(row["gt_json"]),
        "source_numeric_json": resolve_manifest_path(row["source_numeric_json"]),
        "metadata_json": resolve_manifest_path(row["metadata_json"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            reasons.append(f"{label}_missing:{path}")

    mi: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    gt: Dict[str, Any] = {}
    if paths["mi_json"].is_file():
        mi = read_json(paths["mi_json"])
        if mi.get("mi_source") != "synthetic_from_gt":
            reasons.append("mi_source_not_synthetic_from_gt")
        if mi.get("mi_role") != "calibration_input":
            reasons.append("mi_role_not_calibration_input")
        if not isinstance(mi.get("plot_box"), list) or len(mi.get("plot_box", [])) != 4:
            reasons.append("mi_plot_box_missing")
        for key in ("x_axis_points", "x_axis_values", "y_axis_points", "y_axis_values"):
            if key not in mi:
                reasons.append(f"mi_calibration_missing:{key}")
        if mi.get("sample_id") != sample_id:
            reasons.append("mi_sample_id_mismatch")
        if mi.get("domain") != domain:
            reasons.append("mi_domain_mismatch")

    if paths["metadata_json"].is_file():
        metadata = read_json(paths["metadata_json"])
        if metadata.get("sample_id") != sample_id:
            reasons.append("metadata_sample_id_mismatch")
        if metadata.get("domain") != domain:
            reasons.append("metadata_domain_mismatch")
        canonical_paths = metadata.get("canonical_paths", {})
        expected = {
            "input_image": row["input_image"],
            "source_numeric_json": row["source_numeric_json"],
            "gt_json": row["gt_json"],
            "mi_json": row["mi_json"],
        }
        for key, value in expected.items():
            if str(canonical_paths.get(key, "")) != str(value):
                reasons.append(f"metadata_path_mismatch:{key}")

    if paths["gt_json"].is_file():
        gt = read_json(paths["gt_json"])
        if gt.get("sample_id") != sample_id:
            reasons.append("gt_sample_id_mismatch")
        if not isinstance(gt.get("plot_box"), list) or len(gt.get("plot_box", [])) != 4:
            reasons.append("gt_plot_box_missing")
        if not isinstance(gt.get("axis_metadata"), dict):
            reasons.append("gt_axis_metadata_missing")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "paths": paths,
        "mi": mi,
        "metadata": metadata,
        "gt": gt,
        "sample_dir": paths["input_image"].parent,
        "test_id": test_id,
        "sample_id": sample_id,
        "domain": domain,
    }


def make_out_root(base: Path, overwrite: bool) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    has_run_dirs = any(p.is_dir() for p in base.iterdir()) if base.is_dir() else False
    if overwrite or not has_run_dirs:
        base.mkdir(parents=True, exist_ok=True)
        return base
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = base / stamp
    out.mkdir(parents=True)
    return out


def build_cmd(
    *,
    image_path: Path,
    mi_json: Path,
    result_json: Path,
    debug_dir: Path,
    pipeline: str,
    upscale_factor: int,
    final_export_mode: str,
    roi_upscale_method: str,
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
        str(int(upscale_factor)),
        "--roi-upscale-method",
        roi_upscale_method,
        "--final-export-mode",
        final_export_mode,
    ]


def run_one(cmd: List[str], run_dir: Path, dry_run: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    if dry_run:
        return
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)


def is_monotonic(xs: List[Any]) -> bool:
    vals = [float(x) for x in xs]
    return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def arrays_equal(a: List[Any], b: List[Any]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-12):
                return False
        except Exception:
            if x != y:
                return False
    return True


def compute_metrics(result: Dict[str, Any], debug: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from eval.metrics import compute_all_metrics
        from eval.metrics_exp import compute_metrics_v2

        main = compute_all_metrics(result, debug, gt).get("main", {})
        v2 = compute_metrics_v2(result, debug, gt)
        return {
            "curve_y_mae_px": main.get("curve_y_mae_px"),
            "strict_curve_y_mae_px": v2.get("strict_curve_y_mae_px"),
            "numeric_y_mae_norm": main.get("numeric_y_mae_norm"),
            "major_peak_x_error_2theta": main.get("major_peak_x_error_2theta"),
            "peak_recall": main.get("peak_recall"),
        }
    except Exception as exc:
        return {"metrics_error": str(exc)}


def extract_row(
    *,
    canonical_root: Path,
    manifest_path: Path,
    row: Dict[str, Any],
    validation: Dict[str, Any],
    run_mode: str,
    final_export_mode: str,
    upscale_factor: int,
    run_dir: Path,
    result_json: Path,
    debug_json: Path,
) -> Dict[str, Any]:
    result = read_json(result_json)
    debug = read_json(debug_json)
    gt = validation["gt"]
    metrics = compute_metrics(result, debug, gt)

    em = result.get("export_metadata", {}) if isinstance(result.get("export_metadata"), dict) else {}
    ep_eval = result.get("export_points_eval", {}) if isinstance(result.get("export_points_eval"), dict) else {}
    ep_hi = result.get("export_points_highres", {}) if isinstance(result.get("export_points_highres"), dict) else {}
    two_theta = result.get("two_theta_values", [])
    intensities = result.get("intensities", [])

    failure_reasons: List[str] = []
    required_fields = ["two_theta_values", "intensities", "export_points_eval", "export_points_highres", "export_metadata"]
    for field in required_fields:
        if field not in result:
            failure_reasons.append(f"result_missing:{field}")

    if final_export_mode == "eval_grid":
        if not arrays_equal(two_theta, ep_eval.get("two_theta_values", [])):
            failure_reasons.append("root_two_theta_not_eval")
        if not arrays_equal(intensities, ep_eval.get("intensities", [])):
            failure_reasons.append("root_intensities_not_eval")
    if final_export_mode == "highres":
        if not arrays_equal(two_theta, ep_hi.get("two_theta_values", [])):
            failure_reasons.append("root_two_theta_not_highres")
        if not arrays_equal(intensities, ep_hi.get("intensities", [])):
            failure_reasons.append("root_intensities_not_highres")

    roi_w = int(em.get("roi_width_original") or 0)
    raw_n = int(em.get("raw_trace_point_count") or 0)
    eval_n = int(em.get("eval_export_point_count") or ep_eval.get("point_count") or 0)
    hi_n = int(em.get("highres_export_point_count") or ep_hi.get("point_count") or 0)
    final_n = int(em.get("final_export_point_count") or len(two_theta))
    expected_raw = roi_w * int(upscale_factor)
    expected_highres = expected_raw
    expected_final = roi_w if final_export_mode == "eval_grid" else expected_raw
    expected_len = expected_final
    if roi_w <= 0:
        failure_reasons.append("roi_width_missing")
    if raw_n != expected_raw:
        failure_reasons.append(f"raw_trace_point_count_expected_{expected_raw}_got_{raw_n}")
    if hi_n != expected_highres:
        failure_reasons.append(f"highres_export_point_count_expected_{expected_highres}_got_{hi_n}")
    if final_n != expected_final:
        failure_reasons.append(f"final_export_point_count_expected_{expected_final}_got_{final_n}")
    if len(two_theta) != expected_len:
        failure_reasons.append(f"two_theta_len_expected_{expected_len}_got_{len(two_theta)}")
    if len(intensities) != expected_len:
        failure_reasons.append(f"intensity_len_expected_{expected_len}_got_{len(intensities)}")
    if two_theta and not is_monotonic(two_theta):
        failure_reasons.append("x_not_monotonic")

    return {
        "canonical_root": rel(canonical_root),
        "manifest_path": rel(manifest_path),
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "run_mode": run_mode,
        "final_export_mode": final_export_mode,
        "upscale_factor": upscale_factor,
        "input_image": row["input_image"],
        "mi_json": row["mi_json"],
        "gt_json": row["gt_json"],
        "source_numeric_json": row["source_numeric_json"],
        "mi_source": validation["mi"].get("mi_source", ""),
        "mi_role": validation["mi"].get("mi_role", ""),
        "pair_status": "PASS" if not validation["reasons"] else "FAIL",
        "fallback_used": False,
        "roi_width_original": roi_w,
        "roi_width_after_upscale": int(em.get("roi_width_after_upscale") or 0),
        "raw_trace_point_count": raw_n,
        "eval_export_point_count": eval_n,
        "highres_export_point_count": hi_n,
        "final_export_point_count": final_n,
        "two_theta_len": len(two_theta),
        "intensity_len": len(intensities),
        "highres_available": bool(em.get("highres_available", False)),
        "gap_count": int(ep_hi.get("gap_count") or ep_eval.get("gap_count") or 0),
        "valid_point_count": int(ep_hi.get("valid_point_count") or ep_eval.get("valid_point_count") or 0),
        "x_monotonic": bool(two_theta and is_monotonic(two_theta)),
        "strict_curve_y_mae_px": metrics.get("strict_curve_y_mae_px", ""),
        "curve_y_mae_px": metrics.get("curve_y_mae_px", ""),
        "failure_reason": ";".join(failure_reasons),
        "run_dir": rel(run_dir),
        "_metrics_extra": metrics,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric_delta(base: Any, up: Any) -> Optional[float]:
    try:
        return float(up) - float(base)
    except Exception:
        return None


def classify(rows: List[Dict[str, Any]], selected: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    fail_rows = [r for r in rows if r["failure_reason"] or r["pair_status"] != "PASS" or not r["x_monotonic"]]
    eval_pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    domain_judgment: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = f"{r['domain']}_{r['sample_id']}"
        eval_pairs.setdefault(key, {})[r["run_mode"]] = r

    for key, variants in eval_pairs.items():
        if "baseline_1x_eval_grid" not in variants or "upscale_2x_eval_grid" not in variants:
            continue
        base = variants["baseline_1x_eval_grid"]
        up = variants["upscale_2x_eval_grid"]
        d_curve = numeric_delta(base["curve_y_mae_px"], up["curve_y_mae_px"])
        d_strict = numeric_delta(base["strict_curve_y_mae_px"], up["strict_curve_y_mae_px"])
        if d_curve is None:
            verdict = "metric_unavailable"
        elif d_curve < -1e-6:
            verdict = "improved"
        elif d_curve > 1e-6:
            verdict = "worse"
        else:
            verdict = "no_change"
        domain_judgment[key] = {
            "domain": base["domain"],
            "sample_id": base["sample_id"],
            "delta_curve_y_mae_px": d_curve,
            "delta_strict_curve_y_mae_px": d_strict,
            "verdict": verdict,
        }

    if fail_rows:
        decision = "CANONICAL_ROI_UPSCALE_2X_SMOKE_FAIL_BLOCK_BATCH"
    else:
        warnings = [v for v in domain_judgment.values() if v.get("verdict") == "worse"]
        decision = (
            "CANONICAL_ROI_UPSCALE_2X_SMOKE_PASS_WITH_DOMAIN_LIMITATION"
            if warnings
            else "CANONICAL_ROI_UPSCALE_2X_SMOKE_PASS_READY_FOR_SMALL_BATCH"
        )

    return decision, {
        "failed_run_count": len(fail_rows),
        "point_count_failures": [r for r in rows if r["failure_reason"]],
        "domain_judgment": domain_judgment,
        "selected": [
            {
                "test_id": r["test_id"],
                "sample_id": r["sample_id"],
                "domain": r["domain"],
                "input_image": r["input_image"],
                "mi_json": r["mi_json"],
                "gt_json": r["gt_json"],
                "source_numeric_json": r["source_numeric_json"],
            }
            for r in selected
        ],
    }


def write_summaries(out_root: Path, rows: List[Dict[str, Any]], decision: str, details: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    csv_path = out_root / "roi_upscale_smoke_canonical_3domain_summary.csv"
    json_path = out_root / "roi_upscale_smoke_canonical_3domain_summary.json"
    md_path = out_root / "roi_upscale_smoke_canonical_3domain_summary.md"
    write_csv(csv_path, rows)

    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    payload = {
        "final_decision": decision,
        "canonical_root": rel(CANONICAL_ROOT),
        "manifest_path": rel(MANIFEST_PATH),
        "run_count": len(rows),
        "selected_pairs": details["selected"],
        "results": public_rows,
        "pass_fail": {
            "run_pass_count": sum(1 for r in rows if not r["failure_reason"] and r["pair_status"] == "PASS" and r["x_monotonic"]),
            "run_fail_count": details["failed_run_count"],
        },
        "domain_judgment": details["domain_judgment"],
        "point_count_failures": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in details["point_count_failures"]
        ],
        "highres_diagnostic": {
            "all_highres_x_monotonic": all(
                r["x_monotonic"] for r in rows if r["run_mode"] == "upscale_2x_highres"
            ),
            "highres_gap_counts": {
                f"{r['domain']}_{r['sample_id']}": r["gap_count"]
                for r in rows
                if r["run_mode"] == "upscale_2x_highres"
            },
            "note": "highres is diagnostic only; eval_grid metrics are compared separately",
        },
        "not_done": [
            "canonical original files not modified",
            "mi.json not modified",
            "gt.json not modified",
            "source_numeric.json not modified",
            "metadata.json not modified",
            "plot_box/calibration not modified",
            "margin/candidate/DP/tracing scoring not modified",
            "no performance tuning",
            "full 30-run batch not executed",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Canonical ROI Upscale 2x Smoke Test",
        "",
        f"- Final decision: `{decision}`",
        f"- Canonical root: `{rel(CANONICAL_ROOT)}`",
        f"- Manifest: `{rel(MANIFEST_PATH)}`",
        f"- Runs: {len(rows)}",
        "",
        "## Selected Samples",
    ]
    for s in details["selected"]:
        lines.append(f"- `{s['domain']}_{s['sample_id']}`: `{s['input_image']}` + `{s['mi_json']}` + `{s['gt_json']}`")
    lines.extend(["", "## Point Counts"])
    for r in rows:
        lines.append(
            f"- `{r['domain']}_{r['sample_id']}/{r['run_mode']}`: "
            f"raw={r['raw_trace_point_count']}, eval={r['eval_export_point_count']}, "
            f"highres={r['highres_export_point_count']}, final={r['final_export_point_count']}, "
            f"root_len={r['two_theta_len']}, gaps={r['gap_count']}, x_monotonic={r['x_monotonic']}"
        )
    lines.extend(["", "## Eval Grid Comparison"])
    for key, v in details["domain_judgment"].items():
        lines.append(
            f"- `{key}`: delta_curve_y_mae_px={v.get('delta_curve_y_mae_px')}, "
            f"delta_strict_curve_y_mae_px={v.get('delta_strict_curve_y_mae_px')}, verdict={v.get('verdict')}"
        )
    lines.extend(
        [
            "",
            "## Highres Diagnostic",
            "- Highres outputs were used only for point count, gap, monotonicity, and diagnostic review.",
            "- Metrics were not mixed between eval_grid and highres.",
            "",
            "## Outputs",
            f"- CSV: `{rel(csv_path)}`",
            f"- JSON: `{rel(json_path)}`",
            f"- Markdown: `{rel(md_path)}`",
            "",
            "## Not Done",
            "- canonical original files not modified",
            "- mi.json not modified",
            "- gt.json not modified",
            "- source_numeric.json not modified",
            "- metadata.json not modified",
            "- plot_box/calibration not modified",
            "- margin/candidate/DP/tracing scoring not modified",
            "- no performance tuning",
            "- full 30-run batch not executed",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--pipeline", default="v1_2")
    ap.add_argument("--roi-upscale-method", default="lanczos", choices=["lanczos", "bicubic"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze-only", action="store_true", help="Reuse existing run outputs and regenerate summaries")
    args = ap.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    df = load_manifest(manifest_path)
    selected = select_rows(df)
    validations = [validate_row(row, df) for row in selected]
    bad = [v for v in validations if not v["ok"]]
    if bad:
        for v in bad:
            print(f"[VALIDATION_FAIL] {v['test_id']}: {';'.join(v['reasons'])}", file=sys.stderr)
        raise SystemExit(1)

    out_root = make_out_root(args.out_root if args.out_root.is_absolute() else ROOT / args.out_root, bool(args.overwrite))
    (out_root / "selected_pairs.json").write_text(
        json.dumps(
            [
                {
                    "test_id": v["test_id"],
                    "sample_id": v["sample_id"],
                    "domain": v["domain"],
                    "sample_dir": rel(v["sample_dir"]),
                    "input_image": rel(v["paths"]["input_image"]),
                    "mi_json": rel(v["paths"]["mi_json"]),
                    "gt_json": rel(v["paths"]["gt_json"]),
                    "source_numeric_json": rel(v["paths"]["source_numeric_json"]),
                    "metadata_json": rel(v["paths"]["metadata_json"]),
                    "mi_source": v["mi"].get("mi_source"),
                    "mi_role": v["mi"].get("mi_role"),
                    "pair_status": "PASS",
                }
                for v in validations
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: List[Dict[str, Any]] = []
    for row, validation in zip(selected, validations):
        test_key = f"{validation['domain']}_{validation['sample_id']}"
        for run_mode, factor, mode in RUN_MODES:
            run_dir = out_root / test_key / run_mode
            result_json = run_dir / f"{validation['sample_id']}_result.json"
            debug_dir = run_dir / f"debug_{validation['sample_id']}_global"
            debug_json = debug_dir / "debug.json"
            cmd = build_cmd(
                image_path=validation["paths"]["input_image"],
                mi_json=validation["paths"]["mi_json"],
                result_json=result_json,
                debug_dir=debug_dir,
                pipeline=str(args.pipeline),
                upscale_factor=factor,
                final_export_mode=mode,
                roi_upscale_method=str(args.roi_upscale_method),
            )
            print(f"[run] {test_key}/{run_mode}")
            print(f"  input_image={rel(validation['paths']['input_image'])}")
            print(f"  mi_json={rel(validation['paths']['mi_json'])}")
            print(f"  gt_json={rel(validation['paths']['gt_json'])}")
            if not args.dry_run and not args.analyze_only:
                run_one(cmd, run_dir, dry_run=False)
            if not args.dry_run:
                if not result_json.is_file() or not debug_json.is_file():
                    raise FileNotFoundError(f"missing existing output for analysis: {run_dir}")
                rows.append(
                    extract_row(
                        canonical_root=CANONICAL_ROOT,
                        manifest_path=manifest_path,
                        row=row,
                        validation=validation,
                        run_mode=run_mode,
                        final_export_mode=mode,
                        upscale_factor=factor,
                        run_dir=run_dir,
                        result_json=result_json,
                        debug_json=debug_json,
                    )
                )
            else:
                run_one(cmd, run_dir, dry_run=True)

    if args.dry_run:
        print(f"[dry-run] planned_runs={len(selected) * len(RUN_MODES)} out_root={rel(out_root)}")
        return

    decision, details = classify(rows, selected)
    csv_path, json_path, md_path = write_summaries(out_root, rows, decision, details)
    print(f"[saved] {rel(csv_path)}")
    print(f"[saved] {rel(json_path)}")
    print(f"[saved] {rel(md_path)}")
    print(f"final_decision={decision}")


if __name__ == "__main__":
    main()
