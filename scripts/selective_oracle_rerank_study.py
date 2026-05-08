#!/usr/bin/env python3
"""
Selective oracle vs rule vs global oracle — 동일 selected_samples.csv 기준 3-way 비교.

산출: outputs/selective_oracle_rerank_study/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gates import check_gate, label_failures
from eval.metrics import compute_all_metrics

ARM_RULE = "rule"
ARM_GLOBAL_ORACLE = "global_oracle"
ARM_SELECTIVE_ORACLE = "selective_oracle"


def _gate_type_for_domain(domain: str) -> str:
    if domain == "clean":
        return "clean"
    if domain == "styled":
        return "styled"
    if domain in ("real", "real_like"):
        return "real_like"
    raise ValueError(domain)


def _curve_roi_polyline(result: dict, debug: dict) -> List[Tuple[float, float]]:
    cal = debug.get("calibration", {})
    pb = debug.get("plot_box", [0, 0, 0, 0])
    x0, y0 = float(pb[0]), float(pb[1])
    xs = cal.get("x_scale", 1.0)
    ys = cal.get("y_scale", 1.0)
    x_off = cal.get("x_offset", 0.0)
    y_off = cal.get("y_offset", 0.0)
    out: List[Tuple[float, float]] = []
    for tt, inten in zip(result.get("two_theta_values", []), result.get("intensities", [])):
        x_abs = (float(tt) - float(x_off)) / float(xs)
        y_abs = (float(inten) - float(y_off)) / float(ys)
        out.append((x_abs - x0, y_abs - y0))
    return out


def _draw_overlay(roi_path: Path, poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]], out_path: Path) -> None:
    base = Image.open(str(roi_path)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if len(poly_a) >= 2:
        draw.line([(p[0], p[1]) for p in poly_a], fill=(0, 200, 255, 220), width=2)
    if len(poly_b) >= 2:
        draw.line([(p[0], p[1]) for p in poly_b], fill=(255, 140, 0, 220), width=2)
    merged = Image.alpha_composite(base, layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(out_path), format="PNG")


def _eval_arm(result_path: Path, debug_path: Path, gt_path: Path, gate_domain: str) -> Dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    metrics = compute_all_metrics(result, debug, gt)
    failures = label_failures(metrics)
    gates_out: Dict[str, Any] = {}
    for level in ("mvp", "development", "strict"):
        g = check_gate(metrics["main"], gate_domain, gate_level=level)
        gates_out[level] = {"passed": g["passed"]}
    tr = debug.get("trace") or {}
    return {
        "metrics": metrics,
        "gates": gates_out,
        "failure_labels": failures,
        "trace_score": float(tr.get("trace_score", 0.0)),
        "trace_valid_ratio": float(tr.get("valid_ratio", 0.0)),
    }


def _guardrail_ok(rule_main: dict, cur_main: dict, rule_dbg: dict, cur_dbg: dict) -> bool:
    if float(cur_main["numeric_y_mae_norm"]) > float(rule_main["numeric_y_mae_norm"]) + 1e-4:
        return False
    if float(cur_main["max_gap_px"]) > float(rule_main["max_gap_px"]) + 1e-9:
        return False
    rv = float(rule_dbg.get("valid_ratio", 0.0))
    ov = float(cur_dbg.get("valid_ratio", 0.0))
    if ov < rv - 1e-6:
        return False
    return True


def _classify_vs_rule(rule_ev: Dict[str, Any], cur_ev: Dict[str, Any]) -> str:
    rm = rule_ev["metrics"]["main"]
    cm = cur_ev["metrics"]["main"]
    rd = rule_ev["metrics"]["debug"]
    cd = cur_ev["metrics"]["debug"]
    if not _guardrail_ok(rm, cm, rd, cd):
        return "worsened"
    c_down = float(cm["curve_y_mae_px"]) < float(rm["curve_y_mae_px"]) - 1e-9
    p_down = float(cm["major_peak_x_error"]) < float(rm["major_peak_x_error"]) - 1e-9
    c_up = float(cm["curve_y_mae_px"]) > float(rm["curve_y_mae_px"]) + 1e-9
    p_up = float(cm["major_peak_x_error"]) > float(rm["major_peak_x_error"]) + 1e-9
    if c_down and p_down:
        return "improved"
    if c_up or p_up:
        return "worsened"
    return "unchanged"


def _long_row_from_ev(
    arm: str,
    sid: str,
    dom: str,
    tax: str,
    ev: Dict[str, Any],
    *,
    outcome_selective_vs_rule: Optional[str] = None,
    outcome_global_vs_rule: Optional[str] = None,
) -> Dict[str, Any]:
    """한 arm 평가 결과를 한 행으로 평탄화. valid_ratio는 trace에서만 trace_valid_ratio로 둔다."""
    mm = ev["metrics"]["main"]
    dd = ev["metrics"]["debug"]
    dg = ev["metrics"]["diagnosis"]
    row: Dict[str, Any] = {
        "sample_id": sid,
        "pattern_id": sid,
        "domain": dom,
        "taxonomy_prior": tax,
        "arm": arm,
        "trace_valid_ratio": float(ev["trace_valid_ratio"]),
        "failure_labels": ";".join(sorted(ev["failure_labels"])),
    }
    for k, v in mm.items():
        row[k] = v
    for k, v in dd.items():
        if k == "valid_ratio":
            continue
        row[k] = v
    for k, v in dg.items():
        row[k] = v
    row["mvp_pass"] = bool(ev["gates"]["mvp"]["passed"])
    row["development_pass"] = bool(ev["gates"]["development"]["passed"])
    row["strict_pass"] = bool(ev["gates"]["strict"]["passed"])
    if outcome_selective_vs_rule is not None:
        row["outcome_selective_vs_rule"] = outcome_selective_vs_rule
    if outcome_global_vs_rule is not None:
        row["outcome_global_vs_rule"] = outcome_global_vs_rule
    return row


def _scalar_for_delta(x: Any) -> Optional[float]:
    if isinstance(x, (bool, np.bool_)):
        return float(int(bool(x)))
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return None


def _build_summary_df(rdf: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (dom, tax, arm), grp in rdf.groupby(["domain", "taxonomy_prior", "arm"], dropna=False):
        row: Dict[str, Any] = {
            "domain": dom,
            "taxonomy_prior": tax,
            "arm": arm,
            "n": int(len(grp)),
            "strict_pass_mean": float(grp["strict_pass"].astype(float).mean()),
            "mean_curve_y_mae_px": float(grp["curve_y_mae_px"].mean()),
            "mean_trace_valid_ratio": float(grp["trace_valid_ratio"].mean()),
            "median_trace_valid_ratio": float(grp["trace_valid_ratio"].median()),
        }
        if "peak_f1" in grp.columns:
            row["mean_peak_f1"] = float(grp["peak_f1"].mean())
        if "peak_recall" in grp.columns:
            row["mean_peak_recall"] = float(grp["peak_recall"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_delta_df(rdf: pd.DataFrame) -> pd.DataFrame:
    arms_req = (ARM_RULE, ARM_GLOBAL_ORACLE, ARM_SELECTIVE_ORACLE)
    skip_cols = {
        "sample_id",
        "pattern_id",
        "domain",
        "taxonomy_prior",
        "arm",
        "failure_labels",
        "outcome_selective_vs_rule",
        "outcome_global_vs_rule",
    }
    out_rows: List[Dict[str, Any]] = []
    for (sid, dom, tax), grp in rdf.groupby(["sample_id", "domain", "taxonomy_prior"], dropna=False):
        by = grp.set_index("arm")
        if not all(a in by.index for a in arms_req):
            continue
        r0, g0, s0 = by.loc[ARM_RULE], by.loc[ARM_GLOBAL_ORACLE], by.loc[ARM_SELECTIVE_ORACLE]
        row: Dict[str, Any] = {"sample_id": sid, "pattern_id": sid, "domain": dom, "taxonomy_prior": tax}
        for c in rdf.columns:
            if c in skip_cols:
                continue
            vr = _scalar_for_delta(r0.get(c))
            vg = _scalar_for_delta(g0.get(c))
            vs = _scalar_for_delta(s0.get(c))
            if vr is None or vg is None or vs is None:
                continue
            row[f"delta_{c}_selective_minus_rule"] = round(vs - vr, 8)
            row[f"delta_{c}_selective_minus_global"] = round(vs - vg, 8)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _run_local(
    *,
    image_path: str,
    manual_inputs: str,
    out_json: Path,
    dbg_dir: Path,
    pipeline: str,
    extra_args: List[str],
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "runner" / "run_local.py"),
        "--image_path",
        image_path,
        "--manual_inputs_path",
        manual_inputs,
        "--output_json_path",
        str(out_json),
        "--debug_dir",
        str(dbg_dir),
        "--pipeline",
        pipeline,
    ] + extra_args
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _resolve_result_json(preferred: Path) -> Path:
    """기본 경로가 없으면 같은 디렉터리의 *_result.json 1개를 fallback."""
    if preferred.is_file():
        return preferred
    cand = sorted(preferred.parent.glob("*_result.json"))
    if len(cand) == 1:
        return cand[0]
    return preferred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selected_csv",
        type=str,
        default=str(ROOT / "outputs" / "oracle_rerank_batch_study" / "selected_samples.csv"),
    )
    ap.add_argument(
        "--prior_metrics_csv",
        type=str,
        default=str(ROOT / "outputs" / "oracle_rerank_batch_study" / "oracle_batch_metrics.csv"),
    )
    ap.add_argument("--out_root", type=str, default=str(ROOT / "outputs" / "selective_oracle_rerank_study"))
    ap.add_argument("--pipeline", type=str, default="v1_2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--selective-oracle-allow-styled-real",
        action="store_true",
        help="selective oracle risk build를 styled/real_like에도 허용(run_local 플래그 전달)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        metavar="N",
        help="처리할 샘플 행 수 상한(실행 단위 df 확정 직후 적용). 미지정이면 전체.",
    )
    ap.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        dest="sample_offset",
        metavar="K",
        help="selected_samples.csv 행 기준 시작 인덱스 (--max-samples와 함께 df.iloc[offset:offset+N]).",
    )
    ap.add_argument("--selective-risk-dilation-radius", type=int, default=3, metavar="N")
    ap.add_argument("--selective-risk-merge-gap", type=int, default=2, metavar="N")
    ap.add_argument("--selective-risk-min-segment-len", type=int, default=6, metavar="N")
    ap.add_argument("--selective-risk-threshold", type=float, default=0.08, metavar="T")
    ap.add_argument("--selective-risk-disable-taxonomy-prior", action="store_true")
    ap.add_argument("--selective-risk-disable-low-margin", action="store_true")
    ap.add_argument("--selective-risk-disable-candidate-starvation", action="store_true")
    ap.add_argument("--selective-risk-disable-path-instability", action="store_true")
    ap.add_argument("--selective-risk-disable-peak-miss-prior", action="store_true")
    ap.add_argument("--selective-risk-disable-grid-confusion-prior", action="store_true")
    ap.add_argument("--selective-risk-disable-axis-proximity", action="store_true")
    ap.add_argument("--selective-risk-disable-high-entropy", action="store_true")
    ap.add_argument("--selective-risk-disable-large-y-gap", action="store_true")
    ap.add_argument("--selective-risk-disable-peak-window", action="store_true")
    ap.add_argument("--selective-risk-disable-dp-margin-low", action="store_true")
    ap.add_argument("--selective-risk-debug-include-columns", action="store_true")
    ap.add_argument(
        "--selective-risk-taxonomy-require-margin",
        action="store_true",
        help="taxonomy_prior를 저마진 또는 고엔트로피 게이트와 함께만 적용 (run_local 전달)",
    )
    ap.add_argument(
        "--selective-risk-high-entropy-require-low-margin",
        action="store_true",
        help="high_entropy_many_cands에 conf_margin < threshold 요구 (run_local 전달)",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    feat_csv = out_root / "risk_features_columns.csv"
    seg_csv = out_root / "risk_segments.csv"
    if not args.resume:
        for p in (feat_csv, seg_csv):
            if p.exists():
                p.unlink()

    sel_src = Path(args.selected_csv)
    shutil.copy2(sel_src, out_root / "selected_samples.csv")
    df = pd.read_csv(out_root / "selected_samples.csv")
    start = max(0, int(args.sample_offset))
    if args.max_samples is None:
        df = df.iloc[start:].copy()
    else:
        end = start + max(0, int(args.max_samples))
        df = df.iloc[start:end].copy()
    prior = pd.read_csv(args.prior_metrics_csv) if Path(args.prior_metrics_csv).is_file() else None
    tax_map: Dict[Tuple[str, str], str] = {}
    if prior is not None:
        for _, r in prior.iterrows():
            tax_map[(str(r["sample_id"]), str(r["domain"]))] = str(r.get("failure_labels_rule", ""))

    metrics_rows: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        dom = str(row["domain"])
        gt = str(row["gt_path"])
        img = str(row["image_path"])
        mi = str(row["manual_inputs_path"])
        run_key = f"{dom}_{sid}"
        base = out_root / "runs" / run_key
        gate_dom = _gate_type_for_domain(dom)
        tax = tax_map.get((sid, dom), "")

        rule_j = base / "rule" / f"{sid}_result.json"
        rule_d = base / "rule" / f"debug_{sid}_rule" / "debug.json"
        glo_j = base / "global" / f"{sid}_result.json"
        glo_dd = base / "global" / f"debug_{sid}_global" / "debug.json"
        sel_j = base / "selective" / f"{sid}_result.json"
        sel_dd = base / "selective" / f"debug_{sid}_selective" / "debug.json"

        if not args.resume or not rule_j.is_file():
            print(f"[rule] {run_key}")
            _run_local(
                image_path=img,
                manual_inputs=mi,
                out_json=rule_j,
                dbg_dir=base / "rule" / f"debug_{sid}_rule",
                pipeline=args.pipeline,
                extra_args=[],
            )
        if not args.resume or not glo_j.is_file():
            print(f"[global oracle] {run_key}")
            _run_local(
                image_path=img,
                manual_inputs=mi,
                out_json=glo_j,
                dbg_dir=base / "global" / f"debug_{sid}_global",
                pipeline=args.pipeline,
                extra_args=["--oracle-rerank-gt", gt, "--oracle-rerank-sigma", "8"],
            )
        tax_arg: List[str] = []
        if tax:
            tax_arg = ["--selective-oracle-taxonomy-prior", str(tax)]
        if not args.resume or not sel_j.is_file():
            print(f"[selective oracle] {run_key}")
            allow_style_arg = ["--selective-oracle-allow-styled-real"] if args.selective_oracle_allow_styled_real else []
            _run_local(
                image_path=img,
                manual_inputs=mi,
                out_json=sel_j,
                dbg_dir=base / "selective" / f"debug_{sid}_selective",
                pipeline=args.pipeline,
                extra_args=[
                    "--selective-oracle-rerank-gt",
                    gt,
                    "--selective-oracle-sigma",
                    "8",
                    "--run-domain",
                    dom if dom in ("clean", "styled", "real_like") else "clean",
                    "--selective-oracle-risk-features-csv",
                    str(feat_csv),
                    "--selective-risk-dilation-radius",
                    str(int(args.selective_risk_dilation_radius)),
                    "--selective-risk-merge-gap",
                    str(int(args.selective_risk_merge_gap)),
                    "--selective-risk-min-segment-len",
                    str(int(args.selective_risk_min_segment_len)),
                    "--selective-risk-threshold",
                    str(float(args.selective_risk_threshold)),
                ]
                + allow_style_arg
                + (["--selective-risk-disable-taxonomy-prior"] if args.selective_risk_disable_taxonomy_prior else [])
                + (["--selective-risk-disable-low-margin"] if args.selective_risk_disable_low_margin else [])
                + (["--selective-risk-disable-candidate-starvation"] if args.selective_risk_disable_candidate_starvation else [])
                + (["--selective-risk-disable-path-instability"] if args.selective_risk_disable_path_instability else [])
                + (["--selective-risk-disable-peak-miss-prior"] if args.selective_risk_disable_peak_miss_prior else [])
                + (["--selective-risk-disable-grid-confusion-prior"] if args.selective_risk_disable_grid_confusion_prior else [])
                + (["--selective-risk-disable-axis-proximity"] if args.selective_risk_disable_axis_proximity else [])
                + (["--selective-risk-disable-high-entropy"] if args.selective_risk_disable_high_entropy else [])
                + (["--selective-risk-disable-large-y-gap"] if args.selective_risk_disable_large_y_gap else [])
                + (["--selective-risk-disable-peak-window"] if args.selective_risk_disable_peak_window else [])
                + (["--selective-risk-disable-dp-margin-low"] if args.selective_risk_disable_dp_margin_low else [])
                + (["--selective-risk-debug-include-columns"] if args.selective_risk_debug_include_columns else [])
                + (["--selective-risk-taxonomy-require-margin"] if args.selective_risk_taxonomy_require_margin else [])
                + (
                    ["--selective-risk-high-entropy-require-low-margin"]
                    if args.selective_risk_high_entropy_require_low_margin
                    else []
                )
                + tax_arg,
            )

        rule_j_eval = _resolve_result_json(rule_j)
        glo_j_eval = _resolve_result_json(glo_j)
        sel_j_eval = _resolve_result_json(sel_j)
        ev_r = _eval_arm(rule_j_eval, rule_d, Path(gt), gate_dom)
        ev_g = _eval_arm(glo_j_eval, glo_dd, Path(gt), gate_dom)
        ev_s = _eval_arm(sel_j_eval, sel_dd, Path(gt), gate_dom)

        dbg_s = json.loads(sel_dd.read_text(encoding="utf-8"))
        ma = dbg_s.get("model_assist") or {}
        rd = ma.get("risk_detector") or {}
        risk_ratio = float(rd.get("risk_ratio", 0.0))
        risk_seg_n = int(rd.get("risk_segments", 0))
        segs = dbg_s.get("risk_detector_segments") or []

        for sg in segs:
            seg_rows.append(
                {
                    "sample_id": sid,
                    "domain": dom,
                    "segment_start_x": sg.get("segment_start_x"),
                    "segment_end_x": sg.get("segment_end_x"),
                    "segment_len": sg.get("segment_len"),
                    "risk_reasons": sg.get("risk_reasons"),
                    "risk_column_count": rd.get("risk_column_count"),
                    "total_column_count": rd.get("total_column_count"),
                    "risk_ratio": risk_ratio,
                }
            )

        rm = ev_r["metrics"]["main"]
        gm = ev_g["metrics"]["main"]
        sm = ev_s["metrics"]["main"]

        def d(a: dict, b: dict, k: str) -> float:
            return round(float(b[k]) - float(a[k]), 8)

        out_sel = _classify_vs_rule(ev_r, ev_s)
        out_glo = _classify_vs_rule(ev_r, ev_g)

        results_rows.append(_long_row_from_ev(ARM_RULE, sid, dom, tax, ev_r))
        results_rows.append(
            _long_row_from_ev(
                ARM_GLOBAL_ORACLE,
                sid,
                dom,
                tax,
                ev_g,
                outcome_global_vs_rule=out_glo,
            ),
        )
        results_rows.append(
            _long_row_from_ev(
                ARM_SELECTIVE_ORACLE,
                sid,
                dom,
                tax,
                ev_s,
                outcome_selective_vs_rule=out_sel,
            ),
        )

        dg_sel = ev_s["metrics"]["diagnosis"]
        prox_keys = (
            "candidate_recall_per_column",
            "candidate_gt_near_recall_px3",
            "candidate_gt_near_recall_px5",
            "candidate_gt_near_recall_px10",
            "mean_nearest_candidate_gt_dist_px",
            "median_nearest_candidate_gt_dist_px",
            "p90_nearest_candidate_gt_dist_px",
            "columns_evaluated",
            "gt_columns_mapped_meta",
        )
        prox_patch = {k: dg_sel[k] for k in prox_keys if k in dg_sel}

        metrics_rows.append(
            {
                "sample_id": sid,
                "domain": dom,
                "taxonomy_prior": tax,
                "outcome_selective_vs_rule": out_sel,
                "outcome_global_vs_rule": out_glo,
                "risk_ratio": risk_ratio,
                "risk_segment_count": risk_seg_n,
                **prox_patch,
                "curve_y_mae_px_rule": rm["curve_y_mae_px"],
                "curve_y_mae_px_global": gm["curve_y_mae_px"],
                "curve_y_mae_px_selective": sm["curve_y_mae_px"],
                "delta_curve_global_minus_rule": d(rm, gm, "curve_y_mae_px"),
                "delta_curve_selective_minus_rule": d(rm, sm, "curve_y_mae_px"),
                "delta_curve_selective_minus_global": d(gm, sm, "curve_y_mae_px"),
                "major_peak_x_error_rule": rm["major_peak_x_error"],
                "major_peak_x_error_global": gm["major_peak_x_error"],
                "major_peak_x_error_selective": sm["major_peak_x_error"],
                "delta_peak_global_minus_rule": d(rm, gm, "major_peak_x_error"),
                "delta_peak_selective_minus_rule": d(rm, sm, "major_peak_x_error"),
                "delta_peak_selective_minus_global": d(gm, sm, "major_peak_x_error"),
                "numeric_y_mae_norm_rule": rm["numeric_y_mae_norm"],
                "numeric_y_mae_norm_global": gm["numeric_y_mae_norm"],
                "numeric_y_mae_norm_selective": sm["numeric_y_mae_norm"],
                "delta_numeric_global_minus_rule": d(rm, gm, "numeric_y_mae_norm"),
                "delta_numeric_selective_minus_rule": d(rm, sm, "numeric_y_mae_norm"),
                "delta_numeric_selective_minus_global": d(gm, sm, "numeric_y_mae_norm"),
                "max_gap_px_rule": rm["max_gap_px"],
                "max_gap_px_global": gm["max_gap_px"],
                "max_gap_px_selective": sm["max_gap_px"],
                "trace_valid_ratio_rule": ev_r["trace_valid_ratio"],
                "trace_valid_ratio_global": ev_g["trace_valid_ratio"],
                "trace_valid_ratio_selective": ev_s["trace_valid_ratio"],
                "trace_score_rule": ev_r["trace_score"],
                "trace_score_global": ev_g["trace_score"],
                "trace_score_selective": ev_s["trace_score"],
                "mvp_pass_rule": ev_r["gates"]["mvp"]["passed"],
                "mvp_pass_global": ev_g["gates"]["mvp"]["passed"],
                "mvp_pass_selective": ev_s["gates"]["mvp"]["passed"],
                "development_pass_rule": ev_r["gates"]["development"]["passed"],
                "development_pass_global": ev_g["gates"]["development"]["passed"],
                "development_pass_selective": ev_s["gates"]["development"]["passed"],
                "strict_pass_rule": ev_r["gates"]["strict"]["passed"],
                "strict_pass_global": ev_g["gates"]["strict"]["passed"],
                "strict_pass_selective": ev_s["gates"]["strict"]["passed"],
                "failure_labels_rule": ";".join(sorted(ev_r["failure_labels"])),
                "failure_labels_global": ";".join(sorted(ev_g["failure_labels"])),
                "failure_labels_selective": ";".join(sorted(ev_s["failure_labels"])),
            }
        )

    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(out_root / "selective_oracle_metrics.csv", index=False)

    rdf = pd.DataFrame(results_rows)
    rdf.to_csv(out_root / "selective_oracle_rerank_results.csv", index=False)
    _build_summary_df(rdf).to_csv(out_root / "selective_oracle_rerank_summary.csv", index=False)
    _build_delta_df(rdf).to_csv(out_root / "selective_oracle_rerank_delta.csv", index=False)

    pd.DataFrame(seg_rows).to_csv(seg_csv, index=False)

    imp = mdf[mdf["outcome_selective_vs_rule"] == "improved"].copy()
    wor = mdf[mdf["outcome_selective_vs_rule"] == "worsened"].copy()
    unc = mdf[~mdf["outcome_selective_vs_rule"].isin(["improved", "worsened"])].copy()
    imp.to_csv(out_root / "improved_samples.csv", index=False)
    wor.to_csv(out_root / "worsened_samples.csv", index=False)
    unc.to_csv(out_root / "unchanged_samples.csv", index=False)

    reg = mdf[
        (mdf["development_pass_rule"] == True)  # noqa: E712
        & (
            (mdf["development_pass_selective"] == False)  # noqa: E712
            | (mdf["delta_curve_selective_minus_rule"] > 1e-6)
            | (mdf["delta_peak_selective_minus_rule"] > 0.5)
        )
    ]
    reg.to_csv(out_root / "regression_rule_success_samples.csv", index=False)

    # overlays selective vs rule: top5 improved / worsened by curve delta selective-rule
    ovi = out_root / "overlays" / "top5_selective_improved"
    ovw = out_root / "overlays" / "top5_selective_worsened"
    ovi.mkdir(parents=True, exist_ok=True)
    ovw.mkdir(parents=True, exist_ok=True)
    imp5 = imp.sort_values("delta_curve_selective_minus_rule", ascending=True).head(5)
    wor5 = wor.sort_values("delta_curve_selective_minus_rule", ascending=False).head(5)
    for _, r in imp5.iterrows():
        sid, dom = r["sample_id"], r["domain"]
        rk = f"{dom}_{sid}"
        b = out_root / "runs" / rk
        roi = b / "rule" / f"debug_{sid}_rule" / "01_roi_preview.png"
        if roi.is_file():
            rr = json.loads((b / "rule" / f"{sid}_result.json").read_text(encoding="utf-8"))
            rd = json.loads((b / "rule" / f"debug_{sid}_rule" / "debug.json").read_text(encoding="utf-8"))
            ss = json.loads((b / "selective" / f"{sid}_result.json").read_text(encoding="utf-8"))
            sd = json.loads((b / "selective" / f"debug_{sid}_selective" / "debug.json").read_text(encoding="utf-8"))
            _draw_overlay(roi, _curve_roi_polyline(rr, rd), _curve_roi_polyline(ss, sd), ovi / f"{rk}_rule_vs_selective.png")
    for _, r in wor5.iterrows():
        sid, dom = r["sample_id"], r["domain"]
        rk = f"{dom}_{sid}"
        b = out_root / "runs" / rk
        roi = b / "rule" / f"debug_{sid}_rule" / "01_roi_preview.png"
        if roi.is_file():
            rr = json.loads((b / "rule" / f"{sid}_result.json").read_text(encoding="utf-8"))
            rd = json.loads((b / "rule" / f"debug_{sid}_rule" / "debug.json").read_text(encoding="utf-8"))
            ss = json.loads((b / "selective" / f"{sid}_result.json").read_text(encoding="utf-8"))
            sd = json.loads((b / "selective" / f"debug_{sid}_selective" / "debug.json").read_text(encoding="utf-8"))
            _draw_overlay(roi, _curve_roi_polyline(rr, rd), _curve_roi_polyline(ss, sd), ovw / f"{rk}_rule_vs_selective.png")

    # Verdict A/B/C
    n = len(mdf)
    mean_curve_r = float(mdf["curve_y_mae_px_rule"].mean())
    mean_curve_g = float(mdf["curve_y_mae_px_global"].mean())
    mean_curve_s = float(mdf["curve_y_mae_px_selective"].mean())
    mean_peak_r = float(mdf["major_peak_x_error_rule"].mean())
    mean_peak_g = float(mdf["major_peak_x_error_global"].mean())
    mean_peak_s = float(mdf["major_peak_x_error_selective"].mean())
    mean_num_r = float(mdf["numeric_y_mae_norm_rule"].mean())
    mean_num_s = float(mdf["numeric_y_mae_norm_selective"].mean())
    max_gap_inc_g = int((mdf["max_gap_px_global"] > mdf["max_gap_px_rule"]).sum())
    max_gap_inc_s = int((mdf["max_gap_px_selective"] > mdf["max_gap_px_rule"]).sum())
    vr_dec_g = int((mdf["trace_valid_ratio_global"] < mdf["trace_valid_ratio_rule"] - 1e-9).sum())
    vr_dec_s = int((mdf["trace_valid_ratio_selective"] < mdf["trace_valid_ratio_rule"] - 1e-9).sum())
    wors_g = int((mdf["outcome_global_vs_rule"] == "worsened").sum())
    wors_s = int((mdf["outcome_selective_vs_rule"] == "worsened").sum())
    dev_r = float(mdf["development_pass_rule"].mean())
    dev_g = float(mdf["development_pass_global"].mean())
    dev_s = float(mdf["development_pass_selective"].mean())
    reg_g_ct = int(
        (
            (mdf["development_pass_rule"] == True)  # noqa: E712
            & (
                (mdf["development_pass_global"] == False)  # noqa: E712
                | (mdf["delta_curve_global_minus_rule"] > 1e-6)
                | (mdf["delta_peak_global_minus_rule"] > 0.5)
            )
        ).sum()
    )
    reg_s_ct = len(reg)

    verdict = "C"
    if (
        mean_curve_s < mean_curve_r
        and mean_peak_s < mean_peak_r
        and mean_num_s <= mean_num_r + 1e-9
        and max_gap_inc_s <= max(0, max_gap_inc_g)
        and vr_dec_s <= max(0, vr_dec_g)
        and wors_s < wors_g
        and dev_s >= dev_r
        and reg_s_ct <= reg_g_ct
    ):
        verdict = "A"
    elif mean_curve_s < mean_curve_r and mean_peak_s < mean_peak_r and wors_s < wors_g:
        verdict = "B"

    md = out_root / "selective_vs_global_summary.md"
    lines = [
        "# Selective vs global oracle rerank (GT upper bound)",
        "",
        "**B0**(`dist/xrd_digitizer_model_v1_3`)는 공식 성능 기준선이다. 본 selective oracle study는 current branch에서 수행한 GT upper bound 실험으로, 모델 적용 정책을 설계하기 위한 연구 결과이며 B0 대비 공식 성능 개선 주장으로 사용하지 않는다.",
        "",
        "## 실험 목적",
        "전역 oracle은 평균은 좋아도 worsened가 많다. 위험 열에만 oracle을 적용하면 악화 샘플을 줄이면서 rule 대비 개선을 유지할 수 있는지 검증한다.",
        "",
        "## Global oracle batch 요약(참고)",
        "51 샘플 oracle 배치: curve 42.61→37.62, peak 41.61→31.84, worsened 34 / improved 16, strict 0%.",
        "",
        "## Selective 설정",
        "- `core/selective_oracle_settings.py` 기본 임계값 + `trace/risk_detector.py` rule v1",
        "- styled/real_like는 기본 oracle off (`--selective-oracle-allow-styled-real`로 본 스크립트에서만 예외 허용)",
        "",
        "## Risk detector rule (요약)",
        "- conf_margin 낮음, 다후보+고엔트로피, axis 근접, y_gap+저마진, 피크창+저마진, taxonomy prior(조건부), DP block margin 낮음",
        "",
        "## Risk column 비율",
        f"- 평균 risk_ratio: {float(mdf['risk_ratio'].mean()):.4f} (샘플별 `selective_oracle_metrics.csv` 참고)",
        "",
        "## 평균 metric (rule / global / selective)",
        "",
        "| metric | rule | global | selective |",
        "|--------|-----:|-------:|----------:|",
        f"| curve_y_mae_px | {mean_curve_r:.4f} | {mean_curve_g:.4f} | {mean_curve_s:.4f} |",
        f"| major_peak_x_error | {mean_peak_r:.4f} | {mean_peak_g:.4f} | {mean_peak_s:.4f} |",
        f"| numeric_y_mae_norm | {mean_num_r:.6f} | {float(mdf['numeric_y_mae_norm_global'].mean()):.6f} | {mean_num_s:.6f} |",
        "",
        f"- max_gap rule→global 증가 샘플: {max_gap_inc_g}, selective: {max_gap_inc_s}",
        f"- valid_ratio 감소 샘플 global: {vr_dec_g}, selective: {vr_dec_s}",
        "",
        "## improved / worsened (vs rule)",
        f"- selective: improved {len(imp)} / worsened {len(wor)} / 기타 {len(unc)}",
        f"- global: improved {(mdf['outcome_global_vs_rule']=='improved').sum()} / worsened {(mdf['outcome_global_vs_rule']=='worsened').sum()}",
        "",
        "## Pass rate (development / mvp / strict)",
        f"- development: rule {dev_r:.4f}, global {dev_g:.4f}, selective {dev_s:.4f}",
        f"- mvp: rule {float(mdf['mvp_pass_rule'].mean()):.4f}, global {float(mdf['mvp_pass_global'].mean()):.4f}, selective {float(mdf['mvp_pass_selective'].mean()):.4f}",
        f"- strict: 전부 0 또는 동일 트렌드 (표는 CSV 참고)",
        "",
        "## Rule development 성공 → selective 역행",
        f"- selective regression 행 수: {reg_s_ct} (global 유사 지표 행 수 참고: {reg_g_ct})",
        "",
        "## 대표 overlay",
        f"- `overlays/top5_selective_improved/`",
        f"- `overlays/top5_selective_worsened/`",
        "",
        "## 한계",
        "- oracle은 GT 상한; CNN selective과 동일하지 않음.",
        "- risk feature는 rule DP 1회 기반이며 recovery 전 단계 정보다.",
        "",
        "## 최종 판정",
        f"**{verdict}** (A=selective 성공, B=부분, C=실패 기준은 스크립트 내 휴리스틱)",
        "",
        "Selective oracle rerank는 실제 모델이 아니라 GT 기반 상한 실험이다. 이 실험의 목적은 모델 학습 전, 재랭킹을 전체 적용할지 위험 구간에만 적용할지 판단하는 것이다. Selective oracle이 global oracle 대비 worsened 샘플을 줄이면서 rule 대비 주요 지표를 개선할 때만 selective candidate re-ranker 학습 단계로 넘어간다.",
        "",
        "## 다음 단계",
        "- risk rule 튜닝 및 열 단위 지도 시각화",
        "- CNN selective과 동일 마스크로 오프라인 재현",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] verdict={verdict} -> {md}")


if __name__ == "__main__":
    main()
