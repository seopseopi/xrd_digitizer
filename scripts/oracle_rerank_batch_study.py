#!/usr/bin/env python3
"""
Oracle 후보 재랭킹(GT upper bound) 배치 스터디: 샘플 선정 → rule-only vs oracle 동일 조건 실행 → 집계·산출물.

산출: outputs/oracle_rerank_batch_study/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gates import check_gate, label_failures
from eval.metrics import compute_all_metrics
from eval.report import evaluate_single
from runner.batch_run import _ensure_manual_inputs_for_gt


REPORTS_DIR = ROOT / "research_plan_outputs" / "eval_reports"
MANIFEST_DIR = ROOT / "research_plan_outputs" / "02_dataset"
DEFAULT_PROXY_CLEAN_DIR = ROOT / "outputs" / "research_diag" / "baseline_v12_default" / "clean"


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


def _draw_overlay(
    roi_path: Path,
    poly_rule: List[Tuple[float, float]],
    poly_oracle: List[Tuple[float, float]],
    out_path: Path,
) -> None:
    base = Image.open(str(roi_path)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if len(poly_rule) >= 2:
        draw.line([(p[0], p[1]) for p in poly_rule], fill=(0, 200, 255, 220), width=2)
    if len(poly_oracle) >= 2:
        draw.line([(p[0], p[1]) for p in poly_oracle], fill=(255, 0, 180, 220), width=2)
    merged = Image.alpha_composite(base, layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(out_path), format="PNG")


def _load_report_samples(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("samples") or [])


def _gate_type_for_domain(domain: str) -> str:
    if domain == "clean":
        return "clean"
    if domain == "styled":
        return "styled"
    if domain in ("real", "real_like"):
        return "real_like"
    raise ValueError(domain)


def _eval_arm(result_path: Path, debug_path: Path, gt_path: Path, gate_domain: str) -> Dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    metrics = compute_all_metrics(result, debug, gt)
    failures = label_failures(metrics)
    gates_out: Dict[str, Any] = {}
    for level in ("mvp", "development", "strict"):
        g = check_gate(metrics["main"], gate_domain, gate_level=level)
        gates_out[level] = {
            "passed": g["passed"],
            "gate_level": g.get("gate_level"),
            "gate_type": g.get("gate_type"),
        }
    tr = debug.get("trace") or {}
    return {
        "metrics": metrics,
        "gates": gates_out,
        "failure_labels": failures,
        "trace_score": float(tr.get("trace_score", 0.0)),
        "valid_ratio_dbg": float(tr.get("valid_ratio", 0.0)),
    }


def _composite_delta(rule_main: dict, ora_main: dict) -> float:
    return float(ora_main["curve_y_mae_px"]) - float(rule_main["curve_y_mae_px"]) + (
        float(ora_main["major_peak_x_error"]) - float(rule_main["major_peak_x_error"])
    ) / 10.0


def _guardrail_ok(rule_main: dict, ora_main: dict, rule_dbg: dict, ora_dbg: dict) -> bool:
    if float(ora_main["numeric_y_mae_norm"]) > float(rule_main["numeric_y_mae_norm"]) + 1e-4:
        return False
    if float(ora_main["max_gap_px"]) > float(rule_main["max_gap_px"]) + 1e-9:
        return False
    rv = float(rule_dbg.get("valid_ratio", 0.0))
    ov = float(ora_dbg.get("valid_ratio", 0.0))
    if ov < rv - 1e-6:
        return False
    return True


def _classify_outcome(rule_ev: dict, ora_ev: dict) -> Tuple[str, float]:
    """개선: 곡선 MAE·주피크 x 오차가 strict하게 함께 감소 + 가드레일 유지. 그 외 단일 축만 좋아진 경우는 worsened/unchanged로 분류."""
    rm = rule_ev["metrics"]["main"]
    om = ora_ev["metrics"]["main"]
    rd = rule_ev["metrics"]["debug"]
    od = ora_ev["metrics"]["debug"]
    dc = _composite_delta(rm, om)
    gr = _guardrail_ok(rm, om, rd, od)
    curve_better = float(om["curve_y_mae_px"]) < float(rm["curve_y_mae_px"]) - 1e-9
    peak_better = float(om["major_peak_x_error"]) < float(rm["major_peak_x_error"]) - 1e-9
    curve_worse = float(om["curve_y_mae_px"]) > float(rm["curve_y_mae_px"]) + 1e-9
    peak_worse = float(om["major_peak_x_error"]) > float(rm["major_peak_x_error"]) + 1e-9

    if not gr:
        return "worsened", dc
    if curve_better and peak_better:
        return "improved", dc
    if curve_worse or peak_worse:
        return "worsened", dc
    return "unchanged", dc


def _taxonomy_tuple(labels: Sequence[str]) -> str:
    return ";".join(sorted(labels))


def _load_clean_proxy(proxy_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(proxy_dir.glob("pattern_*_result.json")):
        sid = p.name.replace("_result.json", "")
        dpath = proxy_dir / f"debug_{sid}" / "debug.json"
        gt = ROOT / "data" / "gt" / f"{sid}_gt.json"
        if not dpath.exists() or not gt.exists():
            continue
        ev = evaluate_single(str(p), str(dpath), str(gt), gate_type="clean", gate_level="development")
        main = ev["metrics"]["main"]
        out[sid] = {
            "failure_labels": ev["failure_labels"],
            "curve_y_mae_px": float(main["curve_y_mae_px"]),
            "major_peak_x_error": float(main["major_peak_x_error"]),
            "peak_recall": float(main["peak_recall"]),
        }
    return out


def _manifest_image_gt(domain: str, sample_id: str, manifests: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    if domain == "clean":
        row = manifests["clean"][manifests["clean"]["sample_id"] == sample_id].iloc[0]
        return str(row["image_path"]), str(row["gt_path"])
    if domain == "styled":
        row = manifests["styled"][manifests["styled"]["sample_id"] == sample_id].iloc[0]
        return str(row["styled_image_path"]), str(row["gt_path"])
    if domain in ("real", "real_like"):
        row = manifests["real"][manifests["real"]["sample_id"] == sample_id].iloc[0]
        return str(row["real_image_path"]), str(row["gt_path"])
    raise ValueError(domain)


def select_samples(
    proxy_clean_dir: Path,
    n_grid: int,
    n_peak: int,
    n_high_peak: int,
    n_styled: int,
    n_real: int,
    force_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    proxy = _load_clean_proxy(proxy_clean_dir)
    forced = {x.strip() for x in force_ids if x.strip()}

    grid_pool = sorted(
        [s for s, v in proxy.items() if "grid_confusion" in v["failure_labels"]],
        key=lambda s: -proxy[s]["curve_y_mae_px"],
    )
    peak_pool = sorted(
        [s for s, v in proxy.items() if "peak_miss_after_smoothing" in v["failure_labels"]],
        key=lambda s: -proxy[s]["major_peak_x_error"],
    )
    high_peak_pool = sorted(proxy.keys(), key=lambda s: -proxy[s]["major_peak_x_error"])

    picked: Set[Tuple[str, str]] = set()
    rows: List[Dict[str, Any]] = []

    def add_row(sample_id: str, domain: str, reasons: List[str]) -> bool:
        key = (sample_id, domain)
        if key in picked:
            return False
        picked.add(key)
        rows.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "selection_reasons": ";".join(reasons),
            }
        )
        return True

    for sid in grid_pool[:n_grid]:
        add_row(sid, "clean", ["grid_confusion_proxy"])

    peak_added = 0
    for sid in peak_pool:
        if peak_added >= n_peak:
            break
        if (sid, "clean") in picked:
            continue
        if add_row(sid, "clean", ["peak_miss_after_smoothing_proxy"]):
            peak_added += 1

    hp_added = 0
    for sid in high_peak_pool:
        if hp_added >= n_high_peak:
            break
        if (sid, "clean") in picked:
            continue
        if add_row(sid, "clean", ["clean_high_major_peak_x_error_proxy"]):
            hp_added += 1

    styled_samples = _load_report_samples(REPORTS_DIR / "report_styled_development.json")
    styled_ranked = sorted(
        styled_samples,
        key=lambda s: -(
            float(s["metrics"]["main"]["curve_y_mae_px"])
            + float(s["metrics"]["main"]["major_peak_x_error"]) / 10.0
        ),
    )
    for s in styled_ranked[:n_styled]:
        add_row(str(s["sample_id"]), "styled", ["styled_outlier_report_development"])

    real_samples = _load_report_samples(REPORTS_DIR / "report_real_development.json")
    real_ranked = sorted(
        real_samples,
        key=lambda s: -(
            float(s["metrics"]["main"]["curve_y_mae_px"])
            + float(s["metrics"]["main"]["major_peak_x_error"]) / 10.0
        ),
    )
    for s in real_ranked[:n_real]:
        add_row(str(s["sample_id"]), "real_like", ["real_like_outlier_report_development"])

    for fid in forced:
        if fid not in proxy:
            continue
        if (fid, "clean") not in picked:
            add_row(fid, "clean", ["forced_include"])

    while len(rows) < 42:
        grew = False
        for sid in grid_pool:
            if len(rows) >= 42:
                break
            if add_row(sid, "clean", ["grid_confusion_proxy_fill"]):
                grew = True
        if not grew:
            break

    return rows


def _run_local(
    *,
    image_path: str,
    manual_inputs_path: str,
    output_json_path: Path,
    debug_dir: Path,
    pipeline: str,
    oracle_gt: Optional[str],
    oracle_sigma: float,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "runner" / "run_local.py"),
        "--image_path",
        image_path,
        "--manual_inputs_path",
        manual_inputs_path,
        "--output_json_path",
        str(output_json_path),
        "--debug_dir",
        str(debug_dir),
        "--pipeline",
        pipeline,
    ]
    if oracle_gt:
        cmd.extend(["--oracle-rerank-gt", oracle_gt, "--oracle-rerank-sigma", str(oracle_sigma)])
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default=str(ROOT / "outputs" / "oracle_rerank_batch_study"))
    ap.add_argument("--proxy_clean_dir", type=str, default=str(DEFAULT_PROXY_CLEAN_DIR))
    ap.add_argument("--pipeline", type=str, default="v1_2")
    ap.add_argument("--oracle_sigma", type=float, default=8.0)
    ap.add_argument("--select_only", action="store_true")
    ap.add_argument("--run_only", action="store_true", help="selected_samples.csv 가 이미 있을 때 실행만")
    ap.add_argument("--resume", action="store_true", help="결과 JSON이 있으면 해당 샘플 스킵")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mi_root = out_root / "_manual_inputs"
    mi_root.mkdir(parents=True, exist_ok=True)

    manifests = {
        "clean": pd.read_csv(MANIFEST_DIR / "manifest_clean_resolved.csv"),
        "styled": pd.read_csv(MANIFEST_DIR / "manifest_styled_resolved.csv"),
        "real": pd.read_csv(MANIFEST_DIR / "manifest_real_resolved.csv"),
    }

    selected_path = out_root / "selected_samples.csv"
    if not args.run_only:
        selection = select_samples(
            Path(args.proxy_clean_dir),
            n_grid=18,
            n_peak=12,
            n_high_peak=10,
            n_styled=5,
            n_real=5,
            force_ids=("pattern_72296", "pattern_60890"),
        )
        csv_rows = []
        proxy = _load_clean_proxy(Path(args.proxy_clean_dir))
        for r in selection:
            sid = r["sample_id"]
            dom = r["domain"]
            img, gt = _manifest_image_gt(dom, sid, manifests)
            mi_path = mi_root / f"{sid}_{dom}_manual.json"
            _ensure_manual_inputs_for_gt(gt, str(mi_path))
            pr = proxy.get(sid)
            csv_rows.append(
                {
                    "sample_id": sid,
                    "domain": dom,
                    "image_path": img,
                    "gt_path": gt,
                    "manual_inputs_path": str(mi_path.resolve()),
                    "selection_reasons": r["selection_reasons"],
                    "proxy_curve_y_mae_px": pr["curve_y_mae_px"] if pr else "",
                    "proxy_major_peak_x_error": pr["major_peak_x_error"] if pr else "",
                    "proxy_failure_labels": ";".join(pr["failure_labels"]) if pr else "",
                }
            )
        pd.DataFrame(csv_rows).to_csv(selected_path, index=False)
        print(f"[select] wrote {selected_path} n={len(csv_rows)}")
        if args.select_only:
            return

    df_sel = pd.read_csv(selected_path)
    metrics_rows: List[Dict[str, Any]] = []

    for idx, row in df_sel.iterrows():
        sid = str(row["sample_id"])
        dom = str(row["domain"])
        gate_dom = _gate_type_for_domain(dom)
        img = str(row["image_path"])
        gt = str(row["gt_path"])
        mi = str(row["manual_inputs_path"])
        run_key = f"{dom}_{sid}"
        base = out_root / "runs" / run_key
        rule_json = base / "rule" / f"{sid}_result.json"
        rule_dbg_d = base / "rule" / f"debug_{sid}_rule"
        ora_json = base / "oracle" / f"{sid}_result.json"
        ora_dbg_d = base / "oracle" / f"debug_{sid}_oracle"

        if not args.resume or not rule_json.is_file():
            print(f"[run rule] {run_key}")
            _run_local(
                image_path=img,
                manual_inputs_path=mi,
                output_json_path=rule_json,
                debug_dir=rule_dbg_d,
                pipeline=args.pipeline,
                oracle_gt=None,
                oracle_sigma=args.oracle_sigma,
            )
        if not args.resume or not ora_json.is_file():
            print(f"[run oracle] {run_key}")
            _run_local(
                image_path=img,
                manual_inputs_path=mi,
                output_json_path=ora_json,
                debug_dir=ora_dbg_d,
                pipeline=args.pipeline,
                oracle_gt=gt,
                oracle_sigma=args.oracle_sigma,
            )

        rule_ev = _eval_arm(rule_json, rule_dbg_d / "debug.json", Path(gt), gate_dom)
        ora_ev = _eval_arm(ora_json, ora_dbg_d / "debug.json", Path(gt), gate_dom)
        outcome, dc = _classify_outcome(rule_ev, ora_ev)

        rm = rule_ev["metrics"]["main"]
        om = ora_ev["metrics"]["main"]
        rd = rule_ev["metrics"]["debug"]
        od = ora_ev["metrics"]["debug"]

        metrics_rows.append(
            {
                "sample_id": sid,
                "domain": dom,
                "selection_reasons": row.get("selection_reasons", ""),
                "outcome": outcome,
                "composite_delta": round(dc, 6),
                "curve_y_mae_px_rule": rm["curve_y_mae_px"],
                "curve_y_mae_px_oracle": om["curve_y_mae_px"],
                "delta_curve_y_mae_px": round(float(om["curve_y_mae_px"]) - float(rm["curve_y_mae_px"]), 6),
                "major_peak_x_error_rule": rm["major_peak_x_error"],
                "major_peak_x_error_oracle": om["major_peak_x_error"],
                "delta_major_peak_x_error": round(
                    float(om["major_peak_x_error"]) - float(rm["major_peak_x_error"]), 6
                ),
                "numeric_y_mae_norm_rule": rm["numeric_y_mae_norm"],
                "numeric_y_mae_norm_oracle": om["numeric_y_mae_norm"],
                "delta_numeric_y_mae_norm": round(
                    float(om["numeric_y_mae_norm"]) - float(rm["numeric_y_mae_norm"]), 8
                ),
                "major_peak_x_error_2theta_rule": rm["major_peak_x_error_2theta"],
                "major_peak_x_error_2theta_oracle": om["major_peak_x_error_2theta"],
                "delta_major_peak_x_error_2theta": round(
                    float(om["major_peak_x_error_2theta"]) - float(rm["major_peak_x_error_2theta"]), 8
                ),
                "max_gap_px_rule": rm["max_gap_px"],
                "max_gap_px_oracle": om["max_gap_px"],
                "delta_max_gap_px": int(om["max_gap_px"]) - int(rm["max_gap_px"]),
                "valid_ratio_rule": rd["valid_ratio"],
                "valid_ratio_oracle": od["valid_ratio"],
                "delta_valid_ratio": round(float(od["valid_ratio"]) - float(rd["valid_ratio"]), 8),
                "trace_score_rule": rule_ev["trace_score"],
                "trace_score_oracle": ora_ev["trace_score"],
                "delta_trace_score": round(ora_ev["trace_score"] - rule_ev["trace_score"], 6),
                "mvp_pass_rule": rule_ev["gates"]["mvp"]["passed"],
                "mvp_pass_oracle": ora_ev["gates"]["mvp"]["passed"],
                "development_pass_rule": rule_ev["gates"]["development"]["passed"],
                "development_pass_oracle": ora_ev["gates"]["development"]["passed"],
                "strict_pass_rule": rule_ev["gates"]["strict"]["passed"],
                "strict_pass_oracle": ora_ev["gates"]["strict"]["passed"],
                "failure_labels_rule": _taxonomy_tuple(rule_ev["failure_labels"]),
                "failure_labels_oracle": _taxonomy_tuple(ora_ev["failure_labels"]),
                "failure_taxonomy_changed": _taxonomy_tuple(rule_ev["failure_labels"])
                != _taxonomy_tuple(ora_ev["failure_labels"]),
                "guardrail_ok": _guardrail_ok(rm, om, rd, od),
                "rule_result_json": str(rule_json.resolve()),
                "oracle_result_json": str(ora_json.resolve()),
            }
        )

    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(out_root / "oracle_batch_metrics.csv", index=False)

    improved = mdf[mdf["outcome"] == "improved"].copy()
    worsened = mdf[mdf["outcome"] == "worsened"].copy()
    unchanged = mdf[~mdf["outcome"].isin(["improved", "worsened"])].copy()
    improved.to_csv(out_root / "improved_samples.csv", index=False)
    worsened.to_csv(out_root / "worsened_samples.csv", index=False)
    unchanged.to_csv(out_root / "unchanged_samples.csv", index=False)

    reg_rows = mdf[
        (mdf["development_pass_rule"] == True)  # noqa: E712
        & (
            (mdf["development_pass_oracle"] == False)  # noqa: E712
            | (mdf["delta_curve_y_mae_px"] > 1e-6)
            | (mdf["delta_major_peak_x_error"] > 0.5)
            | (mdf["guardrail_ok"] == False)  # noqa: E712
        )
    ]
    reg_rows.to_csv(out_root / "regression_rule_success_samples.csv", index=False)

    overlay_imp = out_root / "overlays" / "top5_improved"
    overlay_worse = out_root / "overlays" / "top5_worsened"
    overlay_imp.mkdir(parents=True, exist_ok=True)
    overlay_worse.mkdir(parents=True, exist_ok=True)

    imp_sort = improved.sort_values("composite_delta", ascending=True).head(5)
    worse_sort = worsened.sort_values("composite_delta", ascending=False).head(5)

    for _, r in imp_sort.iterrows():
        sid = r["sample_id"]
        dom = r["domain"]
        run_key = f"{dom}_{sid}"
        base = out_root / "runs" / run_key
        roi = base / "rule" / f"debug_{sid}_rule" / "01_roi_preview.png"
        rr = json.loads((base / "rule" / f"{sid}_result.json").read_text(encoding="utf-8"))
        rd = json.loads((base / "rule" / f"debug_{sid}_rule" / "debug.json").read_text(encoding="utf-8"))
        oo = json.loads((base / "oracle" / f"{sid}_result.json").read_text(encoding="utf-8"))
        od = json.loads((base / "oracle" / f"debug_{sid}_oracle" / "debug.json").read_text(encoding="utf-8"))
        if roi.is_file():
            _draw_overlay(
                roi,
                _curve_roi_polyline(rr, rd),
                _curve_roi_polyline(oo, od),
                overlay_imp / f"{run_key}_overlay.png",
            )

    for _, r in worse_sort.iterrows():
        sid = r["sample_id"]
        dom = r["domain"]
        run_key = f"{dom}_{sid}"
        base = out_root / "runs" / run_key
        roi = base / "rule" / f"debug_{sid}_rule" / "01_roi_preview.png"
        rr = json.loads((base / "rule" / f"{sid}_result.json").read_text(encoding="utf-8"))
        rd = json.loads((base / "rule" / f"debug_{sid}_rule" / "debug.json").read_text(encoding="utf-8"))
        oo = json.loads((base / "oracle" / f"{sid}_result.json").read_text(encoding="utf-8"))
        od = json.loads((base / "oracle" / f"debug_{sid}_oracle" / "debug.json").read_text(encoding="utf-8"))
        if roi.is_file():
            _draw_overlay(
                roi,
                _curve_roi_polyline(rr, rd),
                _curve_roi_polyline(oo, od),
                overlay_worse / f"{run_key}_overlay.png",
            )

    def _mean(col: str) -> float:
        return float(mdf[col].astype(float).mean())

    by_dom = mdf.groupby("domain").agg(
        n=("sample_id", "count"),
        mean_curve_rule=("curve_y_mae_px_rule", "mean"),
        mean_curve_oracle=("curve_y_mae_px_oracle", "mean"),
        mean_peak_rule=("major_peak_x_error_rule", "mean"),
        mean_peak_oracle=("major_peak_x_error_oracle", "mean"),
        mean_numeric_rule=("numeric_y_mae_norm_rule", "mean"),
        mean_numeric_oracle=("numeric_y_mae_norm_oracle", "mean"),
        dev_pass_rule=("development_pass_rule", "mean"),
        dev_pass_oracle=("development_pass_oracle", "mean"),
    ).reset_index()

    tax_rule = Counter()
    tax_ora = Counter()
    for labels in mdf["failure_labels_rule"]:
        for p in str(labels).split(";"):
            if p:
                tax_rule[p] += 1
    for labels in mdf["failure_labels_oracle"]:
        for p in str(labels).split(";"):
            if p:
                tax_ora[p] += 1

    styled_subset = mdf[mdf["domain"].isin(["styled", "real_like"])]
    styled_mean_delta_curve = (
        float(styled_subset["delta_curve_y_mae_px"].mean()) if len(styled_subset) else float("nan")
    )

    verdict_lines = []
    mean_curve_improved = _mean("curve_y_mae_px_oracle") < _mean("curve_y_mae_px_rule")
    mean_peak_improved = _mean("major_peak_x_error_oracle") < _mean("major_peak_x_error_rule")
    numeric_ok = _mean("numeric_y_mae_norm_oracle") <= _mean("numeric_y_mae_norm_rule") + 1e-9
    max_gap_ok = (mdf["delta_max_gap_px"] > 0).sum() == 0
    vr_ok = (mdf["delta_valid_ratio"] < -1e-9).sum() == 0
    dev_rate_rule = float(mdf["development_pass_rule"].mean())
    dev_rate_ora = float(mdf["development_pass_oracle"].mean())
    tax_total_rule = sum(tax_rule.values())
    tax_total_ora = sum(tax_ora.values())

    a_hints = [
        mean_curve_improved,
        mean_peak_improved,
        numeric_ok,
        max_gap_ok,
        vr_ok,
        dev_rate_ora >= dev_rate_rule or tax_total_ora < tax_total_rule,
    ]
    b_hints = [
        len(improved) <= 3 and len(worsened) > len(improved),
        _mean("curve_y_mae_px_oracle") > _mean("curve_y_mae_px_rule"),
        styled_mean_delta_curve > 0.05 if len(styled_subset) else False,
        not max_gap_ok or not vr_ok,
    ]

    verdict = "A 검토 진행 (oracle 배치 신호)"
    if sum(a_hints) < 4 or sum(b_hints) >= 2:
        verdict = "B 보류 (oracle 배치 신호 약함 또는 역행)"

    md_path = out_root / "oracle_batch_summary.md"

    def _md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(비어 있음)\n"
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, rr in df.iterrows():
            lines.append("| " + " | ".join(str(rr[c]) for c in cols) + " |")
        return "\n".join(lines) + "\n"

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Oracle rerank batch study (GT upper bound)\n\n")
        f.write(
            "- **주의**: oracle은 GT 기반 상한 실험이며 CNN reranker 성능이 아님.\n"
            "- **파이프라인**: `runner/run_local.py` 동일 플래그, `--pipeline "
            + args.pipeline
            + "`, oracle만 `--oracle-rerank-gt` 추가.\n"
            "- **선정**: clean은 `baseline_v12_default/clean` 프록시 메트릭의 failure taxonomy·피크 오류 기준; styled/real_like는 `report_*_development.json` 이상치 상위.\n\n"
        )
        f.write("## 요약 판정\n\n")
        f.write(f"**{verdict}**\n\n")
        f.write("| 항목 | rule 평균 | oracle 평균 |\n|---|---|---|\n")
        f.write(f"| curve_y_mae_px | {_mean('curve_y_mae_px_rule'):.6f} | {_mean('curve_y_mae_px_oracle'):.6f} |\n")
        f.write(
            f"| major_peak_x_error | {_mean('major_peak_x_error_rule'):.6f} | {_mean('major_peak_x_error_oracle'):.6f} |\n"
        )
        f.write(
            f"| numeric_y_mae_norm | {_mean('numeric_y_mae_norm_rule'):.6f} | {_mean('numeric_y_mae_norm_oracle'):.6f} |\n"
        )
        f.write(f"| development pass rate | {dev_rate_rule:.4f} | {dev_rate_ora:.4f} |\n\n")
        f.write(f"- 샘플 수: {len(mdf)} / improved: {len(improved)} / worsened: {len(worsened)} / 기타: {len(unchanged)}\n")
        f.write(f"- max_gap 증가 샘플 수: {(mdf['delta_max_gap_px'] > 0).sum()} / valid_ratio 감소: {(mdf['delta_valid_ratio'] < -1e-9).sum()}\n\n")

        f.write("## 도메인별 평균\n\n")
        f.write(_md_table(by_dom))
        f.write("\n")

        f.write("## Failure taxonomy (라벨 출현 횟수)\n\n")
        f.write("| label | rule | oracle |\n|---|---|---|\n")
        all_labels = sorted(set(tax_rule.keys()) | set(tax_ora.keys()))
        for lb in all_labels:
            f.write(f"| {lb} | {tax_rule.get(lb, 0)} | {tax_ora.get(lb, 0)} |\n")
        f.write("\n")

        f.write("## Rule development 성공 → oracle 역행 후보\n\n")
        f.write(f"- 행 수: {len(reg_rows)} (`regression_rule_success_samples.csv`)\n\n")

        f.write("## 산출물\n\n")
        f.write("- `selected_samples.csv`\n")
        f.write("- `oracle_batch_metrics.csv`\n")
        f.write("- `improved_samples.csv` / `worsened_samples.csv` / `unchanged_samples.csv`\n")
        f.write("- `overlays/top5_improved/` · `overlays/top5_worsened/`\n")

    print(f"[DONE] summary -> {md_path}")


if __name__ == "__main__":
    main()
