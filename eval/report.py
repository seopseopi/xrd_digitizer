"""
$17: Batch evaluation report.

Reads result/debug/GT JSONs, computes metrics, checks gates, generates report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import compute_all_metrics
from eval.gates import check_gate, compute_verdict, label_failures


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _metric_stats_block(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p10": round(_percentile(list(arr), 10), 6),
        "p90": round(_percentile(list(arr), 90), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def evaluate_single(
    result_path: str,
    debug_path: str,
    gt_path: str,
    gate_type: str = "clean",
    gate_level: str = "development",
) -> dict:
    """Evaluate a single sample."""
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    debug = json.loads(Path(debug_path).read_text(encoding="utf-8"))
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))

    metrics = compute_all_metrics(result, debug, gt)
    gate = check_gate(metrics["main"], gate_type, gate_level=gate_level)
    failures = label_failures(metrics)

    return {
        "metrics": metrics,
        "gate": gate,
        "failure_labels": failures,
    }


def evaluate_batch(
    manifest_csv: str,
    output_dir: str,
    gate_type: str = "clean",
    max_samples: int = 0,
    gate_level: str = "development",
    *,
    baseline_report_path: Optional[str] = None,
    run_tag: Optional[str] = None,
    git_commit: Optional[str] = None,
) -> dict:
    """Evaluate all samples in manifest."""
    df = pd.read_csv(manifest_csv)
    out_root = Path(output_dir)

    results: List[dict] = []
    for idx, row in df.iterrows():
        if max_samples > 0 and idx >= max_samples:
            break

        sid = str(row["sample_id"])
        gt_path = str(row["gt_path"])
        result_path = out_root / f"{sid}_result.json"
        debug_path = out_root / f"debug_{sid}" / "debug.json"

        if not result_path.exists() or not debug_path.exists():
            continue

        try:
            ev = evaluate_single(
                str(result_path), str(debug_path), gt_path,
                gate_type=gate_type, gate_level=gate_level,
            )
            ev["sample_id"] = sid
            results.append(ev)
        except Exception as exc:
            print(f"[EVAL SKIP] {sid}: {type(exc).__name__}: {exc}")

    aggregate = _aggregate(results)
    baseline_agg = None
    if baseline_report_path and Path(baseline_report_path).is_file():
        baseline_agg = json.loads(
            Path(baseline_report_path).read_text(encoding="utf-8")
        ).get("aggregate", {})

    pass_rate = float(aggregate.get("pass_rate", 0.0))
    verdict = compute_verdict(pass_rate, gate_level, baseline_agg, aggregate)

    top_failures: Dict[str, int] = {}
    ftc = aggregate.get("failure_taxonomy_counts", {})
    if ftc:
        for k, v in sorted(ftc.items(), key=lambda x: -x[1])[:8]:
            top_failures[k] = v

    summary = {
        "domain": gate_type,
        "gate_level": gate_level,
        "num_samples": aggregate.get("total_samples", 0),
        "pass_rate": pass_rate,
        "verdict": verdict,
    }

    if git_commit is None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None

    metadata = {
        "manifest_csv": str(Path(manifest_csv).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "gate_type": gate_type,
        "gate_level": gate_level,
        "num_samples": aggregate.get("total_samples", 0),
        "git_commit": git_commit,
        "run_tag": run_tag,
    }

    decision = {
        "relative_to_baseline": (
            "improved" if verdict == "fail_but_improved" else "unknown"
        ),
        "next_action": _suggest_next_action(top_failures),
    }

    out: Dict[str, Any] = {
        "summary": summary,
        "metadata": metadata,
        "samples": results,
        "aggregate": aggregate,
        "core_metrics_stats": aggregate.get("main_metrics_stats", {}),
        "debug_metrics_stats": aggregate.get("debug_metrics_stats", {}),
        "diagnosis_metrics_stats": aggregate.get("diagnosis_metrics_stats", {}),
        "failure_taxonomy_counts": aggregate.get("failure_taxonomy_counts", {}),
        "top_failure_labels": top_failures,
        "decision": decision,
        "gate_type": gate_type,
        "gate_level": gate_level,
    }
    return out


def _suggest_next_action(top_failures: Dict[str, int]) -> str:
    if not top_failures:
        return "review_low_pass_rate"
    top = next(iter(top_failures))
    mapping = {
        "tail_collapse": "focus_tail_recovery",
        "candidate_starvation": "focus_candidate_density",
        "grid_confusion": "review_axis_mask_and_color",
        "calibration_mismatch": "review_manual_inputs_calibration",
        "wrong_branch_lock_in": "review_dp_window_and_recovery",
    }
    return mapping.get(top, "review_dominant_failure_taxonomy")


def _aggregate(results: List[dict]) -> dict:
    """Aggregate batch statistics."""
    if not results:
        return {
            "total_samples": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "main_metrics_stats": {},
            "debug_metrics_stats": {},
            "diagnosis_metrics_stats": {},
            "failure_taxonomy_counts": {},
        }

    n = len(results)
    n_pass = sum(1 for r in results if r["gate"]["passed"])

    main_keys = list(results[0]["metrics"]["main"].keys())
    main_stats: Dict[str, Dict[str, float]] = {}
    for key in main_keys:
        vals = [float(r["metrics"]["main"][key]) for r in results]
        main_stats[key] = _metric_stats_block(vals)

    dbg_keys = list(results[0]["metrics"]["debug"].keys())
    dbg_stats: Dict[str, Dict[str, float]] = {}
    for key in dbg_keys:
        vals = [float(r["metrics"]["debug"][key]) for r in results]
        dbg_stats[key] = _metric_stats_block(vals)

    diag_keys = list(results[0]["metrics"]["diagnosis"].keys())
    diag_stats: Dict[str, Dict[str, float]] = {}
    for key in diag_keys:
        vals = [float(r["metrics"]["diagnosis"][key]) for r in results]
        diag_stats[key] = _metric_stats_block(vals)

    all_failures: Dict[str, int] = {}
    for r in results:
        for fl in r.get("failure_labels", []):
            all_failures[fl] = all_failures.get(fl, 0) + 1

    return {
        "total_samples": n,
        "pass_count": n_pass,
        "pass_rate": round(n_pass / max(n, 1), 6),
        "main_metrics_stats": main_stats,
        "debug_metrics_stats": dbg_stats,
        "diagnosis_metrics_stats": diag_stats,
        "failure_taxonomy_counts": all_failures,
    }


def print_report(report: dict) -> None:
    """Print human-readable report to stdout."""
    agg = report.get("aggregate", {})
    summ = report.get("summary", {})
    gate_t = summ.get("domain", report.get("gate_type", "clean"))
    gate_lv = summ.get("gate_level", report.get("gate_level", "development"))
    n = agg.get("total_samples", 0)
    n_pass = agg.get("pass_count", 0)
    rate = agg.get("pass_rate", 0)
    verdict = summ.get("verdict", "?")

    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT — domain={gate_t}  gate_level={gate_lv}")
    print(f"  verdict={verdict}")
    print(f"{'='*60}")
    print(f"  Samples: {n}  |  Pass: {n_pass}  |  Rate: {rate*100:.1f}%")
    print(f"{'-'*60}")

    stats = agg.get("main_metrics_stats", {})
    print(f"\n  Main / core metrics (mean ± std, median):")
    for key, st in stats.items():
        print(
            f"    {key:<38} mean={st.get('mean', 0):.4f} ± {st.get('std', 0):.4f}  "
            f"med={st.get('median', 0):.4f}  p10={st.get('p10', 0):.4f}  p90={st.get('p90', 0):.4f}"
        )

    print(f"\n  Per-sample gate results:")
    for s in report.get("samples", []):
        sid = s.get("sample_id", "?")
        passed = s["gate"]["passed"]
        status = "PASS" if passed else "FAIL"
        fl = ", ".join(s.get("failure_labels", [])) or "-"
        ymae = s["metrics"]["main"]["curve_y_mae_px"]
        pr = s["metrics"]["main"]["peak_recall"]
        print(f"    {sid:<20} [{status}]  y_mae={ymae:.2f}  peak_recall={pr:.2f}  failures={fl}")

    print(f"{'='*60}\n")


def save_report(report: dict, output_path: str) -> None:
    """Save report as JSON (aggregate에 레거시 키 병행)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save = dict(report)
    agg = dict(save.get("aggregate") or {})
    if "core_metrics_stats" not in agg and agg.get("main_metrics_stats"):
        agg["core_metrics_stats"] = agg["main_metrics_stats"]
    save["aggregate"] = agg
    with p.open("w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="$17: Evaluate engine outputs")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gate_type", type=str, default="clean",
                        choices=["clean", "styled", "real_like"])
    parser.add_argument(
        "--gate_level",
        type=str,
        default="development",
        choices=["mvp", "development", "strict"],
        help="임계 세트: mvp / development(기본) / strict(출하급)",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--report_json", type=str, default=None)
    parser.add_argument(
        "--baseline-report-json",
        type=str,
        default=None,
        help="직전 리포트 JSON — fail_but_improved 판정에 사용",
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--git-commit", type=str, default=None)
    args = parser.parse_args()

    report = evaluate_batch(
        args.manifest_csv,
        args.output_dir,
        gate_type=args.gate_type,
        max_samples=args.max_samples,
        gate_level=args.gate_level,
        baseline_report_path=args.baseline_report_json,
        run_tag=args.run_tag,
        git_commit=args.git_commit,
    )

    print_report(report)

    if args.report_json:
        save_report(report, args.report_json)
        print(f"Report saved to: {args.report_json}")


if __name__ == "__main__":
    main()
