#!/usr/bin/env python3
"""
B0(공식 rule-only 스냅샷 기준 eval) vs M1(모델 보조) vs B1(진단 참조, 선택) 통합 비교 리포트.

입력: eval/report.py 가 저장한 report JSON 경로들을 담은 compare manifest JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.model_integration_baseline import (
    BASELINE_B0_RULE_SNAPSHOT_DIR,
    DIAGNOSTIC_REFERENCE_B1_RUN_DIR_EXAMPLE,
    required_final_decision_sentence_ko,
)

DOMAIN_KEYS = ("clean", "styled", "real_like")


def _read_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pass_rate(rep: Dict[str, Any]) -> float:
    return float(rep.get("summary", {}).get("pass_rate", 0.0))


def _metric_mean(rep: Dict[str, Any], key: str) -> float:
    stats = rep.get("aggregate", {}).get("main_metrics_stats", {})
    block = stats.get(key) or {}
    return float(block.get("mean", 0.0))


def _grid_confusion_count(rep: Dict[str, Any]) -> int:
    fc = rep.get("failure_taxonomy_counts") or {}
    return int(fc.get("grid_confusion", 0))


def _paired_regression(
    b0_rep: Dict[str, Any], m1_rep: Dict[str, Any],
) -> Tuple[int, int, float]:
    """B0에서 게이트 통과했으나 M1에서 실패한 샘플 수."""
    b0_samples = {str(s["sample_id"]): s for s in b0_rep.get("samples", [])}
    m1_samples = {str(s["sample_id"]): s for s in m1_rep.get("samples", [])}
    common = sorted(set(b0_samples.keys()) & set(m1_samples.keys()))
    damaged = 0
    b0_pass_total = 0
    for sid in common:
        bp = bool(b0_samples[sid]["gate"]["passed"])
        mp = bool(m1_samples[sid]["gate"]["passed"])
        if bp:
            b0_pass_total += 1
            if not mp:
                damaged += 1
    rate = damaged / max(b0_pass_total, 1)
    return damaged, b0_pass_total, rate


def _load_triplet(block: Mapping[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    tri = block.get(key) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for d in DOMAIN_KEYS:
        p = tri.get(d)
        if not p:
            raise ValueError(f"missing {key}.{d} report path")
        path = Path(str(p))
        if not path.is_file():
            raise FileNotFoundError(path)
        out[d] = _read_report(path)
    return out


def _macro_mean(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _evaluate(
    b0_dev: Dict[str, Dict[str, Any]],
    b0_strict: Dict[str, Dict[str, Any]],
    m1_dev: Dict[str, Dict[str, Any]],
    m1_strict: Dict[str, Dict[str, Any]],
    b1_strict: Optional[Dict[str, Dict[str, Any]]],
    *,
    max_rule_success_corruption_rate: float = 0.0,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    dev_b0 = [_pass_rate(b0_dev[d]) for d in DOMAIN_KEYS]
    dev_m1 = [_pass_rate(m1_dev[d]) for d in DOMAIN_KEYS]
    strict_b0 = [_pass_rate(b0_strict[d]) for d in DOMAIN_KEYS]
    strict_m1 = [_pass_rate(m1_strict[d]) for d in DOMAIN_KEYS]

    checks["1_strict_macro_increase"] = _macro_mean(strict_m1) > _macro_mean(strict_b0)
    checks["2_development_no_decrease_macro"] = _macro_mean(dev_m1) >= _macro_mean(dev_b0)

    curve_ok = True
    gap_ok = True
    numeric_ok = True
    strict_per_domain_ok = True
    for d in DOMAIN_KEYS:
        if _metric_mean(m1_strict[d], "curve_y_mae_px") > _metric_mean(b0_strict[d], "curve_y_mae_px"):
            curve_ok = False
        if _metric_mean(m1_strict[d], "max_gap_px") > _metric_mean(b0_strict[d], "max_gap_px"):
            gap_ok = False
        if _metric_mean(m1_strict[d], "numeric_y_mae_norm") > _metric_mean(b0_strict[d], "numeric_y_mae_norm"):
            numeric_ok = False
        if _pass_rate(m1_strict[d]) < _pass_rate(b0_strict[d]):
            strict_per_domain_ok = False

    checks["3_curve_y_mae_no_worsening_any_domain"] = curve_ok
    checks["4_clean_major_peak_x_improvement"] = (
        _metric_mean(m1_strict["clean"], "major_peak_x_error")
        < _metric_mean(b0_strict["clean"], "major_peak_x_error")
    )
    gc_b0 = sum(_grid_confusion_count(b0_strict[d]) for d in DOMAIN_KEYS)
    gc_m1 = sum(_grid_confusion_count(m1_strict[d]) for d in DOMAIN_KEYS)
    checks["5_grid_confusion_total_decrease"] = gc_m1 < gc_b0
    checks["6_max_gap_no_worsening_any_domain"] = gap_ok
    checks["7_numeric_y_mae_norm_no_worsening_any_domain"] = numeric_ok
    checks["8_no_single_domain_win_others_lose_strict"] = strict_per_domain_ok

    reg_damaged = reg_total = 0
    reg_rates: Dict[str, float] = {}
    for d in DOMAIN_KEYS:
        dmg, tot, rate = _paired_regression(b0_strict[d], m1_strict[d])
        reg_damaged += dmg
        reg_total += tot
        reg_rates[d] = rate
    reg_rate = reg_damaged / max(reg_total, 1)
    checks["9_rule_success_corruption_rate"] = reg_rate
    checks["9_rule_success_corruption_detail"] = {"damaged": reg_damaged, "b0_pass_total": reg_total, "by_domain": reg_rates}
    checks["9_rule_success_corruption_under_threshold"] = reg_rate <= max_rule_success_corruption_rate

    beats_b0 = all(
        checks[k]
        for k in (
            "1_strict_macro_increase",
            "2_development_no_decrease_macro",
            "3_curve_y_mae_no_worsening_any_domain",
            "4_clean_major_peak_x_improvement",
            "5_grid_confusion_total_decrease",
            "6_max_gap_no_worsening_any_domain",
            "7_numeric_y_mae_norm_no_worsening_any_domain",
            "8_no_single_domain_win_others_lose_strict",
            "9_rule_success_corruption_under_threshold",
        )
    )

    beats_b1_only = False
    if b1_strict:
        gc_b1 = sum(_grid_confusion_count(b1_strict[d]) for d in DOMAIN_KEYS)
        strict_b1 = [_pass_rate(b1_strict[d]) for d in DOMAIN_KEYS]
        beats_b1_only = _macro_mean(strict_m1) >= _macro_mean(strict_b1) and gc_m1 <= gc_b1

    verdict = "reject_model_integration"
    if beats_b0:
        verdict = "proceed_model_integration_candidate"
    elif not beats_b0 and b1_strict and beats_b1_only:
        verdict = "reject_model_integration_beats_B1_only_insufficient_vs_B0"

    return {
        "policy_sentence_ko": required_final_decision_sentence_ko,
        "baseline_B0_rule_snapshot_dir": str(BASELINE_B0_RULE_SNAPSHOT_DIR.resolve()),
        "diagnostic_B1_example_dir": str(DIAGNOSTIC_REFERENCE_B1_RUN_DIR_EXAMPLE.resolve()),
        "macro_strict_pass_rate": {"B0": _macro_mean(strict_b0), "M1": _macro_mean(strict_m1)},
        "macro_development_pass_rate": {"B0": _macro_mean(dev_b0), "M1": _macro_mean(dev_m1)},
        "grid_confusion_total": {"B0": gc_b0, "M1": gc_m1},
        "clean_major_peak_x_error_mean": {
            "B0": _metric_mean(b0_strict["clean"], "major_peak_x_error"),
            "M1": _metric_mean(m1_strict["clean"], "major_peak_x_error"),
        },
        "checks": checks,
        "verdict": verdict,
        "beats_B0": beats_b0,
        "beats_B1_diagnostic_only_flag": beats_b1_only if b1_strict else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Model integration compare B0 vs M1 (+ optional B1)")
    ap.add_argument("--manifest", type=str, required=True, help="JSON with B0, M1, optional B1_diagnostic report paths")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    b0 = manifest["B0"]
    m1 = manifest["M1"]
    b1_raw = manifest.get("B1_diagnostic")
    max_corr = float(manifest.get("max_rule_success_corruption_rate", 0.0))

    b0_dev = _load_triplet(b0, "development")
    b0_strict = _load_triplet(b0, "strict")
    m1_dev = _load_triplet(m1, "development")
    m1_strict = _load_triplet(m1, "strict")
    b1_strict: Optional[Dict[str, Dict[str, Any]]] = None
    if b1_raw:
        b1_strict = _load_triplet(b1_raw, "strict")

    result = _evaluate(
        b0_dev, b0_strict, m1_dev, m1_strict, b1_strict,
        max_rule_success_corruption_rate=max_corr,
    )
    result["manifest_path"] = str(Path(args.manifest).resolve())
    result["inputs"] = manifest

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "beats_B0": result["beats_B0"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
