#!/usr/bin/env python3
"""
레거시: 두 개의 eval 리포트만 빠르게 비교할 때 사용.

모델 도입 공식 판정은 `ml.model_integration_compare` (B0 / M1 / 선택 B1)를 사용한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from core.model_integration_baseline import (
    BASELINE_B0_RULE_SNAPSHOT_DIR,
    required_final_decision_sentence_ko,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="(Legacy) 두 리포트 delta — 공식 판정은 ml.model_integration_compare 사용",
    )
    ap.add_argument("--b0_report", type=str, default=None, help="공식 B0 쪽 리포트(JSON)")
    ap.add_argument(
        "--rule_report",
        type=str,
        default=None,
        help="(호환) 예전 이름 — 지정 시 --b0_report 와 동일 취급",
    )
    ap.add_argument("--m1_report", type=str, required=True, help="모델 보조 쪽 리포트(JSON)")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    b0_path = args.b0_report or args.rule_report
    if not b0_path:
        raise SystemExit("Either --b0_report or --rule_report is required")

    b0 = _read_json(Path(b0_path))
    m1 = _read_json(Path(args.m1_report))

    def _m(report: Dict[str, Any], key: str) -> float:
        return float(report.get("aggregate", {}).get("main_metrics_stats", {}).get(key, {}).get("mean", 0.0))

    out = {
        "note": "partial_compare_only_use_model_integration_compare_for_verdict",
        "policy_sentence_ko": required_final_decision_sentence_ko,
        "official_B0_snapshot_dir": str(BASELINE_B0_RULE_SNAPSHOT_DIR.resolve()),
        "b0_report": str(Path(b0_path).resolve()),
        "m1_report": str(Path(args.m1_report).resolve()),
        "pass_rate_B0": float(b0.get("aggregate", {}).get("pass_rate", 0.0)),
        "pass_rate_M1": float(m1.get("aggregate", {}).get("pass_rate", 0.0)),
        "curve_y_mae_px_B0": _m(b0, "curve_y_mae_px"),
        "curve_y_mae_px_M1": _m(m1, "curve_y_mae_px"),
        "major_peak_x_error_B0": _m(b0, "major_peak_x_error"),
        "major_peak_x_error_M1": _m(m1, "major_peak_x_error"),
        "max_gap_px_B0": _m(b0, "max_gap_px"),
        "max_gap_px_M1": _m(m1, "max_gap_px"),
        "numeric_y_mae_norm_B0": _m(b0, "numeric_y_mae_norm"),
        "numeric_y_mae_norm_M1": _m(m1, "numeric_y_mae_norm"),
    }
    out["delta_pass_rate"] = out["pass_rate_M1"] - out["pass_rate_B0"]
    out["delta_curve_y_mae_px"] = out["curve_y_mae_px_M1"] - out["curve_y_mae_px_B0"]
    out["delta_major_peak_x_error"] = out["major_peak_x_error_M1"] - out["major_peak_x_error_B0"]

    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(
        "[WARN] 도메인 통합·B1 대조·grid_confusion·paired regression 은 "
        "ml.model_integration_compare 에서 수행하세요.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
