"""
$17.4: Acceptance gates — domain(clean/styled/real_like) × level(mvp/development/strict).

$17.8: Failure taxonomy auto-labeling.

strict 레벨은 기존 출하급 임계값에 numeric 지표를 포함한 형태로 유지한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# gate_level → gate_type(domain) → metric → (연산자, 임계값)
GATES: Dict[str, Dict[str, Dict[str, Tuple[str, float]]]] = {
    "mvp": {
        "clean": {
            "curve_y_mae_px": ("<=", 18.0),
            "major_peak_x_error": ("<=", 12.0),
            "peak_recall": (">=", 0.50),
            "max_gap_px": ("<=", 30.0),
            "calibration_roundtrip_error": ("<=", 2.0),
            "numeric_y_mae_norm": ("<=", 0.12),
            "major_peak_x_error_2theta": ("<=", 0.8),
        },
        "styled": {
            "curve_y_mae_px": ("<=", 22.0),
            "major_peak_x_error": ("<=", 16.0),
            "peak_recall": (">=", 0.45),
            "max_gap_px": ("<=", 36.0),
            "calibration_roundtrip_error": ("<=", 2.5),
            "numeric_y_mae_norm": ("<=", 0.15),
            "major_peak_x_error_2theta": ("<=", 1.0),
        },
        "real_like": {
            "curve_y_mae_px": ("<=", 28.0),
            "major_peak_x_error": ("<=", 20.0),
            "peak_recall": (">=", 0.35),
            "max_gap_px": ("<=", 44.0),
            "calibration_roundtrip_error": ("<=", 3.0),
            "numeric_y_mae_norm": ("<=", 0.20),
            "major_peak_x_error_2theta": ("<=", 1.2),
        },
    },
    "development": {
        "clean": {
            "curve_y_mae_px": ("<=", 12.0),
            "major_peak_x_error": ("<=", 8.0),
            "peak_recall": (">=", 0.60),
            "max_gap_px": ("<=", 20.0),
            "calibration_roundtrip_error": ("<=", 1.5),
            "numeric_y_mae_norm": ("<=", 0.08),
            "major_peak_x_error_2theta": ("<=", 0.4),
        },
        "styled": {
            "curve_y_mae_px": ("<=", 16.0),
            "major_peak_x_error": ("<=", 11.0),
            "peak_recall": (">=", 0.55),
            "max_gap_px": ("<=", 26.0),
            "calibration_roundtrip_error": ("<=", 2.0),
            "numeric_y_mae_norm": ("<=", 0.10),
            "major_peak_x_error_2theta": ("<=", 0.6),
        },
        "real_like": {
            "curve_y_mae_px": ("<=", 22.0),
            "major_peak_x_error": ("<=", 16.0),
            "peak_recall": (">=", 0.45),
            "max_gap_px": ("<=", 34.0),
            "calibration_roundtrip_error": ("<=", 2.5),
            "numeric_y_mae_norm": ("<=", 0.14),
            "major_peak_x_error_2theta": ("<=", 0.8),
        },
    },
    "strict": {
        "clean": {
            "curve_y_mae_px": ("<=", 5.0),
            "major_peak_x_error": ("<=", 4.0),
            "peak_recall": (">=", 0.70),
            "max_gap_px": ("<=", 10.0),
            "calibration_roundtrip_error": ("<=", 1.0),
            "numeric_y_mae_norm": ("<=", 0.04),
            "major_peak_x_error_2theta": ("<=", 0.2),
        },
        "styled": {
            "curve_y_mae_px": ("<=", 6.0),
            "major_peak_x_error": ("<=", 6.0),
            "peak_recall": (">=", 0.68),
            "max_gap_px": ("<=", 14.0),
            "calibration_roundtrip_error": ("<=", 1.8),
            "numeric_y_mae_norm": ("<=", 0.06),
            "major_peak_x_error_2theta": ("<=", 0.3),
        },
        "real_like": {
            "curve_y_mae_px": ("<=", 8.0),
            "major_peak_x_error": ("<=", 8.0),
            "peak_recall": (">=", 0.60),
            "max_gap_px": ("<=", 18.0),
            "calibration_roundtrip_error": ("<=", 2.5),
            "numeric_y_mae_norm": ("<=", 0.08),
            "major_peak_x_error_2theta": ("<=", 0.5),
        },
    },
}

RECALL_IS_GTE = {"peak_recall"}


def flatten_gate_thresholds(
    gate_type: str = "clean",
    gate_level: str = "strict",
) -> Dict[str, float]:
    """평탄 임계값 (metric → float). 레거시 스크립트·문서 비교용."""
    level = GATES.get(gate_level, GATES["strict"])
    domain = level.get(gate_type, level["clean"])
    return {m: float(tup[1]) for m, tup in domain.items()}


def check_gate(
    main_metrics: dict,
    gate_type: str = "clean",
    gate_level: str = "development",
) -> dict:
    """
    단일 샘플 게이트. gate_level: mvp | development | strict (기본 development).
    """
    level = GATES.get(gate_level)
    if level is None:
        gate_level = "development"
        level = GATES["development"]
    thresholds = level.get(gate_type, level["clean"])
    details: Dict[str, dict] = {}
    all_passed = True

    for metric, (op, thresh) in thresholds.items():
        value = float(main_metrics.get(metric, 0.0))
        if metric in RECALL_IS_GTE:
            passed = value >= thresh
        else:
            passed = value <= thresh
        details[metric] = {
            "value": value,
            "threshold": thresh,
            "operator": op,
            "passed": passed,
        }
        if not passed:
            all_passed = False

    return {
        "passed": all_passed,
        "gate_type": gate_type,
        "gate_level": gate_level,
        "details": details,
    }


def label_failures(metrics: dict, main_metrics: dict = None) -> List[str]:
    """
    $17.8 + $18.4: Auto-label failure taxonomy from metrics.
    8 fixed labels per spec.
    """
    labels: List[str] = []
    diag = metrics.get("diagnosis", {})
    dbg = metrics.get("debug", {})
    main = main_metrics or metrics.get("main", {})

    if diag.get("empty_column_rate", 0) > 0.05:
        labels.append("candidate_starvation")

    if diag.get("path_margin_instability", 0) > 0.6:
        labels.append("wrong_branch_lock_in")

    if dbg.get("tail_mae_px", 0) > 8.0 or dbg.get("tail_collapse_rate", 0) > 0.3:
        labels.append("tail_collapse")

    if main.get("calibration_roundtrip_error", 0) > 1.0:
        labels.append("calibration_mismatch")

    if dbg.get("peak_f1", 1.0) < 0.5 and main.get("peak_recall", 1.0) >= 0.6:
        labels.append("peak_miss_after_smoothing")

    if dbg.get("IoU", 1.0) < 0.7 and diag.get("empty_column_rate", 0) < 0.03:
        labels.append("text_intrusion")

    if diag.get("candidate_recall_per_column", 1.0) > 0.98 and main.get("curve_y_mae_px", 0) > 10:
        labels.append("grid_confusion")

    if diag.get("recovery_success_rate", 1.0) < 0.5 and diag.get("reentry_count", 0) > 0:
        labels.append("legend_capture")

    return list(dict.fromkeys(labels))


def compute_verdict(
    pass_rate: float,
    gate_level: str,
    baseline_aggregate: Optional[dict] = None,
    current_aggregate: Optional[dict] = None,
) -> str:
    """
    배치 요약 verdict. baseline이 있으면 core 4개 중 3개 이상 개선 시 fail_but_improved.
    """
    gl = (gate_level or "development").lower()
    if gl == "strict" and pass_rate >= 0.60:
        return "strict_pass"
    if gl == "development" and pass_rate >= 0.50:
        return "development_pass"
    if gl == "mvp" and pass_rate >= 0.30:
        return "mvp_pass"
    if baseline_aggregate and current_aggregate and _core_metrics_improved(
        baseline_aggregate, current_aggregate
    ):
        return "fail_but_improved"
    return "fail"


CORE_DEV_METRIC_KEYS = (
    "curve_y_mae_px",
    "major_peak_x_error",
    "peak_recall",
)


def _core_metrics_improved(baseline: dict, current: dict) -> bool:
    """main_metrics_stats 평균 기준: recall은 상승이 개선, 나머지는 하락이 개선."""
    b_main = baseline.get("main_metrics_stats") or baseline.get("core_metrics_stats") or {}
    c_main = current.get("main_metrics_stats") or current.get("core_metrics_stats") or {}
    b_dbg = baseline.get("debug_metrics_stats") or {}
    c_dbg = current.get("debug_metrics_stats") or {}

    improved = 0
    total = 0
    for key in ("curve_y_mae_px", "major_peak_x_error"):
        bm = (b_main.get(key) or {}).get("mean")
        cm = (c_main.get(key) or {}).get("mean")
        if bm is None or cm is None:
            continue
        total += 1
        if float(cm) < float(bm) * 0.85:
            improved += 1

    pr_b = (b_main.get("peak_recall") or {}).get("mean")
    pr_c = (c_main.get("peak_recall") or {}).get("mean")
    if pr_b is not None and pr_c is not None:
        total += 1
        if float(pr_c) > float(pr_b) + 0.03:
            improved += 1

    tm_b = (b_dbg.get("tail_mae_px") or {}).get("mean")
    tm_c = (c_dbg.get("tail_mae_px") or {}).get("mean")
    if tm_b is not None and tm_c is not None:
        total += 1
        if float(tm_c) < float(tm_b) * 0.85:
            improved += 1

    return total > 0 and improved >= 3
