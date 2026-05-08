"""§5.5: Scaled strict + band gates."""

from __future__ import annotations

from typing import Any, Dict


def check_gate_v2_strict(main: dict, v2: dict, gate_type: str = "clean") -> Dict[str, Any]:
    """Strict gate using scaled thresholds from document §5.5."""
    s_h = float(v2.get("s_h", 1.0))
    s_w = float(v2.get("s_w", 1.0))
    s = float(v2.get("s", 1.0))

    if gate_type == "clean":
        th = {
            "strict_curve_y_mae_px": 5.0 * s_h,
            "major_peak_x_error": 4.0 * s_w,
            "major_peak_y_error": 6.0 * s_h,
            "peak_recall_fixed": 0.70,
            "max_gap_px": 10.0 * s_w,
            "calibration_roundtrip_error_px": 1.0 * s,
        }
    elif gate_type == "styled":
        th = {
            "strict_curve_y_mae_px": 5.0 * s_h,
            "major_peak_x_error": 5.0 * s_w,
            "major_peak_y_error": 7.0 * s_h,
            "peak_recall_fixed": 0.72,
            "max_gap_px": 12.0 * s_w,
            "calibration_roundtrip_error_px": 1.5 * s,
        }
    else:
        th = {
            "strict_curve_y_mae_px": 7.0 * s_h,
            "major_peak_x_error": 7.0 * s_w,
            "major_peak_y_error": 9.0 * s_h,
            "peak_recall_fixed": 0.65,
            "max_gap_px": 16.0 * s_w,
            "calibration_roundtrip_error_px": 2.0 * s,
        }

    details = {}
    all_ok = True
    # use v2 strict mae if present
    mae_val = float(v2.get("strict_curve_y_mae_px", main.get("curve_y_mae_px", 0)))
    passed = mae_val <= th["strict_curve_y_mae_px"]
    details["strict_curve_y_mae_px"] = {"value": mae_val, "threshold": th["strict_curve_y_mae_px"], "passed": passed}
    if not passed:
        all_ok = False

    mpx = float(main.get("major_peak_x_error", 0))
    passed = mpx <= th["major_peak_x_error"]
    details["major_peak_x_error"] = {"value": mpx, "threshold": th["major_peak_x_error"], "passed": passed}
    if not passed:
        all_ok = False

    mpy = float(main.get("major_peak_y_error", 0))
    passed = mpy <= th["major_peak_y_error"]
    details["major_peak_y_error"] = {"value": mpy, "threshold": th["major_peak_y_error"], "passed": passed}
    if not passed:
        all_ok = False

    pr = float(v2.get("peak_recall_fixed", main.get("peak_recall", 0)))
    passed = pr >= th["peak_recall_fixed"]
    details["peak_recall_fixed"] = {"value": pr, "threshold": th["peak_recall_fixed"], "passed": passed}
    if not passed:
        all_ok = False

    mg = float(main.get("max_gap_px", 0))
    passed = mg <= th["max_gap_px"]
    details["max_gap_px"] = {"value": mg, "threshold": th["max_gap_px"], "passed": passed}
    if not passed:
        all_ok = False

    cal = float(main.get("calibration_roundtrip_error", 0))
    passed = cal <= th["calibration_roundtrip_error_px"]
    details["calibration_roundtrip_error"] = {"value": cal, "threshold": th["calibration_roundtrip_error_px"], "passed": passed}
    if not passed:
        all_ok = False

    return {"passed": all_ok, "gate_type": gate_type + "_v2_strict", "details": details}


def check_gate_v2_band(main: dict, v2: dict, gate_type: str = "clean") -> Dict[str, Any]:
    s_h = float(v2.get("s_h", 1.0))
    s_w = float(v2.get("s_w", 1.0))
    s = float(v2.get("s", 1.0))

    if gate_type == "clean":
        th_mae = 5.0 * s_h
        th_hit = 0.92
        th_pr = 0.70
        th_mpx = 4.0 * s_w
        th_mpy = 6.0 * s_h
        th_gap = 10.0 * s_w
        th_cal = 1.0 * s
    elif gate_type == "styled":
        th_mae = 5.0 * s_h
        th_hit = 0.90
        th_pr = 0.72
        th_mpx = 5.0 * s_w
        th_mpy = 7.0 * s_h
        th_gap = 12.0 * s_w
        th_cal = 1.5 * s
    else:
        th_mae = 7.0 * s_h
        th_hit = 0.87
        th_pr = 0.65
        th_mpx = 7.0 * s_w
        th_mpy = 9.0 * s_h
        th_gap = 16.0 * s_w
        th_cal = 2.0 * s

    band_mae = float(v2.get("curve_band_mae_px", 999))
    hit = float(v2.get("band_hit_rate", 0))

    details = {}
    ok = True
    passed = band_mae <= th_mae
    details["curve_band_mae_px"] = {"value": band_mae, "threshold": th_mae, "passed": passed}
    if not passed:
        ok = False
    passed = hit >= th_hit
    details["band_hit_rate"] = {"value": hit, "threshold": th_hit, "passed": passed}
    if not passed:
        ok = False

    pr = float(v2.get("peak_recall_fixed", 0))
    passed = pr >= th_pr
    details["peak_recall_fixed"] = {"value": pr, "threshold": th_pr, "passed": passed}
    if not passed:
        ok = False

    mpx = float(main.get("major_peak_x_error", 0))
    passed = mpx <= th_mpx
    details["major_peak_x_error"] = {"value": mpx, "threshold": th_mpx, "passed": passed}
    if not passed:
        ok = False

    mpy = float(main.get("major_peak_y_error", 0))
    passed = mpy <= th_mpy
    details["major_peak_y_error"] = {"value": mpy, "threshold": th_mpy, "passed": passed}
    if not passed:
        ok = False

    mg = float(main.get("max_gap_px", 0))
    passed = mg <= th_gap
    details["max_gap_px"] = {"value": mg, "threshold": th_gap, "passed": passed}
    if not passed:
        ok = False

    cal = float(main.get("calibration_roundtrip_error", 0))
    passed = cal <= th_cal
    details["calibration_roundtrip_error"] = {"value": cal, "threshold": th_cal, "passed": passed}
    if not passed:
        ok = False

    return {"passed": ok, "gate_type": gate_type + "_v2_band", "details": details}
