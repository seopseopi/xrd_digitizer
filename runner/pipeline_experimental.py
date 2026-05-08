"""통합 실험 파이프라인 (v2_experimental + 플래그 전용).

운영·기본 최선 트랙은 v1.1(run_local.run_pipeline, calibrate_v1_1)이다.
실험 전처리·후보·양방향 DP 등 (파일명은 exp / experimental).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.plot_scale import compute_plot_scale
from core.exp_params import default_exp_params
from core.types import ManualInputs, RunResult, PIPELINE_STAGES
from preprocess.roi import crop_roi, apply_legend_ignore
from preprocess.perspective import correct_perspective
from preprocess.exp_preprocess import run_preprocess_experimental
from preprocess.morphology import run_morphology_pipeline
from preprocess.masks import mask_axis_lines
from trace.thinning import skeletonize_mask, build_axis_proximity_map
from trace.components import label_components, compute_component_scores, build_component_score_map
from trace.candidates import (
    build_raw_candidates_experimental,
    filter_candidates_experimental,
    build_final_candidates,
    candidates_to_map,
    compute_candidate_stats,
    compute_margin_histogram,
    MAX_CANDIDATES_FOR_DP_EXPERIMENTAL,
)
from trace.dp_trace_experimental import dp_trace_bidirectional
from trace.dp_trace import render_trace_path
from trace.recovery import render_branch_compare, render_candidates_overlay
from trace.recovery_experimental import run_recovery_experimental
from trace.postprocess import (
    blend_sg_toward_gapfill_on_high_curvature,
    repair_isolated_spike_down_y,
    restore_peaks_lowered_by_smoothing,
    gap_fill,
    smooth_trace_experimental,
    detect_peaks_experimental,
    render_smoothed_trace,
    render_peaks_overlay,
)
from calibrate.axis_mapping import build_x_mapping, build_y_mapping
from calibrate.numeric_export import export_numeric
from calibrate.peak_debug_render import render_numeric_peaks_on_roi


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L")


def _to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def render_dual_pass_overlay(
    roi: np.ndarray,
    path_lr: list | None,
    path_rl: list | None,
    roi_w: int,
    roi_h: int,
) -> Image.Image:
    """§8.3: LR=빨강, RL=파랑 겹침."""
    overlay = roi.copy() if roi.ndim == 3 else np.stack([roi] * 3, axis=-1)
    overlay = overlay.astype(np.uint8)
    for col in range(min(roi_w, len(path_lr) if path_lr else 0)):
        y = path_lr[col] if path_lr and col < len(path_lr) else None
        if y is not None and 0 <= int(y) < roi_h:
            yy = int(y)
            for dy in (-1, 0, 1):
                if 0 <= yy + dy < roi_h:
                    overlay[yy + dy, col] = [255, 80, 80]
    for col in range(min(roi_w, len(path_rl) if path_rl else 0)):
        y = path_rl[col] if path_rl and col < len(path_rl) else None
        if y is not None and 0 <= int(y) < roi_h:
            yy = int(y)
            for dy in (-1, 0, 1):
                if 0 <= yy + dy < roi_h:
                    o = overlay[yy + dy, col]
                    if np.array_equal(o, [255, 80, 80]):
                        overlay[yy + dy, col] = [200, 200, 255]
                    else:
                        overlay[yy + dy, col] = [80, 120, 255]
    return Image.fromarray(overlay)


def run_pipeline_experimental(
    image: Image.Image, mi: ManualInputs, params: dict | None = None
) -> Tuple[RunResult, Dict]:
    w, h = image.size
    stage_timings: Dict[str, float] = {}
    warnings = []

    t0 = time.perf_counter()
    roi, plot_box_t = crop_roi(image, mi.plot_box)
    stage_timings["roi_crop"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    roi = correct_perspective(roi, mi.perspective_corners, plot_box_t)
    stage_timings["perspective"] = round(time.perf_counter() - t0, 6)

    roi_h, roi_w = roi.shape[:2]
    sc = compute_plot_scale((0, 0, roi_w - 1, roi_h - 1))
    s = float(sc["s"])
    s_h = float(sc["s_h"])

    pv2 = default_exp_params()
    if params:
        for k, v in params.items():
            if isinstance(v, dict) and isinstance(pv2.get(k), dict):
                pv2[k].update(v)
            else:
                pv2[k] = v

    t0 = time.perf_counter()
    pv = run_preprocess_experimental(
        roi, mi.color_sample_point, plot_box_t,
        legend_ignore_boxes=mi.legend_ignore_boxes,
        s=s,
        params=pv2.get("preprocess", {}),
    )
    color_score = pv["color_score"]
    ridge_score = pv["ridge_score"]
    grid_pen = pv["grid_penalty_mask"]
    edge_support = pv["edge_support"]
    combined = pv["combined_mask"]
    combined = apply_legend_ignore(combined, mi.legend_ignore_boxes, plot_box_t)
    stage_timings["preprocess_experimental"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    morph = run_morphology_pipeline(combined)
    morph["cleaned_mask"] = mask_axis_lines(morph["cleaned_mask"])
    morph["raw_candidate_mask"] = mask_axis_lines(morph["raw_candidate_mask"])
    stage_timings["morphology"] = round(time.perf_counter() - t0, 6)

    raw_fg = int(morph["raw_candidate_mask"].sum())

    skeleton = skeletonize_mask(morph["cleaned_mask"])
    axis_dist = build_axis_proximity_map(roi.shape, (0, 0, roi_w, roi_h))
    labeled, n_comp = label_components(morph["cleaned_mask"])
    comp_scores = compute_component_scores(labeled, n_comp, axis_dist)
    comp_score_map = build_component_score_map(labeled, comp_scores)

    t0 = time.perf_counter()
    raw_cands = build_raw_candidates_experimental(
        morph["raw_candidate_mask"], skeleton,
        color_score, ridge_score, grid_pen, edge_support,
        comp_score_map, axis_dist,
        params=pv2.get("candidates", {}),
    )
    filtered_cands = filter_candidates_experimental(
        raw_cands, ridge_score, grid_pen, roi_h, params=pv2.get("candidates", {})
    )
    final_cands, missing_cols = build_final_candidates(
        filtered_cands,
        comp_score_map,
        roi_w,
        max_for_dp=MAX_CANDIDATES_FOR_DP_EXPERIMENTAL,
        roi_height=roi_h,
    )
    cand_stats = compute_candidate_stats(raw_cands, filtered_cands, final_cands, missing_cols, roi_w)
    cand_margin_hist = compute_margin_histogram(final_cands)
    stage_timings["candidates"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    trace_result = dp_trace_bidirectional(
        final_cands, roi_w, roi_h, ridge_score, grid_pen, s_h, comp_score_map,
    )
    stage_timings["trace_extract"] = round(time.perf_counter() - t0, 6)

    path_before_recovery = list(trace_result["path"])
    trace_path = trace_result["path"]
    trace_confidences = []
    for col_idx, y_val in enumerate(trace_path):
        if y_val is not None and col_idx in final_cands:
            match = [c for c in final_cands[col_idx] if c["y"] == y_val]
            trace_confidences.append(match[0]["confidence"] if match else 0.5)
        else:
            trace_confidences.append(None)

    t0 = time.perf_counter()
    recovery_result = run_recovery_experimental(
        trace_result, final_cands, roi_w, s_h, params=pv2.get("recovery", {})
    )
    stage_timings["recovery"] = round(time.perf_counter() - t0, 6)

    if recovery_result["recovery_triggered"]:
        trace_path = recovery_result["updated_path"]
        trace_confidences = []
        for col_idx, y_val in enumerate(trace_path):
            if y_val is not None and col_idx in final_cands:
                match = [c for c in final_cands[col_idx] if c["y"] == y_val]
                trace_confidences.append(match[0]["confidence"] if match else 0.5)
            else:
                trace_confidences.append(None)

    columns = sorted(final_cands.keys())
    t0 = time.perf_counter()
    y_filled, valid_mask, gap_ranges = gap_fill(trace_path, columns)
    stage_timings["gap_fill"] = round(time.perf_counter() - t0, 6)

    y_filled = repair_isolated_spike_down_y(y_filled, valid_mask)

    gap_filled_set = set()
    for gs, ge in gap_ranges:
        for gi in range(gs, ge):
            gap_filled_set.add(gi)

    t0 = time.perf_counter()
    y_smoothed, sg_window, smooth_gate = smooth_trace_experimental(
        y_filled, valid_mask, roi_w, s_h, params=pv2.get("postprocess", {})
    )
    y_smoothed = blend_sg_toward_gapfill_on_high_curvature(y_filled, y_smoothed, valid_mask)
    y_smoothed = repair_isolated_spike_down_y(y_smoothed, valid_mask)
    y_smoothed = restore_peaks_lowered_by_smoothing(y_filled, y_smoothed, valid_mask)
    stage_timings["smoothing"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    peak_result = detect_peaks_experimental(
        y_filled, y_smoothed, valid_mask, gap_filled_set, params=pv2.get("postprocess", {})
    )
    stage_timings["peak_detection"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    numeric = export_numeric(columns, y_smoothed, valid_mask, mi, peak_result)
    stage_timings["axis_map"] = round(time.perf_counter() - t0, 6)

    cal_meta = numeric["calibration_meta"]
    rt_err = cal_meta["roundtrip_error"]["total_mean_error_px"]
    cal_conf = cal_meta["roundtrip_error"]["calibration_confidence"]

    for stage in PIPELINE_STAGES:
        stage_timings.setdefault(stage, 0.0)

    result = RunResult(
        two_theta_values=numeric["two_theta_values"],
        intensities=numeric["intensities"],
        x_range=numeric["x_range"],
        y_range=numeric["y_range"],
        quality={"pixel_residual_mean": rt_err, "peak_match_score": None},
        confidence=cal_conf,
        warnings=warnings,
        peaks_numeric_curve=numeric.get("peaks_numeric_curve", []),
        used_manual_inputs={
            "plot_box": mi.plot_box,
            "x_axis_points": mi.x_axis_points,
            "x_axis_values": mi.x_axis_values,
            "y_axis_points": mi.y_axis_points,
            "y_axis_values": mi.y_axis_values,
            "color_sample_point": mi.color_sample_point,
            "click_count": mi.click_count,
            "click_budget_status": mi.click_budget_status,
            **(
                {"export_resample_points": mi.export_resample_points}
                if getattr(mi, "export_resample_points", None) is not None
                else {}
            ),
        },
    )

    # Debug readability: raw 후보 전체를 그리면 거의 ROI 전체가 채워져
    # 07->08 단계 전환이 비정상적으로 보일 수 있어, 열별 상위 후보만 시각화한다.
    raw_map = candidates_to_map(raw_cands, (roi_h, roi_w), top_n=1)
    filt_map = candidates_to_map(filtered_cands, (roi_h, roi_w))
    final_map = candidates_to_map(final_cands, (roi_h, roi_w))

    comp_overlay = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for cid in range(1, n_comp + 1):
        color = rng.integers(60, 220, size=3).tolist()
        comp_overlay[labeled == cid] = color

    trace_overlay = render_trace_path(roi, trace_path, trace_confidences)
    smoothed_trace_img = render_smoothed_trace(roi, columns, y_filled, y_smoothed, valid_mask)
    x_map_dbg = build_x_mapping(mi.x_axis_points, mi.x_axis_values, mi.plot_box)
    y_map_dbg = build_y_mapping(mi.y_axis_points, mi.y_axis_values, mi.plot_box)
    numeric_curve_peaks_roi = render_numeric_peaks_on_roi(
        roi, x_map_dbg, y_map_dbg, numeric.get("peaks_numeric_curve", []),
    )
    peaks_overlay_img = render_peaks_overlay(
        roi, columns, y_smoothed, valid_mask,
        peak_result["peaks"], peak_result["major_peaks"],
    )
    branch_compare_img = render_branch_compare(
        roi, path_before_recovery, trace_path, columns,
    )
    recovery_segment_before = render_trace_path(roi, path_before_recovery, trace_confidences)
    recovery_segment_after = render_trace_path(roi, trace_path, trace_confidences)

    # §8.3 시각화: LR(빨강 계열) vs RL(파랑) — merged path는 최종 trace_overlay와 동일
    dual_overlay = render_dual_pass_overlay(roi, trace_result.get("path_lr"), trace_result.get("path_rl"), roi_w, roi_h)

    raw_only = {
        col: [c for c in cands if "raw" in c.get("source_tags", [])]
        for col, cands in raw_cands.items()
    }
    thin_only = {
        col: [c for c in cands if "thin" in c.get("source_tags", [])]
        for col, cands in raw_cands.items()
    }
    ridge_only = {
        col: [c for c in cands if "ridge" in c.get("source_tags", [])]
        for col, cands in raw_cands.items()
    }
    merged_source_map = candidates_to_map(raw_cands, (roi_h, roi_w))
    raw_source_map = candidates_to_map(raw_only, (roi_h, roi_w))
    thin_source_map = candidates_to_map(thin_only, (roi_h, roi_w))
    ridge_source_map = candidates_to_map(ridge_only, (roi_h, roi_w))

    debug_data: Dict = {
        "roi_preview": Image.fromarray(roi),
        "01_roi_preview": Image.fromarray(roi),
        "02_color_score": Image.fromarray(_to_u8(color_score)),
        "03_ridge_score": Image.fromarray(_to_u8(ridge_score)),
        "04_grid_masks": Image.fromarray(_to_u8(np.maximum(pv["strong_grid_mask"], grid_pen / 0.6))),
        "05_combined_mask": _mask_to_image(combined),
        "color_mask": _mask_to_image((color_score > 0.18).astype(np.uint8)),
        "combined_mask": _mask_to_image(combined),
        "raw_candidate_mask": _mask_to_image(morph["raw_candidate_mask"]),
        "pre_skeleton_candidates": _mask_to_image(morph["skeleton_mask"]),
        "skeleton": _mask_to_image(skeleton),
        "components_overlay": Image.fromarray(comp_overlay),
        "07_candidate_map": Image.fromarray(final_map, mode="L"),
        "candidate_map_raw": Image.fromarray(raw_map, mode="L"),
        "candidate_map_thin": Image.fromarray(thin_source_map, mode="L"),
        "candidate_map_ridge": Image.fromarray(ridge_source_map, mode="L"),
        "candidate_map_merged": Image.fromarray(merged_source_map, mode="L"),
        "candidate_map_raw_source": Image.fromarray(raw_source_map, mode="L"),
        "candidate_map_filtered": Image.fromarray(filt_map, mode="L"),
        "candidate_map_final": Image.fromarray(final_map, mode="L"),
        "08_dual_pass_overlay": dual_overlay,
        "trace_path": Image.fromarray(trace_overlay),
        "09_trace_path": Image.fromarray(trace_overlay),
        "smoothed_trace": Image.fromarray(smoothed_trace_img),
        "numeric_curve_peaks_roi": Image.fromarray(numeric_curve_peaks_roi),
        "peaks_overlay": Image.fromarray(peaks_overlay_img),
        "10_peak_overlay": Image.fromarray(peaks_overlay_img),
        "branch_compare": Image.fromarray(branch_compare_img),
        "recovery_segment_before": Image.fromarray(recovery_segment_before),
        "recovery_segment_after": Image.fromarray(recovery_segment_after),
        "debug.json": {
            "image_size": [w, h],
            "plot_box": mi.plot_box,
            "plot_scale": sc,
            "pipeline_version": "v2_integrated",
            "tuned_params": pv2,
            "raw_candidate_pixels": raw_fg,
            "n_components": n_comp,
            "candidate_stats": cand_stats,
            "candidate_conf_margin_histogram": cand_margin_hist,
            "trace": {
                "trace_score": trace_result["trace_score"],
                "valid_ratio": trace_result["valid_ratio"],
                "window_W": trace_result["window_W"],
                "diagnostics": trace_result["diagnostics"],
                "blockwise": trace_result["blockwise"],
                "unstable_blocks": trace_result.get("unstable_blocks", []),
            },
            "recovery": {
                "recovery_triggered": recovery_result["recovery_triggered"],
                "zones": recovery_result.get("zones", []),
                "accepted_count": recovery_result.get("accepted_count", 0),
                "recovery_beam_compare": recovery_result.get("zones", []),
            },
            "postprocess": {
                "gap_fill": {"n_gaps": len(gap_ranges), "gap_ranges": gap_ranges},
                "smoothing": {
                    "sg_window": sg_window,
                    "sg_polyorder": 2,
                    "gated_ratio": round(float(np.mean(smooth_gate)) if smooth_gate.size else 0.0, 4),
                },
                "peaks": peak_result["params"],
                "peak_list": peak_result["peaks"],
                "major_peaks": peak_result["major_peaks"],
            },
            "calibration": {
                "x_scale": cal_meta["x_scale"],
                "x_offset": cal_meta["x_offset"],
                "y_scale": cal_meta["y_scale"],
                "y_offset": cal_meta["y_offset"],
                "roundtrip_error": cal_meta["roundtrip_error"],
                "num_output_points": cal_meta["num_points"],
                "peak_positions_2theta": numeric.get("peak_positions_2theta", []),
                "peaks_numeric_curve": numeric.get("peaks_numeric_curve", []),
                "peaks_numeric_curve_params": numeric.get("peaks_numeric_curve_params", {}),
            },
            "stage_timings": stage_timings,
            "warnings": warnings,
        },
    }

    return result, debug_data
