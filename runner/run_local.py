"""
§9-11: 엔진 단일 이미지 실행.

기본 운영·품질 기준선: v1.1 (--pipeline v1_1, debug.pipeline_version=calibrate_v1_1).
v1.2 (--pipeline v1_2)는 동일 엔진에 calibrate_v1_2 태그만 붙인 고정 스냅샷(배포·재현).
v2_experimental 은 실험 트랙이며 기본 최고 성능으로 쓰지 않는다.

Step 11 전처리: ROI crop -> perspective -> color model -> masks -> morphology -> debug 저장
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline_versions import CALIBRATE_V1_1, CALIBRATE_V1_2
from core.contrast_aux_settings import ContrastAuxSettings, DEFAULT_CONTRAST_AUX_SETTINGS
from core.model_assist_settings import DEFAULT_MODEL_ASSIST_SETTINGS, ModelAssistSettings
from core.oracle_rerank_settings import DEFAULT_ORACLE_RERANK_SETTINGS, OracleRerankSettings
from core.selective_oracle_settings import DEFAULT_SELECTIVE_ORACLE_SETTINGS, SelectiveOracleSettings
from core.sharp_peak_settings import DEFAULT_SHARP_PEAK_SETTINGS, SharpPeakPreserveSettings
from core.types import ManualInputs, RunResult, PIPELINE_STAGES
from core.exp_params import load_exp_params
from core.io import (
    load_image,
    load_manual_inputs,
    validate_manual_inputs,
    save_result_json,
    save_debug_files,
)
from preprocess.roi import crop_roi, apply_legend_ignore
from preprocess.perspective import correct_perspective
from preprocess.color_model import build_color_prototypes, compute_color_distance_map
from preprocess.masks import build_mask_a, build_mask_b, combine_masks, mask_axis_lines
from preprocess.ridge_map import compute_vertical_ridge_response
from preprocess.morphology import run_morphology_pipeline
from preprocess.contrast_aux import (
    apply_contrast_aux_to_raw_candidates,
    build_contrast_aux_map,
    contrast_aux_ab_reference,
    render_trace_on_contrast_aux_map,
)
from trace.thinning import skeletonize_mask, build_axis_proximity_map
from trace.components import label_components, compute_component_scores, build_component_score_map
from trace.candidates import (
    LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT,
    LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT,
    annotate_raw_candidates_confidence_dump,
    bridge_final_candidates_for_dp,
    build_raw_candidates,
    dp_transition_window_width,
    filter_candidates,
    attach_candidate_final_bridge_debug,
    build_final_candidates,
    candidates_to_map,
    compute_candidate_stats,
    skeleton_column_hint_y,
    smooth_hint_y_column,
)
from trace.dp_trace import (
    build_dp_cost_breakdown,
    dp_trace,
    refine_dp_path_column_apex_pull,
    render_trace_path,
)
from trace.peak_apex_refine import refine_major_peaks_roi_profile
from ml.runtime_candidate_rerank import run_dp_with_optional_model_assist
from trace.oracle_rerank import (
    _load_gt_json,
    build_gt_y_roi_per_column,
    run_dp_with_gt_oracle_rerank,
)
from trace.risk_detector import append_risk_features_csv
from trace.selective_oracle_rerank import run_dp_with_gt_selective_oracle_rerank
from trace.recovery import (
    run_recovery, detect_recovery_zones,
    render_branch_compare, render_candidates_overlay,
)
from trace.postprocess import (
    LOOSE_PEAK_PROMINENCE_FACTOR,
    blend_sg_toward_gapfill_on_high_curvature,
    gap_fill,
    smooth_trace,
    detect_peaks,
    repair_isolated_spike_down_y,
    restore_peaks_lowered_by_smoothing,
    render_smoothed_trace,
    render_peaks_overlay,
)
from calibrate.axis_mapping import build_x_mapping, build_y_mapping, convert_trace_upscaled_roi_to_numeric
from calibrate.numeric_export import export_numeric
from calibrate.peak_debug_render import render_numeric_peaks_on_roi
from peaks.sharp_pipeline import run_sharp_peak_preserve
from eval.candidate_gt_proximity import compute_candidate_gt_proximity_diag

from runner.pipeline_experimental import run_pipeline_experimental


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L")


def _scale_px_int(v: int, factor: int, *, min_value: int = 1) -> int:
    return max(min_value, int(round(float(v) * float(factor))))


def _odd_or_min(v: int, *, min_value: int = 3) -> int:
    x = max(int(min_value), int(v))
    if x % 2 == 0:
        x += 1
    return x


def _to_resample(method: str) -> Image.Resampling:
    m = str(method).strip().lower()
    if m == "bicubic":
        return Image.Resampling.BICUBIC
    if m == "nearest":
        return Image.Resampling.NEAREST
    return Image.Resampling.LANCZOS


def _scale_point_in_plot(point_xy: List[int], plot_box_t: tuple[int, int, int, int], factor: int) -> List[int]:
    x0, y0, _, _ = plot_box_t
    px = int(point_xy[0]) - int(x0)
    py = int(point_xy[1]) - int(y0)
    return [int(x0) + int(round(px * factor)), int(y0) + int(round(py * factor))]


def _scale_box_in_plot(box_xyxy: List[int], plot_box_t: tuple[int, int, int, int], factor: int) -> List[int]:
    x0, y0, _, _ = plot_box_t
    bx0, by0, bx1, by1 = [int(v) for v in box_xyxy]
    return [
        int(x0) + int(round((bx0 - int(x0)) * factor)),
        int(y0) + int(round((by0 - int(y0)) * factor)),
        int(x0) + int(round((bx1 - int(x0)) * factor)),
        int(y0) + int(round((by1 - int(y0)) * factor)),
    ]


def _downscale_series_to_original_roi(
    columns: List[int],
    y_values: List[float],
    valid_mask: np.ndarray,
    *,
    original_roi_w: int,
    factor: int,
) -> tuple[List[int], np.ndarray, np.ndarray]:
    if factor <= 1:
        return columns, np.asarray(y_values, dtype=np.float64), np.asarray(valid_mask, dtype=bool)
    if not columns:
        return list(range(original_roi_w)), np.zeros(original_roi_w, dtype=np.float64), np.zeros(original_roi_w, dtype=bool)
    x_src = np.asarray(columns, dtype=np.float64)
    y_src = np.asarray(y_values, dtype=np.float64)
    v_src = np.asarray(valid_mask, dtype=np.float64)
    x_tgt = np.arange(int(original_roi_w), dtype=np.float64) * float(factor)
    y_tgt = np.interp(x_tgt, x_src, y_src)
    v_tgt = np.interp(x_tgt, x_src, v_src) >= 0.5
    return list(range(int(original_roi_w))), y_tgt.astype(np.float64), v_tgt


def _mean_confidence_raw_candidates(raw_candidates: Dict) -> tuple[float, int]:
    s = 0.0
    n = 0
    for _col, cands in raw_candidates.items():
        for c in cands:
            s += float(c["confidence"])
            n += 1
    return (float(s / max(n, 1)), n)


def _preserve_gt_near_final_candidates(
    raw_cands: Dict[int, List[dict]],
    final_cands: Dict[int, List[dict]],
    gt_json_path: str,
    plot_box_t: tuple[int, int, int, int],
    roi_h: int,
    roi_w: int,
    *,
    near_px_primary: float = 5.0,
    near_px_fallback: float = 10.0,
    max_for_dp: int = 8,
) -> Dict[str, int]:
    """진단용: 열별 GT-near raw 후보를 final에 강제 보존."""
    gt = _load_gt_json(gt_json_path)
    gt_by_col, _ = build_gt_y_roi_per_column(gt, plot_box_t, roi_h, roi_w)
    inserted = 0
    replaced = 0
    touched_cols = 0
    for col in range(roi_w):
        gty = gt_by_col.get(col)
        if gty is None:
            continue
        raw_list = raw_cands.get(col, [])
        if not raw_list:
            continue
        near = [
            c for c in raw_list
            if abs(float(c.get("y", 0.0)) - float(gty)) <= float(near_px_primary)
        ]
        if not near:
            near = [
                c for c in raw_list
                if abs(float(c.get("y", 0.0)) - float(gty)) <= float(near_px_fallback)
            ]
        if not near:
            continue
        pick = max(near, key=lambda c: float(c.get("confidence", 0.0)))
        cur = final_cands.get(col, [])
        if any(int(c.get("y", -99999)) == int(pick.get("y", -99998)) for c in cur):
            continue
        touched_cols += 1
        forced = dict(pick)
        forced["source"] = str(forced.get("source", "raw")) + "+debug_gt_near"
        forced["debug_gt_near_preserved"] = True
        cur2 = [dict(c) for c in cur] + [forced]
        gt_near_set = {
            int(c.get("y", 0))
            for c in cur2
            if abs(float(c.get("y", 0.0)) - float(gty)) <= float(near_px_fallback)
        }
        if len(cur2) > int(max_for_dp):
            non_gt = [c for c in cur2 if int(c.get("y", 0)) not in gt_near_set]
            if non_gt:
                worst_non_gt = min(non_gt, key=lambda c: float(c.get("confidence", 0.0)))
                cur2.remove(worst_non_gt)
                replaced += 1
        cur2.sort(key=lambda c: -float(c.get("confidence", 0.0)))
        final_cands[col] = cur2[: int(max_for_dp)]
        inserted += 1
    return {
        "gt_near_preserve_touched_columns": int(touched_cols),
        "gt_near_preserve_inserted": int(inserted),
        "gt_near_preserve_replaced_non_gt": int(replaced),
    }


def run_pipeline(
    image: Image.Image,
    mi: ManualInputs,
    *,
    pipeline_version: str = CALIBRATE_V1_1,
    axis_mask_margin: int = 3,
    use_ridge_candidates: bool = False,
    peak_two_pass: bool = True,
    contrast_aux_settings: ContrastAuxSettings = DEFAULT_CONTRAST_AUX_SETTINGS,
    sample_id: str = "",
    curvature_blend_strength: float = 0.32,
    loose_peak_prominence_factor: Optional[float] = None,
    sharp_peak_settings: SharpPeakPreserveSettings = DEFAULT_SHARP_PEAK_SETTINGS,
    use_dp_candidate_bridge: bool = True,
    use_dp_column_apex_pull: bool = True,
    dump_candidates_json: bool = False,
    debug_dump_raw_confidence_features: bool = False,
    debug_filter_removal_reasons: bool = False,
    debug_filter_keep_gt_near: bool = False,
    candidate_filter_topk_before_final: int = 16,
    candidate_filter_min_conf_keep: float = 0.25,
    candidate_filter_disable_envelope_bonus: bool = False,
    candidate_filter_enable_y_diversity: bool = False,
    candidate_filter_y_diversity_bins: int = 8,
    candidate_filter_enable_source_balance: bool = False,
    candidate_filter_source_balance_raw_quota: int = 2,
    debug_filter_rank_breakdown: bool = False,
    debug_filter_rank_breakdown_max_columns: int = 64,
    candidate_filter_enable_local_evidence_rank: bool = False,
    candidate_filter_local_evidence_weight: float = LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT,
    candidate_filter_local_evidence_tau_px: float = LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT,
    candidate_filter_enable_column_rank_normalization: bool = False,
    candidate_filter_enable_evidence_aware_preserve: bool = False,
    candidate_filter_evidence_preserve_bins: int = 6,
    candidate_filter_evidence_preserve_per_bin: int = 1,
    candidate_filter_evidence_preserve_max_upper_frac: float = 0.5,
    debug_final_selection_reasons: bool = False,
    candidate_final_enable_evidence_aware_preserve: bool = False,
    candidate_final_evidence_preserve_slots: int = 2,
    candidate_final_disable_score_bucket_dedupe: bool = False,
    candidate_final_dedupe_score_decimals: Optional[int] = None,
    candidate_final_enable_continuity_preserve: bool = False,
    candidate_final_continuity_slots: int = 2,
    candidate_final_continuity_window: int = 3,
    candidate_final_continuity_max_jump: int = 8,
    candidate_final_max_dp_bridge_frac: Optional[float] = None,
    debug_preserve_gt_near_final_candidates: bool = False,
    model_assist_settings: ModelAssistSettings = DEFAULT_MODEL_ASSIST_SETTINGS,
    oracle_rerank_settings: OracleRerankSettings = DEFAULT_ORACLE_RERANK_SETTINGS,
    selective_oracle_settings: SelectiveOracleSettings = DEFAULT_SELECTIVE_ORACLE_SETTINGS,
    use_peak_apex_roi_refine: bool = False,
    peak_apex_roi_radius: int = 5,
    debug_dp_cost_breakdown: bool = False,
    dp_confidence_weight_multiplier: float = 1.0,
    dp_transition_penalty_multiplier: float = 1.0,
    dp_curvature_penalty_multiplier: float = 1.0,
    oracle_confidence_sharpening: float = 1.0,
    roi_upscale_factor: int = 1,
    roi_upscale_method: str = "lanczos",
    final_export_mode: str = "eval_grid",
) -> tuple[RunResult, Dict]:
    """Step 11 전처리 + 추적·캘리브레이션 (운영 v1.1 = calibrate_v1_1).

    axis_mask_margin: morphology 이후 `mask_axis_lines` 테두리 제거 폭(px).
    use_ridge_candidates: True면 세로 능선 응답을 후보 신뢰도에 가산 (로드맵 2b, 기본 끔).
    peak_two_pass: False면 피크 검출 단일 prominence 패스만 (로드맵 3a 기본 True).
    """
    w, h = image.size
    stage_timings: Dict[str, float] = {}
    warnings = []

    # --- Step 11: preprocess ---
    t0 = time.perf_counter()
    roi, plot_box_t = crop_roi(image, mi.plot_box)
    stage_timings["roi_crop"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    roi = correct_perspective(roi, mi.perspective_corners, plot_box_t)
    stage_timings["perspective"] = round(time.perf_counter() - t0, 6)

    roi_up_factor = max(1, int(roi_upscale_factor))
    roi_up_method = str(roi_upscale_method).strip().lower()
    proc_plot_box_t = tuple(plot_box_t)
    proc_color_sample_point = list(mi.color_sample_point)
    proc_color_resample_points = list(mi.color_resample_points or [])
    proc_legend_ignore_boxes = list(mi.legend_ignore_boxes or [])
    roi_h_orig, roi_w_orig = int(roi.shape[0]), int(roi.shape[1])
    if roi_up_factor > 1:
        resample = _to_resample(roi_up_method)
        roi = np.asarray(
            Image.fromarray(roi).resize(
                (int(roi_w_orig * roi_up_factor), int(roi_h_orig * roi_up_factor)),
                resample=resample,
            )
        )
        proc_plot_box_t = (
            int(plot_box_t[0]),
            int(plot_box_t[1]),
            int(plot_box_t[0]) + int(roi.shape[1]),
            int(plot_box_t[1]) + int(roi.shape[0]),
        )
        proc_color_sample_point = _scale_point_in_plot(
            list(mi.color_sample_point), tuple(plot_box_t), roi_up_factor
        )
        proc_color_resample_points = [
            _scale_point_in_plot(list(pt), tuple(plot_box_t), roi_up_factor)
            for pt in (mi.color_resample_points or [])
        ]
        proc_legend_ignore_boxes = [
            _scale_box_in_plot(list(box), tuple(plot_box_t), roi_up_factor)
            for box in (mi.legend_ignore_boxes or [])
        ]

    t0 = time.perf_counter()
    prototypes, roi_lab, threshold = build_color_prototypes(
        roi, proc_color_sample_point, proc_plot_box_t,
        color_resample_points=proc_color_resample_points,
    )
    color_dist = compute_color_distance_map(roi_lab, prototypes)
    stage_timings["color_model"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    mask_a = build_mask_a(color_dist, threshold)
    mask_b = build_mask_b(roi)
    combined = combine_masks(mask_a, mask_b)
    combined = apply_legend_ignore(combined, proc_legend_ignore_boxes, proc_plot_box_t)
    stage_timings["masks"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    morph = run_morphology_pipeline(combined)
    amargin = max(1, _scale_px_int(int(axis_mask_margin), roi_up_factor))
    morph["cleaned_mask"] = mask_axis_lines(morph["cleaned_mask"], margin=amargin)
    morph["raw_candidate_mask"] = mask_axis_lines(morph["raw_candidate_mask"], margin=amargin)
    stage_timings["morphology"] = round(time.perf_counter() - t0, 6)

    raw_fg = int(morph["raw_candidate_mask"].sum())
    skel_fg = int(morph["skeleton_mask"].sum())
    if raw_fg == 0:
        warnings.append("preprocess: combined mask has zero foreground pixels")
    if skel_fg == 0 and raw_fg > 0:
        warnings.append("preprocess: thinning removed all candidates (raw still available)")

    # --- Step 12: curve candidates ---
    roi_h, roi_w = roi.shape[:2]

    t0 = time.perf_counter()
    skeleton = skeletonize_mask(morph["cleaned_mask"])
    axis_dist = build_axis_proximity_map(roi.shape, (0, 0, roi_w, roi_h))
    stage_timings["skeletonize"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    labeled, n_comp = label_components(morph["cleaned_mask"])
    comp_scores = compute_component_scores(labeled, n_comp, axis_dist)
    comp_score_map = build_component_score_map(labeled, comp_scores)
    stage_timings["component_score"] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    candidates_total_t0 = t0
    ridge_map = (
        compute_vertical_ridge_response(roi) if use_ridge_candidates else None
    )

    contrast_aux_settings_proc = contrast_aux_settings
    if roi_up_factor > 1:
        contrast_aux_settings_proc = replace(
            contrast_aux_settings,
            contrast_aux_border_suppress_px=_scale_px_int(
                int(contrast_aux_settings.contrast_aux_border_suppress_px), roi_up_factor
            ),
        )

    sharp_peak_settings_proc = sharp_peak_settings
    if roi_up_factor > 1:
        sharp_peak_settings_proc = replace(
            sharp_peak_settings,
            curve_smooth_window=_odd_or_min(
                _scale_px_int(int(sharp_peak_settings.curve_smooth_window), roi_up_factor),
                min_value=3,
            ),
            peak_smooth_window=_odd_or_min(
                _scale_px_int(int(sharp_peak_settings.peak_smooth_window), roi_up_factor),
                min_value=3,
            ),
            peak_preserve_radius=_scale_px_int(int(sharp_peak_settings.peak_preserve_radius), roi_up_factor),
            local_prom_window=_odd_or_min(
                _scale_px_int(int(sharp_peak_settings.local_prom_window), roi_up_factor),
                min_value=5,
            ),
        )

    local_evidence_tau_px_proc = float(candidate_filter_local_evidence_tau_px) * float(roi_up_factor)
    continuity_max_jump_proc = _scale_px_int(int(candidate_final_continuity_max_jump), roi_up_factor)
    peak_apex_roi_radius_proc = _scale_px_int(int(peak_apex_roi_radius), roi_up_factor)

    oracle_rerank_settings_proc = oracle_rerank_settings
    selective_oracle_settings_proc = selective_oracle_settings
    if roi_up_factor > 1:
        oracle_rerank_settings_proc = replace(
            oracle_rerank_settings,
            sigma_px=float(oracle_rerank_settings.sigma_px) * float(roi_up_factor),
        )
        selective_oracle_settings_proc = replace(
            selective_oracle_settings,
            sigma_px=float(selective_oracle_settings.sigma_px) * float(roi_up_factor),
        )

    contrast_aux_map_np: Optional[np.ndarray] = None
    contrast_aux_stats: Dict[str, float] = {}
    if contrast_aux_settings_proc.use_contrast_aux:
        contrast_aux_map_np = build_contrast_aux_map(
            roi,
            (0, 0, roi_w, roi_h),
            proc_legend_ignore_boxes,
            contrast_aux_settings_proc,
            legend_crop_origin=(int(proc_plot_box_t[0]), int(proc_plot_box_t[1])),
        )

    t_candidate_stage = time.perf_counter()
    raw_cands = build_raw_candidates(
        morph["raw_candidate_mask"], skeleton,
        color_dist, comp_score_map, axis_dist,
        ridge_map=ridge_map,
    )
    stage_timings["build_raw_candidates_sec"] = round(time.perf_counter() - t_candidate_stage, 6)
    if contrast_aux_settings_proc.use_contrast_aux and contrast_aux_map_np is not None:
        t_candidate_stage = time.perf_counter()
        contrast_aux_stats = apply_contrast_aux_to_raw_candidates(
            raw_cands, contrast_aux_map_np, contrast_aux_settings_proc,
        )
        stage_timings["apply_contrast_aux_to_raw_candidates_sec"] = round(
            time.perf_counter() - t_candidate_stage, 6
        )

    if debug_dump_raw_confidence_features:
        gt_ann_path = oracle_rerank_settings_proc.gt_json_path or selective_oracle_settings_proc.gt_json_path
        gt_ann: Optional[Dict[int, float]] = None
        if gt_ann_path:
            try:
                gt_ann, _ = build_gt_y_roi_per_column(
                    _load_gt_json(str(gt_ann_path)),
                    tuple(plot_box_t),
                    roi_h,
                    roi_w,
                )
            except Exception as ex:  # pragma: no cover - debug only
                warnings.append(f"debug_dump_raw_confidence_features: gt load {ex}")
        try:
            lab_l = roi_lab[:, :, 0].astype(np.float64) if roi_lab is not None else None
            t_candidate_stage = time.perf_counter()
            annotate_raw_candidates_confidence_dump(
                raw_cands,
                raw_mask=morph["raw_candidate_mask"],
                skeleton_mask=skeleton,
                color_dist_map=color_dist,
                comp_score_map=comp_score_map,
                axis_dist_map=axis_dist,
                labeled=labeled,
                ridge_map=ridge_map,
                roi_lab_l_channel=lab_l,
                gt_y_by_col=gt_ann,
                roi_h=roi_h,
            )
            stage_timings["annotate_raw_candidates_confidence_dump_sec"] = round(
                time.perf_counter() - t_candidate_stage, 6
            )
        except Exception as ex:  # pragma: no cover - debug only
            warnings.append(f"debug_dump_raw_confidence_features: annotate {ex}")

    filter_debug: Dict[str, object] = {}
    gt_for_filter_debug: Optional[Dict[int, float]] = None
    dbg_collect = bool(
        debug_filter_removal_reasons or candidate_filter_enable_evidence_aware_preserve
    )
    if dbg_collect:
        gt_path_dbg = None
        if selective_oracle_settings_proc.enabled and selective_oracle_settings_proc.gt_json_path:
            gt_path_dbg = str(selective_oracle_settings_proc.gt_json_path)
        elif oracle_rerank_settings_proc.enabled and oracle_rerank_settings_proc.gt_json_path:
            gt_path_dbg = str(oracle_rerank_settings_proc.gt_json_path)
        if gt_path_dbg:
            try:
                gt_obj = _load_gt_json(gt_path_dbg)
                gt_for_filter_debug, _ = build_gt_y_roi_per_column(
                    gt_obj, tuple(plot_box_t), roi_h, roi_w,
                )
            except Exception as ex:  # pragma: no cover - debug only
                warnings.append(f"debug_filter_removal_reasons: {ex}")

    gt_for_final_debug: Optional[Dict[int, float]] = None
    if debug_final_selection_reasons:
        if gt_for_filter_debug is not None:
            gt_for_final_debug = gt_for_filter_debug
        else:
            gt_path_fin = None
            if selective_oracle_settings_proc.enabled and selective_oracle_settings_proc.gt_json_path:
                gt_path_fin = str(selective_oracle_settings_proc.gt_json_path)
            elif oracle_rerank_settings_proc.enabled and oracle_rerank_settings_proc.gt_json_path:
                gt_path_fin = str(oracle_rerank_settings_proc.gt_json_path)
            if gt_path_fin:
                try:
                    gt_fin_obj = _load_gt_json(gt_path_fin)
                    gt_for_final_debug, _ = build_gt_y_roi_per_column(
                        gt_fin_obj, tuple(plot_box_t), roi_h, roi_w,
                    )
                except Exception as ex:  # pragma: no cover
                    warnings.append(f"debug_final_selection_reasons: gt load {ex}")

    t_candidate_stage = time.perf_counter()
    filtered_cands = filter_candidates(
        raw_cands,
        min_conf_keep=float(candidate_filter_min_conf_keep),
        max_candidates_after_filter=int(candidate_filter_topk_before_final),
        disable_envelope_bonus=bool(candidate_filter_disable_envelope_bonus),
        debug_gt_y_by_col=gt_for_filter_debug,
        debug_gt_near_px=5.0,
        debug_sink=filter_debug if dbg_collect else None,
        debug_keep_gt_near=bool(debug_filter_keep_gt_near),
        enable_y_diversity=bool(candidate_filter_enable_y_diversity),
        y_diversity_bins=int(candidate_filter_y_diversity_bins),
        enable_source_balance=bool(candidate_filter_enable_source_balance),
        source_balance_raw_quota=int(candidate_filter_source_balance_raw_quota),
        debug_rank_breakdown=bool(debug_filter_rank_breakdown),
        debug_rank_breakdown_max_columns=int(debug_filter_rank_breakdown_max_columns),
        enable_local_evidence_rank=bool(candidate_filter_enable_local_evidence_rank),
        local_evidence_weight=float(candidate_filter_local_evidence_weight),
        local_evidence_tau_px=float(local_evidence_tau_px_proc),
        enable_column_rank_normalization=bool(candidate_filter_enable_column_rank_normalization),
        enable_evidence_aware_preserve=bool(candidate_filter_enable_evidence_aware_preserve),
        evidence_preserve_bins=int(candidate_filter_evidence_preserve_bins),
        evidence_preserve_per_bin=int(candidate_filter_evidence_preserve_per_bin),
        evidence_preserve_max_upper_frac=float(
            candidate_filter_evidence_preserve_max_upper_frac
        ),
        filter_roi_h=int(roi_h),
    )
    stage_timings["filter_candidates_sec"] = round(time.perf_counter() - t_candidate_stage, 6)
    sk_y_hint = smooth_hint_y_column(skeleton_column_hint_y(skeleton), half=5)
    candidate_final_debug_accum: Dict[str, Any] = {}
    t_candidate_stage = time.perf_counter()
    final_pre, missing_cols = build_final_candidates(
        filtered_cands,
        comp_score_map,
        roi_w,
        skeleton_hint_y=sk_y_hint,
        roi_height=roi_h,
        debug_final_sink=candidate_final_debug_accum if debug_final_selection_reasons else None,
        debug_gt_y_by_col=gt_for_final_debug if debug_final_selection_reasons else None,
        debug_gt_near_px=5.0,
        debug_final_selection_reasons=bool(debug_final_selection_reasons),
        candidate_final_enable_evidence_aware_preserve=bool(
            candidate_final_enable_evidence_aware_preserve
        ),
        candidate_final_evidence_preserve_slots=int(candidate_final_evidence_preserve_slots),
        candidate_final_disable_score_bucket_dedupe=bool(
            candidate_final_disable_score_bucket_dedupe
        ),
        candidate_final_dedupe_score_decimals=candidate_final_dedupe_score_decimals,
        candidate_final_enable_continuity_preserve=bool(
            candidate_final_enable_continuity_preserve
        ),
        candidate_final_continuity_slots=int(candidate_final_continuity_slots),
        candidate_final_continuity_window=int(candidate_final_continuity_window),
        candidate_final_continuity_max_jump=int(continuity_max_jump_proc),
    )
    stage_timings["build_final_candidates_sec"] = round(time.perf_counter() - t_candidate_stage, 6)
    final_cands = final_pre
    stage_timings["bridge_final_candidates_for_dp_sec"] = 0.0
    if use_dp_candidate_bridge:
        t_candidate_stage = time.perf_counter()
        final_cands = bridge_final_candidates_for_dp(
            final_pre,
            roi_w,
            roi_h,
            color_dist,
            comp_score_map,
            axis_dist,
            skeleton_hint_y=sk_y_hint,
            max_dp_bridge_frac=candidate_final_max_dp_bridge_frac,
        )
        stage_timings["bridge_final_candidates_for_dp_sec"] = round(
            time.perf_counter() - t_candidate_stage, 6
        )
    if (
        debug_final_selection_reasons
        and gt_for_final_debug is not None
        and "candidate_final_debug" in candidate_final_debug_accum
    ):
        attach_candidate_final_bridge_debug(
            candidate_final_debug_accum["candidate_final_debug"],
            final_pre_bridge=final_pre,
            final_post_bridge=final_cands,
            filtered=filtered_cands,
            debug_gt_y_by_col=gt_for_final_debug,
            roi_h=roi_h,
            w=roi_w,
            gt_near_px=5.0,
            debug_columns=bool(debug_final_selection_reasons),
        )
    preserve_meta: Dict[str, int] = {}
    if debug_preserve_gt_near_final_candidates:
        gt_for_preserve = None
        if selective_oracle_settings_proc.enabled and selective_oracle_settings_proc.gt_json_path:
            gt_for_preserve = str(selective_oracle_settings_proc.gt_json_path)
        elif oracle_rerank_settings_proc.enabled and oracle_rerank_settings_proc.gt_json_path:
            gt_for_preserve = str(oracle_rerank_settings_proc.gt_json_path)
        if gt_for_preserve:
            try:
                preserve_meta = _preserve_gt_near_final_candidates(
                    raw_cands,
                    final_cands,
                    gt_for_preserve,
                    tuple(plot_box_t),
                    roi_h,
                    roi_w,
                    max_for_dp=8,
                )
            except Exception as ex:  # pragma: no cover - debug only
                warnings.append(f"debug_preserve_gt_near_final_candidates: {ex}")
        else:
            warnings.append("debug_preserve_gt_near_final_candidates: gt path missing; skipped")
    cand_stats = compute_candidate_stats(raw_cands, filtered_cands, final_cands, missing_cols, roi_w)
    if use_dp_candidate_bridge:
        cand_stats["dp_candidate_bridge"] = True
        cand_stats["dp_candidate_bridge_window_W"] = int(dp_transition_window_width(roi_w))
    else:
        cand_stats["dp_candidate_bridge"] = False
    if preserve_meta:
        cand_stats.update(preserve_meta)
    stage_timings["candidates_total_sec"] = round(time.perf_counter() - candidates_total_t0, 6)
    stage_timings["candidates"] = stage_timings["candidates_total_sec"]

    # --- Step 13: DP tracing (+ 선택: GT oracle 재랭크 또는 CNN 재랭크 + fallback) ---
    t0 = time.perf_counter()
    selective_risk_rows: Optional[List] = None
    selective_segments: Optional[List] = None
    selective_active = bool(selective_oracle_settings_proc.enabled and selective_oracle_settings_proc.gt_json_path)
    oracle_active = bool(oracle_rerank_settings_proc.enabled and oracle_rerank_settings_proc.gt_json_path) and (
        not selective_active
    )
    ma_active = bool(model_assist_settings.enabled and model_assist_settings.model_ckpt_path)
    if selective_active and oracle_rerank_settings_proc.enabled and oracle_rerank_settings_proc.gt_json_path:
        warnings.append(
            "selective_oracle_rerank: selective GT oracle이 켜져 있어 이번 실행에서는 global --oracle-rerank-gt는 무시됩니다.",
        )
    if (oracle_active or selective_active) and ma_active:
        warnings.append(
            "oracle_rerank: GT oracle 실험이 켜져 있어 이번 실행에서는 runtime model_assist(CNN)를 적용하지 않습니다.",
        )
        ma_active = False

    if selective_active:
        st_sel = selective_oracle_settings_proc
        final_cands, trace_result, sel_meta, selective_risk_rows, selective_segments, risk_summary = (
            run_dp_with_gt_selective_oracle_rerank(
                final_cands,
                roi_w,
                roi_h,
                comp_score_map,
                axis_dist,
                ridge_map,
                str(st_sel.gt_json_path),
                plot_box_t,
                st_sel,
                use_dp_column_apex_pull=use_dp_column_apex_pull,
            )
        )
        style_pol = (
            "clean_only_default"
            if st_sel.styled_real_default_off and not st_sel.allow_styled_real_selective
            else "allow_styled_real_optional"
        )
        model_assist_meta = {
            "mode": "selective_oracle_rerank",
            "enabled": True,
            "style_policy": style_pol,
            "requested": bool(model_assist_settings.enabled),
            "lambda_model": float(model_assist_settings.lambda_model),
            "patch_size": int(model_assist_settings.patch_size),
            "device": str(model_assist_settings.device),
            "checkpoint": model_assist_settings.model_ckpt_path,
            "dp_branch": "selective_oracle_rerank",
            "fallback_reason": None,
            "num_candidates_scored": 0,
            "trace_score_rule": float(sel_meta["trace_score_rule_dp"]),
            "trace_score_model": float(sel_meta["trace_score_selective_oracle_dp"]),
            "valid_ratio_rule": float(sel_meta["valid_ratio_rule_dp"]),
            "valid_ratio_model": float(sel_meta["valid_ratio_selective_oracle_dp"]),
            "fallback_valid_ratio_margin": float(model_assist_settings.fallback_valid_ratio_margin),
            "fallback_trace_score_margin": float(model_assist_settings.fallback_trace_score_margin),
            "risk_detector": {
                "version": "risk_detector_rule_v1",
                "risk_column_count": int(risk_summary.get("risk_column_count_dilated", 0)),
                "total_column_count": int(risk_summary.get("total_column_count", roi_w)),
                "risk_ratio": float(risk_summary.get("risk_ratio", 0.0)),
                "risk_segments": int(risk_summary.get("risk_segments", 0)),
                "settings": {
                    "conf_margin_thr": float(st_sel.conf_margin_thr),
                    "entropy_high_thr": float(st_sel.entropy_high_thr),
                    "risk_dilate_radius_columns": int(st_sel.risk_dilate_radius_columns),
                },
            },
            "selective_oracle": {
                "sigma": float(st_sel.sigma_px),
                "applied_candidate_count": int(
                    float(sel_meta.get("selective_oracle_score_summary", {}).get("applied_candidate_count", 0))
                ),
                "preserved_rule_candidate_count": int(
                    float(sel_meta.get("selective_oracle_score_summary", {}).get("preserved_rule_candidate_count", 0))
                ),
                "trace_score_rule_dp": float(sel_meta["trace_score_rule_dp"]),
                "trace_score_global_oracle_dp": float(sel_meta["trace_score_global_oracle_dp"]),
                "trace_score_selective_oracle_dp": float(sel_meta["trace_score_selective_oracle_dp"]),
            },
            "oracle_rerank": sel_meta,
        }
        if st_sel.risk_features_csv_path and selective_risk_rows:
            try:
                append_risk_features_csv(
                    st_sel.risk_features_csv_path,
                    sample_id or "",
                    str(st_sel.run_domain),
                    selective_risk_rows,
                )
            except Exception as ex:
                warnings.append(f"risk_features_csv append failed: {ex}")
    elif oracle_active:
        final_cands, trace_result, oracle_meta = run_dp_with_gt_oracle_rerank(
            final_cands,
            roi_w,
            roi_h,
            comp_score_map,
            str(oracle_rerank_settings_proc.gt_json_path),
            plot_box_t,
            float(oracle_rerank_settings_proc.sigma_px),
            use_dp_column_apex_pull=use_dp_column_apex_pull,
            confidence_weight_multiplier=float(dp_confidence_weight_multiplier),
            transition_penalty_multiplier=float(dp_transition_penalty_multiplier),
            curvature_penalty_multiplier=float(dp_curvature_penalty_multiplier),
            oracle_confidence_sharpening=float(oracle_confidence_sharpening),
        )
        model_assist_meta = {
            "mode": "oracle_rerank",
            "enabled": False,
            "requested": bool(model_assist_settings.enabled),
            "lambda_model": float(model_assist_settings.lambda_model),
            "patch_size": int(model_assist_settings.patch_size),
            "device": str(model_assist_settings.device),
            "checkpoint": model_assist_settings.model_ckpt_path,
            "dp_branch": "oracle_rerank",
            "fallback_reason": None,
            "num_candidates_scored": 0,
            "trace_score_rule": float(oracle_meta["trace_score_rule_dp"]),
            "trace_score_model": float(oracle_meta["trace_score_oracle_dp"]),
            "valid_ratio_rule": float(oracle_meta["valid_ratio_rule_dp"]),
            "valid_ratio_model": float(oracle_meta["valid_ratio_oracle_dp"]),
            "fallback_valid_ratio_margin": float(model_assist_settings.fallback_valid_ratio_margin),
            "fallback_trace_score_margin": float(model_assist_settings.fallback_trace_score_margin),
            "oracle_rerank": oracle_meta,
        }
    elif ma_active:
        final_cands, trace_result, model_assist_meta = run_dp_with_optional_model_assist(
            final_cands,
            roi,
            roi_w,
            roi_h,
            comp_score_map,
            model_assist_settings,
            use_dp_column_apex_pull=use_dp_column_apex_pull,
        )
        model_assist_meta["mode"] = "cnn_model_assist"
    else:
        trace_result = dp_trace(
            final_cands,
            roi_w,
            roi_h,
            comp_score_map,
            confidence_weight_multiplier=float(dp_confidence_weight_multiplier),
            transition_penalty_multiplier=float(dp_transition_penalty_multiplier),
            curvature_penalty_multiplier=float(dp_curvature_penalty_multiplier),
        )
        if use_dp_column_apex_pull:
            max_pull = max(120, 4 * int(dp_transition_window_width(roi_w)))
            trace_result["path"] = refine_dp_path_column_apex_pull(
                trace_result["path"],
                final_cands,
                conf_slack=0.22,
                max_upward_pull_px=max_pull,
            )
        vr_path = float(
            sum(1 for p in trace_result["path"] if p is not None) / max(len(trace_result["path"]), 1)
        )
        if model_assist_settings.enabled and not model_assist_settings.model_ckpt_path:
            fb_reason = "missing_checkpoint_path"
        else:
            fb_reason = "feature_disabled"
        model_assist_meta = {
            "mode": "rule_only",
            "enabled": False,
            "requested": bool(model_assist_settings.enabled),
            "lambda_model": float(model_assist_settings.lambda_model),
            "patch_size": int(model_assist_settings.patch_size),
            "device": str(model_assist_settings.device),
            "checkpoint": model_assist_settings.model_ckpt_path,
            "dp_branch": "rule_only",
            "fallback_reason": fb_reason,
            "num_candidates_scored": 0,
            "trace_score_rule": float(trace_result["trace_score"]),
            "trace_score_model": None,
            "valid_ratio_rule": vr_path,
            "valid_ratio_model": None,
            "fallback_valid_ratio_margin": float(model_assist_settings.fallback_valid_ratio_margin),
            "fallback_trace_score_margin": float(model_assist_settings.fallback_trace_score_margin),
        }
    stage_timings["trace_extract"] = round(time.perf_counter() - t0, 6)
    model_assist_meta["requested"] = bool(model_assist_settings.enabled)
    model_assist_meta["oracle_rerank_requested"] = bool(
        oracle_rerank_settings.enabled and oracle_rerank_settings.gt_json_path
    )
    model_assist_meta["selective_oracle_rerank_requested"] = bool(
        selective_oracle_settings.enabled and selective_oracle_settings.gt_json_path
    )

    trace_path = trace_result["path"]
    trace_confidences = []
    for col_idx, y_val in enumerate(trace_path):
        if y_val is not None and col_idx in final_cands:
            match = [c for c in final_cands[col_idx] if c["y"] == y_val]
            trace_confidences.append(match[0]["confidence"] if match else 0.5)
        else:
            trace_confidences.append(None)

    if trace_result["valid_ratio"] < 0.5:
        warnings.append(f"trace: low valid_ratio {trace_result['valid_ratio']:.3f}")

    # --- Step 14: recovery / re-entry ---
    t0 = time.perf_counter()
    recovery_result = run_recovery(
        trace_result, final_cands,
        morph["raw_candidate_mask"], skeleton,
        color_dist, comp_score_map, axis_dist,
        threshold, roi_w, roi_h,
        contrast_aux_map=contrast_aux_map_np,
        contrast_aux_settings=contrast_aux_settings,
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
        n_zones = len(recovery_result["zones"])
        n_resolved = sum(1 for z in recovery_result["zones"] if z.get("resolved"))
        warnings.append(f"recovery: {n_zones} zones detected, {n_resolved} resolved")
        for fl in recovery_result.get("failure_labels", []):
            warnings.append(f"recovery: failure [{fl['label']}] at cols {fl['zone']}")

    if contrast_aux_settings_proc.use_contrast_aux:
        cand_report = {
            "sample_id": sample_id,
            "use_contrast_aux": True,
            "contrast_aux_weight": float(contrast_aux_settings_proc.contrast_aux_weight),
            "contrast_aux_min_base_conf": float(contrast_aux_settings_proc.contrast_aux_min_base_conf),
            **contrast_aux_stats,
        }
    else:
        mb, nc = _mean_confidence_raw_candidates(raw_cands)
        cand_report = {
            "sample_id": sample_id,
            "use_contrast_aux": False,
            "contrast_aux_weight": float(contrast_aux_settings_proc.contrast_aux_weight),
            "contrast_aux_min_base_conf": float(contrast_aux_settings_proc.contrast_aux_min_base_conf),
            "mean_base_conf": round(mb, 6),
            "mean_final_conf": round(mb, 6),
            "mean_contrast_bonus": 0.0,
            "num_candidates": float(nc),
            "num_candidates_bonus_applied": 0.0,
            "num_candidates_base_conf_too_low": 0.0,
        }

    # --- Step 15: gap fill / smoothing / peak detection ---
    columns = sorted(final_cands.keys())
    t0 = time.perf_counter()
    y_filled, valid_mask, gap_ranges = gap_fill(trace_path, columns)
    stage_timings["gap_fill"] = round(time.perf_counter() - t0, 6)

    y_filled = repair_isolated_spike_down_y(y_filled, valid_mask)

    gap_filled_set = set()
    for gs, ge in gap_ranges:
        for gi in range(gs, ge):
            gap_filled_set.add(gi)

    loose_pf = (
        float(loose_peak_prominence_factor)
        if loose_peak_prominence_factor is not None
        else float(LOOSE_PEAK_PROMINENCE_FACTOR)
    )

    sharp_debug_pack: Dict = {}

    t0 = time.perf_counter()
    if sharp_peak_settings_proc.use_sharp_peak_preserve:
        y_smoothed, peak_result, sharp_debug_pack = run_sharp_peak_preserve(
            roi,
            y_filled,
            valid_mask,
            gap_filled_set,
            columns,
            roi_w,
            roi_h,
            sharp_peak_settings_proc,
        )
        y_smoothed = repair_isolated_spike_down_y(y_smoothed, valid_mask)
        sg_window = int(sharp_peak_settings_proc.curve_smooth_window)
        stage_timings["smoothing"] = round(time.perf_counter() - t0, 6)
        stage_timings["peak_detection"] = 0.0
    else:
        y_smoothed, sg_window = smooth_trace(y_filled, valid_mask, roi_w)
        y_smoothed = blend_sg_toward_gapfill_on_high_curvature(
            y_filled, y_smoothed, valid_mask, strength=float(curvature_blend_strength),
        )
        y_smoothed = repair_isolated_spike_down_y(y_smoothed, valid_mask)
        y_smoothed = restore_peaks_lowered_by_smoothing(y_filled, y_smoothed, valid_mask)
        stage_timings["smoothing"] = round(time.perf_counter() - t0, 6)

        t0 = time.perf_counter()
        peak_result = detect_peaks(
            y_smoothed,
            valid_mask,
            gap_filled_set,
            columns=columns,
            two_pass_prominence=peak_two_pass,
            loose_prominence_factor=loose_pf,
        )
        stage_timings["peak_detection"] = round(time.perf_counter() - t0, 6)

    if use_peak_apex_roi_refine:
        peak_result = refine_major_peaks_roi_profile(
            peak_result,
            roi,
            columns,
            search_radius_px=int(peak_apex_roi_radius_proc),
            enabled=True,
        )
    model_assist_meta["peak_apex_roi_refine_applied"] = bool(use_peak_apex_roi_refine)
    model_assist_meta["peak_apex_search_radius_px"] = (
        int(peak_apex_roi_radius_proc) if use_peak_apex_roi_refine else None
    )

    n_peaks = len(peak_result["peaks"])
    n_major = len(peak_result["major_peaks"])

    trace_columns_internal = list(columns)
    y_smoothed_internal = np.asarray(y_smoothed, dtype=np.float64).copy()
    valid_internal = np.asarray(valid_mask, dtype=bool).copy()
    raw_trace_valid_point_count_internal = int(np.sum(valid_internal))

    columns_numeric = columns
    y_smoothed_numeric = y_smoothed
    valid_mask_numeric = valid_mask
    if roi_up_factor > 1:
        columns_numeric, y_smoothed_numeric, valid_mask_numeric = _downscale_series_to_original_roi(
            columns,
            y_smoothed,
            valid_mask,
            original_roi_w=roi_w_orig,
            factor=roi_up_factor,
        )
        gap_numeric = {i for i, ok in enumerate(valid_mask_numeric) if not bool(ok)}
        peak_result = detect_peaks(
            y_smoothed_numeric,
            valid_mask_numeric,
            gap_numeric,
            columns=columns_numeric,
            two_pass_prominence=peak_two_pass,
            loose_prominence_factor=loose_pf,
        )
        n_peaks = len(peak_result["peaks"])
        n_major = len(peak_result["major_peaks"])
    if n_peaks == 0:
        warnings.append("postprocess: no peaks detected")

    # --- Step 16: pixel -> numeric conversion ---
    t0 = time.perf_counter()
    numeric = export_numeric(columns_numeric, y_smoothed_numeric, valid_mask_numeric, mi, peak_result)
    stage_timings["axis_map"] = round(time.perf_counter() - t0, 6)

    cal_meta = numeric["calibration_meta"]
    rt_err = cal_meta["roundtrip_error"]["total_mean_error_px"]
    cal_conf = cal_meta["roundtrip_error"]["calibration_confidence"]
    if rt_err > 1.0:
        warnings.append(f"calibration: roundtrip error {rt_err:.4f}px exceeds 1.0")

    x_map_dbg = build_x_mapping(mi.x_axis_points, mi.x_axis_values, mi.plot_box)
    y_map_dbg = build_y_mapping(mi.y_axis_points, mi.y_axis_values, mi.plot_box)
    tt_eval = numeric["two_theta_values"]
    ii_eval = numeric["intensities"]
    export_rs = getattr(mi, "export_resample_points", None)
    export_mode = str(final_export_mode).strip().lower()
    if export_mode not in {"eval_grid", "highres"}:
        raise ValueError(f"unsupported final_export_mode: {final_export_mode}")

    audit_resolution = {
        "input_image_width": int(w),
        "input_image_height": int(h),
        "plot_box_original": list(mi.plot_box),
        "roi_width_original": int(roi_w_orig),
        "roi_height_original": int(roi_h_orig),
        "upscale_factor": int(roi_up_factor),
        "roi_width_after_upscale": int(roi_w),
        "roi_height_after_upscale": int(roi_h),
        "candidate_stats_total_columns": cand_stats.get("total_columns"),
        "final_candidates_distinct_columns": int(len(final_cands)),
        "trace_path_columns_internal": int(len(trace_columns_internal)),
        "raw_trace_valid_point_count_internal": int(raw_trace_valid_point_count_internal),
        "numeric_export_column_count_eval": int(len(columns_numeric)),
        "calibration_num_output_points_eval": int(cal_meta.get("num_points", len(tt_eval))),
        "final_export_point_count_eval": int(len(tt_eval)),
        "eval_export_resample_points_requested": export_rs,
        "numeric_pipeline_downscaled_to_original_roi_for_eval": bool(roi_up_factor > 1),
        "numeric_export_downscale_method": (
            "linear_interp_column_indices_and_y_onto_original_roi_width_grid"
            if roi_up_factor > 1
            else None
        ),
        "output_x_min_eval": float(min(tt_eval)) if tt_eval else None,
        "output_x_max_eval": float(max(tt_eval)) if tt_eval else None,
        "output_y_min_eval": float(min(ii_eval)) if ii_eval else None,
        "output_y_max_eval": float(max(ii_eval)) if ii_eval else None,
    }

    resolution_diag_payload: Dict[str, Any] = {
        "audit": audit_resolution,
        "export_points_eval_note": "same_as_root_fields_two_theta_values_and_intensities",
    }
    export_points_eval = {
        "two_theta_values": tt_eval,
        "intensities": ii_eval,
        "point_count": int(len(tt_eval)),
        "source": "original_roi_eval_grid",
        "upscale_factor": float(roi_up_factor),
        "downscaled_from_highres": bool(roi_up_factor > 1),
        "valid_point_count": int(np.sum(valid_mask_numeric)),
        "gap_count": int(len(valid_mask_numeric) - np.sum(valid_mask_numeric)),
    }
    if roi_up_factor > 1:
        tt_hi, ii_hi = convert_trace_upscaled_roi_to_numeric(
            trace_columns_internal,
            y_smoothed_internal,
            valid_internal,
            x_map_dbg,
            y_map_dbg,
            int(roi_up_factor),
        )
        audit_resolution["highres_export_point_count_diag"] = int(len(tt_hi))
        audit_resolution["output_x_min_highres_diag"] = float(min(tt_hi)) if tt_hi else None
        audit_resolution["output_x_max_highres_diag"] = float(max(tt_hi)) if tt_hi else None
        audit_resolution["output_y_min_highres_diag"] = float(min(ii_hi)) if ii_hi else None
        audit_resolution["output_y_max_highres_diag"] = float(max(ii_hi)) if ii_hi else None
        resolution_diag_payload["raw_trace_points_highres"] = {
            "schema": "roi_upscaled_pixel_trace_columns_v1",
            "upscale_factor": int(roi_up_factor),
            "columns_roi_upscaled": trace_columns_internal,
            "y_px_roi_upscaled": [float(v) for v in y_smoothed_internal],
            "valid": [bool(v) for v in valid_internal],
        }
        resolution_diag_payload["export_points_highres"] = {
            "two_theta_values": tt_hi,
            "intensities": ii_hi,
            "mapping_note": (
                "roi_upscaled_px_scaled_by_inv_factor_then_axis_map;"
                "no_extra_interpolation_beyond_existing_pipeline_gap_fill"
            ),
        }
    else:
        tt_hi, ii_hi = tt_eval, ii_eval
        audit_resolution["highres_export_point_count_diag"] = int(len(tt_eval))
        audit_resolution["note_highres_diag_merged_with_eval"] = True

    valid_internal_list = [bool(v) for v in valid_internal]
    export_points_highres = {
        "two_theta_values": tt_hi,
        "intensities": ii_hi,
        "point_count": int(len(tt_hi)),
        "source": "roi_upscale_trace" if roi_up_factor > 1 else "original_roi_eval_grid",
        "upscale_factor": float(roi_up_factor),
        "downscaled_to_eval_grid": False,
        "trace_column_count": int(len(trace_columns_internal)),
        "valid_point_count": int(np.sum(valid_internal)),
        "gap_count": int(len(valid_internal) - np.sum(valid_internal)),
        "valid": valid_internal_list,
    }
    final_tt = tt_hi if export_mode == "highres" else tt_eval
    final_ii = ii_hi if export_mode == "highres" else ii_eval
    export_metadata = {
        "upscale_factor": float(roi_up_factor),
        "roi_width_original": int(roi_w_orig),
        "roi_height_original": int(roi_h_orig),
        "roi_width_after_upscale": int(roi_w),
        "roi_height_after_upscale": int(roi_h),
        "candidate_column_count": int(cand_stats.get("total_columns") or 0),
        "raw_trace_point_count": int(len(trace_columns_internal)),
        "raw_trace_valid_point_count": int(raw_trace_valid_point_count_internal),
        "eval_export_point_count": int(len(tt_eval)),
        "highres_export_point_count": int(len(tt_hi)),
        "final_export_mode": export_mode,
        "final_export_point_count": int(len(final_tt)),
        "highres_available": bool(len(tt_hi) > 0),
    }
    audit_resolution["final_export_mode"] = export_mode
    audit_resolution["final_export_point_count"] = int(len(final_tt))
    audit_resolution["highres_export_point_count"] = int(len(tt_hi))

    # --- Step 17+ stubs ---
    for stage in PIPELINE_STAGES:
        stage_timings.setdefault(stage, 0.0)
    if not warnings:
        warnings.append("pipeline: numeric conversion complete, evaluation pending")

    result = RunResult(
        two_theta_values=final_tt,
        intensities=final_ii,
        x_range=numeric["x_range"],
        y_range=numeric["y_range"],
        quality={"pixel_residual_mean": rt_err, "peak_match_score": None},
        confidence=cal_conf,
        warnings=warnings,
        peaks_numeric_curve=numeric.get("peaks_numeric_curve", []),
        model_assist=model_assist_meta,
        resolution_diagnostics=resolution_diag_payload,
        export_points_eval=export_points_eval,
        export_points_highres=export_points_highres,
        export_metadata=export_metadata,
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
    numeric_curve_peaks_roi = render_numeric_peaks_on_roi(
        roi, x_map_dbg, y_map_dbg, numeric.get("peaks_numeric_curve", []),
    )
    peaks_overlay_img = render_peaks_overlay(
        roi, columns, y_smoothed, valid_mask,
        peak_result["peaks"], peak_result["major_peaks"],
    )

    branch_compare_img = render_branch_compare(
        roi, trace_result["path"], trace_path, columns,
    )

    recovery_debug = {
        "recovery_triggered": recovery_result["recovery_triggered"],
        "zones": recovery_result.get("zones", []),
        "failure_labels": recovery_result.get("failure_labels", []),
    }

    candidate_gt_proximity_diag = None
    _gtjson_for_diag: Optional[str] = None
    if selective_active and selective_oracle_settings_proc.gt_json_path:
        _gtjson_for_diag = str(selective_oracle_settings_proc.gt_json_path)
    elif oracle_active and oracle_rerank_settings_proc.gt_json_path:
        _gtjson_for_diag = str(oracle_rerank_settings_proc.gt_json_path)
    if _gtjson_for_diag:
        try:
            candidate_gt_proximity_diag = compute_candidate_gt_proximity_diag(
                final_cands,
                _gtjson_for_diag,
                tuple(plot_box_t),
                roi_h,
                roi_w,
            )
        except Exception as ex:  # pragma: no cover - diagnosis only
            warnings.append(f"candidate_gt_proximity: {ex}")

    dp_cost_debug = {"enabled": False}
    if debug_dp_cost_breakdown:
        gt_y_for_dp_cost = None
        if _gtjson_for_diag:
            try:
                gt_y_for_dp_cost, _ = build_gt_y_roi_per_column(
                    _load_gt_json(_gtjson_for_diag),
                    tuple(plot_box_t),
                    roi_h,
                    roi_w,
                )
            except Exception as ex:  # pragma: no cover - diagnosis only
                warnings.append(f"dp_cost_debug gt load: {ex}")
        try:
            dp_cost_debug = build_dp_cost_breakdown(
                final_cands,
                trace_result["path"],
                roi_w,
                roi_h,
                gt_y_by_col=gt_y_for_dp_cost,
                gt_near_px=5.0,
                upper_band_frac=0.2,
                confidence_weight_multiplier=float(dp_confidence_weight_multiplier),
                transition_penalty_multiplier=float(dp_transition_penalty_multiplier),
                curvature_penalty_multiplier=float(dp_curvature_penalty_multiplier),
            )
        except Exception as ex:  # pragma: no cover - diagnosis only
            dp_cost_debug = {"enabled": False, "error": str(ex)}
            warnings.append(f"dp_cost_debug: {ex}")

    debug_data: Dict = {
        "roi_preview": Image.fromarray(roi),
        "color_mask": _mask_to_image(mask_a),
        "combined_mask": _mask_to_image(combined),
        "raw_candidate_mask": _mask_to_image(morph["raw_candidate_mask"]),
        "pre_skeleton_candidates": _mask_to_image(morph["skeleton_mask"]),
        "skeleton": _mask_to_image(skeleton),
        "components_overlay": Image.fromarray(comp_overlay),
        "candidate_map_raw": Image.fromarray(raw_map, mode="L"),
        "candidate_map_filtered": Image.fromarray(filt_map, mode="L"),
        "candidate_map_final": Image.fromarray(final_map, mode="L"),
        "trace_path": Image.fromarray(trace_overlay),
        "smoothed_trace": Image.fromarray(smoothed_trace_img),
        "numeric_curve_peaks_roi": Image.fromarray(numeric_curve_peaks_roi),
        "peaks_overlay": Image.fromarray(peaks_overlay_img),
        "branch_compare": Image.fromarray(branch_compare_img),
        "debug.json": {
            "image_size": [w, h],
            "plot_box": mi.plot_box,
            "color_threshold": round(float(threshold), 2),
            "n_prototypes": len(prototypes),
            "raw_candidate_pixels": raw_fg,
            "skeleton_pixels": skel_fg,
            "n_components": n_comp,
            "component_scores": {str(k): v for k, v in comp_scores.items()},
            "candidate_stats": cand_stats,
            "candidate_filter_debug": filter_debug if filter_debug else None,
            "candidate_final_debug": candidate_final_debug_accum.get("candidate_final_debug"),
            "candidate_gt_proximity": candidate_gt_proximity_diag,
            "dp_cost_debug": dp_cost_debug,
            "trace": {
                "trace_score": trace_result["trace_score"],
                "valid_ratio": trace_result["valid_ratio"],
                "window_W": trace_result["window_W"],
                "diagnostics": trace_result["diagnostics"],
                "blockwise": trace_result["blockwise"],
                "path": trace_result["path"],
            },
            "recovery": recovery_debug,
            "postprocess": {
                "gap_fill": {
                    "n_gaps": len(gap_ranges),
                    "gap_ranges": gap_ranges,
                    "total_filled_px": len(gap_filled_set),
                },
                "smoothing": {
                    "sg_window": sg_window,
                    "sg_polyorder": 2,
                    "use_sharp_peak_preserve": bool(sharp_peak_settings_proc.use_sharp_peak_preserve),
                    "curve_smooth_window": int(sharp_peak_settings_proc.curve_smooth_window),
                    "peak_smooth_window": int(sharp_peak_settings_proc.peak_smooth_window),
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
            "pipeline_version": pipeline_version,
            "warnings": warnings,
            "v1_options": {
                "axis_mask_margin": int(amargin),
                "use_ridge_candidates": bool(use_ridge_candidates),
                "peak_two_pass": bool(peak_two_pass),
                "use_contrast_aux": bool(contrast_aux_settings_proc.use_contrast_aux),
                "contrast_aux_weight": float(contrast_aux_settings_proc.contrast_aux_weight),
                "contrast_aux_min_base_conf": float(contrast_aux_settings_proc.contrast_aux_min_base_conf),
                "contrast_aux_bg_kernel_ratio": float(contrast_aux_settings_proc.contrast_aux_bg_kernel_ratio),
                "contrast_aux_border_suppress_px": int(contrast_aux_settings_proc.contrast_aux_border_suppress_px),
                "curvature_blend_strength": float(curvature_blend_strength),
                "loose_peak_prominence_factor": float(loose_pf),
                "use_sharp_peak_preserve": bool(sharp_peak_settings_proc.use_sharp_peak_preserve),
                "curve_smooth_window": int(sharp_peak_settings_proc.curve_smooth_window),
                "peak_smooth_window": int(sharp_peak_settings_proc.peak_smooth_window),
                "peak_preserve_radius": int(sharp_peak_settings_proc.peak_preserve_radius),
                "peak_blend_raw_weight": float(sharp_peak_settings_proc.peak_blend_raw_weight),
                "sharp_global_prom_ratio": float(sharp_peak_settings_proc.global_prom_ratio),
                "sharp_local_prom_window": int(sharp_peak_settings_proc.local_prom_window),
                "sharp_local_prom_ratio": float(sharp_peak_settings_proc.local_prom_ratio),
                "sharp_local_noise_k": float(sharp_peak_settings_proc.local_noise_k),
                "use_dp_candidate_bridge": bool(use_dp_candidate_bridge),
                "dp_candidate_bridge_window_W": int(dp_transition_window_width(roi_w)),
                "use_dp_column_apex_pull": bool(use_dp_column_apex_pull),
                "debug_preserve_gt_near_final_candidates": bool(debug_preserve_gt_near_final_candidates),
                "debug_dp_cost_breakdown": bool(debug_dp_cost_breakdown),
                "dp_confidence_weight_multiplier": float(dp_confidence_weight_multiplier),
                "dp_transition_penalty_multiplier": float(dp_transition_penalty_multiplier),
                "dp_curvature_penalty_multiplier": float(dp_curvature_penalty_multiplier),
                "oracle_confidence_sharpening": float(oracle_confidence_sharpening),
                "debug_filter_removal_reasons": bool(debug_filter_removal_reasons),
                "debug_filter_keep_gt_near": bool(debug_filter_keep_gt_near),
                "candidate_filter_topk_before_final": int(candidate_filter_topk_before_final),
                "candidate_filter_min_conf_keep": float(candidate_filter_min_conf_keep),
                "candidate_filter_disable_envelope_bonus": bool(candidate_filter_disable_envelope_bonus),
                "candidate_filter_enable_y_diversity": bool(candidate_filter_enable_y_diversity),
                "candidate_filter_y_diversity_bins": int(candidate_filter_y_diversity_bins),
                "candidate_filter_enable_source_balance": bool(candidate_filter_enable_source_balance),
                "candidate_filter_source_balance_raw_quota": int(candidate_filter_source_balance_raw_quota),
                "debug_filter_rank_breakdown": bool(debug_filter_rank_breakdown),
                "debug_filter_rank_breakdown_max_columns": int(debug_filter_rank_breakdown_max_columns),
                "candidate_filter_enable_local_evidence_rank": bool(candidate_filter_enable_local_evidence_rank),
                "candidate_filter_local_evidence_weight": float(candidate_filter_local_evidence_weight),
                "candidate_filter_local_evidence_tau_px": float(local_evidence_tau_px_proc),
                "candidate_filter_enable_column_rank_normalization": bool(
                    candidate_filter_enable_column_rank_normalization
                ),
                "candidate_filter_enable_evidence_aware_preserve": bool(
                    candidate_filter_enable_evidence_aware_preserve
                ),
                "candidate_filter_evidence_preserve_bins": int(
                    candidate_filter_evidence_preserve_bins
                ),
                "candidate_filter_evidence_preserve_per_bin": int(
                    candidate_filter_evidence_preserve_per_bin
                ),
                "candidate_filter_evidence_preserve_max_upper_frac": float(
                    candidate_filter_evidence_preserve_max_upper_frac
                ),
                "debug_final_selection_reasons": bool(debug_final_selection_reasons),
                "candidate_final_enable_evidence_aware_preserve": bool(
                    candidate_final_enable_evidence_aware_preserve
                ),
                "candidate_final_evidence_preserve_slots": int(candidate_final_evidence_preserve_slots),
                "candidate_final_disable_score_bucket_dedupe": bool(
                    candidate_final_disable_score_bucket_dedupe
                ),
                "candidate_final_dedupe_score_decimals": candidate_final_dedupe_score_decimals,
                "candidate_final_enable_continuity_preserve": bool(
                    candidate_final_enable_continuity_preserve
                ),
                "candidate_final_continuity_slots": int(candidate_final_continuity_slots),
                "candidate_final_continuity_window": int(candidate_final_continuity_window),
                "candidate_final_continuity_max_jump": int(continuity_max_jump_proc),
                "candidate_final_max_dp_bridge_frac": candidate_final_max_dp_bridge_frac,
                "debug_dump_raw_confidence_features": bool(debug_dump_raw_confidence_features),
                "runtime_oracle_rerank": bool(oracle_active),
                "runtime_selective_oracle_rerank": bool(selective_active),
                "runtime_model_assist": bool(ma_active),
                "runtime_model_assist_dp_branch": model_assist_meta.get("dp_branch"),
                "peak_apex_roi_refine": bool(use_peak_apex_roi_refine),
                "peak_apex_roi_radius": int(peak_apex_roi_radius_proc),
                "roi_upscale_factor": int(roi_up_factor),
                "roi_upscale_method": str(roi_up_method),
            },
            "model_assist": model_assist_meta,
            "contrast_aux": {
                "enabled": contrast_aux_settings_proc.use_contrast_aux,
                "weight": float(contrast_aux_settings_proc.contrast_aux_weight),
                "min_base_conf": float(contrast_aux_settings_proc.contrast_aux_min_base_conf),
                "bg_kernel_ratio": float(contrast_aux_settings_proc.contrast_aux_bg_kernel_ratio),
                "border_suppress_px": int(contrast_aux_settings_proc.contrast_aux_border_suppress_px),
                "candidate_conf_summary": cand_report,
                "ab_criteria": contrast_aux_ab_reference(),
            },
            "roi_upscale": {
                "enabled": bool(roi_up_factor > 1),
                "factor": int(roi_up_factor),
                "method": str(roi_up_method),
                "original_roi_size": [int(roi_w_orig), int(roi_h_orig)],
                "processed_roi_size": [int(roi_w), int(roi_h)],
                "numeric_export_columns": int(len(columns_numeric)),
                "px_parameter_scaling": "internal_processing_px_scaled_by_factor; numeric_export_back_to_original_roi",
            },
            "resolution_export_audit": audit_resolution,
        },
    }

    debug_data["candidate_conf_before_after"] = cand_report
    if dump_candidates_json:
        # 모델 학습용 데이터셋 생성 시 열별 후보를 직접 재사용할 수 있도록 저장
        debug_data["raw_candidates"] = raw_cands
        debug_data["filtered_candidates"] = filtered_cands
        debug_data["final_candidates"] = final_cands
    debug_data["contrast_aux_ab_log"] = {
        "arm": "experiment" if contrast_aux_settings_proc.use_contrast_aux else "baseline",
        "sample_id": sample_id,
        "settings": {
            "use_contrast_aux": contrast_aux_settings_proc.use_contrast_aux,
            "contrast_aux_weight": contrast_aux_settings_proc.contrast_aux_weight,
            "contrast_aux_min_base_conf": contrast_aux_settings_proc.contrast_aux_min_base_conf,
            "contrast_aux_bg_kernel_ratio": contrast_aux_settings_proc.contrast_aux_bg_kernel_ratio,
            "contrast_aux_border_suppress_px": contrast_aux_settings_proc.contrast_aux_border_suppress_px,
        },
        "criteria": contrast_aux_ab_reference(),
    }
    if contrast_aux_settings_proc.use_contrast_aux and contrast_aux_map_np is not None:
        debug_data["contrast_aux_map"] = np.clip(contrast_aux_map_np * 255.0, 0, 255).astype(np.uint8)
        debug_data["trace_path_contrast_aux_overlay"] = render_trace_on_contrast_aux_map(
            contrast_aux_map_np, trace_path,
        )

    debug_data.update(sharp_debug_pack)

    if selective_risk_rows is not None:
        debug_data["risk_detector_column_feature_count"] = int(len(selective_risk_rows))
        if selective_segments is not None:
            debug_data["risk_detector_segments"] = selective_segments

    if recovery_result["recovery_triggered"]:
        before_cands = recovery_result.get("before_candidates", {})
        after_cands = recovery_result.get("after_candidates", {})
        if before_cands:
            debug_data["recovery_candidates_before"] = Image.fromarray(
                render_candidates_overlay(roi, before_cands, "before"))
        if after_cands:
            debug_data["recovery_candidates_after"] = Image.fromarray(
                render_candidates_overlay(roi, after_cands, "after"))

    return result, debug_data


def _normalize_pipeline_cli(name: str) -> str:
    """CLI 값 정규화: v1 / v1_1 / v1.1 → v1_1, v1_2 / v1.2 → v1_2."""
    x = name.lower().strip().replace(".", "_")
    if x == "v1":
        return "v1_1"
    return x


def run_single(
    image_path: str,
    manual_inputs_path: str,
    output_json_path: str,
    debug_dir: str,
    pipeline: str = "v1_1",
    tune_json: str | None = None,
    allow_experimental_v2: bool = False,
    *,
    axis_mask_margin: int = 3,
    use_ridge_candidates: bool = False,
    peak_two_pass: bool = True,
    contrast_aux_settings: ContrastAuxSettings = DEFAULT_CONTRAST_AUX_SETTINGS,
    curvature_blend_strength: float = 0.32,
    loose_peak_prominence_factor: Optional[float] = None,
    sharp_peak_settings: SharpPeakPreserveSettings = DEFAULT_SHARP_PEAK_SETTINGS,
    use_dp_candidate_bridge: bool = True,
    use_dp_column_apex_pull: bool = True,
    dump_candidates_json: bool = False,
    debug_dump_raw_confidence_features: bool = False,
    debug_filter_removal_reasons: bool = False,
    debug_filter_keep_gt_near: bool = False,
    candidate_filter_topk_before_final: int = 16,
    candidate_filter_min_conf_keep: float = 0.25,
    candidate_filter_disable_envelope_bonus: bool = False,
    candidate_filter_enable_y_diversity: bool = False,
    candidate_filter_y_diversity_bins: int = 8,
    candidate_filter_enable_source_balance: bool = False,
    candidate_filter_source_balance_raw_quota: int = 2,
    debug_filter_rank_breakdown: bool = False,
    debug_filter_rank_breakdown_max_columns: int = 64,
    candidate_filter_enable_local_evidence_rank: bool = False,
    candidate_filter_local_evidence_weight: float = LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT,
    candidate_filter_local_evidence_tau_px: float = LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT,
    candidate_filter_enable_column_rank_normalization: bool = False,
    candidate_filter_enable_evidence_aware_preserve: bool = False,
    candidate_filter_evidence_preserve_bins: int = 6,
    candidate_filter_evidence_preserve_per_bin: int = 1,
    candidate_filter_evidence_preserve_max_upper_frac: float = 0.5,
    debug_final_selection_reasons: bool = False,
    candidate_final_enable_evidence_aware_preserve: bool = False,
    candidate_final_evidence_preserve_slots: int = 2,
    candidate_final_disable_score_bucket_dedupe: bool = False,
    candidate_final_dedupe_score_decimals: Optional[int] = None,
    candidate_final_enable_continuity_preserve: bool = False,
    candidate_final_continuity_slots: int = 2,
    candidate_final_continuity_window: int = 3,
    candidate_final_continuity_max_jump: int = 8,
    candidate_final_max_dp_bridge_frac: Optional[float] = None,
    debug_preserve_gt_near_final_candidates: bool = False,
    model_assist_settings: ModelAssistSettings = DEFAULT_MODEL_ASSIST_SETTINGS,
    oracle_rerank_settings: OracleRerankSettings = DEFAULT_ORACLE_RERANK_SETTINGS,
    selective_oracle_settings: SelectiveOracleSettings = DEFAULT_SELECTIVE_ORACLE_SETTINGS,
    use_peak_apex_roi_refine: bool = False,
    peak_apex_roi_radius: int = 5,
    debug_dp_cost_breakdown: bool = False,
    dp_confidence_weight_multiplier: float = 1.0,
    dp_transition_penalty_multiplier: float = 1.0,
    dp_curvature_penalty_multiplier: float = 1.0,
    oracle_confidence_sharpening: float = 1.0,
    roi_upscale_factor: int = 1,
    roi_upscale_method: str = "lanczos",
    final_export_mode: str = "eval_grid",
    gt_json_path_for_metadata: Optional[str] = None,
) -> RunResult:
    """단일 이미지 처리 진입점.

    운영: v1_1 / v1_2 (동일 엔진, pipeline_version 만 calibrate_v1_1 vs calibrate_v1_2).
    실험: v2_experimental + --allow_experimental_v2 만 허용.
    """
    total_run_t0 = time.perf_counter()
    image = load_image(image_path)
    mi = load_manual_inputs(manual_inputs_path)
    manual_meta: Dict[str, Any] = {}
    try:
        manual_meta = json.loads(Path(manual_inputs_path).read_text(encoding="utf-8"))
    except Exception:
        manual_meta = {}

    errors = validate_manual_inputs(mi, image.size)
    if errors:
        print(f"[WARN] Input validation issues: {errors}", file=sys.stderr)

    pl = _normalize_pipeline_cli(pipeline)

    if pl == "v2":
        raise RuntimeError(
            "pipeline='v2'는 롤백 정책으로 잠겨 있습니다. "
            "운영은 v1_1(v1.1)을 사용하고, 실험은 v2_experimental + --allow_experimental_v2를 사용하세요."
        )
    sample_stem = Path(output_json_path).stem

    if pl == "v2_experimental":
        if not allow_experimental_v2:
            raise RuntimeError("v2_experimental 실행에는 --allow_experimental_v2 플래그가 필요합니다.")
        result, debug_data = run_pipeline_experimental(image, mi, params=load_exp_params(tune_json))
    else:
        cal_ver = CALIBRATE_V1_2 if pl == "v1_2" else CALIBRATE_V1_1
        result, debug_data = run_pipeline(
            image,
            mi,
            pipeline_version=cal_ver,
            axis_mask_margin=axis_mask_margin,
            use_ridge_candidates=use_ridge_candidates,
            peak_two_pass=peak_two_pass,
            contrast_aux_settings=contrast_aux_settings,
            sample_id=sample_stem,
            curvature_blend_strength=curvature_blend_strength,
            loose_peak_prominence_factor=loose_peak_prominence_factor,
            sharp_peak_settings=sharp_peak_settings,
            use_dp_candidate_bridge=use_dp_candidate_bridge,
            use_dp_column_apex_pull=use_dp_column_apex_pull,
            dump_candidates_json=dump_candidates_json,
            debug_dump_raw_confidence_features=debug_dump_raw_confidence_features,
            debug_filter_removal_reasons=debug_filter_removal_reasons,
            debug_filter_keep_gt_near=debug_filter_keep_gt_near,
            candidate_filter_topk_before_final=candidate_filter_topk_before_final,
            candidate_filter_min_conf_keep=candidate_filter_min_conf_keep,
            candidate_filter_disable_envelope_bonus=candidate_filter_disable_envelope_bonus,
            candidate_filter_enable_y_diversity=candidate_filter_enable_y_diversity,
            candidate_filter_y_diversity_bins=candidate_filter_y_diversity_bins,
            candidate_filter_enable_source_balance=candidate_filter_enable_source_balance,
            candidate_filter_source_balance_raw_quota=candidate_filter_source_balance_raw_quota,
            debug_filter_rank_breakdown=debug_filter_rank_breakdown,
            debug_filter_rank_breakdown_max_columns=debug_filter_rank_breakdown_max_columns,
            candidate_filter_enable_local_evidence_rank=candidate_filter_enable_local_evidence_rank,
            candidate_filter_local_evidence_weight=candidate_filter_local_evidence_weight,
            candidate_filter_local_evidence_tau_px=candidate_filter_local_evidence_tau_px,
            candidate_filter_enable_column_rank_normalization=candidate_filter_enable_column_rank_normalization,
            candidate_filter_enable_evidence_aware_preserve=candidate_filter_enable_evidence_aware_preserve,
            candidate_filter_evidence_preserve_bins=candidate_filter_evidence_preserve_bins,
            candidate_filter_evidence_preserve_per_bin=candidate_filter_evidence_preserve_per_bin,
            candidate_filter_evidence_preserve_max_upper_frac=candidate_filter_evidence_preserve_max_upper_frac,
            debug_final_selection_reasons=debug_final_selection_reasons,
            candidate_final_enable_evidence_aware_preserve=candidate_final_enable_evidence_aware_preserve,
            candidate_final_evidence_preserve_slots=candidate_final_evidence_preserve_slots,
            candidate_final_disable_score_bucket_dedupe=candidate_final_disable_score_bucket_dedupe,
            candidate_final_dedupe_score_decimals=candidate_final_dedupe_score_decimals,
            candidate_final_enable_continuity_preserve=(
                candidate_final_enable_continuity_preserve
            ),
            candidate_final_continuity_slots=candidate_final_continuity_slots,
            candidate_final_continuity_window=candidate_final_continuity_window,
            candidate_final_continuity_max_jump=candidate_final_continuity_max_jump,
            candidate_final_max_dp_bridge_frac=candidate_final_max_dp_bridge_frac,
            debug_preserve_gt_near_final_candidates=debug_preserve_gt_near_final_candidates,
            model_assist_settings=model_assist_settings,
            oracle_rerank_settings=oracle_rerank_settings,
            selective_oracle_settings=selective_oracle_settings,
            use_peak_apex_roi_refine=use_peak_apex_roi_refine,
            peak_apex_roi_radius=peak_apex_roi_radius,
            debug_dp_cost_breakdown=debug_dp_cost_breakdown,
            dp_confidence_weight_multiplier=dp_confidence_weight_multiplier,
            dp_transition_penalty_multiplier=dp_transition_penalty_multiplier,
            dp_curvature_penalty_multiplier=dp_curvature_penalty_multiplier,
            oracle_confidence_sharpening=oracle_confidence_sharpening,
            roi_upscale_factor=int(roi_upscale_factor),
            roi_upscale_method=str(roi_upscale_method),
            final_export_mode=str(final_export_mode),
        )

    metadata_gt_json = (
        str(gt_json_path_for_metadata)
        if gt_json_path_for_metadata
        else str(manual_meta.get("gt_json", "") or "")
    )
    if isinstance(getattr(result, "export_metadata", None), dict):
        result.export_metadata.update(
            {
                "sample_id": str(manual_meta.get("sample_id", "") or Path(output_json_path).stem.replace("_result", "")),
                "domain": str(manual_meta.get("domain", "") or ""),
                "input_image": str(image_path),
                "manual_json": str(manual_inputs_path),
                "gt_json": metadata_gt_json,
            }
        )

    if isinstance(debug_data.get("debug.json"), dict):
        debug_data["debug.json"]["run_metadata"] = {
            "input_image": str(image_path),
            "manual_json": str(manual_inputs_path),
            "gt_json": metadata_gt_json or None,
            "output_json_path": str(output_json_path),
            "debug_dir": str(debug_dir),
            "final_export_mode": str(final_export_mode),
        }
    stage_timings_obj: Optional[Dict[str, Any]] = None
    if isinstance(debug_data.get("debug.json"), dict):
        st = debug_data["debug.json"].setdefault("stage_timings", {})
        if isinstance(st, dict):
            stage_timings_obj = st

    t_save = time.perf_counter()
    save_result_json(result, output_json_path)
    if stage_timings_obj is not None:
        stage_timings_obj["result_export_sec"] = round(time.perf_counter() - t_save, 6)

    t_save = time.perf_counter()
    save_debug_files(debug_data, debug_dir)
    if stage_timings_obj is not None:
        stage_timings_obj["debug_dump_sec"] = round(time.perf_counter() - t_save, 6)
        stage_timings_obj["total_run_sec"] = round(time.perf_counter() - total_run_t0, 6)
        debug_json_path = Path(debug_dir) / "debug.json"
        if debug_json_path.is_file():
            with debug_json_path.open("w", encoding="utf-8") as f:
                json.dump(debug_data["debug.json"], f, ensure_ascii=False, indent=2)

    if pl != "v2_experimental":
        arm = "experiment" if contrast_aux_settings.use_contrast_aux else "baseline"
        print(
            f"[contrast_aux_ab] arm={arm} sample={sample_stem} "
            f"-> {debug_dir}/contrast_aux_ab_log.json",
            file=sys.stderr,
        )

    return result


def validate_only(image_path: str, manual_inputs_path: str) -> bool:
    """§10.6: 입력 JSON 검증 전용 모드."""
    image = load_image(image_path)
    mi = load_manual_inputs(manual_inputs_path)

    errors = validate_manual_inputs(mi, image.size)
    clicks = mi.click_count
    status = mi.click_budget_status

    print(f"  image size : {image.size}")
    print(f"  click count: {clicks}")
    print(f"  budget     : {status}")

    if status == "ux_fail":
        print(f"  [UX_FAIL] click count {clicks} > 12")

    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        return False

    print("  [PASS] all validation checks passed")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="§9-10: Run engine on single image")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--manual_inputs_path", type=str, required=True)
    parser.add_argument("--output_json_path", type=str, default=None)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument("--validate_only", action="store_true",
                        help="§10.6: validate manual inputs without running pipeline")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="v1_1",
        choices=["v1_1", "v1", "v1.1", "v1_2", "v1.2", "v2", "v2_experimental"],
        help="v1_1(기본)=v1.1, v1_2=v1.2 동일 엔진·스냅샷 태그, v1=호환, v2 잠금, v2_experimental=실험",
    )
    parser.add_argument("--tune_json", type=str, default=None, help="v2 파라미터 JSON 파일")
    parser.add_argument(
        "--allow_experimental_v2",
        action="store_true",
        help="v2_experimental 실행 잠금 해제 플래그",
    )
    parser.add_argument(
        "--axis-mask-margin",
        type=int,
        default=3,
        metavar="PX",
        help="mask_axis_lines 테두리 제거 폭(px), 로드맵 1c (기본 3)",
    )
    parser.add_argument(
        "--roi-upscale-factor",
        type=int,
        default=1,
        choices=[1, 2],
        metavar="N",
        help="진단용: ROI crop 후 deterministic upscale 배수 (1=기본, 2=ROI 2x)",
    )
    parser.add_argument(
        "--roi-upscale-method",
        type=str,
        default="lanczos",
        choices=["lanczos", "bicubic"],
        help="ROI upscale interpolation method (AI super-resolution 아님)",
    )
    parser.add_argument(
        "--final-export-mode",
        type=str,
        default="eval_grid",
        choices=["eval_grid", "highres"],
        help="final JSON root curve export mode: eval_grid keeps original ROI-width grid, highres uses ROI-upscaled trace points",
    )
    parser.add_argument(
        "--use-ridge-candidates",
        action="store_true",
        help="세로 능선 응답을 후보 신뢰도에 반영, 로드맵 2b (기본 끔)",
    )
    parser.add_argument(
        "--peak-single-pass",
        action="store_true",
        help="피크 검출 시 prominence 완화 2패스·NMS 끄기 (로드맵 3a 기본은 2패스)",
    )
    parser.add_argument(
        "--use-contrast-aux",
        action="store_true",
        help="contrast_aux_v1: 후보 confidence에만 Lab 배경대비 보정(후보·DP 비용 불변)",
    )
    parser.add_argument(
        "--contrast-aux-weight",
        type=float,
        default=0.25,
        metavar="W",
        help="blend 가중치: final=(1-W)*base+W*contrast_bonus (기본 0.25)",
    )
    parser.add_argument(
        "--contrast-aux-min-base-conf",
        type=float,
        default=0.15,
        metavar="T",
        help="base_conf 미만이면 대비 보정 미적용 (기본 0.15)",
    )
    parser.add_argument(
        "--contrast-aux-bg-kernel-ratio",
        type=float,
        default=0.035,
        metavar="R",
        help="local bg median 필터 크기 ≈ max(31, odd(round(R*plot_width)))",
    )
    parser.add_argument(
        "--contrast-aux-border-suppress-px",
        type=int,
        default=8,
        metavar="PX",
        help="plot_box 테두리에서 PX 이내 대비맵 0",
    )
    parser.add_argument(
        "--curvature-blend-strength",
        type=float,
        default=0.32,
        metavar="S",
        help="고곡률 구간에서 gap-fill 쪽 블렌드 강도↑ 시 근접 피크 뭉침 완화(기본 0.32; 예: 0.42~0.48 시도)",
    )
    parser.add_argument(
        "--peak-loose-prominence-factor",
        type=float,
        default=None,
        metavar="F",
        help="2패스 피크 검출 완화 비율(미지정 시 코드 기본값). 낮출수록 약한 피크 더 잡힘(예: 0.42)",
    )
    parser.add_argument(
        "--use-sharp-peak-preserve",
        action="store_true",
        help="후처리 sharp peak preserve (곡선/피크 SG 분리·국소 prominence·apex 스냅)",
    )
    parser.add_argument("--curve-smooth-window", type=int, default=9, metavar="W")
    parser.add_argument("--peak-smooth-window", type=int, default=5, metavar="W")
    parser.add_argument("--peak-preserve-radius", type=int, default=3, metavar="R")
    parser.add_argument("--peak-blend-raw-weight", type=float, default=0.75, metavar="A")
    parser.add_argument("--global-prom-ratio", type=float, default=0.015, metavar="G")
    parser.add_argument("--local-prom-window", type=int, default=61, metavar="W")
    parser.add_argument("--local-prom-ratio", type=float, default=0.12, metavar="L")
    parser.add_argument("--local-noise-k", type=float, default=3.0, metavar="K")
    parser.add_argument(
        "--no-dp-candidate-bridge",
        action="store_true",
        help="DP 직전 이웃 열 y±W 브리지 후보 확장 끔 (기본은 켜짐)",
    )
    parser.add_argument(
        "--no-dp-column-apex-pull",
        action="store_true",
        help="DP 직후 열별 후보 꼭짓점 당김(apex pull) 끔 (기본은 켜짐)",
    )
    parser.add_argument(
        "--dump-candidates-json",
        action="store_true",
        help="debug_dir에 raw/filtered/final candidates JSON 저장 (모델 보조 학습용)",
    )
    parser.add_argument(
        "--debug-dump-raw-confidence-features",
        action="store_true",
        help="raw 후보에 confidence 분해·component_id·GT 거리 등 진단 필드 추가(--dump-candidates-json 시 JSON 반영)",
    )
    parser.add_argument(
        "--debug-filter-removal-reasons",
        action="store_true",
        help="filter_candidates 단계 제거 reason/GT-near 보존 여부를 debug.json에 기록",
    )
    parser.add_argument(
        "--debug-filter-keep-gt-near",
        action="store_true",
        help="진단용: filtered 단계에서 GT-near 후보를 강제 보존(운영 기본 off)",
    )
    parser.add_argument(
        "--candidate-filter-topk-before-final",
        type=int,
        default=16,
        metavar="N",
        help="filtered 단계 열별 상위 후보 수 (기본 16)",
    )
    parser.add_argument(
        "--candidate-filter-min-conf-keep",
        type=float,
        default=0.25,
        metavar="T",
        help="filtered 단계 최소 confidence 임계값 (기본 0.25)",
    )
    parser.add_argument(
        "--candidate-filter-disable-envelope-bonus",
        action="store_true",
        help="filtered 정렬의 envelope bonus 비활성화(진단용)",
    )
    parser.add_argument(
        "--candidate-filter-enable-y-diversity",
        action="store_true",
        help="filtered 선택 시 y-bin 다양성 보존(진단용)",
    )
    parser.add_argument(
        "--candidate-filter-y-diversity-bins",
        type=int,
        default=8,
        metavar="N",
        help="y 다양성 보존용 bin 수 (기본 8)",
    )
    parser.add_argument(
        "--candidate-filter-enable-source-balance",
        action="store_true",
        help="filtered 후보에서 source=raw 최소 quota 보존(진단용)",
    )
    parser.add_argument(
        "--candidate-filter-source-balance-raw-quota",
        type=int,
        default=2,
        metavar="N",
        help="source balance에서 source=raw 최소 보존 수 (기본 2)",
    )
    parser.add_argument(
        "--debug-filter-rank-breakdown",
        action="store_true",
        help="filter 정렬 점수 분해(base/envelope/rank_score) debug 저장",
    )
    parser.add_argument(
        "--debug-filter-rank-breakdown-max-columns",
        type=int,
        default=64,
        metavar="N",
        help="rank breakdown 저장 최대 열 수(기본 64)",
    )
    parser.add_argument(
        "--candidate-filter-enable-local-evidence-rank",
        action="store_true",
        help="filtered 정렬에 이웃 열 top1 y 중앙값과의 일치 보너스 추가(진단·실험, 기본 off)",
    )
    parser.add_argument(
        "--candidate-filter-local-evidence-weight",
        type=float,
        default=None,
        metavar="W",
        help=f"local evidence 가중치(미지정 시 {LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT})",
    )
    parser.add_argument(
        "--candidate-filter-local-evidence-tau-px",
        type=float,
        default=None,
        metavar="PX",
        help=f"local evidence 거리 스케일 px(미지정 시 {LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT})",
    )
    parser.add_argument(
        "--candidate-filter-enable-column-rank-normalization",
        action="store_true",
        help="filtered 게이트 통과 후 열별 confidence를 순위 백분위([0,1])로 치환 후 정렬·클립(진단용)",
    )
    parser.add_argument(
        "--candidate-filter-enable-evidence-aware-preserve",
        action="store_true",
        help="filtered: GT 미사용 수직 bin별 local evidence 후보 보존 + 상단 비율 상한(진단용, 기본 off)",
    )
    parser.add_argument(
        "--candidate-filter-preserve-bins",
        type=int,
        default=6,
        metavar="N",
        help="evidence-aware 보존용 y 구간 수 (기본 6)",
    )
    parser.add_argument(
        "--candidate-filter-preserve-per-bin",
        type=int,
        default=1,
        metavar="N",
        help="구간당 보존 후보 수 (기본 1)",
    )
    parser.add_argument(
        "--candidate-filter-preserve-max-upper-frac",
        type=float,
        default=0.5,
        metavar="F",
        help="선택된 후보 중 ROI 상단 20%% 밴드 비율 상한 (기본 0.5)",
    )
    parser.add_argument(
        "--debug-final-selection-reasons",
        action="store_true",
        help="build_final·브리지 후보 선택 단계 진단을 candidate_final_debug로 저장(열 단위 상세 포함)",
    )
    parser.add_argument(
        "--candidate-final-enable-evidence-aware-preserve",
        action="store_true",
        help="final topK 내 non-upper·local evidence 후보 quota 보존(GT 미사용, 진단용)",
    )
    parser.add_argument(
        "--candidate-final-evidence-preserve-slots",
        type=int,
        default=2,
        metavar="N",
        help="final evidence quota 슬롯 수 (기본 2)",
    )
    parser.add_argument(
        "--candidate-final-enable-continuity-preserve",
        action="store_true",
        help="final: filtered 후보에서 GT 미사용 y-continuous branch 후보를 보존(진단용, 기본 off)",
    )
    parser.add_argument(
        "--candidate-final-continuity-slots",
        type=int,
        default=2,
        metavar="N",
        help="final continuity branch 보존 슬롯 수 (기본 2)",
    )
    parser.add_argument(
        "--candidate-final-continuity-window",
        type=int,
        default=3,
        metavar="N",
        help="continuity branch 연결 시 이전 후보 탐색 column window (기본 3)",
    )
    parser.add_argument(
        "--candidate-final-continuity-max-jump",
        type=int,
        default=8,
        metavar="PX",
        help="continuity branch 연결 허용 column당 최대 y jump px (기본 8)",
    )
    parser.add_argument(
        "--candidate-final-disable-score-bucket-dedupe",
        action="store_true",
        help="final 단계 comp_score 버킷 중복 제거 끔",
    )
    parser.add_argument(
        "--candidate-final-dedupe-score-precision",
        type=int,
        default=None,
        metavar="D",
        help="final dedupe 시 round(comp_score, D) — 미지정 시 2",
    )
    parser.add_argument(
        "--candidate-final-max-dp-bridge-frac",
        type=float,
        default=None,
        metavar="F",
        help="브리지 트림 후 dp_bridge 최대 비율 상한(예: 0.25)",
    )
    parser.add_argument(
        "--debug-preserve-gt-near-final-candidates",
        action="store_true",
        help="진단용: GT-near raw 후보를 final 후보에 강제 보존(운영 기본 off)",
    )
    parser.add_argument(
        "--debug-dp-cost-breakdown",
        action="store_true",
        help="진단용: DP 선택 후보와 GT-near 대안 후보의 confidence/transition/curvature cost를 debug.json에 저장",
    )
    parser.add_argument(
        "--dp-confidence-weight-multiplier",
        type=float,
        default=1.0,
        metavar="M",
        help="진단용: DP confidence penalty 가중치 배수 (기본 1.0)",
    )
    parser.add_argument(
        "--dp-transition-penalty-multiplier",
        type=float,
        default=1.0,
        metavar="M",
        help="진단용: DP dy transition penalty 배수 (기본 1.0)",
    )
    parser.add_argument(
        "--dp-curvature-penalty-multiplier",
        type=float,
        default=1.0,
        metavar="M",
        help="진단용: DP d2y curvature penalty 배수 (기본 1.0)",
    )
    parser.add_argument(
        "--oracle-confidence-sharpening",
        type=float,
        default=1.0,
        metavar="S",
        help="진단용: global oracle confidence를 score**S로 sharpen (기본 1.0)",
    )
    parser.add_argument(
        "--model-assist",
        action="store_true",
        help="DP 직전 SmallCandidateCNN 재랭크 + 규칙 대비 악화 시 자동 fallback",
    )
    parser.add_argument(
        "--model-assist-ckpt",
        type=str,
        default=None,
        metavar="PATH",
        help="candidate_reranker 학습 산출 .pt 경로 (--model-assist 시 필요)",
    )
    parser.add_argument("--model-assist-lambda", type=float, default=0.25, metavar="L")
    parser.add_argument("--model-assist-device", type=str, default="cpu")
    parser.add_argument("--model-assist-patch-size", type=int, default=33, metavar="PX")
    parser.add_argument(
        "--model-assist-fallback-vr-margin",
        type=float,
        default=0.0,
        metavar="EPS",
        help="모델 경로 valid_ratio 가 규칙 경로보다 이만큼 낮아지면 규칙으로 되돌림",
    )
    parser.add_argument(
        "--model-assist-fallback-ts-margin",
        type=float,
        default=0.0,
        metavar="EPS",
        help="모델 trace_score 가 규칙보다 이만큼 크면(더 나쁘면) 규칙으로 되돌림",
    )
    parser.add_argument(
        "--peak-apex-roi-refine",
        action="store_true",
        help="major peak 세로 위치를 ROI 열별 밝기 프로파일로 국소 보정 후 numeric 반영",
    )
    parser.add_argument("--peak-apex-roi-radius", type=int, default=5, metavar="PX")
    parser.add_argument(
        "--oracle-rerank-gt",
        type=str,
        default=None,
        metavar="PATH",
        help="GT JSON 경로 — pixel_curve_path 등으로 열별 GT y 보간 후 oracle confidence 로 DP (학습 불필요)",
    )
    parser.add_argument(
        "--oracle-rerank-sigma",
        type=float,
        default=8.0,
        metavar="PX",
        help="oracle 점수 exp(-(dist/sigma)^2) 의 sigma",
    )
    parser.add_argument(
        "--selective-oracle-rerank-gt",
        type=str,
        default=None,
        metavar="PATH",
        help="GT oracle을 risk 열에만 적용 (global --oracle-rerank-gt 보다 우선)",
    )
    parser.add_argument("--selective-oracle-sigma", type=float, default=8.0, metavar="PX")
    parser.add_argument(
        "--run-domain",
        type=str,
        default="clean",
        choices=["clean", "styled", "real_like"],
        help="selective oracle 스타일 정책용 도메인 태그",
    )
    parser.add_argument(
        "--selective-oracle-taxonomy-prior",
        type=str,
        default=None,
        metavar="LABELS",
        help="세미콜론 구분 failure taxonomy prior (예: grid_confusion;peak_miss_after_smoothing)",
    )
    parser.add_argument(
        "--selective-oracle-allow-styled-real",
        action="store_true",
        help="styled/real_like에서도 위험열 oracle 적용(기본은 off)",
    )
    parser.add_argument(
        "--selective-oracle-risk-features-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="열별 risk feature를 CSV에 append (연구용)",
    )
    parser.add_argument("--selective-risk-dilation-radius", type=int, default=3, metavar="N")
    parser.add_argument("--selective-risk-merge-gap", type=int, default=2, metavar="N")
    parser.add_argument("--selective-risk-min-segment-len", type=int, default=6, metavar="N")
    parser.add_argument(
        "--selective-risk-threshold",
        type=float,
        default=0.08,
        metavar="T",
        help="risk detector conf_margin 임계값(conf_margin < T)",
    )
    parser.add_argument(
        "--selective-risk-disable-taxonomy-prior",
        action="store_true",
        help="risk detector에서 taxonomy prior 영향 제거(진단용)",
    )
    parser.add_argument("--selective-risk-disable-low-margin", action="store_true")
    parser.add_argument("--selective-risk-disable-candidate-starvation", action="store_true")
    parser.add_argument("--selective-risk-disable-path-instability", action="store_true")
    parser.add_argument("--selective-risk-disable-peak-miss-prior", action="store_true")
    parser.add_argument("--selective-risk-disable-grid-confusion-prior", action="store_true")
    parser.add_argument("--selective-risk-disable-axis-proximity", action="store_true")
    parser.add_argument("--selective-risk-disable-high-entropy", action="store_true")
    parser.add_argument("--selective-risk-disable-large-y-gap", action="store_true")
    parser.add_argument("--selective-risk-disable-peak-window", action="store_true")
    parser.add_argument("--selective-risk-disable-dp-margin-low", action="store_true")
    parser.add_argument("--selective-risk-debug-include-columns", action="store_true")
    parser.add_argument(
        "--selective-risk-taxonomy-require-margin",
        action="store_true",
        help="taxonomy_prior reason을 low_margin 또는 high_entropy 조건과 함께만 부착(실험)",
    )
    parser.add_argument(
        "--selective-risk-high-entropy-require-low-margin",
        action="store_true",
        help="high_entropy_many_cands에 conf_margin < threshold 추가 요구(실험)",
    )
    args = parser.parse_args()

    if args.validate_only:
        ok = validate_only(args.image_path, args.manual_inputs_path)
        sys.exit(0 if ok else 1)

    if not args.output_json_path or not args.debug_dir:
        parser.error("--output_json_path and --debug_dir required when not using --validate_only")

    caf = ContrastAuxSettings(
        use_contrast_aux=bool(args.use_contrast_aux),
        contrast_aux_weight=float(args.contrast_aux_weight),
        contrast_aux_min_base_conf=float(args.contrast_aux_min_base_conf),
        contrast_aux_bg_kernel_ratio=float(args.contrast_aux_bg_kernel_ratio),
        contrast_aux_border_suppress_px=int(args.contrast_aux_border_suppress_px),
    )

    sps = SharpPeakPreserveSettings(
        use_sharp_peak_preserve=bool(args.use_sharp_peak_preserve),
        curve_smooth_window=int(args.curve_smooth_window),
        peak_smooth_window=int(args.peak_smooth_window),
        peak_preserve_radius=int(args.peak_preserve_radius),
        peak_blend_raw_weight=float(args.peak_blend_raw_weight),
        global_prom_ratio=float(args.global_prom_ratio),
        local_prom_window=int(args.local_prom_window),
        local_prom_ratio=float(args.local_prom_ratio),
        local_noise_k=float(args.local_noise_k),
    )

    mas = ModelAssistSettings(
        enabled=bool(args.model_assist),
        model_ckpt_path=args.model_assist_ckpt,
        lambda_model=float(args.model_assist_lambda),
        device=str(args.model_assist_device),
        patch_size=int(args.model_assist_patch_size),
        fallback_valid_ratio_margin=float(args.model_assist_fallback_vr_margin),
        fallback_trace_score_margin=float(args.model_assist_fallback_ts_margin),
    )
    sel_gt = bool(args.selective_oracle_rerank_gt)
    orac = OracleRerankSettings(
        enabled=bool(args.oracle_rerank_gt) and not sel_gt,
        gt_json_path=args.oracle_rerank_gt,
        sigma_px=float(args.oracle_rerank_sigma),
    )
    s_orac = SelectiveOracleSettings(
        enabled=sel_gt,
        gt_json_path=args.selective_oracle_rerank_gt,
        sigma_px=float(args.selective_oracle_sigma),
        run_domain=str(args.run_domain),
        apply_to_styles=(
            ("clean", "styled", "real_like")
            if bool(args.selective_oracle_allow_styled_real)
            else ("clean",)
        ),
        conf_margin_thr=float(args.selective_risk_threshold),
        risk_dilate_radius_columns=int(args.selective_risk_dilation_radius),
        merge_gap_columns=int(args.selective_risk_merge_gap),
        min_segment_columns=int(args.selective_risk_min_segment_len),
        disable_low_conf_margin_risk=bool(args.selective_risk_disable_low_margin),
        disable_high_entropy_risk=bool(
            args.selective_risk_disable_candidate_starvation or args.selective_risk_disable_high_entropy
        ),
        disable_axis_proximity_risk=bool(args.selective_risk_disable_axis_proximity),
        disable_large_y_gap_risk=bool(args.selective_risk_disable_large_y_gap),
        disable_peak_window_risk=bool(args.selective_risk_disable_peak_miss_prior or args.selective_risk_disable_peak_window),
        disable_dp_margin_low_risk=bool(args.selective_risk_disable_path_instability or args.selective_risk_disable_dp_margin_low),
        # grid_confusion/peak_miss prior off
        disable_taxonomy_prior_for_risk=bool(
            args.selective_risk_disable_taxonomy_prior or args.selective_risk_disable_grid_confusion_prior
        ),
        risk_debug_include_columns=bool(args.selective_risk_debug_include_columns),
        taxonomy_prior_requires_margin=bool(args.selective_risk_taxonomy_require_margin),
        high_entropy_requires_low_margin=bool(args.selective_risk_high_entropy_require_low_margin),
        taxonomy_prior=args.selective_oracle_taxonomy_prior,
        allow_styled_real_selective=bool(args.selective_oracle_allow_styled_real),
        risk_features_csv_path=args.selective_oracle_risk_features_csv,
    )

    result = run_single(
        args.image_path,
        args.manual_inputs_path,
        args.output_json_path,
        args.debug_dir,
        pipeline=args.pipeline,
        tune_json=args.tune_json,
        allow_experimental_v2=args.allow_experimental_v2,
        axis_mask_margin=args.axis_mask_margin,
        use_ridge_candidates=args.use_ridge_candidates,
        peak_two_pass=not args.peak_single_pass,
        contrast_aux_settings=caf,
        curvature_blend_strength=float(args.curvature_blend_strength),
        loose_peak_prominence_factor=args.peak_loose_prominence_factor,
        sharp_peak_settings=sps,
        use_dp_candidate_bridge=not args.no_dp_candidate_bridge,
        use_dp_column_apex_pull=not args.no_dp_column_apex_pull,
        dump_candidates_json=bool(args.dump_candidates_json),
        debug_dump_raw_confidence_features=bool(args.debug_dump_raw_confidence_features),
        debug_filter_removal_reasons=bool(args.debug_filter_removal_reasons),
        debug_filter_keep_gt_near=bool(args.debug_filter_keep_gt_near),
        candidate_filter_topk_before_final=int(args.candidate_filter_topk_before_final),
        candidate_filter_min_conf_keep=float(args.candidate_filter_min_conf_keep),
        candidate_filter_disable_envelope_bonus=bool(args.candidate_filter_disable_envelope_bonus),
        candidate_filter_enable_y_diversity=bool(args.candidate_filter_enable_y_diversity),
        candidate_filter_y_diversity_bins=int(args.candidate_filter_y_diversity_bins),
        candidate_filter_enable_source_balance=bool(args.candidate_filter_enable_source_balance),
        candidate_filter_source_balance_raw_quota=int(args.candidate_filter_source_balance_raw_quota),
        debug_filter_rank_breakdown=bool(args.debug_filter_rank_breakdown),
        debug_filter_rank_breakdown_max_columns=int(args.debug_filter_rank_breakdown_max_columns),
        candidate_filter_enable_local_evidence_rank=bool(args.candidate_filter_enable_local_evidence_rank),
        candidate_filter_local_evidence_weight=(
            float(args.candidate_filter_local_evidence_weight)
            if args.candidate_filter_local_evidence_weight is not None
            else LOCAL_EVIDENCE_SORT_WEIGHT_DEFAULT
        ),
        candidate_filter_local_evidence_tau_px=(
            float(args.candidate_filter_local_evidence_tau_px)
            if args.candidate_filter_local_evidence_tau_px is not None
            else LOCAL_EVIDENCE_SORT_TAU_PX_DEFAULT
        ),
        candidate_filter_enable_column_rank_normalization=bool(
            args.candidate_filter_enable_column_rank_normalization
        ),
        candidate_filter_enable_evidence_aware_preserve=bool(
            args.candidate_filter_enable_evidence_aware_preserve
        ),
        candidate_filter_evidence_preserve_bins=int(args.candidate_filter_preserve_bins),
        candidate_filter_evidence_preserve_per_bin=int(args.candidate_filter_preserve_per_bin),
        candidate_filter_evidence_preserve_max_upper_frac=float(
            args.candidate_filter_preserve_max_upper_frac
        ),
        debug_final_selection_reasons=bool(args.debug_final_selection_reasons),
        candidate_final_enable_evidence_aware_preserve=bool(
            args.candidate_final_enable_evidence_aware_preserve
        ),
        candidate_final_evidence_preserve_slots=int(args.candidate_final_evidence_preserve_slots),
        candidate_final_disable_score_bucket_dedupe=bool(
            args.candidate_final_disable_score_bucket_dedupe
        ),
        candidate_final_dedupe_score_decimals=args.candidate_final_dedupe_score_precision,
        candidate_final_enable_continuity_preserve=bool(
            args.candidate_final_enable_continuity_preserve
        ),
        candidate_final_continuity_slots=int(args.candidate_final_continuity_slots),
        candidate_final_continuity_window=int(args.candidate_final_continuity_window),
        candidate_final_continuity_max_jump=int(args.candidate_final_continuity_max_jump),
        candidate_final_max_dp_bridge_frac=args.candidate_final_max_dp_bridge_frac,
        debug_preserve_gt_near_final_candidates=bool(args.debug_preserve_gt_near_final_candidates),
        model_assist_settings=mas,
        oracle_rerank_settings=orac,
        selective_oracle_settings=s_orac,
        use_peak_apex_roi_refine=bool(args.peak_apex_roi_refine),
        peak_apex_roi_radius=int(args.peak_apex_roi_radius),
        debug_dp_cost_breakdown=bool(args.debug_dp_cost_breakdown),
        dp_confidence_weight_multiplier=float(args.dp_confidence_weight_multiplier),
        dp_transition_penalty_multiplier=float(args.dp_transition_penalty_multiplier),
        dp_curvature_penalty_multiplier=float(args.dp_curvature_penalty_multiplier),
        oracle_confidence_sharpening=float(args.oracle_confidence_sharpening),
        roi_upscale_factor=int(args.roi_upscale_factor),
        roi_upscale_method=str(args.roi_upscale_method),
        final_export_mode=str(args.final_export_mode),
        gt_json_path_for_metadata=(
            str(args.selective_oracle_rerank_gt)
            if args.selective_oracle_rerank_gt
            else (str(args.oracle_rerank_gt) if args.oracle_rerank_gt else None)
        ),
    )
    print(f"[DONE] confidence={result.confidence}, warnings={len(result.warnings)}")
    print(f"  -> {args.output_json_path}")
    print(f"  -> {args.debug_dir}/")


if __name__ == "__main__":
    main()
