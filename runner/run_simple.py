"""
Simple pixel-based curve runner.

깨끗한 이미지(단색 배경 + 얇은 곡선) 전용 — classical 파이프라인의
candidate / DP / morphology 등을 우회하고 픽셀에서 직접 곡선을 추출한다.

기존 run_local.py 와 동일한 JSON 출력 스키마(RunResult.to_dict()) 생성.

사용법:
  python -m runner.run_simple \\
      --image_path data/.../foo.png \\
      --manual_inputs_path data/.../foo_mi.json \\
      [--stdout] [--output_json_path out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline_versions import CALIBRATE_V1_1
from core.types import ManualInputs, RunResult
from core.io import load_image, load_manual_inputs, validate_manual_inputs, save_result_json
from preprocess.roi import crop_roi
from preprocess.perspective import correct_perspective
from preprocess.simple_trace import (
    extract_curve_simple,
    sample_curve_rgb,
    sample_background_rgb,
    adaptive_max_dist,
)
from trace.postprocess import (
    LOOSE_PEAK_PROMINENCE_FACTOR,
    gap_fill,
    smooth_trace,
    detect_peaks,
    repair_isolated_spike_down_y,
    restore_peaks_lowered_by_smoothing,
    blend_sg_toward_gapfill_on_high_curvature,
)
from calibrate.numeric_export import export_numeric


def _sanitize_nan(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def run_simple_pipeline(
    image: Image.Image,
    mi: ManualInputs,
    *,
    color_max_dist: float = -1.0,  # <=0 이면 bg↔curve 거리 기반 적응형
    pipeline_version: str = CALIBRATE_V1_1,
    curvature_blend_strength: float = 0.32,
    loose_peak_prominence_factor: Optional[float] = None,
    smooth: bool = False,
) -> RunResult:
    stage_timings: dict = {}
    warnings_list: list[str] = []

    # ROI + perspective
    t0 = time.perf_counter()
    roi, plot_box_t = crop_roi(image, mi.plot_box)
    stage_timings['roi_crop'] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    roi = correct_perspective(roi, mi.perspective_corners, plot_box_t)
    stage_timings['perspective'] = round(time.perf_counter() - t0, 6)

    roi_arr = np.asarray(roi)
    if roi_arr.ndim == 2:
        roi_arr = np.stack([roi_arr] * 3, axis=2)
    elif roi_arr.shape[2] == 4:
        roi_arr = roi_arr[:, :, :3]
    roi_h, roi_w = roi_arr.shape[:2]

    # Sample curve RGB + 적응형 max_dist 계산
    t0 = time.perf_counter()
    curve_rgb = sample_curve_rgb(
        roi_arr,
        (int(mi.color_sample_point[0]), int(mi.color_sample_point[1])),
        tuple(int(v) for v in mi.plot_box),
    )
    bg_rgb = sample_background_rgb(roi_arr)
    # color_max_dist > 0 이면 사용자 값 그대로; <=0 이면 적응형 계산
    eff_max_dist = (
        float(color_max_dist) if color_max_dist > 0
        else adaptive_max_dist(curve_rgb, bg_rgb, fraction=0.6, floor=80.0)
    )
    stage_timings['color_sample'] = round(time.perf_counter() - t0, 6)

    # Simple pixel trace
    t0 = time.perf_counter()
    trace_path = extract_curve_simple(roi_arr, curve_rgb, max_dist=eff_max_dist)
    stage_timings['simple_trace'] = round(time.perf_counter() - t0, 6)

    # postprocess: gap_fill / smoothing / peaks
    columns = list(range(roi_w))
    t0 = time.perf_counter()
    y_filled, valid_mask, gap_ranges = gap_fill(trace_path, columns)
    stage_timings['gap_fill'] = round(time.perf_counter() - t0, 6)

    y_filled = repair_isolated_spike_down_y(y_filled, valid_mask)

    gap_filled_set: set[int] = set()
    for gs, ge in gap_ranges:
        for gi in range(gs, ge):
            gap_filled_set.add(gi)

    loose_pf = (
        float(loose_peak_prominence_factor)
        if loose_peak_prominence_factor is not None
        else float(LOOSE_PEAK_PROMINENCE_FACTOR)
    )

    # simple 모드: smoothing은 sharp peak 정점을 평탄화한다 (window=5, 1-2px peak에 치명적).
    # 기본 off — DP fallback 같은 노이즈 원인이 없어 raw trace가 이미 충분히 부드러움.
    t0 = time.perf_counter()
    if smooth:
        y_smoothed, _ = smooth_trace(y_filled, valid_mask, roi_w)
        y_smoothed = blend_sg_toward_gapfill_on_high_curvature(
            y_filled, y_smoothed, valid_mask, strength=float(curvature_blend_strength),
        )
        y_smoothed = repair_isolated_spike_down_y(y_smoothed, valid_mask)
        y_smoothed = restore_peaks_lowered_by_smoothing(y_filled, y_smoothed, valid_mask)
    else:
        y_smoothed = y_filled.copy()
    stage_timings['smoothing'] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    peak_result = detect_peaks(
        y_smoothed,
        valid_mask,
        gap_filled_set,
        columns=columns,
        two_pass_prominence=True,
        loose_prominence_factor=loose_pf,
    )
    stage_timings['peak_detection'] = round(time.perf_counter() - t0, 6)

    n_peaks = len(peak_result['peaks'])
    n_major = len(peak_result['major_peaks'])
    if n_peaks == 0:
        warnings_list.append('postprocess: no peaks detected')

    # numeric export
    t0 = time.perf_counter()
    numeric = export_numeric(columns, y_smoothed, valid_mask, mi, peak_result)
    stage_timings['axis_map'] = round(time.perf_counter() - t0, 6)

    cal_meta = numeric['calibration_meta']
    rt_err = float(cal_meta['roundtrip_error']['total_mean_error_px'])
    cal_conf = float(cal_meta['roundtrip_error']['calibration_confidence'])

    valid_ratio = float(np.mean(valid_mask)) if len(valid_mask) > 0 else 0.0
    if valid_ratio < 0.5:
        warnings_list.append(f'trace: low valid_ratio {valid_ratio:.3f}')
    if rt_err > 1.0:
        warnings_list.append(f'calibration: roundtrip error {rt_err:.4f}px exceeds 1.0')

    return RunResult(
        two_theta_values=numeric['two_theta_values'],
        intensities=numeric['intensities'],
        x_range=numeric['x_range'],
        y_range=numeric['y_range'],
        quality={'pixel_residual_mean': rt_err, 'peak_match_score': None},
        confidence=cal_conf,
        warnings=warnings_list,
        peaks_numeric_curve=numeric.get('peaks_numeric_curve', []),
        model_assist={
            'source': 'simple_pixel_trace',
            'pipeline_version': f'simple_{pipeline_version}',
            'stage_timings': stage_timings,
            'roi_shape': [int(roi_h), int(roi_w)],
            'curve_rgb': list(curve_rgb),
            'bg_rgb': list(bg_rgb),
            'color_max_dist_eff': round(eff_max_dist, 2),
            'color_max_dist_arg': float(color_max_dist),
            'n_peaks': int(n_peaks),
            'n_major_peaks': int(n_major),
            'valid_ratio': round(valid_ratio, 4),
        },
        used_manual_inputs={
            'plot_box': mi.plot_box,
            'x_axis_points': mi.x_axis_points,
            'x_axis_values': mi.x_axis_values,
            'y_axis_points': mi.y_axis_points,
            'y_axis_values': mi.y_axis_values,
            'color_sample_point': mi.color_sample_point,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description='Simple pixel-based XRD curve runner')
    p.add_argument('--image_path',         type=str, required=True)
    p.add_argument('--manual_inputs_path', type=str, required=True)
    p.add_argument('--output_json_path',   type=str, default=None)
    p.add_argument('--color-max-dist',     type=float, default=-1.0,
                   help='>0 이면 고정값 사용; <=0 이면 bg↔curve 거리의 60% 적응형 (기본)')
    p.add_argument('--smooth', action='store_true', help='SG smoothing 활성화 (기본 off — sharp peak 보존)')
    p.add_argument('--stdout', action='store_true')
    p.add_argument('--no-debug', action='store_true', dest='no_debug')
    args = p.parse_args()

    image  = load_image(args.image_path)
    mi     = load_manual_inputs(args.manual_inputs_path)
    issues = validate_manual_inputs(mi, (image.width, image.height))
    for iss in issues:
        print(f'[WARN] {iss}', file=sys.stderr)

    result = run_simple_pipeline(
        image, mi,
        color_max_dist=float(args.color_max_dist),
        smooth=bool(args.smooth),
    )

    print(f'[DONE] confidence={result.confidence}, peaks={len(result.peaks_numeric_curve)}, warnings={len(result.warnings)}',
          file=sys.stderr)

    if args.stdout:
        print(json.dumps(_sanitize_nan(result.to_dict()), ensure_ascii=False, indent=2))
    elif args.output_json_path:
        save_result_json(result, args.output_json_path)
        print(f'  -> {args.output_json_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
