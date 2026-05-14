"""
ML 곡선 추출 러너.

CurveExtractorNet (ResNet-18 + FPN + soft-argmax) 으로 곡선을 추출하고
기존 파이프라인의 gap_fill / smoothing / peak detection / numeric export 를 재사용.

사용법:
  python -m runner.run_ml_curve \\
      --image_path data/rendered_styled/pattern_1076_styled_v1.png \\
      --manual_inputs_path data/manifests/pattern_1076_manual_inputs.json \\
      [--weights ml/weights/curve_extractor.pt] \\
      [--stdout] [--output_json_path out.json] [--debug_dir out_debug/]

기존 run_local.py 와 동일한 JSON 출력 스키마(RunResult.to_dict())를 생성.
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
from ml.infer_curve_extractor import CurveExtractorInfer

_DEFAULT_WEIGHTS = ROOT / 'ml' / 'weights' / 'curve_extractor.pt'


# ── 추론 + 후처리 ─────────────────────────────────────────────────────────────

def run_ml_pipeline(
    image: Image.Image,
    mi: ManualInputs,
    *,
    weights_path: Optional[Path] = None,
    pipeline_version: str = CALIBRATE_V1_1,
    peak_two_pass: bool = True,
    curvature_blend_strength: float = 0.32,
    loose_peak_prominence_factor: Optional[float] = None,
) -> tuple[RunResult, dict]:
    """
    ML 곡선 추출 파이프라인.
    run_pipeline (classical) 과 동일한 (RunResult, debug_data) 튜플을 반환.
    """
    stage_timings: dict = {}
    warnings_list: list[str] = []

    # ── Step 11: preprocess ────────────────────────────────────────────────
    t0 = time.perf_counter()
    roi, plot_box_t = crop_roi(image, mi.plot_box)
    stage_timings['roi_crop'] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    roi = correct_perspective(roi, mi.perspective_corners, plot_box_t)
    stage_timings['perspective'] = round(time.perf_counter() - t0, 6)

    roi_h, roi_w = roi.shape[:2]

    # ── Step ML: curve inference ────────────────────────────────────────────
    t0 = time.perf_counter()
    wp = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
    inf = CurveExtractorInfer(weights_path=wp)
    trace_path_raw = inf.infer(roi)   # list[float], len = roi_w
    stage_timings['ml_infer'] = round(time.perf_counter() - t0, 6)

    # trace_path format: list[Optional[float]] 길이 roi_w
    trace_path: list[Optional[float]] = [float(y) for y in trace_path_raw]

    # ── Step 15: gap fill / smoothing / peak detection ──────────────────────
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

    t0 = time.perf_counter()
    y_smoothed, sg_window = smooth_trace(y_filled, valid_mask, roi_w)
    y_smoothed = blend_sg_toward_gapfill_on_high_curvature(
        y_filled, y_smoothed, valid_mask, strength=float(curvature_blend_strength),
    )
    y_smoothed = repair_isolated_spike_down_y(y_smoothed, valid_mask)
    y_smoothed = restore_peaks_lowered_by_smoothing(y_filled, y_smoothed, valid_mask)
    stage_timings['smoothing'] = round(time.perf_counter() - t0, 6)

    t0 = time.perf_counter()
    peak_result = detect_peaks(
        y_smoothed,
        valid_mask,
        gap_filled_set,
        columns=columns,
        two_pass_prominence=peak_two_pass,
        loose_prominence_factor=loose_pf,
    )
    stage_timings['peak_detection'] = round(time.perf_counter() - t0, 6)

    n_peaks = len(peak_result['peaks'])
    n_major = len(peak_result['major_peaks'])
    if n_peaks == 0:
        warnings_list.append('postprocess: no peaks detected')

    # ── Step 16: pixel → numeric ────────────────────────────────────────────
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

    result = RunResult(
        two_theta_values=numeric['two_theta_values'],
        intensities=numeric['intensities'],
        x_range=numeric['x_range'],
        y_range=numeric['y_range'],
        quality={'pixel_residual_mean': rt_err, 'peak_match_score': None},
        confidence=cal_conf,
        warnings=warnings_list,
        peaks_numeric_curve=numeric.get('peaks_numeric_curve', []),
        model_assist={
            'source': 'ml_curve_extractor',
            'pipeline_version': f'ml_curve_{pipeline_version}',
            'stage_timings': stage_timings,
            'roi_shape': [int(roi_h), int(roi_w)],
            'n_peaks': int(n_peaks),
            'n_major_peaks': int(n_major),
            'valid_ratio': round(valid_ratio, 4),
            'ml_weights': str(wp),
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

    debug_data = dict(
        stage_timings=stage_timings,
        roi_shape=[int(roi_h), int(roi_w)],
    )

    return result, debug_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def _sanitize_nan(obj):
    """JSON에서 허용되지 않는 NaN/Inf float를 None으로 교체한다."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description='ML CurveExtractorNet runner')
    p.add_argument('--image_path',         type=str, required=True)
    p.add_argument('--manual_inputs_path', type=str, required=True)
    p.add_argument('--output_json_path',   type=str, default=None)
    p.add_argument('--debug_dir',          type=str, default=None)
    p.add_argument('--weights',            type=str, default=None,
                   help=f'가중치 경로 (기본: {_DEFAULT_WEIGHTS})')
    p.add_argument('--stdout', action='store_true',
                   help='결과 JSON 을 stdout 으로 출력')
    p.add_argument('--no-debug', action='store_true', dest='no_debug')
    p.add_argument('--peak-single-pass', action='store_true', dest='peak_single_pass')
    p.add_argument('--curvature-blend-strength', type=float, default=0.32, dest='curvature_blend_strength')
    p.add_argument('--peak-loose-prominence-factor', type=float, default=None, dest='loose_peak_pf')
    args = p.parse_args()

    image   = load_image(args.image_path)
    mi      = load_manual_inputs(args.manual_inputs_path)
    issues  = validate_manual_inputs(mi, (image.width, image.height))
    if issues:
        for iss in issues:
            print(f'[WARN] {iss}', file=sys.stderr)

    weights = Path(args.weights) if args.weights else None

    result, debug_data = run_ml_pipeline(
        image, mi,
        weights_path=weights,
        peak_two_pass=not args.peak_single_pass,
        curvature_blend_strength=float(args.curvature_blend_strength),
        loose_peak_prominence_factor=args.loose_peak_pf,
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
