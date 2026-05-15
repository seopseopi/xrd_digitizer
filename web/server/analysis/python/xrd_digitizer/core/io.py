"""
§9: 엔진 I/O 유틸리티.

- 이미지 로딩
- ManualInputs JSON 로딩/검증
- RunResult JSON 저장
- Debug 파일 저장
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.types import ManualInputs, RunResult, DEBUG_OUTPUTS

# run_pipeline 의사 순서: 전처리 → 후보·추적 → (보조) → 후처리·피크 → 산출 오버레이 → 로그 JSON
# debug.json 은 하단에서 따로 저장(파일명 고정, 평가기 호환).
DEBUG_ARTIFACT_SAVE_ORDER: List[str] = [
    "roi_preview",
    "color_mask",
    "combined_mask",
    "raw_candidate_mask",
    "pre_skeleton_candidates",
    "skeleton",
    "components_overlay",
    "candidate_map_raw",
    "candidate_map_filtered",
    "candidate_map_final",
    "trace_path",
    "branch_compare",
    "contrast_aux_map",
    "trace_path_contrast_aux_overlay",
    "recovery_candidates_before",
    "recovery_candidates_after",
    "smooth_for_curve_overlay",
    "smooth_for_peak_overlay",
    "detected_peaks_peak_smooth",
    "near_peak_mask",
    "apex_snapping_overlay",
    "smoothed_trace",
    "peaks_overlay",
    "numeric_curve_peaks_roi",
    "candidate_conf_before_after",
    "contrast_aux_ab_log",
    "sharp_peak_preserve_debug",
]


def _ordered_debug_keys(debug_data: Dict) -> List[str]:
    """번호 부여 순서: 알려진 파이프라인 순 → 나머지는 dict 삽입 순."""
    seen = set()
    out: List[str] = []
    for k in DEBUG_ARTIFACT_SAVE_ORDER:
        if k in debug_data and k != "debug.json":
            out.append(k)
            seen.add(k)
    for k in debug_data:
        if k != "debug.json" and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def load_image(image_path: str) -> Image.Image:
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(str(p)).convert("RGB")
    img.load()
    return img


def load_manual_inputs(json_path: str) -> ManualInputs:
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manual inputs JSON not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "plot_box", "x_axis_points", "x_axis_values",
        "y_axis_points", "y_axis_values", "color_sample_point",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Manual inputs missing required keys: {missing}")

    return ManualInputs(
        plot_box=data["plot_box"],
        x_axis_points=data["x_axis_points"],
        x_axis_values=data["x_axis_values"],
        y_axis_points=data["y_axis_points"],
        y_axis_values=data["y_axis_values"],
        color_sample_point=data["color_sample_point"],
        legend_ignore_boxes=data.get("legend_ignore_boxes"),
        perspective_corners=data.get("perspective_corners"),
        color_resample_points=data.get("color_resample_points"),
        export_resample_points=data.get("export_resample_points"),
    )


def validate_manual_inputs(mi: ManualInputs, image_size: Tuple[int, int]) -> List[str]:
    """§10.3 에러 규칙 검증. 위반 목록을 반환 (빈 리스트 = 정상)."""
    errors: List[str] = []
    w, h = image_size

    pb = mi.plot_box
    if not (isinstance(pb, (list, tuple)) and len(pb) == 4):
        errors.append("plot_box must have 4 elements [x0, y0, x1, y1]")
    else:
        x0, y0, x1, y1 = pb
        if x1 <= x0 or y1 <= y0:
            errors.append("plot_box has zero or negative area (ROI empty)")
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            errors.append("plot_box extends outside image bounds")

    if len(mi.x_axis_points) != len(mi.x_axis_values):
        errors.append("x_axis_points and x_axis_values length mismatch")
    if len(mi.x_axis_points) < 2:
        errors.append("x_axis_points requires at least 2 points")
    elif len(mi.x_axis_points) >= 2:
        px_xs = [p[0] for p in mi.x_axis_points if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(px_xs) >= 2 and abs(px_xs[0] - px_xs[-1]) < 1:
            errors.append("x_axis_points have identical pixel x - axis mapping impossible")
        if len(mi.x_axis_values) >= 2 and abs(float(mi.x_axis_values[0]) - float(mi.x_axis_values[-1])) < 1e-12:
            errors.append("x_axis_values are identical - degenerate x axis")

    if len(mi.y_axis_points) != len(mi.y_axis_values):
        errors.append("y_axis_points and y_axis_values length mismatch")
    if len(mi.y_axis_points) < 2:
        errors.append("y_axis_points requires at least 2 points")
    elif len(mi.y_axis_points) >= 2:
        px_ys = [p[1] for p in mi.y_axis_points if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(px_ys) >= 2 and abs(px_ys[0] - px_ys[-1]) < 1:
            errors.append("y_axis_points have identical pixel y - axis mapping impossible")
        if len(mi.y_axis_values) >= 2 and abs(float(mi.y_axis_values[0]) - float(mi.y_axis_values[-1])) < 1e-12:
            errors.append("y_axis_values are identical - degenerate y axis")

    csp = mi.color_sample_point
    if isinstance(csp, (list, tuple)) and len(csp) == 2 and isinstance(pb, (list, tuple)) and len(pb) == 4:
        cx, cy = csp
        x0, y0, x1, y1 = pb
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            errors.append("color_sample_point is outside plot_box (ROI)")

    return errors


def save_result_json(result: RunResult, output_path: str) -> None:
    """원자적 쓰기: 임시 파일 -> rename."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        if p.exists():
            p.unlink()
        Path(tmp).rename(p)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_debug_files(debug_data: Dict, debug_dir: str) -> None:
    """§9.4 / §11.9 debug 출력 저장.

    PNG·부가 JSON 은 파이프라인 순으로 ``NN_key.ext`` (NN=01,02,…) 접두사를 붙인다.
    ``debug.json`` 만 파일명 고정(eval/report 등이 경로에 의존).
    """
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)

    if "debug.json" in debug_data:
        with (d / "debug.json").open("w", encoding="utf-8") as f:
            json.dump(debug_data["debug.json"], f, ensure_ascii=False, indent=2)

    idx = 1
    for key in _ordered_debug_keys(debug_data):
        val = debug_data[key]
        prefix = f"{idx:02d}_"
        idx += 1
        if isinstance(val, Image.Image):
            val.save(str(d / f"{prefix}{key}.png"), format="PNG")
        elif isinstance(val, np.ndarray):
            Image.fromarray(val).save(str(d / f"{prefix}{key}.png"), format="PNG")
        elif isinstance(val, (dict, list)):
            with (d / f"{prefix}{key}.json").open("w", encoding="utf-8") as f:
                json.dump(val, f, ensure_ascii=False, indent=2)
