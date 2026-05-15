"""
contrast_aux_v1: Lab L 기반 local background 대비 맵 → 후보 confidence 보조만 (후보 생성·DP 비용 불변).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

from core.contrast_aux_settings import ContrastAuxSettings, DEFAULT_CONTRAST_AUX_SETTINGS
from preprocess.color_model import rgb_uint8_to_lab


def _kernel_size_plot_width(plot_width: int, ratio: float) -> int:
    k = int(round(ratio * float(plot_width)))
    if k % 2 == 0:
        k += 1
    return max(31, k)


def build_contrast_aux_map(
    image: np.ndarray,
    plot_box: Tuple[int, int, int, int],
    legend_ignore_boxes: Optional[List[List[int]]] = None,
    cfg: Optional[ContrastAuxSettings] = None,
    *,
    legend_crop_origin: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    RGB ROI 이미지와 plot_box(ROI 내 반개구간 [x0,x1)×[y0,y1)) 기준 대비 점수 맵 [0,1] float32.

    legend_ignore_boxes: 원본 이미지 좌표계 박스 → legend_crop_origin 을 빼 ROI 좌표로 변환 후 0 처리.
    """
    cfg = cfg or DEFAULT_CONTRAST_AUX_SETTINGS
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB uint8")
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in plot_box]
    x0 = max(0, min(x0, w))
    x1 = max(x0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(y0, min(y1, h))

    lab = rgb_uint8_to_lab(image.astype(np.uint8))
    L = lab[:, :, 0].astype(np.float64)

    plot_width = max(1, x1 - x0)
    ksz = _kernel_size_plot_width(plot_width, cfg.contrast_aux_bg_kernel_ratio)
    bg_L = median_filter(L, size=ksz, mode="nearest")
    dark_contrast = np.maximum(0.0, bg_L - L)

    inner = np.zeros((h, w), dtype=bool)
    inner[y0:y1, x0:x1] = True
    roi_vals = dark_contrast[inner]
    if roi_vals.size < 16:
        p05 = float(np.min(roi_vals)) if roi_vals.size else 0.0
        p95 = float(np.max(roi_vals)) if roi_vals.size else 1.0
    else:
        p05 = float(np.percentile(roi_vals, 5))
        p95 = float(np.percentile(roi_vals, 95))
    denom = p95 - p05 + 1e-8
    contrast_score = np.clip((dark_contrast - p05) / denom, 0.0, 1.0).astype(np.float32)

    contrast_score[~inner] = 0.0

    bpx = max(0, int(cfg.contrast_aux_border_suppress_px))
    if bpx > 0 and x1 > x0 and y1 > y0:
        bx0, bx1 = x0 + bpx, x1 - bpx
        by0, by1 = y0 + bpx, y1 - bpx
        if bx1 > bx0 and by1 > by0:
            border_strip = np.zeros_like(inner, dtype=bool)
            border_strip[y0:y1, x0:x1] = True
            border_strip[by0:by1, bx0:bx1] = False
            contrast_score[border_strip] = 0.0
        else:
            contrast_score[inner] = 0.0

    ox, oy = legend_crop_origin
    if legend_ignore_boxes:
        for box in legend_ignore_boxes:
            if not box or len(box) < 4:
                continue
            bx0, by0, bx1, by1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            lx0 = max(0, bx0 - ox)
            ly0 = max(0, by0 - oy)
            lx1 = min(w, bx1 - ox)
            ly1 = min(h, by1 - oy)
            if lx1 > lx0 and ly1 > ly0:
                contrast_score[ly0:ly1, lx0:lx1] = 0.0

    return contrast_score


def blend_confidence_with_contrast_aux(
    base_conf: float,
    col: int,
    row: int,
    contrast_aux_map: np.ndarray,
    cfg: ContrastAuxSettings,
) -> Tuple[float, float]:
    """(final_conf, contrast_bonus). base 낮으면 보정 없음."""
    h, w = contrast_aux_map.shape[:2]
    if row < 0 or col < 0 or row >= h or col >= w:
        return float(base_conf), 0.0
    bonus = float(contrast_aux_map[row, col])
    bc = float(base_conf)
    if bc < cfg.contrast_aux_min_base_conf:
        return bc, bonus
    wgt = float(cfg.contrast_aux_weight)
    final = (1.0 - wgt) * bc + wgt * bonus
    return float(np.clip(final, 0.0, 1.0)), bonus


def apply_contrast_aux_to_raw_candidates(
    raw_candidates: Dict[int, List[dict]],
    contrast_aux_map: np.ndarray,
    cfg: ContrastAuxSettings,
) -> Dict[str, float]:
    """
    각 후보의 confidence를 보정한다. base_confidence 필드 추가 후 재정렬.
    반환: 통계 dict (candidate_conf_before_after.json 용).
    """
    sum_base = 0.0
    sum_final = 0.0
    sum_bonus = 0.0
    n_cand = 0
    n_bonus_applied = 0
    n_base_low = 0

    for col, cands in raw_candidates.items():
        for c in cands:
            base_conf = float(c["confidence"])
            y = int(c["y"])
            final_conf, bonus = blend_confidence_with_contrast_aux(
                base_conf, int(col), y, contrast_aux_map, cfg,
            )
            c["base_confidence"] = base_conf
            c["contrast_bonus"] = bonus
            c["confidence"] = final_conf
            sum_base += base_conf
            sum_final += final_conf
            sum_bonus += bonus
            n_cand += 1
            if base_conf < cfg.contrast_aux_min_base_conf:
                n_base_low += 1
            else:
                n_bonus_applied += 1

        cands.sort(key=lambda x: -x["confidence"])

    mean_bc = float(sum_base / max(n_cand, 1))
    mean_fc = float(sum_final / max(n_cand, 1))
    mean_bo = float(sum_bonus / max(n_cand, 1))
    return {
        "mean_base_conf": round(mean_bc, 6),
        "mean_final_conf": round(mean_fc, 6),
        "mean_contrast_bonus": round(mean_bo, 6),
        "num_candidates": float(n_cand),
        "num_candidates_bonus_applied": float(n_bonus_applied),
        "num_candidates_base_conf_too_low": float(n_base_low),
    }


def render_trace_on_contrast_aux_map(
    contrast_aux_map: np.ndarray,
    path: List[Optional[int]],
) -> np.ndarray:
    """대비 맵을 회색 배경으로 두고 trace path를 녹색 계열로 표시."""
    h, w = contrast_aux_map.shape[:2]
    g = np.clip(contrast_aux_map * 255.0, 0, 255).astype(np.uint8)
    overlay = np.stack([g, g, g], axis=-1).copy()
    for col, y in enumerate(path):
        if y is None or col >= w:
            continue
        yi = int(y)
        if yi < 0 or yi >= h:
            continue
        for dy in range(-1, 2):
            yy = yi + dy
            if 0 <= yy < h:
                overlay[yy, col] = np.array([30, 255, 80], dtype=np.uint8)
    return overlay


def contrast_aux_ab_reference() -> Dict[str, object]:
    """debug.json / 로그용 성공·폐기 판정 기준 텍스트."""
    return {
        "success_any_of": [
            "curve_y_mae_px mean 또는 median 5%+ 개선",
            "max_gap_px median 10%+ 개선",
            "tail_mae_px 8%+ 개선",
            "styled set peak_recall +0.03 이상",
        ],
        "discard_any_of": [
            "major_peak_x_error 5%+ 악화",
            "clean curve_y_mae_px 3%+ 악화",
            "grid_confusion failure count 증가",
            "legend_capture failure count 증가",
            "wrong_branch_lock_in 증가",
        ],
        "compare_protocol": "동일 manifest, 동일 sample_id, 동일 manual input; baseline --use-contrast-aux false vs experiment true",
    }
