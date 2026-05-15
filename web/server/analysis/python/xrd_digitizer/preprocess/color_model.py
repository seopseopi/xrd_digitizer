"""
$11.4: Lab color-space prototype model.

- color space: Lab
- prototype 3+: ROI x 20%, 50%, 80% + color_resample_points
- neighborhood 5x5
- 적응형 threshold: prototype 주변 통계 기반
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

_RGB2LAB_CACHE: dict = {}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIE Lab float64. Shape preserved."""
    arr = rgb.astype(np.float64) / 255.0

    mask = arr > 0.04045
    arr = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    z /= 1.08883

    def _f(t):
        mask_t = t > 0.008856
        return np.where(mask_t, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)

    return np.stack([L, a, b_ch], axis=-1)


def rgb_uint8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 HxWx3 -> CIE Lab float64 HxWx3 (내부 변환 공유)."""
    return _rgb_to_lab(rgb_u8)


def _extract_patch_mean(roi_lab: np.ndarray, cy: int, cx: int, radius: int = 2) -> np.ndarray:
    """5x5 (radius=2) neighborhood mean in Lab."""
    h, w = roi_lab.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    patch = roi_lab[y0:y1, x0:x1]
    return np.mean(patch.reshape(-1, 3), axis=0)


def _search_nearby_dark(roi_lab: np.ndarray, cy: int, cx: int, radius: int = 30) -> np.ndarray:
    """color_sample_point가 배경(밝음)이면 주변에서 가장 어두운 픽셀을 찾는다."""
    h, w = roi_lab.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    patch = roi_lab[y0:y1, x0:x1]
    L_vals = patch[:, :, 0]
    min_idx = np.unravel_index(np.argmin(L_vals), L_vals.shape)
    best_y = y0 + min_idx[0]
    best_x = x0 + min_idx[1]
    return _extract_patch_mean(roi_lab, best_y, best_x, radius=2)


def _find_curve_y_at_x(roi_lab: np.ndarray, ref_lab: np.ndarray, cx: int, search_margin: int = 40) -> int:
    """x=cx 열에서 ref_lab에 가장 가까운 픽셀 y를 찾는다."""
    h = roi_lab.shape[0]
    col = roi_lab[:, cx, :]
    dists = np.linalg.norm(col - ref_lab[None, :], axis=1)
    return int(np.argmin(dists))


def build_color_prototypes(
    roi: np.ndarray,
    color_sample_point: List[int],
    plot_box: Tuple[int, int, int, int],
    color_resample_points: Optional[List[List[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: (prototypes [N,3] Lab, roi_lab [H,W,3], adaptive_threshold)
    """
    roi_lab = _rgb_to_lab(roi)
    h, w = roi.shape[:2]
    x0_pb, y0_pb, _, _ = plot_box

    csp_local_x = int(color_sample_point[0]) - x0_pb
    csp_local_y = int(color_sample_point[1]) - y0_pb
    csp_local_x = max(0, min(w - 1, csp_local_x))
    csp_local_y = max(0, min(h - 1, csp_local_y))

    ref_lab = _extract_patch_mean(roi_lab, csp_local_y, csp_local_x, radius=2)

    if ref_lab[0] > 85.0:
        ref_lab = _search_nearby_dark(roi_lab, csp_local_y, csp_local_x, radius=30)

    target_xs = [int(w * 0.2), int(w * 0.5), int(w * 0.8)]
    proto_positions = []
    for tx in target_xs:
        tx = max(0, min(w - 1, tx))
        best_y = _find_curve_y_at_x(roi_lab, ref_lab, tx)
        proto_positions.append((best_y, tx))

    if color_resample_points:
        for pt in color_resample_points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                rx = max(0, min(w - 1, int(pt[0]) - x0_pb))
                ry = max(0, min(h - 1, int(pt[1]) - y0_pb))
                proto_positions.append((ry, rx))

    protos = []
    for cy, cx in proto_positions:
        cy = max(0, min(h - 1, cy))
        cx = max(0, min(w - 1, cx))
        mean_lab = _extract_patch_mean(roi_lab, cy, cx, radius=2)
        protos.append(mean_lab)

    prototypes = np.array(protos, dtype=np.float64)

    bg_lab = np.median(roi_lab.reshape(-1, 3), axis=0)
    proto_to_bg_dists = [float(np.linalg.norm(p - bg_lab)) for p in prototypes]
    min_gap = float(np.min(proto_to_bg_dists)) if proto_to_bg_dists else 50.0
    adaptive_threshold = min(13.4, max(11.6, min_gap * 0.32))

    return prototypes, roi_lab, adaptive_threshold


def compute_color_distance_map(
    roi_lab: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    """각 픽셀의 nearest prototype Lab distance. Shape: [H, W]."""
    h, w = roi_lab.shape[:2]
    flat = roi_lab.reshape(-1, 3)
    dists = np.full(flat.shape[0], np.inf, dtype=np.float64)
    for proto in prototypes:
        d = np.linalg.norm(flat - proto[None, :], axis=1)
        dists = np.minimum(dists, d)
    return dists.reshape(h, w)
