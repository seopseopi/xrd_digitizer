"""
$12.2: thinning / skeletonization.

- Primary: skimage skeletonize (Zhang-Suen equivalent)
- Fallback: morphological erosion-based centerline
- thinning 결과는 candidate source 중 하나일 뿐 (유일한 원천 아님)
"""

from __future__ import annotations

import numpy as np


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """binary mask -> 1px skeleton."""
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask.astype(bool)).astype(np.uint8)
    except ImportError:
        return _fallback_centerline(mask)


def _fallback_centerline(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    current = mask.astype(bool).copy()
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    prev = current.copy()
    for _ in range(50):
        eroded = binary_erosion(current, structure=struct)
        if eroded.sum() == 0:
            break
        prev = current.copy()
        current = eroded
    return prev.astype(np.uint8)


def build_axis_proximity_map(roi_shape: tuple, plot_box: tuple, axis_width: int = 4) -> np.ndarray:
    """
    축/틱 영역 proximity map (거리가 가까울수록 값이 큼).
    penalty 계산용. 반환: [H, W] float, 각 픽셀에서 가장 가까운 axis/tick까지의 거리.
    """
    h, w = roi_shape[:2]
    x0, y0, x1, y1 = plot_box
    rh, rw = y1 - y0, x1 - x0

    axis_mask = np.zeros((h, w), dtype=np.uint8)
    axis_mask[:axis_width, :] = 1           # top edge
    axis_mask[-axis_width:, :] = 1          # bottom edge (x-axis)
    axis_mask[:, :axis_width] = 1           # left edge (y-axis)
    axis_mask[:, -axis_width:] = 1          # right edge

    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(1 - axis_mask)
    return dist.astype(np.float64)
