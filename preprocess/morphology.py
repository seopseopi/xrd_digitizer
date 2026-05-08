"""
$11.7-11.8: morphology, raw candidate preservation, thinning.

- close kernel 3x3, iter 1
- area < 80 제거
- raw candidate mask 별도 저장 (thinning 전)
- thinning은 후보 정제 수단이지 유일한 원천이 아님
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, label


def apply_morphology_close(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """$11.8: close kernel 3x3, iter 1."""
    struct = np.ones((kernel_size, kernel_size), dtype=bool)
    closed = binary_closing(mask.astype(bool), structure=struct, iterations=iterations)
    return closed.astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = 68) -> np.ndarray:
    """$11.8: connected component area < min_area 제거."""
    labeled, n_labels = label(mask.astype(bool))
    if n_labels == 0:
        return mask.copy()

    areas = np.bincount(labeled.ravel())
    keep = np.zeros(n_labels + 1, dtype=bool)
    for i in range(1, n_labels + 1):
        if areas[i] >= min_area:
            keep[i] = True

    return (keep[labeled]).astype(np.uint8)


def save_raw_candidates(combined_mask: np.ndarray) -> np.ndarray:
    """$11.7: thinning 전 raw candidate 보존."""
    return combined_mask.copy()


def apply_thinning(mask: np.ndarray) -> np.ndarray:
    """
    $11.7: morphological thinning (Zhang-Suen).
    skimage 사용 가능하면 사용, 없으면 iterative erosion fallback.
    """
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
        return skeleton
    except ImportError:
        return _fallback_thinning(mask)


def _fallback_thinning(mask: np.ndarray) -> np.ndarray:
    """skimage 없을 때 간이 thinning (erosion 반복)."""
    from scipy.ndimage import binary_erosion
    current = mask.astype(bool).copy()
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for _ in range(20):
        eroded = binary_erosion(current, structure=struct)
        if np.array_equal(eroded, current) or eroded.sum() == 0:
            break
        current = eroded
    return current.astype(np.uint8)


def run_morphology_pipeline(combined_mask: np.ndarray) -> dict:
    """
    전처리 morphology 전체 파이프라인.
    Returns dict with: raw_candidate_mask, cleaned_mask, skeleton_mask
    """
    raw_candidates = save_raw_candidates(combined_mask)

    closed = apply_morphology_close(combined_mask)
    cleaned = remove_small_components(closed)

    skeleton = apply_thinning(cleaned)

    return {
        "raw_candidate_mask": raw_candidates,
        "cleaned_mask": cleaned,
        "skeleton_mask": skeleton,
    }
