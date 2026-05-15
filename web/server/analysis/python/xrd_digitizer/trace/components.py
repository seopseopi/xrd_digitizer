"""
$12.3: connected component labeling + scoring.

score = 2.0*x_coverage + 1.0*continuity + 0.5*log(1+length)
      - 1.2*edge_penalty - 1.0*text_penalty
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import label


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """binary mask -> labeled array + count."""
    labeled, n = label(mask.astype(bool))
    return labeled.astype(np.int32), int(n)


def compute_component_scores(
    labeled: np.ndarray,
    n_components: int,
    axis_dist_map: np.ndarray,
) -> Dict[int, dict]:
    """
    $12.3 score 계산.
    Returns: {comp_id: {score, x_coverage, continuity, length, ...}}
    """
    h, w = labeled.shape[:2]
    results: Dict[int, dict] = {}

    for cid in range(1, n_components + 1):
        ys, xs = np.where(labeled == cid)
        if len(xs) == 0:
            continue

        length = len(xs)
        x_min_c, x_max_c = int(xs.min()), int(xs.max())
        x_span = x_max_c - x_min_c + 1
        x_coverage = float(x_span) / float(w) if w > 0 else 0.0

        unique_cols = np.unique(xs)
        filled_frac = float(len(unique_cols)) / float(max(1, x_span))
        continuity = filled_frac

        edge_margin = 4
        edge_pixels = int(np.sum(
            (xs < edge_margin) | (xs >= w - edge_margin) |
            (ys < edge_margin) | (ys >= h - edge_margin)
        ))
        edge_penalty = float(edge_pixels) / float(max(1, length))

        close_to_axis = int(np.sum(axis_dist_map[ys, xs] < 5.0))
        text_penalty = float(close_to_axis) / float(max(1, length))

        score = (
            2.0 * x_coverage
            + 1.0 * continuity
            + 0.5 * float(np.log(1.0 + length))
            - 1.2 * edge_penalty
            - 1.0 * text_penalty
        )

        results[cid] = {
            "score": float(score),
            "x_coverage": float(x_coverage),
            "continuity": float(continuity),
            "length": int(length),
            "edge_penalty": float(edge_penalty),
            "text_penalty": float(text_penalty),
            "x_range": (int(x_min_c), int(x_max_c)),
        }

    return results


def build_component_score_map(
    labeled: np.ndarray,
    scores: Dict[int, dict],
) -> np.ndarray:
    """각 픽셀에 해당 component score를 매핑. shape: [H,W] float."""
    out = np.zeros(labeled.shape, dtype=np.float64)
    for cid, info in scores.items():
        out[labeled == cid] = info["score"]
    return out
