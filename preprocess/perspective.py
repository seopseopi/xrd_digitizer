"""
$11.3: perspective correction via homography.
- perspective_corners 있으면 적용, 없으면 pass-through.
- 자동 perspective 탐지 없음 (manual only).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def correct_perspective(
    roi: np.ndarray,
    perspective_corners: Optional[List[List[float]]],
    plot_box: tuple,
) -> np.ndarray:
    """perspective_corners가 제공되면 ROI에 homography 보정 적용."""
    if perspective_corners is None or len(perspective_corners) != 4:
        return roi

    h, w = roi.shape[:2]
    x0, y0, _, _ = plot_box

    src = np.array(
        [[c[0] - x0, c[1] - y0] for c in perspective_corners],
        dtype=np.float64,
    )
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float64,
    )

    H = _homography_from_4pts(src, dst)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return roi

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    hom = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)], axis=0)
    src_h = Hi @ hom
    denom = np.clip(src_h[2:3, :], 1e-9, None)
    src_h /= denom
    sx = src_h[0].reshape(h, w)
    sy = src_h[1].reshape(h, w)

    out = np.full_like(roi, 255, dtype=np.uint8)
    for c in range(roi.shape[2] if roi.ndim == 3 else 1):
        ch = roi[:, :, c] if roi.ndim == 3 else roi
        out_ch = _bilinear_sample(ch.astype(np.float64), sx, sy, 255.0)
        if roi.ndim == 3:
            out[:, :, c] = np.clip(out_ch, 0, 255).astype(np.uint8)
        else:
            out = np.clip(out_ch, 0, 255).astype(np.uint8)
    return out


def _homography_from_4pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        A.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    amat = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(amat)
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    return H


def _bilinear_sample(ch: np.ndarray, sx: np.ndarray, sy: np.ndarray, fill: float) -> np.ndarray:
    h, w = ch.shape
    out = np.full_like(sx, fill, dtype=np.float64)
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    xi = np.floor(sx[valid]).astype(np.int32)
    yi = np.floor(sy[valid]).astype(np.int32)
    xf = sx[valid] - xi
    yf = sy[valid] - yi
    x1 = np.clip(xi + 1, 0, w - 1)
    y1 = np.clip(yi + 1, 0, h - 1)
    out[valid] = (
        (1 - xf) * (1 - yf) * ch[yi, xi]
        + xf * (1 - yf) * ch[yi, x1]
        + (1 - xf) * yf * ch[y1, xi]
        + xf * yf * ch[y1, x1]
    )
    return out
