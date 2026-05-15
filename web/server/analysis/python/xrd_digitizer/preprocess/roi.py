"""
$11.2: ROI crop + legend ignore box masking.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def crop_roi(
    image: Image.Image,
    plot_box: List[int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """plot_box [x0, y0, x1, y1] 기준 crop. 반환: (roi_array_RGB, plot_box_tuple)."""
    x0, y0, x1, y1 = int(plot_box[0]), int(plot_box[1]), int(plot_box[2]), int(plot_box[3])
    w, h = image.size
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    roi = arr[y0c:y1c, x0c:x1c].copy()
    return roi, (x0c, y0c, x1c, y1c)


def apply_legend_ignore(
    mask: np.ndarray,
    legend_ignore_boxes: Optional[List[List[int]]],
    plot_box: Tuple[int, int, int, int],
) -> np.ndarray:
    """legend_ignore_boxes 영역을 mask에서 0으로 클리어."""
    if not legend_ignore_boxes:
        return mask
    x0, y0, _, _ = plot_box
    out = mask.copy()
    for box in legend_ignore_boxes:
        if not box or len(box) < 4:
            continue
        bx0, by0, bx1, by1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        lx0 = max(0, bx0 - x0)
        ly0 = max(0, by0 - y0)
        lx1 = min(out.shape[1], bx1 - x0)
        ly1 = min(out.shape[0], by1 - y0)
        if lx1 > lx0 and ly1 > ly0:
            out[ly0:ly1, lx0:lx1] = 0
    return out
