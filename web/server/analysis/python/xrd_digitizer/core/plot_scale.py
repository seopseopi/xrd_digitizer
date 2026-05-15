"""§4.1 plot_w, plot_h and normalized scales s_w, s_h, s."""

from __future__ import annotations

import math
from typing import Any, Dict


def compute_plot_scale(plot_box_roi: tuple[int, int, int, int] | list[int]) -> Dict[str, Any]:
    """ROI-local plot box: (x0, y0, x1, y1) inclusive."""
    x0, y0, x1, y1 = [int(v) for v in plot_box_roi]
    plot_w = x1 - x0 + 1
    plot_h = y1 - y0 + 1
    s_w = plot_w / 950.0
    s_h = plot_h / 690.0
    s = math.sqrt(s_w * s_h)
    return {
        "plot_w": plot_w,
        "plot_h": plot_h,
        "s_w": float(s_w),
        "s_h": float(s_h),
        "s": float(s),
    }
