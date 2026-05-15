"""

디버그용: 수출 기반 peaks_numeric_curve 를 ROI 이미지 위에 표시.

"""



from __future__ import annotations



from typing import Dict, List



import numpy as np



from calibrate.axis_mapping import value_to_pixel_x, value_to_pixel_y





def render_numeric_peaks_on_roi(

    roi: np.ndarray,

    x_map: Dict,

    y_map: Dict,

    peaks: List[dict],

    *,

    max_marks: int = 220,

    upscale_factor: int = 1,

) -> np.ndarray:

    """ROI(행=h, 열=w) 위에 자홍색 십자로 피크 위치 표시."""

    h, w = roi.shape[:2]

    if roi.ndim == 2:

        base = np.stack([roi] * 3, axis=-1).astype(np.uint8)

    else:

        base = np.asarray(roi).astype(np.uint8).copy()

    arm = max(6, 12 * upscale_factor)  # 십자 팔 길이

    for p in peaks[:max_marks]:

        cx = int(round(value_to_pixel_x(float(p["two_theta"]), x_map)))

        cy = int(round(value_to_pixel_y(float(p["intensity"]), y_map)))

        if not (0 <= cx < w and 0 <= cy < h):

            continue

        for d in range(-arm, arm + 1):

            if 0 <= cx + d < w:

                base[cy, cx + d] = [255, 0, 255]

            if 0 <= cy + d < h:

                base[cy + d, cx] = [255, 0, 255]

    return base

