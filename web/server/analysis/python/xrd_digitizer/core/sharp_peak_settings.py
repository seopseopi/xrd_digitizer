"""sharp peak preserve v1: 후처리만 분리 스무딩·국소 prominence·raw 블렌드."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharpPeakPreserveSettings:
    use_sharp_peak_preserve: bool = False
    curve_smooth_window: int = 9
    peak_smooth_window: int = 5
    peak_preserve_radius: int = 3
    peak_blend_raw_weight: float = 0.75
    global_prom_ratio: float = 0.015
    local_prom_window: int = 61
    local_prom_ratio: float = 0.12
    local_noise_k: float = 3.0


DEFAULT_SHARP_PEAK_SETTINGS = SharpPeakPreserveSettings()
