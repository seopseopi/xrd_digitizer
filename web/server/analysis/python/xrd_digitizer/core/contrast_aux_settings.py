"""contrast_aux_v1: 후보 신뢰도 보조(local background 대비). DP·마스크·후보 소스는 변경하지 않음."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastAuxSettings:
    use_contrast_aux: bool = False
    contrast_aux_weight: float = 0.25
    contrast_aux_min_base_conf: float = 0.15
    contrast_aux_bg_kernel_ratio: float = 0.035
    contrast_aux_border_suppress_px: int = 8


DEFAULT_CONTRAST_AUX_SETTINGS = ContrastAuxSettings()
