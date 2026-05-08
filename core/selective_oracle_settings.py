"""Selective GT oracle 재랭크 실험 설정 — Risk Detector rule 임계값 및 스타일 정책."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SelectiveOracleSettings:
    """GT oracle을 위험 열에만 적용. CNN 없음."""

    enabled: bool = False
    gt_json_path: Optional[str] = None
    sigma_px: float = 8.0

    # Risk rule thresholds (rule-based detector v1)
    conf_margin_thr: float = 0.08
    entropy_high_thr: float = 1.2
    candidate_count_high_thr: int = 6
    y_gap_thr: float = 8.0
    y_gap_margin_thr: float = 0.15  # conf_margin < this with y_gap
    peak_margin_thr: float = 0.18  # near peak window AND conf_margin <
    axis_dist_risk_thr: float = 2.5  # top1 axis_dist below → near axis/border proxy
    dp_margin_low_thr: float = 0.08  # blockwise top1-top2 margin below → unstable

    risk_dilate_radius_columns: int = 3
    min_segment_columns: int = 6  # shorter merged segments dropped
    merge_gap_columns: int = 2  # merge adjacent segments if gap <= this
    disable_taxonomy_prior_for_risk: bool = False
    disable_low_conf_margin_risk: bool = False
    disable_high_entropy_risk: bool = False
    disable_axis_proximity_risk: bool = False
    disable_large_y_gap_risk: bool = False
    disable_peak_window_risk: bool = False
    disable_dp_margin_low_risk: bool = False
    risk_debug_include_columns: bool = False

    # 실험용: taxonomy / 고엔트로피 risk 게이팅 강화 (기본 False = 기존 동작)
    taxonomy_prior_requires_margin: bool = False
    high_entropy_requires_low_margin: bool = False

    # domain: clean | styled | real_like — styled/real_like 기본 oracle off
    run_domain: str = "clean"
    apply_to_styles: Tuple[str, ...] = ("clean",)
    styled_real_default_off: bool = True
    allow_styled_real_selective: bool = False  # True면 styled/real_like에도 위험열 oracle 허용

    # sample-level taxonomy prior (semicolon-separated labels from offline eval)
    taxonomy_prior: Optional[str] = None

    # Optional: append per-column risk features to this CSV (study 스크립트용)
    risk_features_csv_path: Optional[str] = None

    # Optional grid-like proxy at sample level (e.g. candidate_recall) — 없으면 None
    grid_like_proxy: Optional[float] = None


DEFAULT_SELECTIVE_ORACLE_SETTINGS = SelectiveOracleSettings()
