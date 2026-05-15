"""런타임 candidate rerank (DP 직전) 설정 — 학습 스크립트와 λ·패치 크기를 맞출 것."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelAssistSettings:
    """OFF 기본. 켜면 checkpoint 로드 후 후보 confidence를 rule+λ·model 으로 재설정하고 DP를 한 번 더 평가한다."""

    enabled: bool = False
    model_ckpt_path: Optional[str] = None
    lambda_model: float = 0.25
    device: str = "cpu"
    patch_size: int = 33
    # DP 비용 비교: trace_score 는 낮을수록 좋음 / valid_ratio 는 높을수록 좋음
    fallback_valid_ratio_margin: float = 0.0
    fallback_trace_score_margin: float = 0.0


DEFAULT_MODEL_ASSIST_SETTINGS = ModelAssistSettings()
