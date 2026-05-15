"""GT oracle 후보 재랭크 실험 설정 — 학습된 reranker 없이 DP 상한·후보 구조 검증용."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OracleRerankSettings:
    enabled: bool = False
    gt_json_path: Optional[str] = None
    # 작을수록 거리 민감. 후보 confidence = exp(-(dist/sigma)^2)
    sigma_px: float = 8.0


DEFAULT_ORACLE_RERANK_SETTINGS = OracleRerankSettings()
