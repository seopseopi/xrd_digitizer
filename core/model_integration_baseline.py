"""
모델 도입 판단용 기준선(B0) 및 진단 참조(B1) 경로 단일 정의.

B0: 수정 실험 전 최고 성능 rule-only 코드 스냅샷 (배포·회귀 비교용 코드 베이스).
B1: 진단·ablation 산출물 등 참고용 (공식 baseline 아님).
M1: 모델 보조 도입 후보 — 반드시 B0 대비 개선이 입증되어야 통합 가능.

수치·제품 기준선(성능): ROI 2× 업스케일 + final_export_mode=highres 인 출력
(`export_points_highres` vs source_numeric)이 현재까지 최고 성능이므로,
새 실험·튜닝의 성공 여부는 우선 이 트랙의 지표로 판단한다.
1× eval_grid 트랙은 highres 대비 레거시·퇴행 참고용으로만 사용한다.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "repo_root",
    "BASELINE_B0_RULE_SNAPSHOT_DIR",
    "BASELINE_B0_ABSOLUTE_PATH_DEFAULT",
    "DIAGNOSTIC_REFERENCE_B1_RUN_DIR_EXAMPLE",
    "required_final_decision_sentence_ko",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 공식 B0: 최고점 rule-only 스냅샷 디렉터리 (평가 리포트는 이 스냅샷으로 배치·eval 실행 후 생성)
BASELINE_B0_RULE_SNAPSHOT_DIR = repo_root() / "dist" / "xrd_digitizer_model_v1_3"

# 사용자 메시지에 명시된 절대 경로 (문서·템플릿용 동일 의미)
BASELINE_B0_ABSOLUTE_PATH_DEFAULT = Path(
    "/Users/seopseopi/Library/CloudStorage/SynologyDrive-MATERAI/"
    "5. xrd_team/1. XRD/민섭/xrd_digitizer_v1/dist/xrd_digitizer_model_v1_3"
)

# B1 예시: 진단용 ablation 배치 산출물 (공식 baseline으로 사용 금지)
DIAGNOSTIC_REFERENCE_B1_RUN_DIR_EXAMPLE = (
    repo_root() / "outputs" / "0504" / "runs" / "eval_ablation_08_ridge_m4_peak1"
)

required_final_decision_sentence_ko = (
    "모델 도입은 진단용 조합 대비 개선 여부가 아니라, 수정 전 최고 성능 rule-only 스냅샷인 "
    "B0(dist/xrd_digitizer_model_v1_3) 대비 개선 여부로 판단한다."
)
