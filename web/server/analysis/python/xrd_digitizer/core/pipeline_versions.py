"""운영·실험 파이프라인 버전 문자열 (debug.json pipeline_version 과 일치).

정책: 운영 기본 CLI는 v1.1 (--pipeline v1_1, calibrate_v1_1).
v1.2(calibrate_v1_2)는 동일 엔진·동일 `run_pipeline`에 대한 고정 스냅샷 라벨(배포·재현용)이다.
v2_integrated 는 통합 실험 트랙일 뿐이며, 수치상·운영상 기본 최선으로 간주하지 않는다.
"""

from __future__ import annotations

# 생산·품질 기준선 (run_local 기본 --pipeline v1_1)
CALIBRATE_V1_1 = "calibrate_v1_1"

# 현재 성능 고정 스냅샷 (엔진은 v1_1과 동일, 결과·디버그에만 v1.2 태그)
CALIBRATE_V1_2 = "calibrate_v1_2"

# 구버전 JSON 과의 호환 (아카이브 평가용)
CALIBRATE_V1_LEGACY = "calibrate_v1"

# 실험용 통합 파이프라인 (v2_experimental 전용, 운영 기본 아님)
V2_INTEGRATED = "v2_integrated"
