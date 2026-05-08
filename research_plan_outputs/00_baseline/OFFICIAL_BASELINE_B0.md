# 공식 모델 도입 기준선 B0

모델 도입 여부는 진단용 산출물(B1) 대비가 아니라, **수정 실험 전 최고 성능 rule-only 코드 스냅샷 B0** 대비로만 판단한다.

## B0 경로 (단일 출처)

코드 상 상수: `core/model_integration_baseline.py` 의 `BASELINE_B0_RULE_SNAPSHOT_DIR`.

디렉터리:

`dist/xrd_digitizer_model_v1_3`

(동일 의미 절대 경로:  
`/Users/seopseopi/Library/CloudStorage/SynologyDrive-MATERAI/5. xrd_team/1. XRD/민섭/xrd_digitizer_v1/dist/xrd_digitizer_model_v1_3`)

## 평가 절차

1. 위 스냅샷을 워킹 트리로 사용하거나, 동일 커밋/번들로 재현 가능한 환경에서 배치를 실행한다.
2. **동일 매니페스트·동일 eval 게이트 정책**으로 `eval/report.py` 리포트를 clean / styled / real_like 각각 생성한다.
3. development·strict 게이트별 리포트를 각각 확보한다.
4. `python3 -m ml.model_integration_compare --manifest ... --out_json ...` 로 M1과 비교한다.

## B1 (진단 참조)

예: `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1` — 실패 원인 분석·ablation 참고용이며 **공식 baseline이 아니다.**

필수 문장은 `research_plan_outputs/05_final_decision/final_decision.md` 및 `core/model_integration_baseline.required_final_decision_sentence_ko` 에 동일하게 있다.
