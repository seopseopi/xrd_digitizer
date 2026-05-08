# 모델 통합 판단 (현 시점)

모델 도입은 진단용 조합 대비 개선 여부가 아니라, 수정 전 최고 성능 rule-only 스냅샷인 **B0(`dist/xrd_digitizer_model_v1_3`)** 대비 개선 여부로 판단한다. 모델은 해당 baseline보다 clean/styled/real_like 통합 성능에서 명확히 개선될 때만 도입한다.

## 역할 구분

| ID | 설명 |
| --- | --- |
| **B0** | `dist/xrd_digitizer_model_v1_3` — 공식 baseline (`core/model_integration_baseline.py`) |
| **B1** | 예: `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1` — diagnostic reference (실패 원인·ablation 참고) |
| **M1** | 모델 보조 후보 — **B0 대비** 통합 조건을 만족해야 함 |

**M1이 B1보다 좋아도 B0보다 나쁘면 `reject_model_integration` 으로 보류한다.**

## 공식 판정 도구

`ml/model_integration_compare.py` + `research_plan_outputs/model_eval/compare_manifest.example.json` 참고.

## 진단 요약 (참고)

이전 진단에서 후보 컬럼 recall은 높은 편이었고, 자동 taxonomy 상 **`grid_confusion`**, **`peak_miss_after_smoothing`** 이 두드러졌다. 이는 **규칙 기반 튜닝·모델 보조 설계 방향** 참고용이며, 모델 통과 여부는 위 B0 대비 비교로만 결정한다.
