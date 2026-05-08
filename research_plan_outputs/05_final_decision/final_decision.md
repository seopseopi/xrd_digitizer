# 최종 의사결정 (모델 도입 기준선)

모델 도입은 진단용 조합 대비 개선 여부가 아니라, 수정 전 최고 성능 rule-only 스냅샷인 **B0(`dist/xrd_digitizer_model_v1_3`)** 대비 개선 여부로 판단한다. 모델은 해당 baseline보다 clean/styled/real_like 통합 성능에서 명확히 개선될 때만 도입한다.

## 비교군 정의

| ID | 설명 |
| --- | --- |
| **B0** | `dist/xrd_digitizer_model_v1_3` — 공식 rule-only baseline |
| **B1** | 현재 워크트리 진단·ablation 산출물(예: `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1`) — **diagnostic reference만** |
| **M1** | B0와 동일 평가 조건에서 모델 보조를 붙인 후보 |

## 통합 조건 요약

- M1이 B1보다 좋아도, **B0보다 나쁘면 `reject_model_integration`** (보류).
- 상세 체크리스트·판정은 `ml/model_integration_compare.py` 출력 JSON을 따른다.

## 판정 실행

```bash
cd /path/to/xrd_digitizer_v1
PYTHONPATH=. python3 -m ml.model_integration_compare \
  --manifest research_plan_outputs/model_eval/compare_manifest.filled.json \
  --out_json research_plan_outputs/model_eval/model_integration_verdict.json
```
