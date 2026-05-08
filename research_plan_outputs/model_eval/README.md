# 모델 통합 평가 (B0 / B1 / M1)

## 정의

| ID | 의미 |
| --- | --- |
| **B0** | `dist/xrd_digitizer_model_v1_3` — 수정 전 최고 성능 rule-only 스냅샷. 공식 baseline. |
| **B1** | 진단·ablation 산출물 (예: `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1`) — 참고만. |
| **M1** | 동일 평가 조건에서 모델 보조 적용 후보. |

필수 원칙 문장: `core.model_integration_baseline.required_final_decision_sentence_ko`

## 절차

1. B0 스냅샷 환경에서 매니페스트 고정 → 배치 → `eval/report.py` 로 도메인별 `development`·`strict` 리포트 각 3개 생성.
2. M1 (통합 브랜치)에서 동일 매니페스트·동일 게이트로 동일 6+6개 리포트 생성.
3. (선택) B1 진단 리포트 경로를 manifest에 넣어 참고 비교.
4. 실행:

```bash
cd /path/to/xrd_digitizer_v1
PYTHONPATH=. python3 -m ml.model_integration_compare \
  --manifest research_plan_outputs/model_eval/compare_manifest.filled.json \
  --out_json research_plan_outputs/model_eval/model_integration_verdict.json
```

5. 사람이 읽는 요약은 `model_integration_comparison_REPORT.md` 템플릿에 verdict 요약을 붙인다.

## 파일

- `compare_manifest.example.json`: 경로 채워 넣을 템플릿
- `model_integration_comparison_REPORT.md`: 리포트 스켈레톤
