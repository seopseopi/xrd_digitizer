# 모델 통합 비교 리포트 (B0 / B1 / M1)

모델 도입은 진단용 조합 대비 개선 여부가 아니라, 수정 전 최고 성능 rule-only 스냅샷인 **B0(`dist/xrd_digitizer_model_v1_3`)** 대비 개선 여부로 판단한다.

## 입력

- Manifest: `compare_manifest.filled.json` (예시: `compare_manifest.example.json` 복사)
- 기계 판정 출력: `model_integration_verdict.json` (`python3 -m ml.model_integration_compare` 결과)

## 요약 표 (수동 기입 또는 verdict JSON에서 복사)

| 지표 (예: strict 기준 macro 또는 도메인별) | B0 | M1 | B1 (참고) |
| --- | --- | --- | --- |
| macro strict pass rate | | | |
| macro development pass rate | | | |
| grid_confusion 합계 | | | |
| clean major_peak_x_error 평균 | | | |
| curve_y_mae_px (도메인별) | | | |
| max_gap_px (도메인별) | | | |
| numeric_y_mae_norm (도메인별) | | | |
| B0 통과 → M1 실패 샘플 비율 | | | |

## 자동 판정 결과

`model_integration_verdict.json` 의 `verdict` 필드를 아래에 붙여 넣는다.

- `proceed_model_integration_candidate`: B0 대비 조건 충족
- `reject_model_integration`: B0 미달
- `reject_model_integration_beats_B1_only_insufficient_vs_B0`: B1보다만 나음

```
(paste verdict JSON snippet here)
```

## 결론

- 최종 코드 리뷰·릴리즈 전 **사람 판단**은 별도.
- B0를 이기지 못하면 모델 통합은 보류한다.
