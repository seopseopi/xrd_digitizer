# Ablation 요약 (`ablation_matrix.tsv`)

동일 10장 부분집합·phase-b 플래그 조합 결과이다. **도메인마다 최적 조합이 다를 수 있음.**

이 표와 태그(`eval_ablation_*`)는 **B1 진단·실험 참조용**이다. 모델 도입 baseline(**B0 = `dist/xrd_digitizer_model_v1_3`**)과 혼동하지 않는다.

관찰:

- **ridge 단독(`eval_ablation_02`)**: styled에서 평균 `curve_y_mae_px`가 **개선**(31.2→23.9)되는 반면, clean에서 `major_peak_x_error`는 **악화**(30.3→37.9) 경향 → 계획서의 폐기 조건(한 스타일 개선에 다른 스타일 악화) 검토 대상.
- **`axis-mask-margin 4` + gap 메트릭**: 일부 조합에서 `max_gap_px`가 0→1로 바뀜 (`metrics._max_gap_px`가 gap_ranges 존재 여부에 민감).
- **`peak-single-pass`**: `peak_recall`와 `major_peak_x_error` 트레이드오프 (예: real_like에서 recall 하락).

다음 단계: 계획서대로 **한 번에 한 플래그**만 바꾼 재실행 + 전체 매니페스트(50~100장)로 재검증이 필요하다.
