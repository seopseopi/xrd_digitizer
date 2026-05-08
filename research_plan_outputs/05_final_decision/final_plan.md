# 다음 실행 우선순위 (단일 개발자용)

1. **전 매니페스트 규모로 재평가**: 현재 스냅샷은 도메인당 **10장**(과거 phase-b `max_samples`). `research_plan_outputs/02_dataset/manifest_*_resolved.csv` 전체에 대해 배치+eval을 한 번 고정한다.
2. **격자·축 혼선**: `axis_mask_margin`, 색 마스크, `contrast_aux`를 **각각 단일 ablation**으로 분리해 `grid_confusion` 비중 변화를 본다.
3. **피크 정렬**: `major_peak_x_error`와 `peak_miss_after_smoothing`를 줄이기 위해 **피크 검출 패스·sharp preserve**를 트레이드오프 분석한다.
4. **이상치 샘플**: styled `pattern_72296`, real_like `pattern_60890` 등 **y_mae 특이치**를 우선 시각화한다.
5. **공식 B0 baseline 리포트 확보**: `dist/xrd_digitizer_model_v1_3` 스냅샷으로 동일 매니페스트·동일 eval 설정에서 clean/styled/real_like 리포트를 생성한다. 모델(M1) 통합 여부는 **항상 이 B0 대비**로만 판단한다. `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1` 등은 **B1 진단 참조**로만 남긴다.

산출물 인덱스: 상위 `DIAGNOSTIC_REPORT_KR.md`.
