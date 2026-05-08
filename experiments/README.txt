experiments/archive/
  예전 루트에 흩어져 있던 배치 결과(outputs_baseline_*, outputs_v2_* 등),
  구버전 최소 번들(xrd_calibrate_v1_bundle), legacy_compare, ml_rescue_v15,
  outputs/ 안의 v2·튜닝·비교 실행 폴더(outputs_legacy_runs) 를 한곳으로 옮긴 보관소입니다.

운영·기본 최선: v1.1 — runner/run_local.py --pipeline v1_1 (debug.pipeline_version = calibrate_v1_1).
v2 는 「더 나은 기본 모델」이 아니라 통합 실험용: --pipeline v2_experimental --allow_experimental_v2 (+ 튜닝 JSON).

팀원 분석에서 말한 H7_region_tail_anchor 등은 아직 본 저장소에 미구현일 수 있습니다.
구현 시 trace/ 또는 experiments/ 하위에 플래그 실험 모듈로 추가하는 것을 권장합니다.

phase1_curve_continuity.txt — 1차 곡선 정합 목표·외부 연구 링크·로컬 ablation 기록 (eval_* 태그와 함께 참고).

improvement_roadmap.txt — 단계별 개선 체크리스트(Step 1a 마스크 등). 완료 시 [x] 로 갱신.

scripts/analyze_gt_peak_spacing.py — GT peak 간격 통계 → experiments/gt_peak_spacing_stats.txt (로드맵 3b).
scripts/run_eval_ablation_matrix.py — 플래그 8조합 phase-b 연속 실행 후 summarize.
scripts/summarize_eval_reports.py — outputs/runs/<tag>/report_*.json 집계 → experiments/ablation_eval_summary.tsv
scripts/verdict_eval_reports.py — 게이트 대비 pass_rate + 배치 평균 FAIL 이유 → experiments/verdict_ablation.txt
experiments/ablation_eval_analysis.txt — 위 TSV 기준 운영 권장 요약.
experiments/verdict_ablation.txt — 되는지/안되는지(게이트 기준) 한눈에.
experiments/eval_methodology_ko.txt — 평가 파이프라인·지표·게이트·실패 라벨 전체 정리 (연구용).
experiments/code_improvement_targets_ko.txt — 증상별로 고칠 코드 위치·알고리즘 변경 방향 (연구용).
experiments/drill_pattern_760_checklist_ko.txt — pattern_760 드릴 실행 경로·지표·P0/P1 관찰 포인트.
experiments/perspective_ui_guide.txt — perspective + 축 클릭 UI 가이드 (로드맵 4a).
