# Failure taxonomy (`eval/gates.label_failures`)

규칙 기반 자동 라벨이며, 한 샘플에 여러 개 붙을 수 있다.

| 라벨 | 대략적 해석 |
| --- | --- |
| `candidate_starvation` | 빈 열 비율 높음 → 후보/브리지 |
| `wrong_branch_lock_in` | 경로 margin 불안정 → DP |
| `tail_collapse` | tail MAE·collapse 높음 → 후처리·복구 |
| `calibration_mismatch` | 라운드트립 오차 큼 → 축/ManualInputs |
| `peak_miss_after_smoothing` | recall은 있는데 F1 낮음 → 스무딩·피크 분기 |
| `text_intrusion` | IoU 낮은데 후보는 있음 → 마스크/ROI |
| `grid_confusion` | 후보 recall 높은데 곡선 MAE 큼 → 격자·축·색 혼선 |
| `legend_capture` | 복구 성공률 낮음 등 |

이번 배치에서 eval 리포트 9개를 합친 빈도: **`grid_confusion` ≫ `peak_miss_after_smoothing`** (`03_stage_diagnosis/failure_taxonomy_from_eval.md` 참고).
