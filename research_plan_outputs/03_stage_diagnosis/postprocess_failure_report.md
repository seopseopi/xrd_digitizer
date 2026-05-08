# 후처리·피크 보존 진단

## 관측

- **`peak_miss_after_smoothing`** 라벨이 반복된다 → 스무딩/sharp 분기·피크 prominence 설정이 주 피크 검출·정렬에 영향.
- `max_gap_px`가 스냅샷에서 **1**로 고정에 가깝게 관측 → 긴 연속 단절보다는 **국소 오차·피크 위치 문제**가 게이트를 지배한다는 신호로 해석 가능 (단, gap_ranges 미보고 시 메트릭 정의상 0이 될 수 있어 다른 ablation 조합과 비교 필요).

## 다음 실험

1. `--use-sharp-peak-preserve` 단독 on  
2. `peak-single-pass` 유무에 따른 peak_recall vs major_peak_x_error 트레이드오프 (`ablation_matrix.tsv` 04행 vs 23행 등)  
