# Main metric 정책 (코드 기준)

출처: `eval/metrics.py` 의 `main` 블록, 게이트용 판정은 `eval/gates.py`.

| 키 | 의미 |
| --- | --- |
| `curve_y_mae_px` | GT 픽셀 경로 대비 복원 곡선 y 평균 오차 |
| `major_peak_x_error` | 주요 피크 **x(픽셀)** 위치 오차 |
| `peak_recall` | prominence 기반 피크 매칭 recall |
| `max_gap_px` | 후처리 gap-fill에서 보고된 연속 구간 최대 길이(px) |
| `calibration_roundtrip_error` | 축 매핑 라운드트립 평균 오차(px) |
| `numeric_y_mae_norm` | 정규화된 수치 세기 오차 |
| `major_peak_x_error_2theta` | 주요 피크 **2θ** 위치 오차 |

Debug·diagnosis는 게이트 합격 판단에 쓰지 않고 원인 분석용이다 (`candidate_recall_per_column`, `empty_column_rate`, `recovery_success_rate`, `path_margin_instability`, `tail_mae_px`, `peak_f1` 등).
