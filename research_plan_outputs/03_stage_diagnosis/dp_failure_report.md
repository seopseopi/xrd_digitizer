# DP 추적 단계 진단

## 관측

- `path_margin_instability`은 샘플별로 중간 수준(예: 0.47)인 경우가 있어 **`wrong_branch_lock_in` 라벨은 상대적으로 드묾**. 즉 튜닝 여지는 있으나 유일 병목은 아님.
- **`major_peak_x_error`** 불량과 결합해 볼 때, 비용함수가 **세로 격자/인접 열의 가짜 능선**을 선호하는 구간이 있는지 점검이 필요하다.

## 다음 실험

1. `apex_pull` / transition 관련 파라미터 단일 변수 ablation  
2. blockwise margin 로그가 큰 구간과 피크 오차 피크가 일치하는지 오버레이  
