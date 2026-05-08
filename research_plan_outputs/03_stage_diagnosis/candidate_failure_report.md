# 후보 생성 단계 진단

## 관측

- diagnosis의 **`candidate_recall_per_column`** 다수 샘플에서 **0.99 근처** (`eval_reports` 내 per-sample 확인). 계획서 관점에서는 후보 단계 **굶주림은 주 원인이 아님**.
- 그럼에도 **`major_peak_x_error` 평균이 매우 큼**(clean 평균 약 39px): 후보는 있으나 **DP가 주 피크 경로를 선택하지 못하거나**, 피크 정렬 메트릭이 추적 경로와 어긋난다.

## 다음 실험

1. `bridge_final_candidates` on/off는 이미 기본 on — 잘못된 브리지 증폭 여부를 소수 실패 샘플에서 시각화  
2. ridge·contrast_aux 단일 플래그 ablation (`experiments/ablation_eval_summary.tsv` 참조)  
