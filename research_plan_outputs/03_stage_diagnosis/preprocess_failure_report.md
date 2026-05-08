# 전처리 단계 진단 (색·ROI·마스크·축)

## 관측 (eval 스냅샷)

- 자동 라벨 **`grid_confusion`**가 압도적이다. 정의상 후보 컬럼 커버는 높은데 픽셀 곡선 MAE가 크게 나는 패턴으로, **격자선·축 영역이 곡선 후보로 남거나 DP가 격자 쪽으로 붙는** 경우를 의심한다.
- `calibration_roundtrip_error≈0` 이므로 **축 2점 매핑 자체는 GT에서 유도한 ManualInputs 기준으로 안정**이다.

## 우선 조치 제안 (계획서 8.1과 정렬)

1. `axis_mask_margin` ablation (이미 ablation 매트릭스에 m4 조합 존재)  
2. 색 거리 임계·마스크 A/B 결합 재확인 (디버그 마스크 이미지)  
3. `contrast_aux` 단독 on/off로 저대비 styled/real_like만 재측정  
