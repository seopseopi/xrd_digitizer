# XRD Digitizer

> XRD 패턴 이미지에서 `(2θ, intensity)` 수치 데이터를 자동으로 복원하는 파이프라인

**현재 성능 (canonical-30 benchmark, clean domain, 20개 패턴):**  
MAE mean **0.0323** · ptp\_r mean **0.804** (2× ROI upscale 기준)

---

## 어떤 문제를 푸는가

연구실에 쌓인 XRD 논문 그래프, 스캔본, 오래된 PDF — 이 이미지들에서 실제 수치 데이터를 뽑아내는 일은 지금도 수작업에 의존한다. 이 프로젝트는 **이미지 → JSON(two_theta, intensity 배열)** 변환을 자동화한다.

```
입력: XRD 패턴 이미지 (PNG / JPEG)
출력: { "two_theta_values": [...], "intensities": [...], "peaks_numeric_curve": [...] }
```

---

## 출력 예시

### 복원된 곡선 + 피크 검출

![peak markers](docs/assets/output_peak_markers.png)

### GT vs Predicted 비교 (pattern\_11832, MAE = 0.0038)

![gt vs pred](docs/assets/output_gt_vs_pred.png)

---

## 파이프라인 — 6단계

```
이미지 입력
   ↓ ① ROI crop & 2× upscale
   ↓ ② color mask (curve 픽셀 추출)
   ↓ ③ candidate map 생성 (ridge + contrast-aux)
   ↓ ④ DP tracing (최적 경로 추적)
   ↓ ⑤ 곡선 smoothing + 피크 검출
   ↓ ⑥ 축 calibration → JSON export
```

| 단계 | 설명 | 시각화 |
|------|------|--------|
| ① ROI 입력 | 축 좌표를 기준으로 그래프 영역만 잘라내고 2× lanczos upscale | ![](docs/assets/step1_roi_input.png) |
| ② Color mask | HSV 색상 범위로 curve 픽셀만 분리 | ![](docs/assets/step2_color_mask.png) |
| ③ Candidate map | ridge 응답 + contrast-aux 신뢰도 결합 | ![](docs/assets/step3_candidates.png) |
| ④ DP trace | column-wise dynamic programming으로 최적 경로 탐색 | ![](docs/assets/step4_trace.png) |
| ⑤ Peaks overlay | smoothed 곡선 위 major/minor 피크 표시 | ![](docs/assets/step5_peaks.png) |

---

## 성능

### Benchmark 결과 (canonical-30, 2× ROI upscale)

![benchmark chart](docs/assets/benchmark_mae_chart.png)

| domain | n | MAE mean | MAE median | ptp\_r mean |
|--------|---|----------:|----------:|------------:|
| clean | 20 | **0.0323** | 0.0351 | 0.804 |

- **MAE** = 정규화된 강도 오차 평균 (GT range 기준)  
- **ptp\_r** = 예측 강도 range / GT 강도 range (1.0이 이상적)

---

## 개발 여정

이 섹션은 프로젝트를 진행하면서 마주친 주요 문제와 해결 과정을 기록한다.

---

### Phase 1 — 초기 구현 · 기반 파이프라인

**목표:** XRD 이미지에서 curve trace를 추출하는 최소 동작 파이프라인 구성

- DP tracing 기본 알고리즘 구현  
- `run_local.py` CLI 진입점, `batch_run.py` 배치 실행기 구성  
- canonical-30 benchmark 데이터셋 설계 (clean 20 / styled 5 / real\_like 5)

**남은 문제:** 해상도가 낮아 얇은 피크의 꼭대기를 픽셀 단위로 정확히 추적하기 어려움

---

### Phase 2 — 2× Highres ROI Upscale 도입

> commit `8568720`, `b27220c`

**문제:** 원본 ROI (약 950×690 px)에서 얇은 피크의 apex가 1–2픽셀 범위에 몰려 DP가 정확한 y 위치를 찾지 못함

**해결:** ROI를 Lanczos 2× upscale(→ 1900×1380 px)한 뒤 내부 처리를 고해상도에서 수행하고, 결과를 다시 원본 좌표계로 downscale해서 export

```bash
--roi-upscale-factor 2 --final-export-mode highres
```

**효과:** 피크 apex 추적 정밀도 향상, sub-pixel 단위 보간 가능

---

### Phase 3 — Amorphous Hump 처리

> commit `3dc7715`

**문제:** 결정질 피크 아래에 넓은 비정질(amorphous) hump가 있는 패턴에서 DP가 hump baseline을 따라가거나 피크를 무시하는 실패 발생

**해결:** wide-band trend attraction 도입 — 저주파 배경(trend)을 추정하고, DP cost에 trend와의 거리를 반영해 hump 위의 피크를 올바르게 따라가도록 유도

---

### Phase 4 — Debug 렌더링 좌표 오류 수정

> commit `9f6e49e`, `fefdce5`

**문제:** 2× upscale 모드에서 debug 이미지 #15(numeric curve peaks)의 마커가 실제 피크가 아닌 엉뚱한 위치에 그려짐

**원인:** debug axis map이 1× 픽셀 기준으로 빌드되어 있었는데, 2× ROI 이미지에 그대로 사용 → 좌표가 절반 위치에 렌더링

**수정:**
```python
# 2× ROI 렌더링 시 axis map scale을 upscale_factor로 나눠 보정
x_map_dbg_render = dict(x_map_dbg, scale=x_map_dbg["scale"] / roi_up_factor)
y_map_dbg_render = dict(y_map_dbg, scale=y_map_dbg["scale"] / roi_up_factor)
```

---

### Phase 5 — 가장 중요한 버그 수정: y-pixel 단위 변환 오류

> commit `7cba173`

**문제:** 2× upscale 모드에서 마커 위치가 여전히 틀렸고, 피크들이 baseline 근처에 몰려 표시됨. 조사하자 intensity 계산 자체가 틀린 것을 발견.

**원인:** `_downscale_series_to_original_roi` 함수가 2× 픽셀 좌표를 그대로 반환

```python
# ❌ 버그: x는 1× 공간으로 변환했지만 y는 2× px 그대로
x_tgt = np.arange(original_roi_w) * float(factor)   # [0, 2, 4, ...]
y_tgt = np.interp(x_tgt, x_src, y_src)              # 2× y px 그대로 반환!
return list(range(original_roi_w)), y_tgt, ...
```

이 때문에 1× 기준 y\_map에 2× 픽셀 y가 입력되어 intensity 계산이 최대 2배 오차 발생

**수정:**
```python
# ✅ 수정: y_tgt를 factor로 나눠 1× 픽셀로 변환
y_tgt = np.interp(x_tgt, x_src, y_src) / float(factor)  # 2× → 1×
```

**효과:**

| 지표 | 수정 전 | 수정 후 | 변화 |
|------|--------:|--------:|-----:|
| MAE mean | 0.0334 | **0.0323** | −3.3% ✓ |
| 마커 정확도 | 절반 높이 | 피크 꼭대기 정확 | ✓ |

**수정 전/후 debug 이미지 #14 비교:**

| 수정 전 (마커가 baseline 근처에 몰림) | 수정 후 (피크 꼭대기에 정확히 위치) |
|------|------|
| 마커들이 그래프 하단 baseline에 집중 | 각 피크의 꼭대기에 빨간/노란 원 표시 |

![peaks overlay (latest)](docs/assets/step5_peaks.png)

---

## 남은 한계 / Known Issues

| 항목 | 현상 | 원인 분석 |
|------|------|----------|
| **ptp\_r < 1** | 예측 intensity range가 GT의 약 80% | DP가 피크 apex를 1–2px 낮게 추적, y축 calibration 점이 실제 데이터 최댓값과 다소 다름 |
| **근접 피크 미분리** | 2θ 간격 < 1° 인 doublet을 하나로 합침 | DP column resolution 한계 |
| **노이즈 심한 패턴** | trace가 노이즈 스파이크를 따라가 MAE 상승 | candidate confidence가 노이즈와 signal을 구분 못함 |
| **Manual input 필요** | plot\_box, 축 calibration 좌표를 사용자가 직접 입력 | 완전 자동화 미구현 |

---

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `torch`는 선택 사항입니다. rule-based 파이프라인은 학습된 모델 없이 실행됩니다.

---

## 사용법

### 단일 이미지

```bash
python3 -m runner.run_local \
  --image_path path/to/input.png \
  --manual_inputs_path path/to/mi.json \
  --output_json_path outputs/result.json \
  --debug_dir outputs/debug \
  --roi-upscale-factor 2 \
  --final-export-mode highres
```

### 배치 실행

```bash
python3 -m runner.batch_run \
  --manifest_csv data/test_canonical_30/manifest.csv \
  --output_dir outputs/ \
  --roi-upscale-factor 2
```

### manual_inputs 형식 예시

```json
{
  "plot_box": [170, 90, 1120, 780],
  "x_axis_points": [[170, 780], [1120, 780]],
  "x_axis_values": [5.0, 70.4],
  "y_axis_points": [[170, 780], [170, 90]],
  "y_axis_values": [0.0, 1.0],
  "color_sample_point": [500, 400]
}
```

---

## 프로젝트 구조

```
core/        설정, 타입, IO, 파이프라인 버전
preprocess/  ROI crop, perspective 보정, mask, morphology, ridge map
trace/       후보 생성, bridge, DP tracing, recovery, postprocess
calibrate/   축 보정, numeric export, 피크 디버그 렌더
peaks/       피크 검출과 smoothing
runner/      CLI / 배치 실행 진입점
eval/        평가 지표와 진단 유틸리티
data/        canonical-30 benchmark 데이터셋
docs/        README 이미지 asset
```

---

## 테스트

```bash
python3 -m pytest
python3 -m py_compile runner/run_local.py trace/candidates.py
```
