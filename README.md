# XRD Graph Image → Numeric JSON Reconstruction

> XRD 패턴 이미지에서 `(2θ, intensity)` 수치 데이터를 자동으로 복원하는 **deterministic scientific digitizer**

이 프로젝트는 초기에 end-to-end 방식의 이미지 기반 곡선 복원으로 시작했지만, 시각적으로 유사한 결과가 실제 수치 복원 정확도를 보장하지 않는다는 한계를 확인했다. 이후 좌표 보정, 다중 후보 생성, Dynamic Programming 기반 경로 선택, 피크 보존, 고해상도 수치 평가, 실패 원인 분석이 가능한 **deterministic XRD graph digitizer** 구조로 전환했다.

**현재 성능 (canonical-30, clean 20개, 2× ROI upscale):**  
MAE mean **0.0323** · ptp\_r mean **0.804**

---

## 출력 예시

### 복원 곡선 + 피크 검출

![peak markers](docs/assets/output_peak_markers.png)

### GT vs Predicted 비교 (pattern\_11832, MAE = 0.0038)

![gt vs pred](docs/assets/output_gt_vs_pred.png)

---

## 현재 파이프라인

```
입력: XRD 이미지 + manual_inputs.json (축 보정 좌표)
  ↓ ① ROI crop & 2× upscale
  ↓ ② color mask (curve 픽셀 추출)
  ↓ ③ multi-source candidate generation
  ↓ ④ candidate filtering & ranking
  ↓ ⑤ DP path selection
  ↓ ⑥ smoothing + peak detection
  ↓ ⑦ axis calibration → JSON export
출력: { two_theta_values, intensities, peaks_numeric_curve }
```

| 단계 | 시각화 |
|------|--------|
| ① ROI 입력 | ![](docs/assets/step1_roi_input.png) |
| ② Color mask | ![](docs/assets/step2_color_mask.png) |
| ③ Candidate map | ![](docs/assets/step3_candidates.png) |
| ④ DP trace (smoothed) | ![](docs/assets/step4_trace.png) |
| ⑤ Peaks overlay | ![](docs/assets/step5_peaks.png) |

---

## 성능 (canonical-30 benchmark)

![benchmark chart](docs/assets/benchmark_mae_chart.png)

| domain | n | MAE mean | MAE median | ptp\_r mean |
|--------|--:|----------:|----------:|------------:|
| clean | 20 | **0.0323** | 0.0351 | 0.804 |

- **MAE** = normalized 강도 오차 평균 (GT intensity range 기준)
- **ptp\_r** = pred range / GT range (1.0이 이상적, 현재 약 80% 커버)

---

---

# 개발 히스토리

이 섹션은 프로젝트를 진행하면서 마주친 문제와 구조 전환 과정을 시간 순으로 기록한다.  
단순히 "성능이 좋아졌다"가 아니라 **왜 접근을 바꿨고, 어떤 기준으로 판단했으며, 무엇이 해결되고 무엇이 남았는지**를 남긴다.

---

## 전체 흐름 요약

| 단계 | 접근 방식 | 핵심 문제 | 전환 이유 |
|------|----------|----------|----------|
| 1 | End-to-End 모델 | 설명 불가능, 수치 오차 큼 | 실패 원인 추적 불가 |
| 2 | Heatmap / 곡선 검출 | 곡선은 보이나 y값 틀림 | 좌표계 보정 구조 필요 |
| 3 | Numeric reconstruction 전환 | x/y 좌표 오차, y값 저하 | calibration 구조 도입 |
| 4 | 수동 좌표 입력 구조 | pair integrity 부재 | manifest 기반 고정 |
| 5 | 다중 후보 생성 | peak top 손실, 후보 부족 | multi-source candidate |
| 6 | DP 경로 선택 | bottom branch lock | 후보 ranking + DP cost 개선 |
| 7 | 평가 기준 재설계 | 시각 평가 ≠ 수치 정확도 | highres metric 도입 |
| 8 | canonical test set | 실험 재현성 없음 | manifest 기반 30개 고정 |
| 9 | 실패 원인 진단 | real_like 실패 분석 | failure taxonomy 도입 |
| 10 | 2× highres upscale | 저해상도 apex 손실 | 내부 처리 해상도 2배 |
| 11 | 버그 수정 (y-pixel) | intensity 계산 오류 | MAE 0.0334 → 0.0323 |

---

## Phase 1 — End-to-End 기반 접근

### 목표

```
XRD graph image → numeric sequence / reconstructed curve
```

이미지 전체를 모델이 보고 곡선 또는 수치 시퀀스를 직접 예측하는 구조였다.  
빠르게 MVP를 만들기에는 좋았지만 실제 XRD 복원 문제에서는 한계가 컸다.

### 확인된 문제

**수치 정확도 문제**
- main peak 높이가 낮게 복원됨
- shoulder 구간이 부드럽게 뭉개짐
- tail fluctuation이 사라짐
- peak 위치가 약간씩 밀림
- x축 간격이 실제 값과 정확히 맞지 않음
- y축 amplitude가 원본보다 낮게 나옴

**설명 불가능성 문제**

결과가 틀렸을 때 원인을 분리하기 어려웠다:
```
모델이 곡선을 못 본 것인가?
좌표 변환이 틀린 것인가?
피크를 낮게 예측한 것인가?
후처리에서 망가진 것인가?
```

### 결론

End-to-end 방식은 이 프로젝트의 목표와 맞지 않았다.  
단순 이미지 복원이 아니라 **과학 그래프의 수치 복원**이 목적이므로, 각 단계를 분리해 검증 가능한 구조가 필요했다.

---

## Phase 2 — Heatmap / 곡선 검출 기반 접근

### 목표

```
XRD image → curve heatmap / mask → post-processing → numeric data
```

### 남은 문제

**y값이 틀림**

육안으로 비슷한 곡선이 나와도 실제 수치로 비교하면 y값이 낮거나 peak가 눌리는 문제:
```
원본 peak:       /\
복원 peak:      /--\
```
skeleton이나 centerline 기반 접근은 peak의 중앙 또는 아래쪽을 따라가는 문제가 있었다.

**x축 좌표 변환 불안정**  
plot area, 기준점, tick spacing이 조금만 틀어져도 전체 x값이 밀렸다.

**y축 amplitude 불일치**  
모양은 맞아도 y값 스케일이 달라지는 문제 — numeric JSON 복원에서 치명적.

### 결론

곡선을 검출하는 것만으로는 충분하지 않았다.  
프로젝트 방향이 **시각적 복원 중심** → **수치 복원 중심**으로 전환되었다.

---

## Phase 3 — Numeric JSON Reconstruction 중심으로 전환

### 핵심 목표 재정의

```
이전: 이미지와 비슷한 그래프 만들기
이후: XRD graph image → source numeric JSON reconstruction
```

수치 복원의 5가지 기준을 정의했다:

1. **Curve shape fidelity** — 전체 모양
2. **X-axis coordinate fidelity** — 2θ 위치 정확도
3. **Y-axis amplitude fidelity** — intensity 값 정확도
4. **Peak preservation** — 피크 개수, 위치, 높이
5. **Failure explainability** — 실패 원인 분리 가능성

이 기준 때문에 단순 ML 모델보다 **검증 가능한 deterministic pipeline**이 더 적합하다고 판단했다.

---

## Phase 4 — 좌표 보정 입력 구조 도입

### 배경

numeric reconstruction을 하려면 이미지 pixel 좌표를 실제 그래프 좌표로 변환해야 한다.

### 설계 원칙

이미지 단독 실행을 허용하지 않고, 항상 다음 pair를 요구하는 구조로 변경:

```
input.png + mi.json (manual inputs)
```

**mi.json 포함 정보:**
- plot box 좌상단 / 우하단
- x축 기준점 2개 + 실제 2θ 값
- y축 기준점 2개 + 실제 intensity 값
- curve 색상 샘플 좌표

**실행 원칙:**
```
이미지만으로 실행하지 않는다.
sample name으로 경로를 추론하지 않는다.
manifest에 명시된 pair만 실행한다.
```

---

## Phase 5 — Multi-source Candidate Generation

### 문제

초기에는 skeleton/thinning 기반으로 곡선 중심선을 따라가는 방식이 강했다.  
하지만 다음 문제가 있었다:

- 굵은 peak에서 top을 놓침 → 선의 중앙/아래를 따라감
- low-contrast 구간에서 곡선이 끊김
- baseline 근처 edge를 잘못 선택함

### 해결

정답 후보를 최대한 candidate pool 안에 포함시키기 위해 다중 소스 구조로 변경:

```
raw candidate
+ thinning candidate
+ edge/connected component candidate
+ peak/top candidate
```

**핵심 전환:**  
"가장 깔끔한 선 하나를 찾는 문제" → **"많은 후보 중 가장 과학적으로 그럴듯한 경로를 선택하는 문제"**

---

## Phase 6 — Dynamic Programming 기반 경로 선택

### DP의 역할

각 column마다 존재하는 여러 후보 중 하나를 선택해 최종 curve path를 구성.  
단순히 가장 진한 점이 아니라 다음을 함께 고려:

- 후보 confidence
- 이전 column과의 y 연속성
- 급격한 y 변화 penalty
- peak 근처 sharpness 허용
- bottom branch 회피

### 새로 생긴 문제

**bottom branch lock:**  
flat 또는 low-contrast 구간에서 baseline 근처 후보를 잘못 선택:
```
실제 곡선:  ───────
선택 경로: _______
```

**peak top flattening:**
```
원본:      /\
복원:     /--\
```
XRD에서 peak height와 peak width는 물질 분석에서 중요한 정보이기 때문에 치명적.

**continuity penalty의 부작용:**  
안정성을 위한 연속성 penalty가 실제 sharp peak나 작은 peak를 noise처럼 취급해 무시.

---

## Phase 7 — 평가 체계 재설계

### 문제

초기에는 결과 그래프를 눈으로 비교해 성능을 판단했다.  
하지만 다음 문제가 있었다:

- 전체 shape은 유사하지만 y amplitude가 낮음
- peak center가 몇 pixel 밀림
- **highres에서 개선됐지만 950-point grid로 평가하면 개선이 숨겨짐**

### 개선

```
주요 지표:
- highres normalized y MAE
- shape correlation
- peak center error / peak height error / peak width error

디버그 지표:
- candidate recall
- selected path behavior
- bottom branch score
- failure subtype
```

2x highres 기반 평가를 primary metric으로 올리고, 950-point eval grid는 legacy로 낮췄다.

---

## Phase 8 — Canonical Test Set 구축

### 목표

실험이 많아질수록 샘플 경로와 metadata가 섞일 위험이 커졌다.  
`data/test_canonical_30/manifest.csv`를 기준으로 고정된 30개 평가셋을 만들었다.

**구성:**

| domain | 수 | 특징 |
|--------|---|------|
| clean | 20 | 노이즈 없는 이상적 패턴 |
| styled | 5 | 색상/배경 변형 |
| real\_like | 5 | 실제 스캔에 가까운 패턴 |

각 샘플은 `input.png`, `mi.json`, `gt.json`, `source_numeric.json`, `metadata.json`을 독립적으로 포함한다.

**효과:**
- 같은 입력으로 반복 평가 가능
- domain별 성능 비교 가능
- 실험 결과를 객관적으로 비교 가능

---

## Phase 9 — Real-like 실패 진단

### 대표 실패 샘플: `real_like_pattern_83398`

real_like domain에서는 clean 이미지와 다른 문제가 나타났다.  
긴 구간에서 실제 곡선을 따라가지 못하고 잘못된 branch에 lock-in 되는 문제.

### 진단 결과

```
true-like raw candidates missing in many columns
raw extraction component fragmentation failure
low-contrast true curve issue
```

**핵심 발견:**  
DP가 틀린 선택을 한 것도 문제였지만, 더 근본적으로는 **선택할 수 있는 좋은 후보가 없었다.**

```
이전 가정: "후보는 충분한데 DP가 잘못 고른다."
새로운 결론: "일부 실패는 DP 문제가 아니라 candidate generation 문제다."
```

---

## Phase 10 — Recovery 실험

### 시도한 방향

| 실험 | 내용 | 결과 |
|------|------|------|
| Selective oracle / rerank | 위험 column에서만 후보 순위 조정 | domain policy로 일부 no-op |
| Fragmentation-aware bridge | 끊어진 후보 재연결 | 후보 수 증가했지만 MAE 무변화 |
| Shell retention | filtering에서 버려지는 후보 유지 | wrong branch도 함께 유지됨 |
| Path-level recovery v0 | 실패 구간만 local DP 재실행 | CANDIDATE\_POOL\_NO\_ALTERNATIVE |

### 결론

단순히 후보를 더 넣는 것은 해결책이 아니었다.

```
candidate count 증가 ≠ 성능 개선
true-like candidate recall 증가 = 실제 개선 가능성
```

이후 방향은 **selected-path monitor** 기반의 실패 원인 진단으로 이동했다.

---

## Phase 11 — Failure Taxonomy 도입

### 목적

단순히 "실패했다"가 아니라 실패를 다음처럼 분류하기 시작했다:

```
RAW_EXTRACTION_COMPONENT_FRAGMENTATION_FAILURE
RAW_EXTRACTION_LOW_CONTRAST_TRUE_CURVE
CANDIDATE_POOL_NO_ALTERNATIVE
BOTTOM_BRANCH_LOCK
PEAK_TOP_FLATTENING
WRONG_BRANCH_SELECTION
```

이 덕분에 실험 결과가 나빠졌을 때도 무작정 튜닝하지 않고 원인을 분리할 수 있게 되었다.

**selected path monitor 도입:**  
최종 DP가 선택한 경로가 실제 곡선 근처인지, bottom branch인지, 후보 자체가 없었는지 column 단위로 추적.

---

## Phase 12 — 2× Highres ROI Upscale 도입

> commit `8568720`, `b27220c`

### 문제

원본 ROI(약 950×690 px)에서 얇은 peak의 apex가 1–2픽셀 범위에 몰려  
DP가 정확한 y 위치를 찾지 못함.

### 해결

ROI를 Lanczos 2× upscale(→ 1900×1380 px)한 뒤 내부 처리를 고해상도에서 수행하고,  
결과를 다시 원본 좌표계로 downscale해서 export.

```bash
--roi-upscale-factor 2 --final-export-mode highres
```

**효과:** 피크 apex 추적 정밀도 향상, sub-pixel 단위 보간 가능

---

## Phase 13 — Amorphous Hump 처리

> commit `3dc7715`

### 문제

결정질 피크 아래에 넓은 비정질(amorphous) hump가 있는 패턴에서  
DP가 hump baseline을 따라가거나 피크를 무시하는 실패 발생.

### 해결

wide-band trend attraction 도입 — 저주파 배경(trend)을 추정하고,  
DP cost에 trend와의 거리를 반영해 hump 위의 피크를 올바르게 따라가도록 유도.

---

## Phase 14 — Debug 렌더링 좌표 오류 수정

> commit `9f6e49e`, `fefdce5`

### 문제

2× upscale 모드에서 debug 이미지의 peak 마커가 실제 피크가 아닌 엉뚱한 위치에 표시됨.

### 원인

debug axis map이 1× 픽셀 기준으로 빌드되어 있었는데,  
2× ROI 이미지에 그대로 사용 → 좌표가 절반 위치에 렌더링.

### 수정

```python
# 2× ROI 렌더링 시 axis map scale을 upscale_factor로 나눠 보정
x_map_dbg_render = dict(x_map_dbg, scale=x_map_dbg["scale"] / roi_up_factor)
y_map_dbg_render = dict(y_map_dbg, scale=y_map_dbg["scale"] / roi_up_factor)
```

---

## Phase 15 — 핵심 버그 수정: y-pixel 단위 변환 오류

> commit `7cba173`

### 문제

2× upscale 모드에서 피크 마커 위치가 여전히 틀렸고,  
intensity 계산 자체가 틀린 것을 발견.

### 원인

`_downscale_series_to_original_roi` 함수가 x 좌표는 1× 공간으로 변환했지만  
**y값은 2× 픽셀 그대로 반환** (`/ factor` 누락):

```python
# ❌ 버그: y_tgt가 2× 픽셀 y 값 그대로
x_tgt = np.arange(original_roi_w) * float(factor)   # [0, 2, 4, ...]
y_tgt = np.interp(x_tgt, x_src, y_src)              # 2× y px 그대로!
return list(range(original_roi_w)), y_tgt, ...
```

이 때문에 1× 기준 y\_map에 2× 픽셀 y가 입력되어 intensity 계산에 최대 2배 오차.

### 수정

```python
# ✅ 수정: y_tgt를 factor로 나눠 1× 픽셀로 변환
y_tgt = np.interp(x_tgt, x_src, y_src) / float(factor)
```

### 효과

| 지표 | 수정 전 | 수정 후 |
|------|--------:|--------:|
| MAE mean | 0.0334 | **0.0323** (−3.3%) |
| 피크 마커 | 절반 높이에 렌더링 | 피크 꼭대기에 정확히 위치 |

---

## 현재 남은 한계

| 항목 | 현상 | 원인 |
|------|------|------|
| **ptp\_r ≈ 0.80** | 예측 intensity range가 GT의 약 80% | DP가 peak apex를 1–2px 낮게 추적 |
| **근접 피크 미분리** | 2θ 간격 < 1° doublet을 하나로 합침 | DP column resolution 한계 |
| **low-contrast 패턴** | true curve candidate 자체가 누락 | raw extraction 단계의 한계 |
| **wrong branch retention** | 후보를 늘려도 wrong branch가 유지됨 | candidate quality 문제 |
| **manual input 필요** | 축 calibration 좌표를 직접 입력해야 함 | 완전 자동화 미구현 |

---

## 이 프로젝트에서 강조할 수 있는 점

| 역량 | 내용 |
|------|------|
| **문제 재정의** | 이미지 복원 문제처럼 보였지만 scientific numeric reconstruction 문제로 재정의하고 목표를 바꿈 |
| **실험 기반 구조 전환** | end-to-end 한계 확인 후 무리한 모델 튜닝 대신 deterministic pipeline으로 전환 |
| **디버깅 가능한 시스템** | candidate dump, selected path monitor, failure taxonomy, debug report로 각 단계 실패 분리 가능 |
| **평가 기준 설계** | visual similarity가 아니라 source numeric 기준의 highres metric 도입 |
| **재현 가능한 실험 환경** | canonical test set + manifest 기반 pair integrity로 실험 재현성 확보 |
| **버그 발견 및 수정** | y-pixel 단위 변환 버그처럼 겉으로 드러나지 않는 systemic bug를 실험 결과 분석을 통해 발견 |

---

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `torch`는 선택 사항입니다. rule-based 파이프라인은 모델 없이 실행됩니다.

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

### manual\_inputs 예시

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
