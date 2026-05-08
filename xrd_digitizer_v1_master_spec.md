# XRD Graph Image → Numeric JSON Reconstruction V1 독립 엔진 조립 설명서

## 이 문서를 어떻게 써야 하는가

이 문서는 아이디어 문서가 아니다.  
이 문서는 앞에서부터 순서대로 따라 하면서 실제 결과물을 하나씩 만드는 조립 설명서다.

중요한 규칙은 아래와 같다.

- 뒤 단계로 먼저 가지 않는다.
- 지금 단계의 파일, 폴더, 출력물이 실제로 생기기 전에는 다음 단계로 넘어가지 않는다.
- 추측으로 넘어가지 않는다. 저장 경로, 파일 이름, 컬럼 이름, 필드 이름을 실제로 맞춘다.
- 모든 중간 결과를 저장한다.
- 사이트 전체를 만들지 않는다. 독립 엔진만 완성한다.
- ML은 baseline이 끝나기 전에는 시작하지 않는다.

이 문서의 목표는 다음 하나다.

이미지 1장을 넣으면, 그 그래프를 만든 원래 측정 데이터에 최대한 가까운 numeric JSON 1개를 안정적으로 뽑아내는 독립 엔진 V1을 완성하는 것

---

## 목차

0. 최종 목표 고정  
1. 지금 필요한 재료 확인  
2. 작업 폴더 만들기  
3. 원본 JSON 전수 조사  
4. 개발용 300개 subset 만들기  
5. clean 이미지와 GT 만들기  
6. styled 이미지 만들기  
7. real-like 이미지 만들기  
8. train/val/test split 만들기  
9. 엔진 입출력 골격 만들기  
10. 사용자 입력 단계 만들기  
11. 전처리 단계 만들기  
12. 곡선 후보 만들기  
13. DP tracing 만들기  
14. recovery / re-entry 만들기  
15. gap fill / smoothing / peak detection 만들기  
16. pixel → numeric 변환 만들기  
17. 평가기 만들기  
18. baseline 실제 실행  
19. 그 다음에만 ML rescue 시작  
20. 지금 따로 조사해야 할 수 있는 것  
21. 별도 심층 리서치용 프롬프트  
22. 파일 포맷 예시 모음  
23. 스크립트별 상세 명세  
24. 최종 체크리스트  
25. 자체 검토 결과

---

## 0. 최종 목표 고정

이 프로젝트의 최종 목표는 아래 하나다.

입력: XRD 그래프 이미지  
출력: 그 그래프를 만든 원래 측정 데이터에 최대한 가까운 numeric JSON

여기서 중요한 점은 다음과 같다.

- 목표는 그림을 다시 그리는 것이 아니다.
- 목표는 이미지처럼 보이는 좌표를 대충 만드는 것이 아니다.
- 목표는 후속 분석에 쓸 수 있는 수치 데이터 JSON을 만드는 것이다.
- 따라서 최종 평가는 “예쁘게 보이냐”가 아니라 “숫자가 맞느냐” 기준이다.

이 장이 끝났는지 확인하는 방법:

팀원 모두가 아래 문장을 그대로 말할 수 있어야 한다.

우리는 사이트 전체를 만드는 것이 아니라, XRD 이미지에서 원래 측정 데이터에 가까운 JSON을 복원하는 독립 엔진을 만든다.

---

## 1. 지금 필요한 재료 확인

### 1.1 이미 가진 것

원본 XRD JSON 약 9만 개  
이건 프로젝트의 가장 중요한 자산이다.

### 1.2 새로 만들어야 하는 것

- 원본 JSON 메타 요약표
- 개발용 subset 300개
- clean 이미지
- styled 이미지
- real-like 이미지
- GT JSON
- split CSV
- baseline 엔진 코드
- evaluator 코드
- debug 산출물

### 1.3 지금 절대 하지 말아야 할 것

- 9만 개 전체를 한 번에 렌더링하지 않는다.
- baseline 없이 ML부터 하지 않는다.
- 웹사이트 전체를 만들지 않는다.
- fancy GUI를 만들지 않는다.

이 장이 끝났는지 확인하는 방법:

- JSON 중심 전략으로 간다는 데 팀이 합의했다.
- 필요한 결과물이 정리되어 있다.

---

## 2. 작업 폴더 만들기

### 2.1 왜 이걸 먼저 하는가

폴더 구조를 먼저 안 만들면 이미지, GT, 메타, split이 뒤섞인다. 그러면 나중에 데이터가 망가진다.

### 2.2 최종 폴더 구조

```text
xrd_digitizer_v1/
├─ data/
│  ├─ source_json/
│  ├─ metadata/
│  │  ├─ all_samples.csv
│  │  ├─ dev_subset.csv
│  │  ├─ split_train.csv
│  │  ├─ split_val.csv
│  │  └─ split_test.csv
│  ├─ rendered_clean/
│  ├─ rendered_styled/
│  ├─ rendered_real_like/
│  ├─ gt/
│  ├─ debug/
│  └─ manifests/
├─ scripts/
│  ├─ scan_source_json.py
│  ├─ build_dev_subset.py
│  ├─ render_clean_dataset.py
│  ├─ render_styled_dataset.py
│  ├─ render_real_like_dataset.py
│  ├─ build_splits.py
│  ├─ validate_dataset_integrity.py
│  └─ summarize_dataset_distribution.py
├─ runner/
│  ├─ run_local.py
│  └─ batch_run.py
├─ core/
│  ├─ io.py
│  ├─ types.py
│  ├─ config.py
│  └─ utils.py
├─ preprocess/
│  ├─ roi.py
│  ├─ perspective.py
│  ├─ color_model.py
│  ├─ masks.py
│  └─ morphology.py
├─ trace/
│  ├─ thinning.py
│  ├─ components.py
│  ├─ candidates.py
│  ├─ dp_trace.py
│  └─ recovery.py
├─ calibrate/
│  ├─ axis_mapping.py
│  └─ numeric_export.py
├─ peaks/
│  ├─ smooth.py
│  └─ detect_peaks.py
├─ eval/
│  ├─ metrics.py
│  ├─ gates.py
│  └─ report.py
├─ debug_artifacts/
│  ├─ save_artifacts.py
│  └─ visualize.py
├─ legacy_compare/
│  └─ compare_with_resnet50.py
├─ ml_rescue_v15/
└─ integration_stub/
   └─ interface_spec.md
```

### 2.3 폴더 역할

- `data/source_json/` : 원본 JSON 저장
- `data/metadata/` : 전수 스캔 결과, subset, split 저장
- `data/rendered_clean/` : clean synthetic 이미지 저장
- `data/rendered_styled/` : styled synthetic 이미지 저장
- `data/rendered_real_like/` : real-like synthetic 이미지 저장
- `data/gt/` : GT JSON 저장
- `data/manifests/` : 이미지-GT-원본JSON 연결표 저장

이 장이 끝났는지 확인하는 방법:

- 모든 폴더가 실제로 생성됐다.
- 빈 폴더라도 이름이 정확히 맞다.

---

## 3. 원본 JSON 전수 조사

이 단계에서는 9만 JSON을 전부 훑어서 메타 표를 만든다.

### 3.1 지금 만들어야 하는 것

- `scripts/scan_source_json.py`
- `data/metadata/all_samples.csv`

### 3.2 입력 JSON에서 반드시 읽을 필드

기본 키 이름은 아래로 가정한다.

- x key: `two_theta_values`
- y key: `intensities`

만약 실제 JSON 키가 다르면, `core/config.py` 에 아래처럼 고정한다.

```python
X_KEY = "two_theta_values"
Y_KEY = "intensities"
```

### 3.3 all_samples.csv 컬럼 고정

아래 컬럼을 그대로 만든다.

- `sample_id`
- `source_json_path`
- `num_points`
- `x_min`
- `x_max`
- `y_min`
- `y_max`
- `y_dynamic_range`
- `dynamic_range_log`
- `peak_count_est`
- `peak_height_ratio`
- `mean_peak_spacing_norm`
- `tail_energy_ratio`
- `fwhm_mean_est`
- `family_id_raw`
- `is_valid`
- `invalid_reason`

### 3.4 sample_id 규칙

기본: 파일명 stem  
충돌 시 뒤에 4자리 해시 추가

예:
- `sample_000001`
- `sample_000001_ab12`

### 3.5 invalid 규칙

아래 중 하나면 invalid다.

- x 배열 없음
- y 배열 없음
- 길이 다름
- `num_points < 50`
- x 비단조 증가
- NaN / Inf 포함
- `y_max - y_min <= 0`

### 3.6 shape diversity feature 계산 규칙

#### peak_count_est
- y를 Savitzky-Golay로 1회 smoothing
- 파라미터: `window=7`, `polyorder=2`
- prominence threshold:

```text
prominence >= max(0.07 * (y_max - y_min), local_noise_floor)
local_noise_floor = max(3*sigma_local, 0.01*(y_max-y_min))
```

- 검출 peak 개수를 저장

#### peak_height_ratio
- `max_peak_height / mean_peak_height`
- peak 0개면 0

#### mean_peak_spacing_norm
- peak x 위치 정렬
- 인접 peak 간 거리 평균 계산
- `(x_max - x_min)` 로 나눔
- peak 2개 미만이면 0

#### tail_energy_ratio
- 마지막 20% x구간 intensity 합 / 전체 intensity 합

#### dynamic_range_log
- `log10((y_max - y_min) + 1)`

#### fwhm_mean_est
- major peak 후보 각각의 half-max width 대략 계산
- 평균 저장
- 구현이 복잡하면 1차에서는 빈 칼럼 허용. 단, 칼럼 자체는 만든다.

### 3.7 scan_source_json.py 함수 명세

필수 함수:

- `load_json(path) -> dict`
- `validate_xy(x, y) -> (bool, str)`
- `estimate_peaks(y) -> dict`
- `compute_shape_features(x, y, peak_info) -> dict`
- `build_row(path, data) -> dict`
- `main() -> None`

### 3.8 scan_source_json.py 실행 예시

```bash
python scripts/scan_source_json.py \
  --source_root data/source_json \
  --output_csv data/metadata/all_samples.csv
```

### 3.9 이 단계가 끝났는지 확인하는 법

- `all_samples.csv` 가 생성됨
- 20개 샘플을 열어 수치가 말이 됨
- `invalid_reason` 이 채워짐

이게 끝나야 다음 단계로 넘어간다.

---

## 4. 개발용 300개 subset 만들기

### 4.1 지금 만들어야 하는 것

- `scripts/build_dev_subset.py`
- `scripts/summarize_dataset_distribution.py`
- `data/metadata/dev_subset.csv`

### 4.2 왜 300개인가

- 너무 적으면 다양성이 부족함
- 너무 많으면 처음 디버깅이 느려짐
- 300개면 clean/styled/real-like 확장 전 개발용으로 충분함

### 4.3 subset 추출 규칙

랜덤 추출 금지. 아래를 따른다.

1단계  
valid 샘플만 남긴다.

2단계  
아래 3개를 각각 3구간으로 나눈다.

- `peak_count_est` : low / mid / high
- `tail_energy_ratio` : low / mid / high
- `dynamic_range_log` : low / mid / high

즉 최대 27개 bin이 생긴다.

3단계  
각 bin에서 가능한 한 비슷한 수로 뽑는다.

4단계  
같은 bin 안에서는 아래 우선순위로 고른다.

- `peak_height_ratio` 다른 것 우선
- `mean_peak_spacing_norm` 다른 것 우선
- 같은 family 추정치 중복 적은 것 우선

### 4.4 최종 분할

- debug: 100
- validation: 100
- holdout: 100

### 4.5 build_dev_subset.py 함수 명세

- `load_metadata(csv_path) -> DataFrame`
- `bin_column(values, n_bins=3) -> Series`
- `build_strat_bins(df) -> DataFrame`
- `sample_balanced_subset(df, total_n=300) -> DataFrame`
- `split_debug_val_holdout(df) -> dict`
- `main() -> None`

### 4.6 실행 예시

```bash
python scripts/build_dev_subset.py \
  --input_csv data/metadata/all_samples.csv \
  --output_csv data/metadata/dev_subset.csv \
  --total_n 300
```

### 4.7 이 단계가 끝났는지 확인하는 법

- `dev_subset.csv` 생성됨
- 300개 있음
- debug/validation/holdout 각 100개
- 분포 요약표 생성됨

이게 끝나야 다음 단계로 넘어간다.

---

## 5. clean 이미지와 GT 만들기

### 5.1 지금 만들어야 하는 것

- `scripts/render_clean_dataset.py`
- `data/rendered_clean/{sample_id}_clean_v1.png`
- `data/gt/{sample_id}_gt.json`
- `data/manifests/clean_manifest.csv`

### 5.2 clean 이미지 고정 규칙

- canvas: `1200 x 900`
- plot_box: `[170, 90, 1120, 780]`
- background: `(255,255,255)`
- axis color: `(30,30,30)`
- axis thickness: `2 px`
- tick length: `6 px`
- x ticks: `8`
- y ticks: `6`
- curve color: `(20,20,20)`
- curve thickness: `2 px`
- font: `Noto Sans 16 px`
- grid: off
- legend: off
- blur: none
- JPEG degradation: none
- perspective: none

### 5.3 clean 생성 순서

1. 원본 JSON에서 x,y 읽기
2. x_min,x_max,y_min,y_max 계산
3. 흰 캔버스 생성
4. plot_box 고정
5. x/y axis 그리기
6. tick 배치
7. tick label 그리기
8. x,y를 픽셀 좌표로 선형 변환
9. anti-aliased polyline으로 curve 그리기
10. png 저장
11. GT JSON 저장
12. manifest 한 줄 기록

### 5.4 tick label 규칙

- x label: 정수 또는 소수 둘째 자리
- y label: 일반 숫자 문자열
- scientific notation 기본 금지
- 라벨이 plot 안을 침범하지 않게 margin 8 px 이상 유지

### 5.5 GT JSON 필드 고정

- `sample_id`
- `source_json_path`
- `x_values`
- `y_values`
- `plot_box`
- `pixel_curve_path`
- `per_column_y_gt`
- `axis_metadata`
- `peak_indices`
- `peak_x_values`
- `peak_y_values`
- `peak_prominences`
- `peak_pixel_points`
- `major_peak_indices`
- `render_variant`

### 5.6 pixel_curve_path 규칙

픽셀 좌표로 변환된 점을 순서대로 저장  
형식: `[[x1,y1],[x2,y2], ...]`

### 5.7 per_column_y_gt 규칙

- plot_box 내부 모든 정수 x column마다 y 한 개 저장
- 직접 지나지 않으면 선형 보간
- 값이 없을 때 nearest 금지, 선형 보간 우선
- plot_box 밖은 저장 안 함

### 5.8 render_clean_dataset.py 함수 명세

- `load_subset(csv_path) -> DataFrame`
- `load_sample_json(path) -> dict`
- `map_xy_to_pixels(x, y, plot_box) -> tuple`
- `render_clean_image(sample) -> PIL.Image | np.ndarray`
- `build_gt(sample, pixel_curve_path, plot_box) -> dict`
- `append_manifest_row(row) -> None`
- `main() -> None`

### 5.9 실행 예시

```bash
python scripts/render_clean_dataset.py \
  --subset_csv data/metadata/dev_subset.csv \
  --output_dir data/rendered_clean \
  --gt_dir data/gt \
  --manifest_csv data/manifests/clean_manifest.csv \
  --max_samples 100
```

### 5.10 이 단계가 끝났는지 확인하는 법

- clean 100장 생성됨
- 각 이미지마다 GT 존재
- manifest 생성됨
- 5개 이상 직접 열어 확인함

이게 끝나야 다음 단계로 넘어간다.

---

## 6. styled 이미지 만들기

### 6.1 지금 만들어야 하는 것

- `scripts/render_styled_dataset.py`
- `data/rendered_styled/{sample_id}_styled_v{n}.png`
- `data/manifests/styled_manifest.csv`

### 6.2 styled의 원칙

styled는 수치 데이터는 그대로 두고 스타일만 바꾸는 것이다. GT는 바뀌지 않는다.

### 6.3 styled variant 고정

#### styled_v1 : 기본 논문형
- background: pure white
- curve color: black or dark navy
- curve thickness: 2.0 px
- font: Noto Sans 16 px
- grid: off
- legend: off
- axis thickness: 2 px
- tick count: x=8, y=6

#### styled_v2 : 컬러 라인 + grid형
- background: white
- curve color: dark blue / dark red / dark green 중 1개
- curve thickness: 2.0 px
- font: DejaVu Sans 17 px
- grid: on
- grid color: light gray
- legend: off
- tick count: x=8, y=6

#### styled_v3 : 보고서 범례형
- background: white
- curve color: dark red or dark blue
- curve thickness: 2.5 px
- font: Arial-like 16 px
- grid: off
- legend: on
- legend 위치: top-right 또는 top-left
- legend box 크기: plot width 18~25%, plot height 8~12%
- legend text 길이: 8~20 chars

#### styled_v4 : 옅은 배경 + 얇은 선형
- background: 밝은 회색/청회색
- curve color: medium gray / muted blue / muted green
- curve thickness: 1.5 px
- font: Noto Sans 16 px
- grid: on
- grid color: very light gray
- legend: off
- tick count: x=10, y=8

#### styled_v5 : 저대비 tail 테스트형
- background: white
- curve color: dark gray
- curve thickness: 1.8 px
- font: DejaVu Sans 16 px
- grid: off 또는 매우 약하게 on
- legend: optional
- 오른쪽 마지막 20% x 구간에서 curve contrast 10~20% 약화

### 6.4 color 팔레트 고정

- black: `(20,20,20)`
- dark navy: `(25,45,90)`
- dark red: `(140,35,35)`
- dark green: `(35,110,60)`
- medium gray: `(90,90,90)`
- muted blue: `(90,120,170)`
- muted green: `(90,140,110)`

### 6.5 legend 규칙

- 위치 후보: top-left, top-right, upper-center-right
- box padding: 8 px
- line 길이: 28~36 px
- text 후보:
  - Sample A
  - XRD Pattern
  - Measured
  - Run 01
  - Pattern-1
- line color = curve color
- text color = dark gray

### 6.6 grid 규칙

- x grid는 x tick 위치와 일치
- y grid는 y tick 위치와 일치
- 색: `(220,220,220)` 또는 `(230,230,235)`
- 두께: `1 px`

### 6.7 styled 생성 순서

1. clean 기반 렌더 호출
2. variant 선택
3. style 파라미터 적용
4. grid 적용
5. legend 적용
6. anti-alias line 적용
7. png 저장
8. manifest 저장

### 6.8 render_styled_dataset.py 함수 명세

- `load_clean_manifest(path) -> DataFrame`
- `get_style_variant(variant_id) -> dict`
- `render_styled_from_gt(gt, variant_cfg) -> image`
- `write_manifest_row(row) -> None`
- `main() -> None`

### 6.9 실행 예시

```bash
python scripts/render_styled_dataset.py \
  --clean_manifest data/manifests/clean_manifest.csv \
  --output_dir data/rendered_styled \
  --manifest_csv data/manifests/styled_manifest.csv \
  --variants styled_v1 styled_v3 styled_v5 \
  --max_samples 50
```

### 6.10 이 단계가 끝났는지 확인하는 법

- styled 50장 이상 생성됨
- v1,v3,v5 최소 포함
- GT 안 바뀜
- baseline에 넣으면 clean보다 어려움

이게 끝나야 다음 단계로 넘어간다.

---

## 7. real-like 이미지 만들기

### 7.1 지금 만들어야 하는 것

- `scripts/render_real_like_dataset.py`
- `data/rendered_real_like/{sample_id}_real_v{n}.png`
- `data/manifests/real_manifest.csv`

### 7.2 real-like 원칙

real-like는 styled에 현실적인 품질 저하를 더한 것이다. 정답 수치열은 그대로다.

### 7.3 real-like variant 고정

#### real_v1 : JPEG 압축형
- base: styled_v2 or styled_v3
- JPEG quality: 75
- blur: none
- perspective: none

#### real_v2 : blur + compression형
- base: styled_v4 or styled_v5
- JPEG quality: 55
- Gaussian blur sigma: 0.6
- brightness: -3% ~ +3%
- contrast: -5% ~ 0%

#### real_v3 : screenshot/photo-like형
- base: styled_v2 or styled_v3
- JPEG quality: 75
- blur sigma: 0.6 or 1.0
- border 가능
- shadow 가능
- brightness shift: -5% ~ +5%

#### real_v4 : weak tail + slight perspective형
- base: styled_v5
- JPEG quality: 75 or 55
- blur sigma: 0.6
- tail contrast drop 적용
- perspective distortion 적용

### 7.4 JPEG 규칙

- quality 후보: 95, 75, 55
- 개발 초기에는 75와 55만 사용
- 구현은 jpg 저장 -> 다시 읽기 -> png 저장 가능

### 7.5 blur 규칙

- sigma 후보: 0, 0.6, 1.0
- 0.6 기본 hard case
- 1.0 더 어려운 case

### 7.6 brightness / contrast 규칙

- brightness shift: -5% ~ +5%
- contrast scale: 0.9 ~ 1.0

### 7.7 border / shadow 규칙

- outer border: 10~30 px
- shadow는 매우 약하게
- plot 전체를 가리는 shadow 금지

### 7.8 perspective 규칙

- 확률: 0.25
- 각 코너 오프셋: 가로/세로 길이의 1~3%

### 7.9 tail contrast drop 규칙

- 적용 구간: 마지막 20% x span
- 감소량: 10~20%

### 7.10 real-like 생성 순서

1. styled 이미지 선택
2. variant 선택
3. jpg compression 적용 여부 결정
4. blur 적용 여부 결정
5. brightness/contrast 적용
6. border/shadow 적용
7. perspective 적용
8. 저장 및 manifest 기록

### 7.11 render_real_like_dataset.py 함수 명세

- `load_styled_manifest(path) -> DataFrame`
- `get_real_variant(variant_id) -> dict`
- `apply_jpeg_artifact(image, quality) -> image`
- `apply_blur(image, sigma) -> image`
- `apply_perspective(image, offset_ratio) -> image`
- `apply_tail_drop(image, plot_box, ratio) -> image`
- `main() -> None`

### 7.12 실행 예시

```bash
python scripts/render_real_like_dataset.py \
  --styled_manifest data/manifests/styled_manifest.csv \
  --output_dir data/rendered_real_like \
  --manifest_csv data/manifests/real_manifest.csv \
  --variants real_v2 real_v4 \
  --max_samples 50
```

### 7.13 이 단계가 끝났는지 확인하는 법

- real-like 50장 이상 생성됨
- v2,v4 포함
- blur/JPEG/tail drop 케이스가 있음
- baseline failure가 기록되기 시작함

이게 끝나야 다음 단계로 넘어간다.

---

## 8. train/val/test split 만들기

### 8.1 지금 만들어야 하는 것

- `scripts/build_splits.py`
- `data/metadata/split_train.csv`
- `data/metadata/split_val.csv`
- `data/metadata/split_test.csv`

### 8.2 family_id 규칙

1순위  
실제 metadata에 동일 시료 / 동일 조성 / 동일 실험 세트 정보가 있으면 그걸 family_id로 쓴다.

2순위 surrogate family_id

- `peak_count_bin`
- `major_peak_position_signature`
- `dynamic_range_bin`

#### peak_count_bin
- low / mid / high

#### major_peak_position_signature
- major peak x 위치를 x_range로 정규화
- 상위 3개를 0.05 단위 양자화
- 예: `0.15_0.32_0.71`

#### dynamic_range_bin
- low / mid / high

최종 예:

`pc_mid__pp_0.15_0.32_0.71__dr_high`

### 8.3 split 비율

- train 80%
- val 10%
- test 10%

### 8.4 build_splits 함수 명세

- `load_subset(csv_path) -> DataFrame`
- `assign_family_id(df) -> DataFrame`
- `group_split_by_family(df, ratios=(0.8,0.1,0.1)) -> dict`
- `write_split_csv(df, out_path) -> None`
- `main() -> None`

### 8.5 실행 예시

```bash
python scripts/build_splits.py \
  --input_csv data/metadata/dev_subset.csv \
  --train_csv data/metadata/split_train.csv \
  --val_csv data/metadata/split_val.csv \
  --test_csv data/metadata/split_test.csv
```

### 8.6 이 단계가 끝났는지 확인하는 법

- 같은 family가 둘 이상의 split에 없음
- split CSV가 실제 생성됨

이게 끝나야 다음 단계로 넘어간다.

---

## 9. 엔진 입출력 골격 만들기

### 9.1 지금 만들어야 하는 것

- `runner/run_local.py`
- `runner/batch_run.py`
- `core/types.py`
- `core/io.py`
- `core/config.py`

### 9.2 최소 동작 목표

이미지 1장을 넣으면
- 입력 메타를 읽고
- 처리 함수 틀을 호출하고
- 결과 JSON 껍데기와 debug 파일을 저장한다.

### 9.3 입력 형식 고정

필수:
- `image_path`
- `plot_box`
- `x_axis_points`
- `x_axis_values`
- `y_axis_points`
- `y_axis_values`
- `color_sample_point`

선택:
- `legend_ignore_boxes`
- `perspective_corners`
- `color_resample_points`

### 9.4 출력 형식 고정

최종 JSON 필수 필드:
- `two_theta_values`
- `intensities`
- `x_range`
- `y_range`
- `quality`
- `confidence`
- `warnings`
- `used_manual_inputs`

debug 출력:
- `overlay`
- `color_mask`
- `combined_mask`
- `skeleton`
- `candidate_map`
- `trace_path`
- `peaks_overlay`
- `debug.json`

### 9.5 타입 명세

`core/types.py` 에 아래 dataclass를 만든다.

```python
@dataclass
class ManualInputs:
    plot_box: list
    x_axis_points: list
    x_axis_values: list
    y_axis_points: list
    y_axis_values: list
    color_sample_point: list
    legend_ignore_boxes: list | None = None
    perspective_corners: list | None = None
    color_resample_points: list | None = None

@dataclass
class RunResult:
    two_theta_values: list
    intensities: list
    x_range: list
    y_range: list
    quality: dict
    confidence: float
    warnings: list
    used_manual_inputs: dict
```

### 9.6 실행 예시

```bash
python runner/run_local.py \
  --image_path data/rendered_clean/sample_000123_clean_v1.png \
  --manual_inputs_path examples/manual_input_sample.json \
  --output_json_path outputs/sample_000123_result.json \
  --debug_dir outputs/debug_sample_000123
```

### 9.7 이 단계가 끝났는지 확인하는 법

- 비어 있는 pipeline이어도 JSON 껍데기를 저장함
- debug 디렉토리에 파일이 생김

이게 끝나야 다음 단계로 넘어간다.

---

## 10. 사용자 입력 단계 만들기

### 10.1 왜 필요한가

V1은 semi-automatic이므로 사용자 입력이 반드시 필요하다.

### 10.2 입력 규칙

- 좌표는 원본 이미지 기준 픽셀 좌표
- 축 값은 float
- 클릭 순서와 사용 여부를 `used_manual_inputs` 에 기록

### 10.3 에러 규칙

- ROI 비어 있음 -> 에러
- x축/y축 기준점 해석 불가 -> 에러
- color sample이 ROI 밖 -> 에러

### 10.4 클릭 예산 고정

정상:
- plot box 1
- x축 2점
- y축 2점
- color 1
- 총 6 clicks

정상 최대:
- 기본 6 + legend ignore 1 = 7

하드:
- 기본 6 + legend ignore 1 + perspective 4 = 11

하드 최대:
- 기본 11 + color resample 1 = 12

`12 초과 -> UX fail 기록`

### 10.5 예시 입력 JSON

```json
{
  "plot_box": [170, 90, 1120, 780],
  "x_axis_points": [[170, 780], [1120, 780]],
  "x_axis_values": [10.0, 80.0],
  "y_axis_points": [[170, 780], [170, 90]],
  "y_axis_values": [0.0, 15000.0],
  "color_sample_point": [420, 310],
  "legend_ignore_boxes": [],
  "perspective_corners": null,
  "color_resample_points": []
}
```

### 10.6 이 단계가 끝났는지 확인하는 법

- 입력 JSON을 읽고 검증 가능
- 클릭 수를 계산 가능
- 오류를 제대로 띄움

이게 끝나야 다음 단계로 넘어간다.

---

## 11. 전처리 단계 만들기

### 11.1 지금 만들어야 하는 것

- `preprocess/roi.py`
- `preprocess/perspective.py`
- `preprocess/color_model.py`
- `preprocess/masks.py`
- `preprocess/morphology.py`

### 11.2 ROI 규칙

- 기본은 plot_box crop
- plot_box 밖은 무시

### 11.3 perspective 규칙

- perspective_corners 있으면 homography 적용
- 없으면 적용 안 함
- 자동 perspective는 하지 않음

### 11.4 color prototype 규칙

- color space: Lab
- prototype 3개
- neighborhood 5x5
- 위치: ROI x 20%, 50%, 80%

### 11.5 mask 규칙

#### mask_A
- color-distance foreground mask

#### mask_B
- edge/gradient 보조 mask

#### combine

```text
mask = mask_A OR (mask_B AND dilate(mask_A, 3x3))
```

### 11.6 전처리 단계 수정 원칙

기존 combined_mask는 유지한다.  
다만 combined_mask 이후 결과를 바로 thinning의 입력으로만 해석하지 않는다.

전처리 단계에서는 아래 두 종류의 candidate source를 분리해 유지한다.

- combined_mask 기반 raw candidate
- thinning / centerline 기반 candidate

즉, 전처리 단계의 목적은 아래 둘을 분리하는 것이다.

- 후보를 보존하는 것
- 후보를 정제하는 것

### 11.7 raw candidate 보존 규칙

combined_mask에서 살아남은 curve-like pixel은 raw candidate 자산으로 본다.  
thinning 전에 raw candidate 상태를 별도 저장한다.

약한 tail, 얇은 peak shoulder, blur로 두꺼워진 선은 thinning에서 손실될 수 있으므로 raw candidate를 기준 자산으로 유지한다.

thinning 결과는 raw candidate를 더 날씬하게 만드는 후처리로만 해석한다.  
thinning 결과는 후보 정제 수단이지, 후보의 유일한 원천이 아니다.

### 11.8 morphology 규칙

- close kernel 3x3
- iter 1
- area < 80 제거

단, morphology는 noise 제거를 위한 것이며 candidate recall을 과도하게 깎지 않도록 해석한다.  
세부 threshold는 empirical tuning 대상으로 남긴다.

### 11.9 저장해야 하는 debug 파일

- `color_mask.png`
- `combined_mask.png`
- `roi_preview.png`
- `raw_candidate_mask.png`
- `pre_skeleton_candidates.png`

### 11.10 이 단계가 끝났는지 확인하는 법

- tail 일부가 mask_B로 살아남음
- ignore box 영역이 비어 있음
- combined_mask가 저장됨
- raw candidate와 thinning 결과를 비교할 수 있음
- thinning 후 사라진 후보가 raw candidate에서는 보이는지 확인 가능

이게 끝나야 다음 단계로 넘어간다.

---

## 12. 곡선 후보 만들기

### 12.1 지금 만들어야 하는 것

- `trace/thinning.py`
- `trace/components.py`
- `trace/candidates.py`

### 12.2 thinning 규칙

- 기본: `ximgproc.thinning`
- fallback: morphological centerline approximation
- 단, thinning 결과는 최종 후보의 유일한 source가 아니다.

### 12.3 component score 고정식

```text
score = 2.0*x_coverage + 1.0*continuity + 0.5*log(1+length)
      - 1.2*edge_penalty - 1.0*text_penalty
```

### 12.4 candidate 생성 수정 원칙

candidate 생성은 아래 3단계 구조로 고정한다.

1) raw candidate (recall 보존용)  
2) filtered candidate (precision 향상용)  
3) final DP candidate (tractable search용)

중요 규칙:

- column별 top-K 제한은 raw 단계에서 절대 적용하지 않는다.
- pruning은 filtered 이후에만 적용한다.
- thinning은 candidate source 중 하나일 뿐이며 raw candidate와 동등하게 취급한다.

### 12.5 candidate 규칙

초기 후보 풀 단계:
- column별 후보를 넓게 수집한다.
- source는 아래 둘을 모두 사용한다.
  - raw candidate
  - thinning / centerline candidate
- component 정보를 후보 confidence 계산에 반영한다.

필터링 단계:
- confidence 기반으로 후보를 줄인다.
- 이 단계 이후에만 DP 입력용 candidate 수를 제한한다.

### 12.6 candidate confidence 구성 요소

candidate confidence는 단일 휴리스틱이 아니라 아래 항목의 조합으로 계산한다.

- color consistency
- local continuity
- component support
- text / axis / grid distance penalty

즉, “가늘고 긴 component에 속한다”만으로 후보를 고르지 않는다.  
곡선 색 일관성, 열 간 연속성, 텍스트/축/그리드와의 거리까지 같이 본다.

candidate_confidence ∈ [0, 1]

계산식은 아래로 고정한다.

```text
confidence =
  0.35 * color_consistency
+ 0.25 * local_continuity
+ 0.20 * component_support
+ 0.20 * (1 - penalty)
```

각 항목 정의:

#### 1) color_consistency ∈ [0,1]
- Lab color distance 기반
- prototype 3개 평균 거리 d 사용

```text
color_consistency = exp(-d / 20)
```

#### 2) local_continuity ∈ [0,1]
- 이전 column best candidate와의 y 차이 기반

```text
dy = |y_t - y_{t-1}|
local_continuity = exp(-dy / 8)
```

- 첫 column은 `1.0`으로 고정한다.

#### 3) component_support ∈ [0,1]
- component score를 logistic으로 normalize
- component_score_raw 사용

```text
component_support = 1 / (1 + exp(-0.8*(score - 2)))
```

#### 4) penalty ∈ [0,1]
- text / axis / grid proximity penalty
- `d_min` = 가장 가까운 text/axis/grid 거리(px)

```text
penalty = exp(-d_min / 6)
```

### 12.7 empty column 처리 규칙

후보가 없는 column은 즉시 missing 처리하지 않는다.  
인접 column 기반 보조 후보 탐색 개념을 둔다.  
인접 column에서도 적절한 후보가 없을 때만 missing으로 기록한다.

추가 고정 규칙:

```text
neighbor_radius = 3 columns
```

동작:
1. 현재 column 후보 없음
2. ±3 column 범위에서 후보 수집
3. 인접 후보 기반 linear interpolation으로 임시 후보 생성
4. 그래도 없으면 missing으로 기록

### 12.8 candidate map 저장 규칙

candidate map은 단일 이미지가 아니라 최소 아래 3단계를 구분해 저장한다.

- raw candidate
- filtered candidate
- final DP candidate

구현 규칙:
- `candidate_map_raw.png`
- `candidate_map_filtered.png`
- `candidate_map_final.png`

### 12.9 filtering 및 final DP candidate 규칙

filtered candidate 생성 규칙:

```text
min_conf_keep = 0.25
```

- `confidence < 0.25` 이면 제거

추가 pruning:

```text
max_candidates_after_filter = 12
```

- column별 confidence 상위 12개만 유지

DP 입력 전 최종 제한:

```text
max_candidates_for_DP = 6
```

선택 기준:
- confidence 상위 6개
- 동일 component 중복 제거 (1개만 유지)

### 12.10 저장해야 하는 debug 파일

- `skeleton.png`
- `components_overlay.png`
- `candidate_map_raw.png`
- `candidate_map_filtered.png`
- `candidate_map_final.png`

### 12.11 debug 해석 규칙

candidate 관련 디버그 해석 기준:

- raw에 정답 후보가 있어야 한다. 이것은 recall 기준이다.
- filtered에서 사라지면 filtering 문제로 본다.
- final에서 사라지면 pruning 문제로 본다.
- DP 실패 시:
  - raw 없음 -> `candidate_starvation`
  - raw 있음 / final 없음 -> filtering 또는 pruning 과도
  - final 있음 / tracing 실패 -> DP 문제

### 12.12 이 단계가 끝났는지 확인하는 법

- skeleton 생성됨
- component score 계산됨
- raw / filtered / final candidate map이 각각 생성됨
- 정답 후보가 초기 단계에서 너무 빨리 사라지지 않음
- missing column 비율을 측정할 수 있음

이게 끝나야 다음 단계로 넘어간다.

---

## 13. DP tracing 만들기

### 13.1 지금 만들어야 하는 것

- `trace/dp_trace.py`

### 13.2 DP cost

```text
J = α*|Δy| + β*|Δ²y| + γ*(1-candidate_conf) + δ*component_switch_penalty
```

고정값:

- `W = max(25, round(0.03 * Pw))`
- `α = 1.0`
- `β = 0.35`
- `γ = 0.8`
- `δ = 1.2`
- `K = 3`

위 cost form은 유지한다.

### 13.3 DP 수정 원칙

DP 자체를 더 복잡하게 만드는 것이 1순위가 아니다.  
우선순위는 아래와 같다.

- candidate recall 부족을 드러내는 것
- candidate가 충분한지 진단 가능한 구조를 두는 것
- hard case에서 단일 경로만 남기지 않는 것

### 13.4 candidate recall 해석 규칙

DP는 존재하지 않는 정답 후보를 복원할 수 없다.  
따라서 candidate recall은 DP 이전의 1급 진단 항목으로 본다.

### 13.5 multi-hypothesis 규칙

기본은 단일 최적 경로 tracing으로 시작할 수 있다.  
단, hard case에서는 아래 구조를 허용한다.

- multi-hypothesis tracing
- beam-style tracing

이는 baseline을 복잡하게 만들기 위한 것이 아니라, 잘못된 경로가 얼마나 불안정한지 진단하기 위한 장치다.

### 13.6 출력

- traced curve
- trace score
- valid_ratio
- path margin 관련 진단값

### 13.7 blockwise trace 진단 규칙

아래 항목은 block 단위로 저장 가능해야 한다.

- top-1 cumulative cost
- top-2 cumulative cost
- top-1 / top-2 margin

이를 통해 “후보 부족”, “branch lock-in”, “경로 불안정”을 구분할 수 있어야 한다.

### 13.8 저장해야 하는 debug 파일

- `trace_path.png`
- `trace_debug.json`

`trace_debug.json` 에는 가능하면 아래를 포함한다.

- `blockwise_top1_cost`
- `blockwise_top2_cost`
- `path_margin`

### 13.9 이 단계가 끝났는지 확인하는 법

- clean 10장에서 큰 오류 없이 tracing 됨
- trace_path가 저장됨
- hard case에서 path margin 불안정 여부를 볼 수 있음
- DP 실패가 candidate 부족 때문인지 경로 선택 문제인지 분리해 볼 수 있음

이게 끝나야 다음 단계로 넘어간다.

---

## 14. recovery / re-entry 만들기

### 14.1 왜 필요한가

DP는 오접속이 길게 전파될 수 있다.

### 14.2 지금 만들어야 하는 것

- `trace/recovery.py`

### 14.3 trigger 규칙

아래 중 하나면 recovery 진입.

- 최근 40 columns에서 `valid_ratio < 0.85`
- 연속 missing > `max(8, round(0.015*Pw))`
- local trace score 30% 이상 급락
- best vs second-best cost margin 12 columns 연속 0.15 이하

### 14.4 recovery 수정 원칙

recovery의 1순위 목적은 잘못된 branch로 갈아타는 것이 아니라, candidate를 다시 확보하고 국소 구간을 재추적하는 것이다.

### 14.5 recovery 행동

아래 순서로 해석한다.

1. 최근 구간 candidate 재탐색
2. 후보 재점수화
3. local re-trace
4. 그 뒤에 second-best branch 비교
5. 그래도 실패하면 fail-fast
6. 사용자 추가 입력 요청

즉, second-best branch 재평가는 recovery의 첫 단계가 아니라 후보 재획득 이후의 단계다.

### 14.6 추가 입력 순서

- legend ignore
- perspective
- color resample
- ROI re-box

### 14.7 failure taxonomy 연계 규칙

recovery 실패가 반복되면 단순 실패로 끝내지 않는다.  
구조화된 failure taxonomy label을 남긴다.

예:
- `candidate_starvation`
- `wrong_branch_lock_in`
- `legend_capture`
- `grid_confusion`

### 14.8 저장해야 하는 debug 파일

- `recovery_log.json`
- `branch_compare.png`
- `recovery_candidates_before.png`
- `recovery_candidates_after.png`

### 14.9 이 단계가 끝났는지 확인하는 법

- failure-driven 이미지에서 recovery가 한 번 이상 작동함
- recovery 로그가 남음
- recovery 전후 candidate 상태를 비교할 수 있음
- branch 교체 이전에 candidate 재획득이 시도되었는지 확인 가능

이게 끝나야 다음 단계로 넘어간다.

---

## 15. gap fill / smoothing / peak detection 만들기

### 15.1 gap fill 규칙

- linear interpolation only
- `gap ≤ 10 px` 만 허용

### 15.2 smoothing 규칙

Savitzky-Golay

- `window = nearest odd integer to max(5, min(11, round(0.012*Pw)))`
- `polyorder = 2`
- peak-preservation 실패 시 window 2 감소

### 15.3 peak GT 및 peak detection 규칙

smoothing 후 peak detection

```text
prominence >= max(0.07 * (y_max - y_min), local_noise_floor)
local_noise_floor = max(3*sigma_local, 0.01*(y_max-y_min))
min_peak_distance = max(3, round(0.004 * num_points))
min_peak_height = y_min + 0.03 * (y_max - y_min)
```

### 15.4 major peak 규칙

- prominence 순 정렬
- `top max(3, ceil(0.1 * num_peaks_detected))`
- max 8개 cap

### 15.5 저장해야 하는 debug 파일

- `smoothed_trace.png`
- `peaks_overlay.png`
- `peak_debug.json`

### 15.6 이 단계가 끝났는지 확인하는 법

- peak list 생성됨
- major peak 생성됨
- overlay로 확인 가능

이게 끝나야 다음 단계로 넘어간다.

---

## 16. pixel → numeric 변환 만들기

### 16.1 지금 만들어야 하는 것

- `calibrate/axis_mapping.py`
- `calibrate/numeric_export.py`

### 16.2 규칙

- x mapping: 2-point linear
- y mapping: 2-point linear + inversion
- calibration roundtrip error 필수 계산

### 16.3 출력 필드

- `two_theta_values`
- `intensities`
- `x_range`
- `y_range`
- `calibration_meta`

### 16.4 이 단계가 끝났는지 확인하는 법

- traced curve가 numeric JSON으로 변환됨
- x_range, y_range 나옴
- roundtrip error 계산됨

이게 끝나야 다음 단계로 넘어간다.

---

## 17. 평가기 만들기

### 17.1 지금 만들어야 하는 것

- `eval/metrics.py`
- `eval/gates.py`
- `eval/report.py`

### 17.2 metrics 구조 분리

metrics는 3종류로 구분한다.

#### 1) main metrics (gate 판단용)
- `curve_y_mae_px`
- `major_peak_x_error`
- `major_peak_y_error`
- `peak_recall@fixed_prominence`
- `max_gap_px`
- `calibration_roundtrip_error`

#### 2) debug metrics (분석용)
- `IoU`
- `valid_ratio`
- `trace_score`
- `tail_mae_px`
- `tail_collapse_rate`
- `peak_precision`
- `peak_f1`
- `prominence_preservation`

#### 3) diagnosis-only metrics (필수)
다음은 mandatory diagnosis metric 으로 본다.

- `candidate_recall_per_column`
- `empty_column_rate`
- `recovery_success_rate`
- `reentry_count`
- `path_margin_instability`

### 17.3 debug metrics 해석 규칙

기존 debug metric은 유지한다.  
다만 역할을 아래처럼 명확히 분리한다.

- `peak_f1` 는 gate 금지
- `peak_f1` 는 debug / 분석용으로만 사용
- diagnosis-only metrics 는 baseline diagnosis용 mandatory metric 으로 본다.

### 17.4 acceptance gate

#### clean (baseline 1차 목표)
- `y_mae_px ≤ 5`
- `major_peak_x_error ≤ 4 px`
- `major_peak_y_error ≤ 6 px`
- `peak_recall ≥ 0.70`
- `max_gap_px ≤ 10`
- `calibration_roundtrip_error ≤ 1 px`

#### styled (robustness 확인용, gate 완화)
- `y_mae_px ≤ 6`
- `major_peak_x_error ≤ 6 px`
- `major_peak_y_error ≤ 8 px`
- `peak_recall ≥ 0.68`
- `max_gap_px ≤ 14`
- `calibration_roundtrip_error ≤ 1.8 px`

#### real-like (stress-test, gate 느슨)
- `y_mae_px ≤ 8`
- `major_peak_x_error ≤ 8 px`
- `major_peak_y_error ≤ 10 px`
- `peak_recall ≥ 0.60`
- `max_gap_px ≤ 18`
- `calibration_roundtrip_error ≤ 2.5 px`

### 17.5 hard pass/fail 해석 규칙

- clean 통과 -> baseline 정상 동작
- styled/real-like fail -> 허용 가능

즉, 초기 baseline 해석에서는 clean을 가장 먼저 안정적으로 통과시키는 것이 우선이다.

### 17.6 diagnostic interpretation 규칙

styled / real-like는 아래 목적으로 해석한다.

- robustness 약점 탐지
- failure taxonomy 수집
- candidate starvation / branch lock-in 분석

즉:
- styled / real-like fail은 “문제”가 아니라 “데이터”다.

### 17.7 peak_f1 규칙

- `peak_f1` 는 절대 gate에 사용하지 않는다.
- `peak_f1` 는 debug / 분석용으로만 사용한다.

### 17.8 failure 해석 연결 규칙

다음 metric과 failure taxonomy를 직접 연결한다.

- `empty_column_rate` 증가 -> `candidate_starvation`
- `path_margin_instability` 증가 -> `wrong_branch_lock_in`
- `recovery_success_rate` 저하 -> recovery 실패
- `tail_mae_px` 증가 -> `tail_collapse`

### 17.9 최종 해석 우선순위

baseline 해석 우선순위:

1. clean 통과 여부
2. candidate recall 상태
3. DP 안정성 (path margin)
4. recovery 동작 여부
5. styled / real-like failure taxonomy

### 17.10 이 단계가 끝났는지 확인하는 법

- metric 계산 가능
- pass/fail 출력 가능
- 리포트 저장 가능
- candidate starvation 계열 failure를 수치로 볼 수 있음

이게 끝나야 다음 단계로 넘어간다.

---

## 18. baseline 실제 실행

### 18.1 실행 순서

- clean 10장 실행
- debug 확인
- clean 100장 batch
- styled 50장 batch
- real-like 50장 batch
- failure taxonomy 기록

### 18.2 지금 중요한 기준

- 끝까지 도는가
- 왜 실패했는지 보이는가
- JSON이 분석에 쓸 만한가

### 18.3 baseline 실행 해석 원칙

초기 baseline 단계에서는 아래 우선순위를 따른다.

- clean을 안정적으로 통과시키는가
- styled에서 robustness 약점을 드러낼 수 있는가
- real-like에서 stress-test failure를 수집할 수 있는가

즉 styled / real-like는 초기 acceptance보다 failure-driven development 용도가 더 강하다.

### 18.4 expected failure taxonomy

최소 아래 label을 고정한다.

- `tail_collapse` : tail 구간이 약해지며 후보가 무너져 수치 복원이 붕괴되는 현상
- `text_intrusion` : 글자 stroke가 curve 후보로 섞여 들어오는 현상
- `grid_confusion` : grid line이 후보로 잘못 선택되는 현상
- `legend_capture` : legend 선 또는 legend box 주변 요소를 curve로 잡는 현상
- `peak_miss_after_smoothing` : tracing은 되었지만 smoothing 이후 peak가 사라지는 현상
- `candidate_starvation` : 특정 연속 구간에서 정답 curve 후보 자체가 거의 남지 않는 현상
- `wrong_branch_lock_in` : 초반 오접속 이후 잘못된 branch에 계속 묶이는 현상
- `calibration_mismatch` : tracing은 되었지만 축 매핑이 어긋나 numeric export가 틀어지는 현상

### 18.5 failure 기록 규칙

failure 기록은 자유 메모가 아니라 label 기반으로 축적한다.  
한 샘플에 복수 label이 붙을 수 있다.  
failure taxonomy는 이후 ML rescue 필요성 판단의 근거 데이터로 사용한다.

### 18.6 이 단계가 끝났는지 확인하는 법

- clean에서 baseline 동작
- styled/real-like 실패 유형 파악
- failure taxonomy 문서 시작
- candidate starvation / wrong branch / calibration mismatch를 분리 기록 가능

이게 끝나야 다음 단계로 넘어간다.

---

## 19. 그 다음에만 ML rescue 시작

### 19.1 ML 시작 전 조건

- baseline 끝까지 동작
- debug 충분
- synthetic renderer 있음
- evaluator 있음
- failure 분류 끝남

### 19.2 ML이 필요한 경우

- thin peak 반복 miss
- low-contrast tail collapse
- styled/real-like에서 mask 불안정

### 19.3 초기 ML rescue 방향

1순위 후보:
- 1D CNN + BiLSTM 보조 구조

대안:
- lightweight 2D U-Net / HRNet 계열

단, baseline 전에는 금지.

### 19.4 이 단계가 끝났는지 확인하는 법

- baseline이 아직 끝나지 않았으면 진행 금지
- baseline 끝났으면 rescue 필요성 문서 작성

---

## 20. 지금 따로 조사해야 할 수 있는 것

- shape diversity feature 추가 필요 여부
- usable metadata 존재 여부
- family별 prominence 조정 필요 여부
- styled/real-like 강도 empirical tuning
- 1D CNN+BiLSTM vs 2D lightweight 비교
- legend/font/grid 실제 분포
- JPEG/blur/perspective 최적 범위
- major peak domain-specific 정교화
- stratified binning 최적화

---

## 21. 별도 심층 리서치용 프롬프트

당신은 XRD graph image → numeric JSON reconstruction 프로젝트의 데이터 전략 검증자다.

배경:
- 현재 팀은 원본 XRD JSON 약 9만 개를 보유하고 있다.
- 목표는 이 JSON으로부터 clean / styled / real-like synthetic 이미지를 만들어, rule-based baseline 개발과 이후 ML rescue 학습에 쓰는 것이다.
- 현재 필요한 것은 구현 일반론이 아니라, 아래 특정 판단 항목에 대한 결정 근거다.

조사할 질문:
1. XRD JSON의 shape diversity를 계량적으로 분류하려면 peak_count, peak_height_ratio, mean_peak_spacing_norm, tail_energy_ratio, dynamic_range_log 외에 추가할 가치가 큰 feature가 무엇인가?
2. 현재 JSON 메타에 composition / sample / experiment identifiers가 없다면 surrogate family_id로 `peak_count_bin + major_peak_position_signature + dynamic_range_bin`을 쓰는 것이 타당한가? 더 나은 규칙이 있는가?
3. peak GT 생성 시 `prominence >= max(0.07*(y_max-y_min), local_noise_floor)` 와 `major_peaks = top max(3, ceil(0.1*num_peaks)), cap 8` 규칙이 XRD 특성상 적절한가?
4. styled / real-like synthetic 변형 강도는 현재 제안값으로 실제 캡처/스크린샷/보고서 분포를 얼마나 잘 근사하는가? 더 적절한 수치 범위가 있는가?
5. baseline 이후 thin peak / low-contrast tail failure를 보완하기 위한 가장 실용적인 ML rescue 초기 구조는 1D CNN+BiLSTM인가, lightweight 2D U-Net/HRNet 계열인가?
6. XRD 그래프에서 legend, font, grid, line style 분포를 synthetic 렌더링에 반영하려면 어떤 스타일 템플릿 세트가 가장 실용적인가?
7. JPEG quality, blur sigma, perspective distortion offset, tail contrast drop 비율을 현재 제안값보다 더 실무적으로 고정하려면 어떤 범위가 적절한가?
8. XRD용 major peak 정의를 prominence 상위 10% + 최소 3개 + 최대 8개 규칙으로 둘 때의 장단점은 무엇이고, 더 적합한 대안이 있는가?
9. 9만 JSON 규모에서 stratified subset 300개와 이후 확장 1500/1000/1000 구성을 만들 때, feature binning을 어떤 방식으로 하면 가장 단순하면서 leakage와 분포 치우침을 줄일 수 있는가?

출력 요구:
- 각 질문마다 실무적으로 바로 채택 가능한 1순위 권고안 제시
- 대안 2개까지 비교
- 왜 그 선택이 현재 프로젝트 제약(1인 개발, 빠른 measurable result, JSON 풍부/이미지 부족)에 맞는지 설명
- 가능하면 XRD, chart digitization, signal extraction, peak detection, lightweight vision model 관점에서 근거 제시
- 모호한 조언 금지
- 최종적으로 “지금 당장 채택할 값/규칙” 형태로 정리

---

## 22. 파일 포맷 예시 모음

### 22.1 all_samples.csv 한 줄 예시

```text
sample_id,source_json_path,num_points,x_min,x_max,y_min,y_max,y_dynamic_range,dynamic_range_log,peak_count_est,peak_height_ratio,mean_peak_spacing_norm,tail_energy_ratio,fwhm_mean_est,family_id_raw,is_valid,invalid_reason
sample_000123,data/source_json/sample_000123.json,2048,10.0,80.0,0.0,15342.0,15342.0,4.1859,12,3.42,0.071,0.083,0.54,,True,
```

### 22.2 dev_subset.csv 한 줄 예시

```text
sample_id,source_json_path,debug_split,peak_count_est,tail_energy_ratio,dynamic_range_log
sample_000123,data/source_json/sample_000123.json,debug,12,0.083,4.1859
```

### 22.3 clean_manifest.csv 한 줄 예시

```text
sample_id,image_path,gt_path,source_json_path,variant_type,variant_id,family_id,split
sample_000123,data/rendered_clean/sample_000123_clean_v1.png,data/gt/sample_000123_gt.json,data/source_json/sample_000123.json,clean,clean_v1,pc_mid__pp_0.15_0.32_0.71__dr_high,debug
```

### 22.4 GT JSON 예시

```json
{
  "sample_id": "sample_000123",
  "source_json_path": "data/source_json/sample_000123.json",
  "x_values": [10.0, 10.034, 10.068],
  "y_values": [123.0, 125.0, 127.0],
  "plot_box": [170, 90, 1120, 780],
  "pixel_curve_path": [[170, 770], [171, 769], [172, 768]],
  "per_column_y_gt": {"170": 770, "171": 769, "172": 768},
  "axis_metadata": {
    "x_min": 10.0,
    "x_max": 80.0,
    "y_min": 0.0,
    "y_max": 15342.0
  },
  "peak_indices": [120, 248, 455],
  "peak_x_values": [14.1, 18.6, 25.7],
  "peak_y_values": [4021.0, 8320.0, 12011.0],
  "peak_prominences": [401.2, 811.7, 1330.4],
  "peak_pixel_points": [[221, 540], [317, 410], [510, 250]],
  "major_peak_indices": [248, 455],
  "render_variant": "clean_v1"
}
```

### 22.5 엔진 출력 JSON 예시

```json
{
  "two_theta_values": [10.0, 10.034, 10.068],
  "intensities": [122.1, 124.7, 126.9],
  "x_range": [10.0, 80.0],
  "y_range": [0.0, 15342.0],
  "quality": {
    "curve_y_mae_px": 3.4,
    "major_peak_x_error": 2.1,
    "major_peak_y_error": 4.8,
    "peak_recall": 0.83,
    "max_gap_px": 6,
    "calibration_roundtrip_error": 0.7
  },
  "confidence": 0.87,
  "warnings": [],
  "used_manual_inputs": {
    "plot_box": true,
    "x_axis_points": true,
    "y_axis_points": true,
    "color_sample_point": true,
    "legend_ignore_boxes": 0,
    "perspective_corners": false,
    "color_resample_points": 0
  }
}
```

---

## 23. 스크립트별 상세 명세

### 23.1 scan_source_json.py
입력:
- `--source_root`
- `--output_csv`

출력:
- `all_samples.csv`

종료 조건:
- source_root 아래 JSON 전수 스캔 완료
- CSV에 모든 valid/invalid 샘플 기록

### 23.2 build_dev_subset.py
입력:
- `--input_csv`
- `--output_csv`
- `--total_n`

출력:
- `dev_subset.csv`

종료 조건:
- 300개 샘플 생성
- debug/validation/holdout 분리 완료

### 23.3 render_clean_dataset.py
입력:
- `--subset_csv`
- `--output_dir`
- `--gt_dir`
- `--manifest_csv`
- `--max_samples`

출력:
- clean png
- gt json
- manifest csv

종료 조건:
- 지정 수량 clean 생성 완료

### 23.4 render_styled_dataset.py
입력:
- `--clean_manifest`
- `--output_dir`
- `--manifest_csv`
- `--variants`
- `--max_samples`

출력:
- styled png
- styled manifest

### 23.5 render_real_like_dataset.py
입력:
- `--styled_manifest`
- `--output_dir`
- `--manifest_csv`
- `--variants`
- `--max_samples`

출력:
- real-like png
- real manifest

### 23.6 build_splits.py
입력:
- `--input_csv`
- `--train_csv`
- `--val_csv`
- `--test_csv`

출력:
- split csv 3개

### 23.7 validate_dataset_integrity.py
입력:
- `--manifest_csv`

출력:
- 누락 파일 검사 리포트

검사 항목:
- image 존재
- gt 존재
- source json 존재
- sample_id 중복 여부

### 23.8 summarize_dataset_distribution.py
입력:
- `--input_csv`

출력:
- peak_count, tail_energy, dynamic_range 분포 요약
- family 분포 요약

---

## 24. 최종 체크리스트

### 데이터 준비
- [ ] 폴더 구조 생성
- [ ] all_samples.csv 생성
- [ ] dev_subset.csv 생성
- [ ] clean 이미지 생성
- [ ] styled 이미지 생성
- [ ] real-like 이미지 생성
- [ ] GT JSON 생성
- [ ] split CSV 생성

### 엔진 구현
- [ ] run_local.py 생성
- [ ] 입력 처리 구현
- [ ] ROI/perspective 구현
- [ ] color prototype 구현
- [ ] mask combine 구현
- [ ] thinning 구현
- [ ] component scoring 구현
- [ ] candidate generation 구현
- [ ] DP tracing 구현
- [ ] recovery 구현
- [ ] gap fill 구현
- [ ] smoothing 구현
- [ ] peak detection 구현
- [ ] calibration 구현
- [ ] evaluator 구현
- [ ] debug artifacts 저장 구현

### 규칙 고정 여부
- [ ] click budget 고정
- [ ] peak GT 규칙 고정
- [ ] family_id 규칙 고정
- [ ] acceptance gate 고정
- [ ] peak_f1 debug-only 분리
- [ ] baseline-before-ML 유지

---

## 25. 자체 검토 결과

이 문서를 아래 기준으로 다시 점검했다.

### 점검 기준 1: 앞에서부터 따라 만들 수 있는가
판정: 통과

이유:
- 목표 → 재료 → 폴더 → 데이터 → 엔진 → 평가 → ML 순으로 재배열함
- 각 단계마다 “끝났는지 확인하는 방법”을 넣음

### 점검 기준 2: 추상적인 표현이 남아 있는가
판정: 대부분 제거됨

구체화한 것:
- 폴더 구조
- CSV 컬럼
- GT JSON 필드
- clean/styled/real-like variant 규칙
- peak GT 규칙
- family_id surrogate 규칙
- acceptance gate
- script별 인자
- 예시 파일 포맷
- candidate confidence / filtering / DP 입력 제한 수치
- main / debug / diagnosis metric 역할 분리

남겨둔 것:
- fwhm_mean_est 의 2차 정밀화
- 실제 metadata에서 composition id를 뽑는 규칙
- ML rescue 구조 최종 선택

이 세 가지는 구현 막힘 시 별도 심층 리서치 대상으로 분리했다.

### 점검 기준 3: 내용이 줄어들었는가
판정: 아니오

기존 정보를 유지했고  
파일 예시, 스크립트 명세, 포맷 예시, 종료 조건, 그리고 12/17단계 수치 고정 규칙을 추가했다.

### 최종 판정

이 문서는 이제 “전략 요약”이 아니라 실제 조립 설명서로 사용 가능한 수준이다.  
다만 가장 좋은 다음 단계는, 이 문서를 더 늘리는 것이 아니라 스크립트 1개씩 실제 코드 작성 단계로 내려가는 것이다.

