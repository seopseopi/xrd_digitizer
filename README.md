<div align="center">

# XRD Digitizer

**논문 이미지의 XRD 패턴을 머신러닝 + 컴퓨터비전 파이프라인으로 수치 데이터로 자동 복원하는 풀스택 도구**

*Reconstruct numerical `(2θ, intensity)` arrays from XRD chart images — and analyze them in the browser.*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-Express-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img src="docs/assets/poster_gt_vs_pred.png" alt="GT vs Predicted" width="80%"/>

</div>

---

## TL;DR

재료공학 논문에 게재되는 XRD 패턴은 대부분 **이미지로만** 공유됩니다. 수치 원본이 없어 재해석·재학습·DB 구축이 불가능하고, 수동 디지타이즈는 한 패턴당 15–30분 소요됩니다.

본 프로젝트는 **이미지 → 수치 → 결정학적 분석** 전 과정을 자동화한 풀스택 도구입니다.

- **이미지 처리 파이프라인** — K-Means · Sub-pixel · Dynamic Programming · Linear Regression
- **웹 UI** — React + Node 기반 디지타이저 + Williamson-Hall · QPA · 텍스처 분석기
- **벤치마크** — `Intensity MAE 0.0079`, `Peak Recall 99.7%`, 90,000+ 패턴 검증

## ✨ Highlights

| Why this matters | What I built |
|---|---|
| 논문 XRD 이미지를 수치로 되살리지 못해 ML 학습 데이터 수급이 막혀있다 | End-to-End deterministic 파이프라인 (7단계) |
| 수동 디지타이즈는 1장 15–30분, 오차 ±2% | **약 3초/장 · 수동 대비 300배 빠름** |
| 단순 색상 추출은 비표준 차트에서 깨진다 | K-Means 비지도 분리 + DP 최적 경로로 강인성 확보 |
| 추출 후 후속 분석 도구가 따로 필요 | React 웹앱에 디지타이저 + 결정학 분석기 통합 |

---

## 🖼️ Demo

### Pipeline visualization — image to numbers in 5 steps

<table>
<tr>
<td width="50%"><img src="docs/assets/step1_roi_input.png" alt="Step 1"/></td>
<td width="50%"><img src="docs/assets/step2_color_mask.png" alt="Step 2"/></td>
</tr>
<tr>
<td><b>① ROI 자동 감지</b><br/>입력 차트 영역만 잘라내 후속 단계에 전달</td>
<td><b>② K-Means 색상 클러스터링</b><br/>배경·곡선·격자선을 RGB 공간에서 비지도 분리</td>
</tr>
<tr>
<td><img src="docs/assets/step3_candidates.png" alt="Step 3"/></td>
<td><img src="docs/assets/step4_trace.png" alt="Step 4"/></td>
</tr>
<tr>
<td><b>③ Sub-pixel 후보 추출</b><br/>ridge + edge + apex + band 4종 후보 풀</td>
<td><b>④ Dynamic Programming 추적</b><br/>연속성 + 신뢰도 기반 전역 최적 경로</td>
</tr>
<tr>
<td colspan="2" align="center"><img src="docs/assets/step5_peaks.png" alt="Step 5" width="60%"/><br/><b>⑤ 피크 검출 + 수치 데이터 출력</b> · Prominence + NMS</td>
</tr>
</table>

### Result on `pattern_11832`

| Reconstructed vs Ground Truth | Auto-detected Peaks |
|:---:|:---:|
| <img src="docs/assets/poster_gt_vs_pred.png"/> | <img src="docs/assets/poster_peaks_major.png"/> |
| Intensity MAE = **0.0079** | Recall = **99.7%** on 90,000+ patterns |

### Benchmark — validated on 90,000+ patterns

<img src="docs/assets/benchmark_mae_chart.png" alt="Benchmark" width="100%"/>

> 좌: Normalized MAE 분포 (75%가 MAE < 0.010) · 우: Intensity range coverage (pred/GT)

### Web app — XRD Digitizer + Analyzer

<!--
👇 실제 배포/실행 후 스크린샷을 docs/screenshots/ 에 캡처해 넣어주세요.
   추천 컷:
   - digitizer.png        : 이미지 업로드 + 3점/4점 캘리브레이션 UI
   - analyzer.png         : 피크 표 / Williamson-Hall 플롯 / 결정성 결과
-->

| Digitizer | Analyzer |
|:---:|:---:|
| 이미지 업로드 → 3점 영역 + 4점 축 캘리브레이션 → 수치 추출 | 피크 fitting · 결정성 · Williamson-Hall · QPA · 텍스처 |
| <img src="docs/screenshots/digitizer.png" alt="Digitizer screenshot" width="100%"/> | <img src="docs/screenshots/analyzer.png" alt="Analyzer screenshot" width="100%"/> |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Web (React SPA)                         │
│   ┌─────────────────┐    ┌──────────────────────────────────┐    │
│   │  XRD Digitizer  │ →  │  XRD Analyzer                    │    │
│   │  (3pt + 4pt UI) │    │  Peaks · Crystallinity · W-H ·   │    │
│   │                 │    │  QPA · Texture · Stress          │    │
│   └────────┬────────┘    └──────────────────────────────────┘    │
└────────────┼─────────────────────────────────────────────────────┘
             │  REST  /api/analysis/xrd/*
┌────────────▼─────────────────────────────────────────────────────┐
│                  Node.js / Express  (web/server)                 │
│   File upload · OCR axis hint (pytesseract) · child_process →    │
└────────────┬─────────────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────────────┐
│        Python pipeline  (core / preprocess / trace / ...)        │
│   ROI ▶ Mask ▶ Candidates ▶ DP Trace ▶ Smooth ▶ Peaks ▶ JSON     │
└──────────────────────────────────────────────────────────────────┘
```

## 🧠 Pipeline (deterministic, 7 stages)

```
① ROI crop & 2× Lanczos upscale     (690px → 1380px)
② HSV / K-Means color mask          → curve 픽셀 분리
③ Multi-source candidate generation (ridge + edge + peak-apex + band-midline)
④ Candidate filtering + confidence ranking
⑤ Column-wise Dynamic Programming    → 전역 최적 경로
⑥ Smoothing + 피크 검출              (prominence + NMS)
⑦ Linear-regression axis calibration → JSON export  (1900pt highres)
```

| Stage | Technique | ML Category |
|---|---|---|
| ROI 감지 | Hough Transform + Edge Detection | CV |
| 색상 분리 | K-Means Clustering | Unsupervised ML |
| 곡선 추출 | Sub-pixel Anti-Aliasing Edge Interpolation | CV / Signal |
| 경로 최적화 | Dynamic Programming | Algorithm |
| 축 캘리브레이션 | Linear Regression (scikit-learn) | Supervised ML |
| 피크 검출 | Prominence Scoring + NMS | Signal |

## 🛠️ Tech Stack

**Core pipeline** — Python 3.9+ · NumPy · SciPy · OpenCV · scikit-learn · scikit-image
**Web client**    — React 18 · React Router · Chart.js + zoom plugin · `ml-levenberg-marquardt`
**Web server**    — Node.js · Express · Multer · Tesseract OCR (axis hinting)
**Tooling**       — pytest · ruff · GitHub-Actions ready

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.9
- Node.js ≥ 18
- (선택) Tesseract OCR — 축 라벨 자동 인식용

### 1. Clone & install

```bash
git clone https://github.com/seopseopi/xrd_digitizer.git
cd xrd_digitizer

# Python pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web app (optional — only if you want the UI)
cd web && npm run install:all && cd ..
```

### 2. Run the CLI on a single image

```bash
python3 -m runner.run_local \
  --image_path examples/sample.png \
  --manual_inputs_path examples/sample_mi.json \
  --output_json_path result.json \
  --debug_dir debug/ \
  --roi-upscale-factor 2
```

### 3. Run the web app

```bash
# terminal 1 — Express API
cd web && npm run start:server

# terminal 2 — React dev server
cd web && npm run start:client
# → http://localhost:3000
```

> 자세한 웹 가이드: [`web/README.md`](web/README.md)

### Minimal `mi.json` format

```json
{
  "plot_box": [170, 90, 1120, 780],
  "x_axis_points": [[170, 780], [1120, 780]],
  "x_axis_values": [5.0, 70.4],
  "y_axis_points":  [[170, 780], [170, 90]],
  "y_axis_values":  [0.0, 1.0],
  "color_sample_point": [500, 400]
}
```

---

## 📊 Results

### Canonical-30 benchmark (2× ROI upscale)

| Domain | n | MAE mean | MAE median | MAE max | ptp_r mean |
|---|---:|---:|---:|---:|---:|
| clean      | 20 | **0.0323** | 0.0351 | 0.0501 | 0.804 |
| styled     | 5  | 0.0839     | 0.0343 | 0.2770 | —     |
| real_like  | 5  | 0.0448     | 0.0346 | 0.0832 | —     |

> **MAE** = normalized intensity error (GT range 기준) · **ptp_r** = pred range / GT range (1.0이 이상)
> Peak recall = **1.000** (clean domain, n=10)

### Large-scale benchmark (rendered set)

| Metric | Value |
|---|---:|
| Patterns evaluated | **90,000+** |
| Intensity MAE (mean) | **0.0079** |
| 95th percentile MAE  | 0.0166 |
| Patterns with MAE < 0.010 | **75%** |
| Peak Recall | **99.7%** |
| Per-image processing time | ≈ **3 s** |

---

## 🗺️ Engineering Journey

> 처음에는 end-to-end ML로 풀려고 했지만, 학습 실패 원인을 분리할 수 없었습니다.
> "보기엔 비슷한 곡선"과 "수치적으로 정확한 곡선"은 다르다는 점을 깨닫고
> deterministic pipeline + 명확한 평가 지표로 방향을 바꿨습니다.

| # | 시도 | 부딪힌 문제 | 배운 것 |
|---|---|---|---|
| 1 | End-to-end ML | 실패 원인 추적 불가, y amplitude 오차 | **관측 가능한 단계로 분해** |
| 2 | Heatmap / skeleton | skeleton이 peak 중앙 아래를 추적 | **시각 평가 ≠ 수치 정확도** |
| 3 | Numeric JSON 정의 | 평가 지표 부재 | MAE / ptp_r / peak error 도입 |
| 4 | Manual mi.json + manifest | x/y 캘리브레이션 오차 | **기준 좌표계 고정** |
| 5 | Multi-source candidates | 단일 소스로 peak top 누락 | 후보 풀 다양화 |
| 6 | DP 비용 함수 설계 | bottom branch lock, peak flattening | continuity / sharpness 균형 |
| 7 | Highres numeric metric | 시각 평가로 2× 효과 숨겨짐 | **무엇을 측정하는지가 절반** |
| 8 | Canonical-30 고정 셋 | 실험 재현성 부재 | manifest 기반 회귀 테스트 |
| 9 | Failure taxonomy | "실패"를 분류 불가 | 6개 subtype 정의 |
| 10 | 2× highres ROI upscale | 1–2 px apex 손실 | 내부 1900×1380 처리 |
| 11 | Amorphous hump 처리 | hump 위 peak 무시 | wide-band trend attraction |
| 12 | Debug 좌표 오프셋 수정 | 2× 이미지에 1× axis map 적용 | scale 보정 |
| 13 | **y-pixel 단위 변환 버그** | 2× y-px → 1× y_map 적용 → intensity 2× 오차 | **MAE 0.0334 → 0.0323 (−3.3%)** |

### Phase 13 — single-line fix worth the whole epic

```python
# ❌ before  — y_tgt comes out in 2× pixels
x_tgt = np.arange(original_roi_w) * float(factor)
y_tgt = np.interp(x_tgt, x_src, y_src)
return list(range(original_roi_w)), y_tgt, ...

# ✅ after   — bring y back to 1× before handing off to calibration
y_tgt = np.interp(x_tgt, x_src, y_src) / float(factor)
```

| Metric | before | after |
|---|---:|---:|
| MAE mean | 0.0334 | **0.0323** |
| Peak markers | apex 절반 높이 | apex 정확히 위치 |

### Phase 14 –20 — bringing the engine into the browser

Phase 1–13까지는 CLI 파이프라인의 **수치 정확도**를 끌어올리는 엔진 작업이었다.
이후 단계는 같은 파이프라인을 웹으로 옮기면서 부딪힌, 본질적으로 **"사용자 입력의 오차를 어떻게 흡수할 것인가"** 의 문제였다.
CLI는 `mi.json`을 사람이 직접 만들었지만, 웹에서는 그 부담을 0에 가깝게 줄이되 — 자동이 틀렸을 때 **사용자가 즉시 미세조정**할 수 있어야 했다.

| # | 단계 | 부딪힌 문제 | 처리 방식 |
|---|---|---|---|
| 14 | **독립 웹앱 분리**<br/>(`f50c264`) | CLI 만으로는 검증 사이클이 느리고 도구를 다른 사람이 못 씀 | React + Express + Python `child_process` 의 풀스택 standalone — 로그인 없이 단독 동작 |
| 15 | 초기 통합 안정화<br/>(`7663cfa`·`7be5149`·`29e6e8d`·`288449f`) | toolbar 비어있음 · CSS/proxy 충돌 · 캔버스 높이 0 | 분석/디지타이저 컴포넌트 정확한 마운트 지점 확보, dev proxy 정리, layout root wrapper |
| 16 | **축 좌표 자동 추출**<br/>(`ef8634c`) | 사용자가 ① plot_box (3점) + ② x/y 축 끝점 좌표값(4점) 을 매번 픽셀 단위로 손수 클릭해야 함 | (a) Hough 기반 ROI 자동 감지 + (b) **pytesseract OCR로 축 tick 라벨을 읽어 X1·X2·Y1·Y2의 실제 수치까지 자동 채움**.<br/>한 번의 "🔍 영역 + 축 자동 감지" 클릭으로 7점이 한꺼번에 잡힘 |
| 17 | **자동값 사용자 보정 UX**<br/>(`XRDDigitizer.js`) | OCR 실패 · 비표준 차트 스타일에서 ±수 px 오차 | 영역 3점(원점·X끝·Y끝) + 캘리브 4점(X1·X2·Y1·Y2)을 **재클릭 가능한 핸들**로 노출. 자동값에서 시작 → 점을 클릭하면 활성 상태가 되고, 캔버스에서 다시 클릭하면 픽셀 단위로 nudge. 인접 tick으로 옮길 수도 있어 차트 끝점이 아니어도 됨 |
| 18 | **곡선 색상 자동 + 사용자 클릭 선택**<br/>(`color_sample_point`) | 차트마다 곡선 색이 다르고, 동일 톤이 라벨·격자에도 섞여 단순 색 필터로는 깨짐 | ROI 중앙 픽셀에서 후보 색을 자동 추정 → K-Means seed 로 전달. 사용자가 곡선 위 임의 픽셀을 클릭하면 그 RGB를 `color_sample_point`로 교체해 재분리. **자동 + 수동의 양방향 입력** |
| 19 | 결과 검증 인터랙션<br/>(`631497c`·`2943455`·`7146ef8`) | 디지타이즈 직후 추출이 맞는지 픽셀 단위로 확인 불가 · 휠 줌과 트랙패드 핀치 충돌 | 추출 곡선을 캔버스에 오버레이로 그려 GT 위에 직접 비교. wheel zoom → pan으로 교체, 트랙패드 핀치 명시적 무시 (의도치 않은 줌 차단) |
| 20 | UI 모던화<br/>(`581b161`·`9091f54`·`40881e3`·`c644493`) | 기능 위주 UI 라 첫 사용자 학습곡선이 가파름 | Pretendard 폰트 + glassmorphism + indigo/violet accent. 단계 스테퍼 재설계, 업로드 파일명 pill 뱃지, accordion 설정 패널 + 슬라이더(피크 높이·간격)로 노출 |

### What I learned

- **시각적으로 그럴듯한 결과는 측정 가능한 정확도가 아니다.** 픽셀 시각화와 수치 metric을 분리해서 보지 않으면, 회귀를 놓친다.
- **End-to-end보다 분해 가능한 파이프라인이 디버깅 가능성을 만든다.** ML 단계는 정말 필요한 곳(K-Means, Linear Regression)에만 두고, 결정론적 단계와 명확히 분리했다.
- **고정된 평가 셋(canonical-30) + manifest** 가 실험 속도를 가장 크게 끌어올렸다.

---

## 📁 Project Structure

```
xrd_digitizer/
├── core/            # 설정, 타입, IO, 파이프라인 버전
├── preprocess/      # ROI crop, mask, morphology, ridge map
├── trace/           # Candidate 생성, DP tracing, recovery
├── calibrate/       # 축 보정, numeric export, 피크 렌더
├── peaks/           # 피크 검출 + smoothing
├── runner/          # CLI / 배치 실행
├── eval/            # 평가 지표, 진단
├── examples/        # 샘플 입력 (mi.json 등)
├── tests/           # pytest
├── docs/
│   └── assets/      # README 이미지
└── web/
    ├── client/      # React SPA  (Digitizer + Analyzer)
    └── server/      # Express + Python child_process
```

## 🧭 Roadmap & Limitations

| 항목 | 현재 수치 | 원인 / 다음 단계 |
|---|---|---|
| ptp_r ≈ 0.80 | 예측 intensity range가 GT의 약 80% | DP가 apex를 1–2 px 낮게 추적 |
| styled MAE max 0.277 | 색상 변형 시 mask 실패 | HSV 임계값 학습화 |
| real_like MAE max 0.083 | 저대비 구간 candidate 누락 | raw extraction 강화 |
| 근접 피크 미분리 | 2θ 간격 < 1° doublet | DP column resolution 한계 |
| Manual mi.json 필요 | 축 보정 좌표 직접 입력 | 축 라벨 OCR + 자동 ROI |

---

## 📜 License

MIT — see [`LICENSE`](LICENSE).

## 👋 Author

**이민섭 (seopseopi)** — Materials Informatics / ML Engineer
[GitHub](https://github.com/seopseopi)

> 본 프로젝트는 deterministic pipeline 설계 + 풀스택 통합 경험을 담은 **개인 포트폴리오**입니다.
> 코드·평가 셋·README 모두 재현 가능하도록 정리되어 있으니, 자유롭게 clone 해서 돌려보시기 바랍니다.
