# XRD Web App

React (client) + Express (server) 기반의 XRD 디지타이저 & 분석 도구.
Python 파이프라인을 `child_process`로 호출해 이미지 → 수치 변환을 처리하고,
결정학적 분석은 Node 측에서 직접 수행합니다.

```
web/
├── client/   # React 18 SPA
│   └── src/pages/XRD/
│       ├── XRDDigitizer.js        # 이미지 → 수치 (3점 + 4점 캘리브)
│       ├── XRDAnalyzer.js         # 수치 → 피크/결정성/W-H/QPA/...
│       └── XrdAdvancedAnalysisSection.js
└── server/   # Node.js / Express
    ├── index.js                   # 서버 엔트리
    └── analysis/
        ├── routes/xrd.routes.js   # REST 엔드포인트
        ├── core/xrd.js            # 결정학적 분석 (JS)
        └── core/xrdDigitizer.js   # Python 파이프라인 호출 wrapper
```

## 🚀 Quick Start

### 1. 설치

```bash
# 레포 루트에서
cd web
npm run install:all
```

### 2. 실행 (두 개의 터미널)

```bash
# Terminal 1 — Express API (포트 5000)
npm run start:server

# Terminal 2 — React dev server (포트 3000)
npm run start:client
```

브라우저에서 <http://localhost:3000> 접속.
React dev server는 `/api/*` 요청을 `http://localhost:5000`으로 자동 프록시합니다 ([`src/setupProxy.js`](client/src/setupProxy.js)).

### 3. Production 빌드

```bash
npm run --prefix client build      # client/build 생성
npm run start:server               # Express가 build를 정적 서빙
```

## ⚙️ 환경변수 (`server/.env`)

```env
PORT=5000

# (선택) 디지타이저 Python 경로 — 콜드 venv보다 워밍업된 레포 루트 venv가 ~14× 빠릅니다
XRD_DIGITIZER_PATH=/absolute/path/to/xrd_digitizer
XRD_DIGITIZER_PYTHON=/absolute/path/to/xrd_digitizer/.venv/bin/python3
```

값을 비워두면 `web/server/analysis/python/xrd_digitizer/.venv` 를 사용합니다 (자동 검색).
샘플은 [`server/.env.example`](server/.env.example) 참고.

## 🔌 API Endpoints

모든 엔드포인트는 `POST /api/analysis/xrd/*`.

| Endpoint | 역할 |
|---|---|
| `/parse` | `.xy` / `.dat` / `.csv` 등 텍스트 데이터 파싱 |
| `/detect-roi` | 입력 이미지에서 ROI 자동 감지 |
| `/digitize` | 이미지 + `mi.json` → 수치 곡선 + 피크 |
| `/detect-peaks` | 수치 곡선 → 피크 좌표 + prominence |
| `/fit-peaks` | Pseudo-Voigt / Gaussian 피크 피팅 |
| `/calculate-crystallinity` | 결정성 (crystalline / amorphous 분리) |
| `/calculate-crystallite-sizes` | Scherrer 결정자 크기 |
| `/williamson-hall-fit` | Williamson-Hall 분석 |
| `/identify-phase-candidates` | 상(phase) 후보 식별 |
| `/compute-texture-indices` | Texture 지수 |
| `/estimate-qpa-phase-fractions` | QPA 정량 상분석 |
| `/fit-residual-stress-sin2-psi` | sin²ψ 잔류 응력 |
| `/index-miller` | Miller 지수 인덱싱 |
| `/analyze-dislocation` | 전위 밀도 분석 |
| `/get-rietveld-guidance` | Rietveld 가이던스 |

## 🧭 Tech Stack

- **Client** — React 18 · React Router 7 · Chart.js + zoom plugin · `ml-levenberg-marquardt` · `regression` · `crystcif-parse`
- **Server** — Express · Multer (파일 업로드) · `child_process` (Python 호출) · `pytesseract` (축 라벨 OCR · 선택)

## 📸 Screenshots

> 실제 화면 캡처는 [`../docs/screenshots/`](../docs/screenshots/) 에 추가해주세요.

| Digitizer | Analyzer |
|:---:|:---:|
| ![digitizer](../docs/screenshots/digitizer.png) | ![analyzer](../docs/screenshots/analyzer.png) |
