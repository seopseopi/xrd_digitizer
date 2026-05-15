# XRD Analyzer Web

XRD 분석 + 이미지 디지타이저 독립 웹 앱 (로그인 없음)

## 실행 방법

```bash
# 1. 의존성 설치
cd web
npm run install:all

# 2. Python 패키지 설치 (디지타이저용)
npm run setup:python

# 3. 터미널 2개로 각각 실행
npm run start:server   # Express 서버 (localhost:5000)
npm run start:client   # React 앱 (localhost:3000)
```

브라우저에서 http://localhost:3000 접속하면 바로 XRD 분석 화면이 나옵니다.

## 구조

```
web/
├── client/        # React (CRA)
│   └── src/
│       ├── pages/XRD/           # XRD Analyzer + Digitizer UI
│       └── analysis/core/xrd/   # 분석 알고리즘 (프론트엔드)
└── server/        # Express
    ├── index.js
    └── analysis/
        ├── routes/xrd.routes.js
        ├── core/xrd.js          # 백엔드 분석 함수 래퍼
        ├── core/xrdDigitizer.js # Python subprocess 브릿지
        └── python/xrd_digitizer/ # Python 파이프라인
```
