"""Generate comprehensive HTML + PDF report for XRD Digitizer V1 project.

Covers Step 1 through V2 optimization, with embedded images and
detailed problem analysis / solution roadmap.
"""
from __future__ import annotations

import base64, json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def img_b64(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

def img_tag(path: pathlib.Path, caption: str = "", width: str = "100%") -> str:
    b64 = img_b64(path)
    if not b64:
        return f'<p class="missing">[이미지 없음: {path.name}]</p>'
    html = f'<img src="{b64}" style="width:{width}; border:1px solid #ddd; border-radius:6px;" />'
    if caption:
        html += f'<p class="caption">{caption}</p>'
    return html

def debug_section(base: pathlib.Path, label: str) -> str:
    files = [
        ("roi_preview.png", "ROI 크롭"),
        ("color_mask.png", "색상 마스크 (mask_A)"),
        ("combined_mask.png", "결합 마스크 (mask_A + mask_B)"),
        ("raw_candidate_mask.png", "형태학 처리 후 원시 후보"),
        ("skeleton.png", "스켈레톤화"),
        ("components_overlay.png", "연결 성분 라벨"),
        ("candidate_map_raw.png", "Raw 후보맵"),
        ("candidate_map_filtered.png", "Filtered 후보맵"),
        ("candidate_map_final.png", "Final 후보맵 (DP 입력)"),
        ("trace_path.png", "DP 추적 경로"),
        ("branch_compare.png", "복구 분기 비교"),
        ("smoothed_trace.png", "Raw vs Smoothed"),
        ("peaks_overlay.png", "피크 검출 오버레이"),
    ]
    rows = []
    for i in range(0, len(files), 2):
        cells = ""
        for j in range(2):
            if i + j < len(files):
                fname, cap = files[i + j]
                cells += f'<td style="width:50%;vertical-align:top;padding:6px;">{img_tag(base / fname, f"{label} — {cap}", "100%")}</td>'
            else:
                cells += "<td></td>"
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="img-grid">{"".join(rows)}</table>'

# ── Eval data ────────────────────────────────────────────────────
def load_eval(path: pathlib.Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

v1_clean = load_eval(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "eval_report.json")
v2_clean = load_eval(ROOT / "experiments" / "archive" / "outputs_v2_clean" / "eval_report.json")
v2_styled = load_eval(ROOT / "experiments" / "archive" / "outputs_v2_styled" / "eval_report.json")
v2_real = load_eval(ROOT / "experiments" / "archive" / "outputs_v2_real" / "eval_report.json")

# ── HTML ─────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>XRD Digitizer V1 — 전체 프로젝트 종합 보고서</title>
<style>
:root {{
  --primary: #1e40af; --primary-light: #3b82f6; --primary-bg: #eff6ff;
  --text: #1f2937; --text-light: #6b7280; --bg: #ffffff;
  --code-bg: #f3f4f6; --border: #e5e7eb;
  --green: #166534; --green-bg: #dcfce7;
  --red: #991b1b; --red-bg: #fee2e2;
  --orange: #92400e; --orange-bg: #fef3c7;
  --purple: #6b21a8; --purple-bg: #f3e8ff;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif;
  color: var(--text); background: #f8fafc; line-height:1.7; font-size:14px;
}}
.container {{ max-width:960px; margin:0 auto; background:var(--bg); box-shadow:0 0 40px rgba(0,0,0,.08); }}
.cover {{
  background: linear-gradient(135deg,#1e3a8a 0%,#3b82f6 50%,#60a5fa 100%);
  color:white; padding:70px 60px; text-align:center;
  min-height:100vh; display:flex; flex-direction:column; justify-content:center;
}}
.cover h1 {{ font-size:40px; font-weight:800; margin-bottom:8px; }}
.cover h2 {{ font-size:22px; font-weight:400; opacity:.9; margin-bottom:24px; }}
.cover .divider {{ width:80px; height:3px; background:rgba(255,255,255,.6); margin:0 auto 24px; }}
.cover .subtitle {{ font-size:14px; opacity:.8; line-height:1.9; }}
.cover .meta {{ margin-top:40px; font-size:12px; opacity:.55; }}
.content {{ padding:45px 55px; }}
@media (max-width:768px) {{ .content {{ padding:24px 16px; }} }}
.toc {{ background:var(--primary-bg); border-radius:10px; padding:24px 30px; margin-bottom:40px; }}
.toc h2 {{ color:var(--primary); font-size:20px; margin-bottom:14px; }}
.toc ol {{ padding-left:20px; }} .toc li {{ margin-bottom:4px; font-size:13px; }}
.toc a {{ color:var(--text); text-decoration:none; }} .toc a:hover {{ color:var(--primary-light); }}
.chapter {{ margin-bottom:50px; page-break-before:always; }}
.chapter:first-of-type {{ page-break-before:auto; }}
.chapter-title {{
  font-size:22px; font-weight:800; color:var(--primary);
  border-bottom:3px solid var(--primary-light); padding-bottom:8px; margin-bottom:20px;
}}
.section-title {{
  font-size:17px; font-weight:700; color:#374151;
  margin:22px 0 10px; padding-left:11px; border-left:4px solid var(--primary-light);
}}
.sub-section {{ font-size:14px; font-weight:700; color:#4b5563; margin:16px 0 8px; }}
p {{ margin-bottom:10px; }}
pre {{
  background:var(--code-bg); border:1px solid var(--border); border-radius:7px;
  padding:14px 18px; font-family:'Consolas','D2Coding',monospace; font-size:12px;
  line-height:1.6; overflow-x:auto; margin:10px 0 16px; white-space:pre-wrap; word-break:break-word;
}}
code {{ background:var(--code-bg); padding:1px 5px; border-radius:3px; font-family:'Consolas','D2Coding',monospace; font-size:12px; }}
table {{ width:100%; border-collapse:collapse; margin:10px 0 18px; font-size:12.5px; }}
th {{ background:var(--primary); color:white; padding:8px 12px; text-align:left; font-weight:600; }}
td {{ padding:7px 12px; border-bottom:1px solid var(--border); }}
tr:hover td {{ background:#f9fafb; }}
.tag {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:10px; font-weight:600; }}
.tag.blue {{ background:#dbeafe; color:#1e40af; }}
.tag.green {{ background:#dcfce7; color:#166534; }}
.tag.red {{ background:#fee2e2; color:#991b1b; }}
.tag.purple {{ background:#f3e8ff; color:#6b21a8; }}
.tag.orange {{ background:#fef3c7; color:#92400e; }}
.info-box {{
  background:var(--primary-bg); border-left:4px solid var(--primary-light);
  border-radius:0 7px 7px 0; padding:14px 18px; margin:14px 0 18px;
}}
.info-box .title {{ font-weight:700; color:var(--primary); margin-bottom:4px; font-size:13px; }}
.info-box p {{ font-size:12.5px; margin-bottom:3px; }}
.warn-box {{
  background:var(--red-bg); border-left:4px solid #ef4444;
  border-radius:0 7px 7px 0; padding:14px 18px; margin:14px 0 18px;
}}
.warn-box .title {{ font-weight:700; color:var(--red); margin-bottom:4px; font-size:13px; }}
.warn-box p {{ font-size:12.5px; margin-bottom:3px; }}
.success-box {{
  background:var(--green-bg); border-left:4px solid #22c55e;
  border-radius:0 7px 7px 0; padding:14px 18px; margin:14px 0 18px;
}}
.success-box .title {{ font-weight:700; color:var(--green); margin-bottom:4px; font-size:13px; }}
.caption {{ font-size:11px; color:var(--text-light); margin:2px 0 10px; text-align:center; }}
.missing {{ font-size:11px; color:#999; font-style:italic; }}
.img-grid {{ border:none; }} .img-grid td {{ border:none; padding:4px; }}
.pipeline {{
  background:#f9fafb; border:1px solid var(--border); border-radius:8px;
  padding:16px 20px; margin:14px 0 18px; font-family:'Consolas','D2Coding',monospace; font-size:12px; line-height:1.9;
}}
.kpi {{ display:inline-block; background:var(--primary-bg); border:1px solid #bfdbfe; border-radius:8px; padding:12px 20px; margin:4px 6px 4px 0; text-align:center; }}
.kpi .val {{ font-size:28px; font-weight:800; color:var(--primary); }}
.kpi .label {{ font-size:11px; color:var(--text-light); }}
@media print {{
  body {{ background:white; font-size:11px; }}
  .container {{ box-shadow:none; }}
  .cover {{ min-height:auto; padding:50px 40px; page-break-after:always; }}
  .content {{ padding:25px 35px; }}
  .chapter {{ page-break-before:always; }}
}}
</style>
</head>
<body>
<div class="container">

<!-- ========== COVER ========== -->
<div class="cover">
  <h1>XRD Digitizer V1</h1>
  <h2>전체 프로젝트 종합 보고서</h2>
  <div class="divider"></div>
  <div class="subtitle">
    Step 1 ~ V2 최적화 전 과정<br>
    데이터 준비 · 엔진 구현 · 평가 · 심층 조사 · Compliance · 최적화 · 문제 분석 · 해결 방향
  </div>
  <div class="meta">
    프로젝트: XRD 그래프 이미지 → 수치 JSON 복원 엔진&emsp;|&emsp;작성일: 2026-04-14<br>
    기준 문서: xrd_digitizer_v1_master_spec.md (2,066 lines)&emsp;|&emsp;구현 파일: 32개
  </div>
</div>

<div class="content">

<!-- ========== TOC ========== -->
<div class="toc">
  <h2>목차</h2>
  <ol>
    <li><a href="#ch1">프로젝트 개요 및 아키텍처</a></li>
    <li><a href="#ch2">데이터 준비 (Step 3~8)</a></li>
    <li><a href="#ch3">엔진 구현 (Step 9~16)</a></li>
    <li><a href="#ch4">평가 체계 (Step 17~18)</a></li>
    <li><a href="#ch5">Baseline 실행 결과</a></li>
    <li><a href="#ch6">심층 조사 및 리서치 (Step 19~21)</a></li>
    <li><a href="#ch7">Compliance 검증 (Step 22~25)</a></li>
    <li><a href="#ch8">V2 최적화 사이클</a></li>
    <li><a href="#ch9">V1 vs V2 성능 비교</a></li>
    <li><a href="#ch10">외부 XRD 분석 시스템 검토</a></li>
    <li><a href="#ch11">근본 문제 심층 분석</a></li>
    <li><a href="#ch12">해결 방향 및 실행 로드맵</a></li>
    <li><a href="#ch13">부록</a></li>
  </ol>
</div>

<!-- ================================================================ -->
<!--  1. 프로젝트 개요                                                -->
<!-- ================================================================ -->
<div class="chapter" id="ch1">
<h2 class="chapter-title">1. 프로젝트 개요 및 아키텍처</h2>

<h3 class="section-title">1.1 목표</h3>
<p>90,279개의 원본 XRD(X-Ray Diffraction) JSON 데이터를 기반으로 합성 이미지를 생성하고, 이를 역으로 복원하여 <strong>2-theta / Intensity 수치 데이터</strong>를 자동 추출하는 독립 엔진을 구축한다. ML 모델 없이 <strong>rule-based 알고리즘만</strong>으로 동작한다.</p>

<h3 class="section-title">1.2 KPI 현황</h3>
<div style="text-align:center; margin:16px 0;">
  <div class="kpi"><div class="val">33%</div><div class="label">Clean Pass Rate (V2)</div></div>
  <div class="kpi"><div class="val">0%</div><div class="label">Styled Pass Rate</div></div>
  <div class="kpi"><div class="val">4%</div><div class="label">Real-like Pass Rate</div></div>
  <div class="kpi"><div class="val">200/200</div><div class="label">파이프라인 안정성</div></div>
  <div class="kpi"><div class="val">85/85</div><div class="label">Compliance PASS</div></div>
</div>

<h3 class="section-title">1.3 전체 아키텍처</h3>
<div class="pipeline">
Source JSON (90,279) → [Step 3] Metadata Scan → all_samples.csv<br>
&emsp;→ [Step 4] Stratified Subset → dev_subset.csv (300)<br>
&emsp;→ [Step 5] Clean Rendering → 100 PNG + GT JSON<br>
&emsp;→ [Step 6] Styled Rendering → 150 PNG (5 variants × 30)<br>
&emsp;→ [Step 7] Real-like Rendering → 100 PNG (2 variants × 50)<br>
&emsp;→ [Step 8] Train/Val/Test Split<br>
<br>
Engine Pipeline:<br>
&emsp;[Step 11] ROI Crop → Lab Color Model → Mask Generation → Morphology<br>
&emsp;[Step 12] Skeleton → Component Scoring → Candidate Generation<br>
&emsp;[Step 13] DP Tracing → [Step 14] Recovery<br>
&emsp;[Step 15] Gap Fill / Smoothing / Peak Detection<br>
&emsp;[Step 16] Pixel → Numeric Calibration<br>
<br>
[Step 17-18] Evaluator: 6 Main Metrics + 5 Diagnosis + 8 Failure Labels
</div>

<h3 class="section-title">1.4 프로젝트 폴더 구조</h3>
<pre>c:\\xrd_digitizer_v1\\
├── xrd_digitizer_v1_master_spec.md    # 마스터 스펙 (2066 lines)
├── scripts\\          # Step 3~8: 데이터 준비 스크립트 (8개)
├── core\\             # Step 9~10: 엔진 코어 (3개)
├── preprocess\\       # Step 11: 전처리 모듈 (5개)
├── trace\\            # Step 12~15: 추적 모듈 (6개)
├── calibrate\\        # Step 16: 좌표 변환 (2개)
├── runner\\           # 실행기 (2개)
├── eval\\             # Step 17~18: 평가 (4개)
├── data\\             # 이미지, GT, 매니페스트
├── experiments/archive/outputs_baseline_*/  # V1 배치(아카이브)
├── experiments/archive/outputs_v2_*/        # V2 배치(아카이브)
└── outputs\\            # 보고서</pre>
</div>

<!-- ================================================================ -->
<!--  2. 데이터 준비                                                   -->
<!-- ================================================================ -->
<div class="chapter" id="ch2">
<h2 class="chapter-title">2. 데이터 준비 (Step 3~8)</h2>

<h3 class="section-title">2.1 Step 3: 메타데이터 스캔</h3>
<p><code>scripts/scan_source_json.py</code> — 90,279개 JSON에서 17개 컬럼 메타데이터 추출.</p>
<table>
<tr><th>항목</th><th>내용</th></tr>
<tr><td>입력</td><td><code>data/source_json/</code> 90,279개 JSON</td></tr>
<tr><td>출력</td><td><code>data/metadata/all_samples.csv</code> (17 컬럼)</td></tr>
<tr><td>유효성 검증</td><td>7가지 invalid rule 적용</td></tr>
</table>

<h3 class="section-title">2.2 Step 4: 층화 개발 서브셋</h3>
<p><code>scripts/build_dev_subset.py</code> — 3×3×3 = 27 bin에서 균등 추출하여 300 샘플 서브셋 구성.</p>
<table>
<tr><th>분할</th><th>수</th></tr>
<tr><td>debug_n</td><td>100</td></tr>
<tr><td>val_n</td><td>100</td></tr>
<tr><td>holdout_n</td><td>100</td></tr>
<tr><td><strong>합계</strong></td><td><strong>300</strong></td></tr>
</table>

<h3 class="section-title">2.3 Step 5: Clean 이미지 렌더링</h3>
<p>캔버스 1200×900px, plot box [170,90,1120,780], 흰 배경, 검정 곡선(lw=1.5), HEADROOM_RATIO=0.08.</p>
{img_tag(ROOT / "data" / "rendered_clean" / "pattern_1076_clean_v1.png", "Clean 렌더링 예시 (pattern_1076)", "80%")}

<h3 class="section-title">2.4 Step 6: Styled 이미지 렌더링</h3>
<table>
<tr><th>Variant</th><th>스타일 특징</th></tr>
<tr><td>styled_v1</td><td>파란색 곡선, 옅은 격자, 범례</td></tr>
<tr><td>styled_v2</td><td>빨간색 곡선, 박스 범례</td></tr>
<tr><td>styled_v3</td><td>녹색 곡선, 두꺼운 격자, 배경색 변경</td></tr>
<tr><td>styled_v4</td><td>보라색 곡선, 범례 없음, 부분 격자</td></tr>
<tr><td>styled_v5</td><td>주황색 곡선, tail contrast drop 10-20%</td></tr>
</table>
{img_tag(ROOT / "data" / "rendered_styled" / "pattern_11832_styled_v1.png", "Styled 렌더링 예시 (styled_v1)", "80%")}
{img_tag(ROOT / "data" / "rendered_styled" / "pattern_11832_styled_v3.png", "Styled 렌더링 예시 (styled_v3)", "80%")}

<h3 class="section-title">2.5 Step 7: Real-like 이미지 렌더링</h3>
<table>
<tr><th>Variant</th><th>열화 특징</th></tr>
<tr><td>real_v2</td><td>JPEG 압축(Q60-80), 가우시안 블러(σ=0.5-1.5)</td></tr>
<tr><td>real_v4</td><td>밝기/대비 변화, 그림자 오버레이, 원근 왜곡(25%), border</td></tr>
</table>
{img_tag(ROOT / "data" / "rendered_real_like" / "pattern_11832_real_v2.png", "Real-like 렌더링 예시 (real_v2)", "80%")}
{img_tag(ROOT / "data" / "rendered_real_like" / "pattern_11832_real_v4.png", "Real-like 렌더링 예시 (real_v4)", "80%")}

<h3 class="section-title">2.6 Step 8: 데이터 분할</h3>
<p><code>family_id</code> 기반 분할로 동일 패턴의 다른 variant가 다른 split에 섞이는 것을 방지.</p>
<table>
<tr><th>분할</th><th>비율</th><th>수</th></tr>
<tr><td>Train</td><td>80%</td><td>240</td></tr>
<tr><td>Val</td><td>10%</td><td>30</td></tr>
<tr><td>Test</td><td>10%</td><td>30</td></tr>
</table>
</div>

<!-- ================================================================ -->
<!--  3. 엔진 구현                                                     -->
<!-- ================================================================ -->
<div class="chapter" id="ch3">
<h2 class="chapter-title">3. 엔진 구현 (Step 9~16)</h2>

<h3 class="section-title">3.1 Step 9~10: 엔진 코어 및 사용자 입력</h3>
<p>핵심 데이터 구조: <code>ManualInputs</code> (plot_box, axis points, color_sample_point 등), <code>RunResult</code> (two_theta_values, intensities, confidence 등). click budget 12회.</p>

<h3 class="section-title">3.2 Step 11: 전처리 파이프라인</h3>
<div class="pipeline">
입력 이미지 → ROI 크롭 → 원근 보정 → Lab 색공간 변환<br>
&emsp;→ 색상 프로토타입 추출 → 색상 거리맵<br>
&emsp;→ mask_A (색상 근접) + mask_B (에지/그래디언트) → 결합 마스크<br>
&emsp;→ 형태학 처리 (closing, 소성분 제거) → 세선화
</div>

<h4 class="sub-section">PASS 샘플 전처리 시각화 (pattern_1, y_mae=2.45px)</h4>
{debug_section(ROOT / "experiments" / "archive" / "outputs_v2_clean" / "debug_pattern_1", "pattern_1 (PASS)")}

<h4 class="sub-section">FAIL 샘플 전처리 시각화 (pattern_1360, y_mae=39.64px)</h4>
{debug_section(ROOT / "experiments" / "archive" / "outputs_v2_clean" / "debug_pattern_1360", "pattern_1360 (FAIL)")}

<h3 class="section-title">3.3 Step 12: 곡선 후보 생성</h3>
<p>3단계 후보 파이프라인: Raw (스켈레톤+마스크 합집합) → Filtered (x_coverage, length, edge_penalty) → Final DP (confidence, 빈 열 보간).</p>

<h3 class="section-title">3.4 Step 13: DP 경로 추적</h3>
<p>비용 함수: <code>cost = α·|dy| + β·|d²y| + γ·(1-conf) + δ·comp_switch + border_penalty</code></p>
<table>
<tr><th>파라미터</th><th>값</th><th>역할</th></tr>
<tr><td>α (ALPHA)</td><td>1.0</td><td>y 변화 페널티</td></tr>
<tr><td>β (BETA)</td><td>0.35</td><td>y 가속도 페널티</td></tr>
<tr><td>γ (GAMMA)</td><td>0.8</td><td>낮은 confidence 페널티</td></tr>
<tr><td>δ (DELTA)</td><td>1.2</td><td>성분 전환 페널티</td></tr>
<tr><td>ε (EPSILON)</td><td>3.0</td><td>경계 근접 페널티 (V2 추가)</td></tr>
</table>

<h3 class="section-title">3.5 Step 14: 복구 / 재진입</h3>
<p>4가지 트리거: low_valid_ratio, consecutive_missing, score_spike, low_margin_streak.<br>
6단계 복구: 후보 재탐색 → 재점수화 → 로컬 재추적 → 분기 비교 → fail-fast → 사용자 입력 요청.</p>

<h3 class="section-title">3.6 Step 15: 갭필 / 스무딩 / 피크 검출</h3>
<table>
<tr><th>처리</th><th>방법</th><th>파라미터</th></tr>
<tr><td>Gap Fill</td><td>선형 보간</td><td>gap ≤ 10px</td></tr>
<tr><td>Smoothing</td><td>Savitzky-Golay</td><td>window 동적(0.012×Pw), 피크 보존 재시도</td></tr>
<tr><td>Peak Detection</td><td>scipy find_peaks</td><td>적응형 prominence, 로컬 노이즈 추정</td></tr>
</table>

<h3 class="section-title">3.7 Step 16: 픽셀 → 수치 변환</h3>
<p>2점 선형 매핑 (X: pixel→2θ, Y: pixel→Intensity 역방향). 왕복 오차 < 0.5px → confidence=1.0.</p>
</div>

<!-- ================================================================ -->
<!--  4. 평가 체계                                                     -->
<!-- ================================================================ -->
<div class="chapter" id="ch4">
<h2 class="chapter-title">4. 평가 체계 (Step 17~18)</h2>

<h3 class="section-title">4.1 메트릭 체계</h3>
<table>
<tr><th>카테고리</th><th>메트릭</th><th>합격 기준 (clean)</th></tr>
<tr><td rowspan="6"><span class="tag blue">Main (Gate)</span></td>
    <td>curve_y_mae_px</td><td>≤ 5.0</td></tr>
<tr><td>major_peak_x_error</td><td>≤ 3.0</td></tr>
<tr><td>major_peak_y_error</td><td>≤ 5.0</td></tr>
<tr><td>peak_recall</td><td>≥ 0.8</td></tr>
<tr><td>max_gap_px</td><td>≤ 10.0</td></tr>
<tr><td>calibration_roundtrip_error</td><td>≤ 1.0</td></tr>
<tr><td><span class="tag purple">Diagnosis</span></td>
    <td colspan="2">candidate_recall, empty_column_rate, recovery_success_rate, reentry_count, path_margin_instability</td></tr>
</table>

<h3 class="section-title">4.2 실패 분류 체계 (8개 라벨)</h3>
<table>
<tr><th>Label</th><th>자동 감지 조건</th></tr>
<tr><td><span class="tag red">grid_confusion</span></td><td>candidate_recall &gt; 0.98 and y_mae &gt; 10</td></tr>
<tr><td><span class="tag orange">candidate_starvation</span></td><td>empty_column_rate &gt; 0.05</td></tr>
<tr><td><span class="tag purple">legend_capture</span></td><td>recovery_success_rate &lt; 0.5 and reentry_count &gt; 0</td></tr>
<tr><td><span class="tag blue">text_intrusion</span></td><td>IoU &lt; 0.7 and empty_column_rate &lt; 0.03</td></tr>
<tr><td>tail_collapse</td><td>tail_mae_px &gt; 8.0 or tail_collapse_rate &gt; 0.3</td></tr>
<tr><td>wrong_branch_lock_in</td><td>path_margin_instability &gt; 0.6</td></tr>
<tr><td>calibration_mismatch</td><td>calibration_roundtrip_error &gt; 1.0</td></tr>
<tr><td>peak_miss_after_smoothing</td><td>peak_f1 &lt; 0.5 and peak_recall ≥ 0.6</td></tr>
</table>
</div>

<!-- ================================================================ -->
<!--  5. Baseline 실행 결과                                            -->
<!-- ================================================================ -->
<div class="chapter" id="ch5">
<h2 class="chapter-title">5. Baseline (V1) 실행 결과</h2>

<h3 class="section-title">5.1 실행 요약</h3>
<table>
<tr><th>구분</th><th>샘플 수</th><th>결과</th><th>소요 시간</th></tr>
<tr><td>Clean 100</td><td>100장</td><td>ok=100, fail=0</td><td>58.9초</td></tr>
<tr><td>Styled 50</td><td>50장</td><td>ok=50, fail=0</td><td>31.2초</td></tr>
<tr><td>Real-like 50</td><td>50장</td><td>ok=50, fail=0</td><td>31.2초</td></tr>
</table>
<div class="success-box">
  <div class="title">파이프라인 안정성</div>
  <p>200장 전부 crash 없이 완료 (fail=0). §18.2 충족.</p>
</div>

<h3 class="section-title">5.2 Baseline 평가 결과</h3>
<table>
<tr><th>Dataset</th><th>Pass</th><th>Rate</th><th>y_mae mean</th><th>y_mae max</th><th>Top Failure</th></tr>
<tr><td><strong>clean</strong></td><td>31</td><td><strong>31.0%</strong></td><td>26.58 px</td><td>682.38 px</td><td>grid_confusion (48)</td></tr>
<tr><td><strong>styled</strong></td><td>0</td><td><strong>0.0%</strong></td><td>16.37 px</td><td>39.16 px</td><td>grid_confusion (50)</td></tr>
<tr><td><strong>real_like</strong></td><td>2</td><td><strong>4.0%</strong></td><td>16.78 px</td><td>39.16 px</td><td>grid_confusion (48)</td></tr>
</table>

<h3 class="section-title">5.3 Failure Taxonomy 분포</h3>
<table>
<tr><th>Label</th><th>clean</th><th>styled</th><th>real</th><th>Total</th><th>비율</th></tr>
<tr style="background:#fee2e2;"><td><strong>grid_confusion</strong></td><td>48</td><td>50</td><td>48</td><td><strong>146</strong></td><td><strong>88.5%</strong></td></tr>
<tr><td>candidate_starvation</td><td>9</td><td>0</td><td>2</td><td>11</td><td>6.7%</td></tr>
<tr><td>legend_capture</td><td>7</td><td>0</td><td>0</td><td>7</td><td>4.2%</td></tr>
<tr><td>text_intrusion</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0.6%</td></tr>
</table>

<h3 class="section-title">5.4 PASS 대표 샘플 (pattern_1, y_mae=2.45px)</h3>
<table class="img-grid">
<tr>
  <td style="width:50%">{img_tag(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "debug_pattern_1" / "trace_path.png", "DP 추적 경로 — 곡선을 정확히 따라감")}</td>
  <td style="width:50%">{img_tag(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "debug_pattern_1" / "peaks_overlay.png", "피크 검출 — 모든 피크 정확히 탐지")}</td>
</tr>
</table>

<h3 class="section-title">5.5 FAIL 대표 샘플 (pattern_1360, y_mae=39.64px)</h3>
<table class="img-grid">
<tr>
  <td style="width:50%">{img_tag(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "debug_pattern_1360" / "trace_path.png", "DP 추적 경로 — grid line/축 선을 추종")}</td>
  <td style="width:50%">{img_tag(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "debug_pattern_1360" / "candidate_map_final.png", "후보맵 — 축 선 픽셀이 후보에 포함")}</td>
</tr>
</table>
</div>

<!-- ================================================================ -->
<!--  6. 심층 조사                                                     -->
<!-- ================================================================ -->
<div class="chapter" id="ch6">
<h2 class="chapter-title">6. 심층 조사 및 리서치 (Step 19~21)</h2>

<h3 class="section-title">6.1 §19 ML Rescue 필요성 평가</h3>
<div class="info-box">
  <div class="title">판정: ML rescue 현시점 불필요</div>
  <p>§19.2의 3가지 ML 필요 조건 (thin peak miss, tail collapse, mask 불안정) 어느 것도 발생하지 않음.</p>
  <p>현재 실패의 100%가 rule-based로 해결 가능한 유형.</p>
</div>

<h3 class="section-title">6.2 Step 20~21 심층 조사 (9개 항목)</h3>
<table>
<tr><th>#</th><th>항목</th><th>판정</th><th>근거</th></tr>
<tr><td>1</td><td>shape diversity feature</td><td><span class="tag green">현행 유지</span></td><td>상관계수 &lt; 0.6</td></tr>
<tr><td>2</td><td>surrogate family_id</td><td><span class="tag green">현행 유지</span></td><td>leakage 미발생</td></tr>
<tr><td>3</td><td>prominence 규칙</td><td><span class="tag green">현행 유지</span></td><td>recall = 1.00</td></tr>
<tr><td>4</td><td>styled/real 강도</td><td><span class="tag green">현행 유지</span></td><td>고유 실패 미발생</td></tr>
<tr><td>5</td><td>ML 구조 비교</td><td><span class="tag orange">보류</span></td><td>ML 불필요</td></tr>
<tr style="background:#fee2e2;"><td><strong>6</strong></td><td><strong>grid/axis line 대응</strong></td><td><span class="tag red">즉시 구현</span></td><td><strong>실패의 88.5%</strong></td></tr>
<tr><td>7</td><td>JPEG/blur 범위</td><td><span class="tag green">현행 유지</span></td><td>현행 적정</td></tr>
<tr><td>8</td><td>major peak 규칙</td><td><span class="tag green">현행 유지</span></td><td>f1 = 1.00</td></tr>
<tr><td>9</td><td>stratified binning</td><td><span class="tag green">현행 유지</span></td><td>27 bin 적정</td></tr>
</table>

<h3 class="section-title">6.3 핵심 발견: grid_confusion 원인 특정</h3>
<div class="warn-box">
  <div class="title">grid_confusion 샘플의 candidate_recall = 0.9992</div>
  <p>거의 모든 열에 후보가 존재 = axis line 자체가 후보로 잡힘.</p>
  <p>clean 이미지에 grid line이 없으므로 <strong>축 선(axis line) 자체를 DP가 추종</strong>하는 것이 원인.</p>
  <p>grid_confusion만 제거 시 pass rate 31% → 60% 예상.</p>
</div>
</div>

<!-- ================================================================ -->
<!--  7. Compliance                                                    -->
<!-- ================================================================ -->
<div class="chapter" id="ch7">
<h2 class="chapter-title">7. Compliance 검증 (Step 22~25)</h2>

<table>
<tr><th>영역</th><th>항목 수</th><th>PASS</th><th>FAIL</th></tr>
<tr><td>§22 파일 포맷</td><td>5</td><td>5</td><td>0</td></tr>
<tr><td>§23 스크립트 인터페이스</td><td>29</td><td>29</td><td>0</td></tr>
<tr><td>§24 최종 체크리스트</td><td>30</td><td>30</td><td>0</td></tr>
<tr><td>§25 자체 검토</td><td>21</td><td>21</td><td>0</td></tr>
<tr style="background:#dcfce7;"><td><strong>합계</strong></td><td><strong>85</strong></td><td><strong>85</strong></td><td><strong>0</strong></td></tr>
</table>

<div class="success-box">
  <div class="title">마스터 스펙 전 섹션(§3~25) 구현 및 검증 완료</div>
  <p>데이터 준비 8항목 + 엔진 구현 16항목 + 규칙 고정 6항목 전부 통과.</p>
</div>
</div>

<!-- ================================================================ -->
<!--  8. V2 최적화                                                     -->
<!-- ================================================================ -->
<div class="chapter" id="ch8">
<h2 class="chapter-title">8. V2 최적화 사이클</h2>

<h3 class="section-title">8.1 적용된 변경사항 (3건)</h3>

<h4 class="sub-section">변경 1: Axis Line 마스킹</h4>
<p><code>preprocess/masks.py</code> — ROI 경계 3px 하드 스트리핑, morphology 이후 적용.</p>
<pre>def mask_axis_lines(mask, margin=3):
    h, w = mask.shape[:2]
    out = mask.copy()
    out[:margin, :] = 0           # 상단 3px
    out[max(0, h-margin):, :] = 0 # 하단 3px
    out[:, :margin] = 0           # 좌측 3px
    out[:, max(0, w-margin):] = 0 # 우측 3px
    return out</pre>

<h4 class="sub-section">변경 2: Adaptive Threshold 상한 캡핑</h4>
<p><code>preprocess/color_model.py</code> — 과도한 임계값 방지.</p>
<pre># V1: adaptive_threshold = max(12.0, min_gap * 0.35)
# V2: adaptive_threshold = min(14.0, max(12.0, min_gap * 0.35))</pre>

<h4 class="sub-section">변경 3: DP Border Penalty</h4>
<p><code>trace/dp_trace.py</code> — ROI 상/하단 12px 이내 경로에 가중 패널티.</p>
<pre>EPSILON = 3.0; BORDER_RADIUS = 12
def _border_penalty(y, roi_height):
    dist = min(y, roi_height - 1 - y)
    if dist >= BORDER_RADIUS: return 0.0
    return EPSILON * (1.0 - dist / BORDER_RADIUS)</pre>

<h3 class="section-title">8.2 시행착오 전체 기록 (8회 실험)</h3>
<table>
<tr><th>#</th><th>실험</th><th>결과</th><th>최종</th></tr>
<tr><td>1</td><td>mask_axis_lines margin=4, morphology 이전</td><td>효과 없음 (31%)</td><td><span class="tag red">되돌림</span></td></tr>
<tr><td>2</td><td>margin=6 + 밀도 기반 검출</td><td>미미 (31→33%)</td><td><span class="tag orange">단순화</span></td></tr>
<tr><td>3</td><td>threshold = min(15, max(12, gap*0.22))</td><td>악화 (26%)</td><td><span class="tag red">되돌림</span></td></tr>
<tr><td>4</td><td>threshold = min(14, max(12, gap*0.35))</td><td>미미 (33%)</td><td><span class="tag green">채택</span></td></tr>
<tr><td>5</td><td>적응형 fg_ratio 캡핑</td><td>무효 (33%)</td><td><span class="tag red">되돌림</span></td></tr>
<tr><td>6</td><td>mask_B 비활성화</td><td>악화 (31%)</td><td><span class="tag red">되돌림</span></td></tr>
<tr><td>7</td><td>ALPHA=0.5, BETA=0.2, K=3</td><td>무효 (33%)</td><td><span class="tag red">되돌림</span></td></tr>
<tr><td>8</td><td>_local_continuity 제거</td><td>악화 (30%)</td><td><span class="tag red">되돌림</span></td></tr>
</table>

<h3 class="section-title">8.3 실험에서 얻은 교훈</h3>
<div class="warn-box">
  <div class="title">핵심 발견 3가지</div>
  <p><strong>1.</strong> foreground 밀도를 줄여도 (70k→16k) y_mae 불변 — grid line 픽셀이 아닌 추적 정확도 문제</p>
  <p><strong>2.</strong> DP smoothness 조정은 양날의 검 — 낮추면 피크 추적 ↑ but 노이즈 취약 ↑</p>
  <p><strong>3.</strong> border penalty만 유의미 — 경계 축 선 추종 억제 (31→33%), 내부 grid에는 무력</p>
</div>

<h3 class="section-title">8.4 V2 전처리 비교 (pattern_1360)</h3>
<table class="img-grid">
<tr>
  <td style="width:50%">
    {img_tag(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "debug_pattern_1360" / "combined_mask.png", "V1 — 결합 마스크")}
  </td>
  <td style="width:50%">
    {img_tag(ROOT / "experiments" / "archive" / "outputs_v2_clean" / "debug_pattern_1360" / "combined_mask.png", "V2 — 결합 마스크 (threshold 캡핑)")}
  </td>
</tr>
<tr>
  <td>{img_tag(ROOT / "experiments" / "archive" / "outputs_baseline_clean" / "debug_pattern_1360" / "trace_path.png", "V1 — DP 추적 경로")}</td>
  <td>{img_tag(ROOT / "experiments" / "archive" / "outputs_v2_clean" / "debug_pattern_1360" / "trace_path.png", "V2 — DP 추적 경로 (border penalty)")}</td>
</tr>
</table>
</div>

<!-- ================================================================ -->
<!--  9. V1 vs V2 비교                                                 -->
<!-- ================================================================ -->
<div class="chapter" id="ch9">
<h2 class="chapter-title">9. V1 vs V2 성능 비교</h2>

<h3 class="section-title">9.1 전체 결과</h3>
<table>
<tr><th>데이터셋</th><th>V1 (Baseline)</th><th>V2 (최적화)</th><th>변화</th></tr>
<tr><td>clean Pass Rate</td><td>31.0%</td><td><strong>33.0%</strong></td><td style="color:green;">+2pp</td></tr>
<tr><td>styled Pass Rate</td><td>0.0%</td><td>0.0%</td><td>—</td></tr>
<tr><td>real_like Pass Rate</td><td>4.0%</td><td>4.0%</td><td>—</td></tr>
<tr><td>clean y_mae mean</td><td>26.58 px</td><td><strong>13.33 px</strong></td><td style="color:green;">−13.25 px</td></tr>
<tr><td>clean y_mae max</td><td>682.38 px</td><td><strong>338.19 px</strong></td><td style="color:green;">−344 px</td></tr>
</table>

<h3 class="section-title">9.2 Failure Taxonomy 변화 (clean)</h3>
<table>
<tr><th>Label</th><th>V1</th><th>V2</th><th>변화</th></tr>
<tr><td>grid_confusion</td><td>48</td><td><strong>34</strong></td><td style="color:green;">−14</td></tr>
<tr><td>candidate_starvation</td><td>9</td><td>14</td><td style="color:red;">+5</td></tr>
<tr><td>legend_capture</td><td>7</td><td>9</td><td style="color:red;">+2</td></tr>
<tr><td>text_intrusion</td><td>1</td><td>1</td><td>—</td></tr>
</table>

<h3 class="section-title">9.3 진단 메트릭 비교</h3>
<table>
<tr><th>메트릭</th><th>V1</th><th>V2</th></tr>
<tr><td>candidate_recall</td><td>0.9487</td><td>0.9209</td></tr>
<tr><td>empty_column_rate</td><td>0.0314</td><td>0.0477</td></tr>
<tr><td>recovery_success_rate</td><td>0.9300</td><td>0.9100</td></tr>
<tr><td>path_margin_instability</td><td>0.4466</td><td>0.4616</td></tr>
</table>
</div>

<!-- ================================================================ -->
<!--  10. 외부 시스템 검토                                              -->
<!-- ================================================================ -->
<div class="chapter" id="ch10">
<h2 class="chapter-title">10. 외부 XRD 분석 시스템 검토</h2>

<p><strong>대상</strong>: MaterIAI 프로젝트 — React 기반 브라우저 내 XRD 분석 도구 (9개 모듈).</p>

<table>
<tr><th>모듈</th><th>기능</th><th>우리 프로젝트 활용</th></tr>
<tr><td>xrdParser.js</td><td>XRD 파일 파싱</td><td><span class="tag red">불필요</span> — 우리는 이미지 입력</td></tr>
<tr><td>peakDetection.js</td><td>S-G 스무딩 + 피크 탐지</td><td><span class="tag orange">제한적</span></td></tr>
<tr><td>peakFitting.js</td><td>Gaussian/Voigt 피팅</td><td><span class="tag orange">제한적</span> — 후처리 참고 가능</td></tr>
<tr><td>millerIndex.js</td><td>d-spacing, 밀러지수</td><td><span class="tag red">불필요</span> — V1 스코프 밖</td></tr>
<tr><td>crystallinity.js</td><td>결정화도, Scherrer</td><td><span class="tag red">불필요</span></td></tr>
<tr><td>dislocationAnalysis.js</td><td>mWH/mWA 전위 밀도</td><td><span class="tag red">불필요</span></td></tr>
</table>

<div class="info-box">
  <div class="title">결론</div>
  <p>근본적으로 다른 단계의 도구. 우리 프로젝트가 <strong>출력하려는 결과물을 입력으로 받는</strong> 다운스트림 시스템. 직접 활용할 코드/알고리즘 없음.</p>
</div>
</div>

<!-- ================================================================ -->
<!--  11. 근본 문제 심층 분석                                           -->
<!-- ================================================================ -->
<div class="chapter" id="ch11">
<h2 class="chapter-title">11. 근본 문제 심층 분석</h2>

<h3 class="section-title">11.1 문제 계층 구조</h3>
<div class="pipeline" style="background:#fff5f5; border-color:#fca5a5;">
<strong style="color:#991b1b;">[Level 1]</strong> 평가 체계의 구조적 한계 — <strong>GT 미스매치</strong><br>
&emsp;&emsp;↓ 완벽한 엔진도 100% 달성 불가<br>
<strong style="color:#92400e;">[Level 2]</strong> 전처리의 색상 모델 한계 — <strong>Lab 단일 모델</strong><br>
&emsp;&emsp;↓ styled/real 대응 불가<br>
<strong style="color:#6b21a8;">[Level 3]</strong> DP 추적의 구조적 한계 — <strong>smoothness bias</strong><br>
&emsp;&emsp;↓ grid_confusion 완전 해소 불가
</div>

<!-- Level 1 -->
<h3 class="section-title">11.2 [Level 1] GT 미스매치 — 가장 근본적 문제</h3>

<div class="warn-box">
  <div class="title">문제 정의</div>
  <p>Ground Truth는 각 x 열(column)에 대해 <strong>단일 보간 y값</strong>을 저장하지만, 렌더링된 이미지에서 날카로운 피크는 <strong>수직 픽셀 밴드</strong>를 형성한다.</p>
</div>

<pre>GT가 기대하는 것:           실제 렌더링:
  |                            |
  |    *                       |   **
  |   * *                      |  ****
  |  *   *                     | ******
  | *     *                    |********
  |*       *                   |*       *

x=50에서 GT y=10 (단일)      x=50에서 곡선 픽셀 y=[5,6,...,15] (밴드)</pre>

<p><strong>영향</strong>:</p>
<ul style="margin:8px 0 16px 20px;">
  <li>엔진이 피크의 어떤 픽셀을 선택하든 GT 점과 오차 발생</li>
  <li>날카로운 피크가 많은 샘플일수록 y_mae가 체계적으로 높음</li>
  <li><strong>완벽한 엔진이라도</strong> 이 샘플들에서 y_mae ≤ 5px 달성 불가</li>
  <li>현재 추정 상한: clean 기준 약 <strong>70~80%</strong></li>
</ul>

<p><strong>실측 증거</strong>: 디버그 분석에서 특정 FAIL 샘플의 trace path가 <em>시각적으로는 곡선 위에 있으나</em> y_mae가 높게 측정됨. 피크 부근 GT y값과 실제 렌더링 픽셀 y값 사이에 5~15px 차이.</p>

<!-- Level 2 -->
<h3 class="section-title">11.3 [Level 2] 색상 모델 한계 — styled 0% 통과의 원인</h3>

<div class="warn-box">
  <div class="title">문제 정의</div>
  <p>현재 색상 모델은 <code>color_sample_point</code>에서 <strong>단일 색상 프로토타입</strong>을 추출하고, Lab 거리로 전경/배경 분류. styled의 색상 곡선 + 색상 배경에서 분류 실패.</p>
</div>

<table>
<tr><th>상황</th><th>현재 모델 대응</th><th>결과</th></tr>
<tr><td>검정 곡선 + 흰 배경</td><td>Lab 거리 큼 → 정확 분류</td><td>Clean: <span class="tag green">작동</span></td></tr>
<tr><td>파란/빨간 곡선 + 색상 배경</td><td>Lab 거리 작음 → 분류 실패</td><td>Styled: <span class="tag red">실패</span></td></tr>
<tr><td>그래디언트 배경</td><td>배경 색상이 위치마다 다름</td><td>Styled: <span class="tag red">실패</span></td></tr>
<tr><td>곡선 색상 ≈ 격자선 색상</td><td>둘 다 전경으로 잡힘</td><td>모든 셋: <span class="tag red">grid_confusion</span></td></tr>
</table>

<p><strong>실측</strong>: styled candidate_recall = 0.9920 (후보 풍부) but 전부 잘못된 후보. styled 평균 y_mae = 15.96px.</p>

<!-- Level 3 -->
<h3 class="section-title">11.4 [Level 3] DP 추적의 구조적 한계</h3>

<div class="warn-box">
  <div class="title">문제 정의</div>
  <p>DP는 smoothness(dy, d²y)에 강한 가중치 → <strong>수평에 가까운 경로를 선호</strong>. 날카로운 피크는 높은 비용, grid line은 최저 비용.</p>
</div>

<p><strong>구조적 문제 3가지</strong>:</p>
<table>
<tr><th>문제</th><th>설명</th><th>결과</th></tr>
<tr><td><strong>Smoothness-accuracy trade-off</strong></td><td>피크 = 큰 dy = 높은 비용, grid = dy=0 = 최저 비용</td><td>DP가 grid를 선호</td></tr>
<tr><td><strong>Local greedy</strong></td><td>직전 열의 최적만 참조, 한번 잘못 진입하면 복구 어려움</td><td>wrong_branch_lock_in</td></tr>
<tr><td><strong>Confidence 피드백 루프</strong></td><td>local_continuity가 이전 y에 의존 → 오류 강화</td><td>잘못된 경로 지속</td></tr>
</table>

<h3 class="section-title">11.5 문제별 영향 범위 정리</h3>
<table>
<tr><th>문제</th><th>영향 대상</th><th>영향 규모</th><th>해결 난이도</th></tr>
<tr style="background:#fee2e2;"><td>GT 미스매치</td><td>피크 있는 모든 샘플</td><td>y_mae에 5~15px 체계적 오차</td><td><span class="tag red">높음</span></td></tr>
<tr style="background:#fef3c7;"><td>색상 모델 한계</td><td>styled/real 전체</td><td>styled 0%, real 4%</td><td><span class="tag orange">중간</span></td></tr>
<tr style="background:#f3e8ff;"><td>DP smoothness bias</td><td>grid_confusion 전체</td><td>clean 34건 (51%)</td><td><span class="tag red">높음</span></td></tr>
<tr><td>Border axis 추종</td><td>ROI 경계 근처</td><td>V2에서 부분 해소</td><td><span class="tag green">해결됨</span></td></tr>
<tr><td>Threshold 과잉</td><td>어두운 곡선 패턴</td><td>V2에서 캡핑</td><td><span class="tag green">해결됨</span></td></tr>
</table>
</div>

<!-- ================================================================ -->
<!--  12. 해결 방향                                                    -->
<!-- ================================================================ -->
<div class="chapter" id="ch12">
<h2 class="chapter-title">12. 해결 방향 및 실행 로드맵</h2>

<!-- 방향 A -->
<h3 class="section-title">12.1 방향 A: GT 평가 체계 개선 (Level 1 해결)</h3>
<div class="info-box">
  <div class="title">목표: 피크 영역의 체계적 오차를 평가 체계가 허용하도록 수정</div>
</div>

<h4 class="sub-section">A-1. 피크 밴드 허용 오차 (Peak Band Tolerance)</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>렌더링 시 각 x열의 곡선 픽셀 범위 [y_min, y_max]를 GT에 저장</li>
  <li>엔진 trace y가 이 범위 안이면 오차 = 0</li>
  <li>구현: <code>scripts/render_clean_dataset.py</code> + <code>eval/metrics.py</code></li>
</ul>

<h4 class="sub-section">A-2. 가중 y_mae (Weighted MAE)</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>피크 영역(높은 dy)의 y 오차에 낮은 가중치, 평탄 구간의 정확도 중시</li>
</ul>

<table>
<tr><th>항목</th><th>내용</th></tr>
<tr><td>구현 위치</td><td><code>scripts/render_clean_dataset.py</code>, <code>eval/metrics.py</code></td></tr>
<tr><td>난이도</td><td><span class="tag orange">중</span> (렌더러 + 평가기 수정, 데이터 재렌더링)</td></tr>
<tr><td>예상 효과</td><td>clean 33% → <strong>50~60%</strong></td></tr>
</table>

<!-- 방향 B -->
<h3 class="section-title">12.2 방향 B: 다중 색상 모델 (Level 2 해결)</h3>
<div class="info-box">
  <div class="title">목표: styled/real-like에서 곡선 색상을 정확히 분리</div>
</div>

<h4 class="sub-section">B-1. 다중 프로토타입 클러스터링</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>color_sample_point 주변 패치에서 K-means (K=3~5) 클러스터링</li>
  <li>가장 배경과 먼 클러스터를 곡선 프로토타입으로 선택</li>
</ul>

<h4 class="sub-section">B-2. 적응형 배경 모델</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>ROI를 격자로 분할, 각 영역별 로컬 배경 추출</li>
  <li>그래디언트 배경에서도 전경/배경 분리 가능</li>
</ul>

<h4 class="sub-section">B-3. HSV 기반 색상 채널 분리</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>styled 곡선은 특정 Hue에 집중, 격자선은 회색(낮은 채도)</li>
  <li>Hue 채널 필터링으로 곡선/격자 분리</li>
</ul>

<table>
<tr><th>항목</th><th>내용</th></tr>
<tr><td>구현 위치</td><td><code>preprocess/color_model.py</code>, <code>preprocess/masks.py</code></td></tr>
<tr><td>난이도</td><td><span class="tag orange">중</span></td></tr>
<tr><td>예상 효과</td><td>styled 0% → <strong>20~40%</strong>, real 4% → <strong>15~30%</strong></td></tr>
</table>

<!-- 방향 C -->
<h3 class="section-title">12.3 방향 C: DP 추적 알고리즘 개선 (Level 3 해결)</h3>
<div class="info-box">
  <div class="title">목표: grid_confusion을 구조적으로 해소</div>
</div>

<h4 class="sub-section">C-1. Grid Line 사전 검출 마스킹 (최우선)</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>Hough Line Transform으로 수평/수직 직선 검출</li>
  <li>검출된 직선 ±2px을 combined_mask에서 제거</li>
  <li>axis line뿐 아니라 plot 내부 격자선도 대응</li>
  <li>구현: <code>preprocess/masks.py</code>에 <code>mask_grid_lines()</code> 추가</li>
  <li>난이도: <span class="tag green">낮음</span>, 소요: 2~3시간</li>
</ul>

<h4 class="sub-section">C-2. 양방향 DP (Bidirectional DP)</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>좌→우 + 우→좌 두 경로를 계산하고 결합</li>
  <li>한쪽에서 잘못 잠긴 경로를 반대 방향이 보정</li>
  <li>구현: <code>trace/dp_trace.py</code></li>
  <li>난이도: <span class="tag orange">중</span>, 소요: 4~6시간</li>
</ul>

<h4 class="sub-section">C-3. Multi-scale 추적</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>1차: 4x 축소 저해상도에서 global path (grid line이 사라짐)</li>
  <li>2차: 원본 해상도에서 1차 경로 ±W만 탐색</li>
  <li>구현: <code>trace/dp_trace.py</code> multi-scale wrapper</li>
  <li>난이도: <span class="tag red">중~높</span>, 소요: 6~10시간</li>
</ul>

<h4 class="sub-section">C-4. 2-pass Confidence 재평가</h4>
<ul style="margin:8px 0 16px 20px;">
  <li>1차 DP 완료 후 전체 경로 통계로 confidence 재계산</li>
  <li>피드백 루프 차단 → 2차 DP 개선</li>
  <li>난이도: <span class="tag orange">중</span></li>
</ul>

<table>
<tr><th>하위 방향</th><th>예상 효과</th><th>난이도</th></tr>
<tr style="background:#dcfce7;"><td><strong>C-1 Hough 마스킹</strong></td><td>grid_confusion 50% 감소 → clean 45%</td><td><span class="tag green">낮음</span></td></tr>
<tr><td>C-2 양방향 DP</td><td>추가 10~15% 해소 → clean 55%</td><td><span class="tag orange">중</span></td></tr>
<tr><td>C-3 Multi-scale</td><td>추가 10% → clean 65%</td><td><span class="tag red">중~높</span></td></tr>
<tr><td>C-4 2-pass</td><td>피드백 루프 차단 → clean 60~65%</td><td><span class="tag orange">중</span></td></tr>
</table>

<!-- 방향 D -->
<h3 class="section-title">12.4 방향 D: ML Rescue (최후 수단)</h3>
<div class="info-box">
  <div class="title">전제 조건: 방향 A~C 적용 후 clean pass rate &lt; 80%</div>
</div>

<table>
<tr><th>구조</th><th>적합 시나리오</th><th>입력</th></tr>
<tr><td>1D CNN + BiLSTM</td><td>signal-level 보정</td><td>per-column y + confidence</td></tr>
<tr><td>Lightweight 2D U-Net</td><td>mask refinement</td><td>ROI 이미지 (RGB)</td></tr>
</table>

<!-- 로드맵 -->
<h3 class="section-title">12.5 권장 실행 로드맵</h3>

<div class="pipeline" style="background:#eff6ff; border-color:#3b82f6;">
<strong>[Phase 1 — 즉시]</strong> 방향 C-1: Hough Grid 마스킹<br>
&emsp;예상: clean 33% → 45% &emsp;|&emsp; 난이도: 낮음 &emsp;|&emsp; 소요: 2~3시간<br><br>

<strong>[Phase 2 — 단기]</strong> 방향 A: GT 평가 체계 개선<br>
&emsp;예상: 45% → 55~60% &emsp;|&emsp; 난이도: 중 &emsp;|&emsp; 소요: 4~6시간<br><br>

<strong>[Phase 3 — 단기]</strong> 방향 B: 다중 색상 모델<br>
&emsp;예상: styled 0% → 20~40% &emsp;|&emsp; 난이도: 중 &emsp;|&emsp; 소요: 4~6시간<br><br>

<strong>[Phase 4 — 중기]</strong> 방향 C-2/3: 양방향 DP 또는 Multi-scale<br>
&emsp;예상: clean 60% → 70% &emsp;|&emsp; 난이도: 중~높 &emsp;|&emsp; 소요: 6~10시간<br><br>

<strong>[Phase 5 — 최종]</strong> 방향 D: ML Rescue (필요 시)<br>
&emsp;예상: 70% → 85%+ &emsp;|&emsp; 난이도: 높 &emsp;|&emsp; 소요: 1~2주
</div>

<div style="text-align:center; margin:20px 0;">
  <div class="kpi"><div class="val">33%</div><div class="label">현재 (V2)</div></div>
  <div class="kpi" style="background:#fef3c7;"><div class="val">45%</div><div class="label">Phase 1 후</div></div>
  <div class="kpi" style="background:#dcfce7;"><div class="val">60%</div><div class="label">Phase 2 후</div></div>
  <div class="kpi" style="background:#dcfce7;"><div class="val">70%</div><div class="label">Phase 4 후</div></div>
  <div class="kpi" style="background:#dbeafe;"><div class="val">85%+</div><div class="label">Phase 5 후</div></div>
</div>
</div>

<!-- ================================================================ -->
<!--  13. 부록                                                         -->
<!-- ================================================================ -->
<div class="chapter" id="ch13">
<h2 class="chapter-title">13. 부록</h2>

<h3 class="section-title">13.1 V2 최종 엔진 설정</h3>
<table>
<tr><th>모듈</th><th>파라미터</th><th>V1</th><th>V2</th></tr>
<tr><td>color_model.py</td><td>adaptive_threshold</td><td>max(12, gap*0.35)</td><td><strong>min(14, max(12, gap*0.35))</strong></td></tr>
<tr><td>masks.py</td><td>mask_axis_lines margin</td><td>(없음)</td><td><strong>3px, morphology 이후</strong></td></tr>
<tr><td>dp_trace.py</td><td>EPSILON / BORDER_RADIUS</td><td>(없음)</td><td><strong>3.0 / 12</strong></td></tr>
<tr><td>dp_trace.py</td><td>ALPHA / BETA</td><td>1.0 / 0.35</td><td>1.0 / 0.35 (변경 없음)</td></tr>
<tr><td>candidates.py</td><td>MAX_CANDIDATES_FOR_DP</td><td>6</td><td>6 (변경 없음)</td></tr>
</table>

<h3 class="section-title">13.2 참고 문서</h3>
<table>
<tr><th>문서</th><th>내용</th></tr>
<tr><td><code>xrd_digitizer_v1_master_spec.md</code></td><td>전체 아키텍처 및 요구사항 (2066 lines)</td></tr>
<tr><td><code>outputs/implementation_report_steps_3_to_18.md</code></td><td>Step 3~18 구현 상세</td></tr>
<tr><td><code>outputs/step22_25_compliance_report.md</code></td><td>§22~25 검증 (85/85)</td></tr>
<tr><td><code>outputs/step20_research_findings.md</code></td><td>9개 항목 조사 결과</td></tr>
<tr><td><code>outputs/step21_deep_research_report.md</code></td><td>리서치 기반 규칙 확정</td></tr>
<tr><td><code>outputs/ml_rescue_necessity_assessment.md</code></td><td>ML rescue 불필요 판정</td></tr>
</table>

<h3 class="section-title">13.3 실행 환경</h3>
<table>
<tr><th>항목</th><th>내용</th></tr>
<tr><td>OS</td><td>Windows 10 (build 26200)</td></tr>
<tr><td>Python</td><td>3.13</td></tr>
<tr><td>주요 라이브러리</td><td>NumPy, Pandas, Pillow, SciPy, scikit-image</td></tr>
</table>
</div>

</div><!-- /.content -->
</div><!-- /.container -->
</body>
</html>
"""

# Write HTML
html_path = OUT / "xrd_digitizer_v1_full_report.html"
html_path.write_text(html, encoding="utf-8")
print(f"HTML saved: {html_path}")

# ── PDF via Playwright (headless Chromium) ──
try:
    from playwright.sync_api import sync_playwright
    pdf_path = OUT / "xrd_digitizer_v1_full_report.pdf"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri())
        page.pdf(path=str(pdf_path), format="A4", print_background=True,
                 margin={"top": "15mm", "bottom": "15mm", "left": "10mm", "right": "10mm"})
        browser.close()
    print(f"PDF saved: {pdf_path}")
except Exception as e:
    print(f"PDF generation failed: {e}")
    print("HTML report is ready. Open in browser and use Ctrl+P to save as PDF.")
