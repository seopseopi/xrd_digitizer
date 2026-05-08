from __future__ import annotations

import argparse
import base64
import json
import pathlib
import statistics
import sys
from typing import Any

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHIVE_BATCH = ROOT / "experiments" / "archive" / "outputs_legacy_runs"
RUNS = ROOT / "outputs" / "runs"
sys.path.insert(0, str(ROOT))

from eval.gates_exp import check_gate_v2_band, check_gate_v2_strict
from eval.metrics import compute_all_metrics
from eval.metrics_exp import compute_metrics_v2, merge_main_with_v2


def build_domains(
    prefix: str,
    batch_root: pathlib.Path | None = None,
) -> list[tuple[str, pathlib.Path, pathlib.Path, str]]:
    root = batch_root if batch_root is not None else ARCHIVE_BATCH
    return [
        ("clean", ROOT / "data/manifests/clean_manifest.csv", root / f"{prefix}_clean", "clean"),
        ("styled", ROOT / "data/manifests/styled_manifest.csv", root / f"{prefix}_styled", "styled"),
        ("real-like", ROOT / "data/manifests/real_manifest.csv", root / f"{prefix}_real", "real-like"),
    ]


def img_b64(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b}"


def gather_domain(name: str, manifest_path: pathlib.Path, out_dir: pathlib.Path, gate_type: str) -> dict[str, Any]:
    df = pd.read_csv(manifest_path)
    samples = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        rp = out_dir / f"{sid}_result.json"
        dp = out_dir / f"debug_{sid}" / "debug.json"
        if not rp.exists() or not dp.exists():
            continue
        result = json.loads(rp.read_text(encoding="utf-8"))
        debug = json.loads(dp.read_text(encoding="utf-8"))
        gt = json.loads(pathlib.Path(str(row["gt_path"])).read_text(encoding="utf-8"))
        m = compute_all_metrics(result, debug, gt)
        v2 = compute_metrics_v2(result, debug, gt)
        mm = merge_main_with_v2(m["main"], v2)
        gs = check_gate_v2_strict(mm, v2, gate_type)
        gb = check_gate_v2_band(mm, v2, gate_type)
        samples.append(
            {
                "sample_id": sid,
                "main": mm,
                "v2": v2,
                "strict_pass": bool(gs["passed"]),
                "band_pass": bool(gb["passed"]),
                "both_pass": bool(gs["passed"] and gb["passed"]),
                "debug_dir": out_dir / f"debug_{sid}",
            }
        )

    if not samples:
        return {"name": name, "count": 0, "samples": []}

    strict_mae = [s["v2"]["strict_curve_y_mae_px"] for s in samples]
    band_mae = [s["v2"]["curve_band_mae_px"] for s in samples]
    hit = [s["v2"]["band_hit_rate"] for s in samples]
    peak_recall = [s["v2"]["peak_recall_fixed"] for s in samples]
    both_pass = sum(1 for s in samples if s["both_pass"])
    strict_pass = sum(1 for s in samples if s["strict_pass"])
    band_pass = sum(1 for s in samples if s["band_pass"])
    worst = max(samples, key=lambda s: s["v2"]["strict_curve_y_mae_px"])
    median_s = sorted(samples, key=lambda s: s["v2"]["strict_curve_y_mae_px"])[len(samples) // 2]

    return {
        "name": name,
        "count": len(samples),
        "strict_pass": strict_pass,
        "band_pass": band_pass,
        "both_pass": both_pass,
        "strict_mae_mean": statistics.mean(strict_mae),
        "strict_mae_std": statistics.pstdev(strict_mae) if len(strict_mae) > 1 else 0.0,
        "band_mae_mean": statistics.mean(band_mae),
        "band_hit_mean": statistics.mean(hit),
        "peak_recall_mean": statistics.mean(peak_recall),
        "worst": worst,
        "median": median_s,
        "samples": samples,
    }


def section_images(sample: dict[str, Any], title: str) -> str:
    d: pathlib.Path = sample["debug_dir"]
    files = [
        ("01_roi_preview.png", "ROI"),
        ("02_color_score.png", "Color score"),
        ("03_ridge_score.png", "Ridge score"),
        ("05_combined_mask.png", "Combined mask"),
        ("07_candidate_map.png", "Candidate map"),
        ("08_dual_pass_overlay.png", "Dual-pass overlay"),
        ("09_trace_path.png", "Trace path"),
        ("10_peak_overlay.png", "Peak overlay"),
    ]
    cards = []
    for fn, cap in files:
        p = d / fn
        b64 = img_b64(p)
        if not b64:
            cards.append(f"<div class='card'><div class='missing'>missing: {fn}</div><div class='cap'>{cap}</div></div>")
            continue
        cards.append(
            "<div class='card'>"
            f"<img src='{b64}'/>"
            f"<div class='cap'>{cap}</div>"
            "</div>"
        )
    m = sample["v2"]
    return (
        f"<h4>{title}: {sample['sample_id']}</h4>"
        f"<p class='meta'>strict_mae={m['strict_curve_y_mae_px']:.3f}, "
        f"band_mae={m['curve_band_mae_px']:.3f}, band_hit={m['band_hit_rate']:.3f}, "
        f"peak_recall={m['peak_recall_fixed']:.3f}</p>"
        f"<div class='grid'>{''.join(cards)}</div>"
    )


def build_html(results: list[dict[str, Any]]) -> str:
    rows = []
    for r in results:
        rows.append(
            "<tr>"
            f"<td>{r['name']}</td><td>{r['count']}</td>"
            f"<td>{r['strict_pass']}</td><td>{r['band_pass']}</td><td>{r['both_pass']}</td>"
            f"<td>{r['strict_mae_mean']:.3f} ± {r['strict_mae_std']:.3f}</td>"
            f"<td>{r['band_mae_mean']:.3f}</td><td>{r['band_hit_mean']:.3f}</td><td>{r['peak_recall_mean']:.3f}</td>"
            "</tr>"
        )

    impl_rows = [
        ("§5 평가기 v2(strict/band/gate)", "완료"),
        ("§6 전처리 v2(color/ridge/grid/combined)", "완료(근사 포함)"),
        ("§7 후보 생성 v2(conf/filter/final)", "완료"),
        ("§8 추적 v2(dual-pass/disagreement/unstable-ambiguous)", "완료"),
        ("§8.5 rescue 채택 규칙(gain/coverage/boundary)", "완료(로컬 구현)"),
        ("§9 선택적 smoothing gate", "완료"),
        ("§9 raw+smoothed peak 병합/NMS", "완료(어깨 억제 일부 근사)"),
        ("§10 ML rescue v1.5 실행 경로", "부분(인터페이스/게이트 중심)"),
        ("§12 디버그 판독 자동화", "부분(시각/수치 혼합)"),
        ("§13 파라미터 조정 순서 기반 자동 튜닝", "완료: tuning/sequential_v13.py + --tune_json 배치"),
    ]
    impl_table = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in impl_rows)

    details = []
    for r in results:
        details.append(f"<h3>{r['name']} 상세</h3>")
        details.append(section_images(r["worst"], "Worst sample"))
        details.append(section_images(r["median"], "Median sample"))

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>V2 상세 검증 보고서</title>
<style>
body{{font-family:'Malgun Gothic',sans-serif;margin:24px;color:#111;line-height:1.45}}
h1{{font-size:22px;border-bottom:3px solid #1e40af;padding-bottom:8px}}
h2{{margin-top:24px;font-size:17px;color:#1e3a8a}}
h3{{margin-top:20px;font-size:15px}}
h4{{margin:12px 0 6px 0;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:10pt}}
th,td{{border:1px solid #d1d5db;padding:6px 8px;vertical-align:top}}
th{{background:#eff6ff}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.card{{border:1px solid #ddd;border-radius:6px;padding:6px;background:#fff}}
.card img{{width:100%;display:block}}
.cap{{font-size:9pt;color:#374151;margin-top:4px}}
.missing{{color:#991b1b;font-size:9pt}}
.meta{{font-size:10pt;color:#374151}}
.warn{{background:#fff7ed;border:1px solid #fdba74;padding:10px;border-radius:6px}}
@page{{size:A4;margin:12mm}}
</style></head><body>
<h1>XRD V2 통합 상세 검증 보고서</h1>
<p>요청사항: 계획서 항목별 구현 상태 검증 + 전량 실행 결과 + 디버그 근거 이미지 포함 정식 보고.</p>

<h2>1) 전량 실행 요약</h2>
<table>
<tr><th>도메인</th><th>샘플 수</th><th>strict pass</th><th>band pass</th><th>both pass</th><th>strict mae(mean±std)</th><th>band mae mean</th><th>band hit mean</th><th>peak recall mean</th></tr>
{''.join(rows)}
</table>

<h2>2) 계획서 항목 구현 상태</h2>
<table><tr><th>항목</th><th>상태</th></tr>{impl_table}</table>

<h2>3) 핵심 결론</h2>
<div class='warn'>
<b>실행 안정성</b>은 확보되었으나, strict/band 게이트 통과율은 모든 도메인에서 0으로 성능 미달 상태.<br/>
현재 구현은 계획서의 큰 구조는 연결되었지만, 수치 임계/세부 규칙(특히 피크 및 튜닝 순서 자동화) 보강이 필요.
</div>

<h2>4) 도메인별 대표 샘플 근거</h2>
{''.join(details)}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_prefix", type=str, default="exp_full")
    ap.add_argument("--report_name", type=str, default="exp_detailed_validation_report")
    ap.add_argument(
        "--batch_root",
        type=pathlib.Path,
        default=None,
        help=f"V2 배치 산출물 상위 폴더 (기본: {ARCHIVE_BATCH})",
    )
    args = ap.parse_args()

    batch_root = args.batch_root
    if batch_root is None and args.output_prefix.startswith("exp_tuned"):
        batch_root = RUNS
    domains = build_domains(args.output_prefix, batch_root)
    results = []
    for name, manifest, out_dir, gate in domains:
        results.append(gather_domain(name, manifest, out_dir, gate))

    html = build_html(results)
    out_html = ROOT / f"outputs/{args.report_name}.html"
    out_pdf = ROOT / f"outputs/{args.report_name}.pdf"
    out_html.write_text(html, encoding="utf-8")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(out_html.resolve().as_uri())
        page.pdf(path=str(out_pdf), format="A4", print_background=True)
        browser.close()

    print(f"HTML: {out_html}")
    print(f"PDF: {out_pdf}")


if __name__ == "__main__":
    main()
