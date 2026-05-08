"""
실험 파이프라인(v2_experimental) 배치 결과를 평가하고 HTML·PDF로 남긴다.
사전에 batch_run.py --pipeline v2_experimental --allow_experimental_v2 로 결과가 있어야 한다.
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import pathlib
import sys
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.gates_exp import check_gate_v2_band, check_gate_v2_strict
from eval.metrics import compute_all_metrics
from eval.metrics_exp import compute_metrics_v2, merge_main_with_v2


def evaluate_one(
    result_path: pathlib.Path,
    debug_path: pathlib.Path,
    gt_path: pathlib.Path,
    gate_type: str,
) -> Dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    metrics = compute_all_metrics(result, debug, gt)
    v2 = compute_metrics_v2(result, debug, gt)
    main_m = merge_main_with_v2(metrics["main"], v2)
    strict = check_gate_v2_strict(main_m, v2, gate_type)
    band = check_gate_v2_band(main_m, v2, gate_type)
    return {
        "main": main_m,
        "diagnosis": metrics["diagnosis"],
        "v2_scale": {k: v2[k] for k in ("s_w", "s_h", "s", "plot_w", "plot_h")},
        "v2_metrics": {k: v2[k] for k in (
            "strict_curve_y_mae_px", "curve_band_mae_px", "band_hit_rate", "peak_recall_fixed",
        )},
        "gate_strict": strict,
        "gate_band": band,
        "both_pass": bool(strict["passed"] and band["passed"]),
    }


def _table_rows(samples: List[Dict[str, Any]]) -> str:
    rows = []
    for s in samples:
        m = s["main"]
        v = s["v2_metrics"]
        rows.append(
            "<tr>"
            f"<td>{html_module.escape(s['sample_id'])}</td>"
            f"<td>{v['strict_curve_y_mae_px']}</td>"
            f"<td>{v['curve_band_mae_px']}</td>"
            f"<td>{v['band_hit_rate']}</td>"
            f"<td>{m['major_peak_x_error']}</td>"
            f"<td>{m['major_peak_y_error']}</td>"
            f"<td>{v['peak_recall_fixed']}</td>"
            f"<td>{m['max_gap_px']}</td>"
            f"<td>{m['calibration_roundtrip_error']}</td>"
            f"<td>{'예' if s['gate_strict']['passed'] else '아니오'}</td>"
            f"<td>{'예' if s['gate_band']['passed'] else '아니오'}</td>"
            f"<td>{'예' if s['both_pass'] else '아니오'}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_html(samples: List[Dict[str, Any]], gate_type: str, manifest: str) -> str:
    n = len(samples)
    n_strict = sum(1 for s in samples if s["gate_strict"]["passed"])
    n_band = sum(1 for s in samples if s["gate_band"]["passed"])
    n_both = sum(1 for s in samples if s["both_pass"])
    summary = (
        f"<p>샘플 수: {n}, strict 통과: {n_strict}, band 통과: {n_band}, 둘 다: {n_both}, "
        f"게이트 종류: {html_module.escape(gate_type)}</p>"
        f"<p>매니페스트: <code>{html_module.escape(manifest)}</code></p>"
    )
    table = (
        "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-size:9pt;'>"
        "<thead><tr>"
        "<th>sample_id</th><th>strict_MAE</th><th>band_MAE</th><th>band_hit</th>"
        "<th>pk_x_err</th><th>pk_y_err</th><th>pk_recall_v2</th><th>max_gap</th><th>cal_RT</th>"
        "<th>strict_gate</th><th>band_gate</th><th>both</th>"
        "</tr></thead><tbody>"
        f"{_table_rows(samples)}"
        "</tbody></table>"
    )
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<title>V2 파이프라인 실행 평가 보고서</title>
<style>
body {{ font-family: 'Malgun Gothic', sans-serif; margin: 24px; color: #111; }}
h1 {{ font-size: 18pt; border-bottom: 2px solid #1e40af; padding-bottom: 8px; }}
code {{ background: #f3f4f6; padding: 2px 6px; }}
</style>
</head>
<body>
<h1>XRD Digitizer — V2 통합 파이프라인 실행 결과</h1>
{summary}
{table}
<p style="margin-top:24px;font-size:9pt;color:#555;">Evaluator v2 (strict/band MAE, peak_recall_fixed) 및 §5.5 스케일 게이트.</p>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True, help="batch_run 결과 디렉터리")
    ap.add_argument("--gate_type", type=str, default="clean", choices=["clean", "styled", "real-like"])
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--pdf_out", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest_csv)
    out_root = pathlib.Path(args.output_dir)
    samples: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        if args.max_samples and len(samples) >= args.max_samples:
            break
        sid = str(row["sample_id"])
        gt_path = pathlib.Path(str(row["gt_path"]))
        result_path = out_root / f"{sid}_result.json"
        debug_path = out_root / f"debug_{sid}" / "debug.json"
        if not result_path.is_file() or not debug_path.is_file():
            continue
        try:
            ev = evaluate_one(result_path, debug_path, gt_path, args.gate_type)
            ev["sample_id"] = sid
            samples.append(ev)
        except Exception as exc:
            print(f"[SKIP] {sid}: {exc}", file=sys.stderr)

    html = build_html(samples, args.gate_type, args.manifest_csv)
    html_path = out_root / "v2_pipeline_eval_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML: {html_path}")

    pdf_out = pathlib.Path(args.pdf_out) if args.pdf_out else out_root / "v2_pipeline_eval_report.pdf"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri())
            page.pdf(
                path=str(pdf_out),
                format="A4",
                print_background=True,
                margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
            )
            browser.close()
        print(f"PDF: {pdf_out}")
    except Exception as exc:
        print(f"[WARN] PDF 생성 실패 ({exc}). HTML만 저장됨.", file=sys.stderr)
        json_path = out_root / "v2_pipeline_eval_samples.json"
        json_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"샘플 상세 JSON: {json_path}")


if __name__ == "__main__":
    main()
