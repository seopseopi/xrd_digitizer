from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main() -> None:
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Experimental Full Run Master Report</title>
<style>
body{{font-family:'Malgun Gothic',sans-serif;margin:28px;color:#111}}
table{{border-collapse:collapse;font-size:10pt}}th,td{{border:1px solid #ccc;padding:6px 10px}}
h1{{font-size:18pt}}
</style></head><body>
<h1>XRD 실험 파이프라인 전량 실행 마스터 보고서</h1>
<table>
<tr><th>도메인</th><th>평가 샘플</th><th>strict 통과</th><th>band 통과</th><th>both 통과</th><th>세부 PDF</th></tr>
<tr><td>clean</td><td>100</td><td>0</td><td>0</td><td>0</td><td>experiments/archive/outputs_legacy_runs/v2_full_clean/v2_full_clean_report.pdf</td></tr>
<tr><td>styled</td><td>150</td><td>0</td><td>0</td><td>0</td><td>experiments/archive/outputs_legacy_runs/v2_full_styled/v2_full_styled_report.pdf</td></tr>
<tr><td>real-like</td><td>100</td><td>0</td><td>0</td><td>0</td><td>experiments/archive/outputs_legacy_runs/v2_full_real/v2_full_real_report.pdf</td></tr>
</table>
<p style="margin-top:16px">이번 실행은 계획서 전 항목(특히 §8 rescue/§9 peak gate)을 코드 경로에 반영한 뒤 전량 배치로 재검증한 결과다.</p>
</body></html>"""
    html_path = OUT / "exp_master_summary_report.html"
    pdf_path = OUT / "exp_master_summary_report.pdf"
    html_path.write_text(html, encoding="utf-8")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.pdf(path=str(pdf_path), format="A4", print_background=True)
        browser.close()
    print(f"HTML: {html_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    sys.exit(main())
