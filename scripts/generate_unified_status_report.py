from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.gates_exp import check_gate_v2_strict
from eval.gt_compat import normalize_gt_for_eval
from eval.metrics import compute_all_metrics
from eval.metrics_exp import compute_metrics_v2, merge_main_with_v2


def eval_dir(
    manifest: pathlib.Path,
    out_dir: pathlib.Path,
    expected_pipeline: str | tuple[str, ...],
) -> dict:
    expected = (
        (expected_pipeline,)
        if isinstance(expected_pipeline, str)
        else tuple(expected_pipeline)
    )
    df = pd.read_csv(manifest)
    maes = []
    strict_ok = 0
    n = 0
    misrouted = 0
    fail_hist = Counter()
    rec_activated = 0
    rec_improved = 0
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        rp = out_dir / f"{sid}_result.json"
        dp = out_dir / f"debug_{sid}" / "debug.json"
        if not rp.exists() or not dp.exists():
            continue
        res = json.loads(rp.read_text(encoding="utf-8"))
        dbg = json.loads(dp.read_text(encoding="utf-8"))
        if str(dbg.get("pipeline_version", "")) not in expected:
            misrouted += 1
            continue
        gt = normalize_gt_for_eval(json.loads(pathlib.Path(str(row["gt_path"])).read_text(encoding="utf-8")))
        m = compute_all_metrics(res, dbg, gt)
        v2 = compute_metrics_v2(res, dbg, gt)
        mm = merge_main_with_v2(m["main"], v2)
        g = check_gate_v2_strict(mm, v2, "clean")
        if g["passed"]:
            strict_ok += 1
        else:
            for k, info in g["details"].items():
                if not info.get("passed", True):
                    fail_hist[k] += 1
        rec = dbg.get("recovery", {})
        if rec.get("recovery_triggered"):
            rec_activated += 1
            if int(rec.get("accepted_count", 0)) > 0:
                rec_improved += 1
        maes.append(float(v2["strict_curve_y_mae_px"]))
        n += 1
    if maes:
        mean_mae = float(np.mean(maes))
        min_mae = float(np.min(maes))
        p_lt15 = 100.0 * sum(1 for x in maes if x < 15.0) / len(maes)
    else:
        mean_mae = min_mae = 999.0
        p_lt15 = 0.0
    return {
        "n": n,
        "misrouted": misrouted,
        "strict_ok": strict_ok,
        "strict_rate": 100.0 * strict_ok / max(n, 1),
        "mean_mae": mean_mae,
        "min_mae": min_mae,
        "pct_lt15": p_lt15,
        "rec_activated": rec_activated,
        "rec_improved": rec_improved,
        "rec_rate": 100.0 * rec_improved / max(rec_activated, 1),
        "fail_hist": fail_hist,
    }


def main() -> None:
    manifest = ROOT / "data/manifests/clean_manifest.csv"
    arch = ROOT / "experiments" / "archive"
    legacy_runs = arch / "outputs_legacy_runs"
    stable = eval_dir(
        manifest,
        legacy_runs / "stable_baseline_clean",
        ("calibrate_v1", "calibrate_v1_1"),
    )
    v2_now = eval_dir(manifest, legacy_runs / "v2_tuned_clean", "v2_integrated")
    v2_struct_path = legacy_runs / "v2_structfix_clean30_r2"
    v2_struct_r2 = (
        eval_dir(manifest, v2_struct_path, "v2_integrated")
        if v2_struct_path.is_dir()
        else {
            "n": 0,
            "misrouted": 0,
            "strict_ok": 0,
            "strict_rate": 0.0,
            "mean_mae": 999.0,
            "min_mae": 999.0,
            "pct_lt15": 0.0,
            "rec_activated": 0,
            "rec_improved": 0,
            "rec_rate": 0.0,
            "fail_hist": Counter(),
        }
    )

    fail_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in v2_now["fail_hist"].most_common()
    ) or "<tr><td colspan='2'>(none)</td></tr>"

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
<title>통합 현황 보고서</title>
<style>
body{{font-family:'Malgun Gothic',sans-serif;margin:24px;line-height:1.55;color:#111}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:10pt}}
th,td{{border:1px solid #ccc;padding:6px 8px}} th{{background:#eff6ff}}
code{{background:#f3f4f6;padding:2px 6px}}
</style></head><body>
<h1>통합 현황 보고서 (단일본)</h1>
<p>역할별 분리 없이 현재 상태, 원인, 조치, 결과를 하나로 통합했습니다.</p>

<h2>1) 현재 기준선 및 운영 규칙</h2>
<ul>
<li>운영 기준선(아카이브): <code>experiments/archive/outputs_legacy_runs/stable_baseline_clean</code> — pipeline_version <code>calibrate_v1</code> 또는 <code>calibrate_v1_1</code></li>
<li>신규 실행·품질 기준선: <code>--pipeline v1_1</code> (v1.1) → debug 의 <code>calibrate_v1_1</code> — v2 는 기본 최고 성능이 아님</li>
<li>운영 경로 잠금: <code>pipeline=v2</code> 금지, 실험은 <code>v2_experimental</code> + 명시적 플래그만</li>
<li>보고서/평가는 pipeline version mismatch를 <code>misrouted</code>로 별도 집계</li>
</ul>

<h2>2) 핵심 지표 요약 (clean manifest)</h2>
<table>
<tr><th>구분</th><th>valid n</th><th>misrouted</th><th>strict pass</th><th>strict rate</th><th>mean MAE</th><th>min MAE</th><th>MAE&lt;15</th></tr>
<tr><td>Stable baseline (v1 / v1.1 JSON)</td><td>{stable['n']}</td><td>{stable['misrouted']}</td><td>{stable['strict_ok']}</td><td>{stable['strict_rate']:.1f}%</td><td>{stable['mean_mae']:.3f}</td><td>{stable['min_mae']:.3f}</td><td>{stable['pct_lt15']:.2f}%</td></tr>
<tr><td>V2 current tuned</td><td>{v2_now['n']}</td><td>{v2_now['misrouted']}</td><td>{v2_now['strict_ok']}</td><td>{v2_now['strict_rate']:.1f}%</td><td>{v2_now['mean_mae']:.3f}</td><td>{v2_now['min_mae']:.3f}</td><td>{v2_now['pct_lt15']:.2f}%</td></tr>
<tr><td>V2 structural fix r2 (clean30 subset)</td><td>{v2_struct_r2['n']}</td><td>{v2_struct_r2['misrouted']}</td><td>{v2_struct_r2['strict_ok']}</td><td>{v2_struct_r2['strict_rate']:.1f}%</td><td>{v2_struct_r2['mean_mae']:.3f}</td><td>{v2_struct_r2['min_mae']:.3f}</td><td>{v2_struct_r2['pct_lt15']:.2f}%</td></tr>
</table>

<h2>3) 실패 원인(현재 V2 기준)</h2>
<table><tr><th>지표</th><th>실패 수</th></tr>{fail_rows}</table>

<h2>4) 실험 결과 판정</h2>
<ul>
<li>구조 수정 실험은 <code>min MAE</code> 일부 개선이 있으나, <code>mean MAE</code> 악화로 실패.</li>
<li>따라서 현재는 구조 실험 확장보다 기준선 재점검/실험 격리를 우선 유지.</li>
</ul>

<h2>5) 다음 실행 원칙</h2>
<ul>
<li>운영 배치는 <code>pipeline=v1_1</code> (또는 호환 <code>v1</code>) 사용</li>
<li>실험 배치는 <code>pipeline=v2_experimental --allow_experimental_v2</code>로만 실행</li>
<li>새 실험은 baseline 대비 mean MAE 악화 시 즉시 폐기</li>
</ul>
</body></html>"""
    out = ROOT / "outputs" / "unified_project_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

