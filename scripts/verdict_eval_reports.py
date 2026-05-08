#!/usr/bin/env python3
"""
outputs/runs/<tag>/report_*.json 을 읽어 게이트 대비 판정 요약을 낸다.

- pass_rate: 리포트에 있는 샘플별 AND 게이트 통과 비율(공식 eval).
- mean_ok: 배치 평균 지표만 임계값에 대입했을 때 전부 통과하면 OK (참고용, 엄격 게이트와 다름).

NOTE: DEFAULT_TAGS 의 eval_ablation_* 결과는 ablation·진단 참조용이다.
모델 통합 여부는 core/model_integration_baseline 의 B0 스냅샷 대비 비교(ml.model_integration_compare)로 판단한다.

예:
  python3 scripts/verdict_eval_reports.py --repo-root . --out experiments/verdict_ablation.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# eval 패키지 (repo 루트에서 실행)
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.gates import RECALL_IS_GTE, flatten_gate_thresholds  # noqa: E402

GATE_FILES = [
    ("clean", "report_clean.json"),
    ("styled", "report_styled.json"),
    ("real_like", "report_real_like.json"),
]

DEFAULT_TAGS = [
    "eval_ablation_01_default",
    "eval_ablation_02_ridge",
    "eval_ablation_03_m4",
    "eval_ablation_04_peak1",
    "eval_ablation_05_ridge_m4",
    "eval_ablation_06_ridge_peak1",
    "eval_ablation_07_m4_peak1",
    "eval_ablation_08_ridge_m4_peak1",
]


def _mean_gate_fails(stats: dict, gate: str, gate_level: str = "strict") -> List[str]:
    th = flatten_gate_thresholds(gate, gate_level)
    fails: List[str] = []
    for metric, t in th.items():
        block = stats.get(metric) or {}
        if "mean" not in block:
            continue
        v = float(block["mean"])
        if metric in RECALL_IS_GTE:
            ok = v >= t
        else:
            ok = v <= t
        if not ok:
            cmp_ = ">=" if metric in RECALL_IS_GTE else "<="
            fails.append(f"{metric} mean={v:.4g} need {cmp_} {t}")
    return fails


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    runs = root / "outputs" / "runs"

    lines: List[str] = []
    lines.append("eval verdict — batch mean vs gate + official pass_rate")
    lines.append(f"repo: {root}")
    lines.append("")

    wtag = max(len(t) for t in args.tags) + 1

    for gate, fname in GATE_FILES:
        lines.append(f"=== gate: {gate} ===")
        lines.append(
            f"{'tag':<{wtag}} pass_rate  mean_ok  (mean fails if any)"
        )
        for tag in args.tags:
            p = runs / tag / fname
            if not p.is_file():
                lines.append(f"{tag:<{wtag}} MISSING {p.name}")
                continue
            rep = _load_report(p)
            agg = rep.get("aggregate") or {}
            pr = float(agg.get("pass_rate", -1.0))
            stats = agg.get("main_metrics_stats") or {}
            gl = (rep.get("metadata") or {}).get("gate_level") or rep.get("gate_level") or "strict"
            fails = _mean_gate_fails(stats, gate, gate_level=gl)
            mean_ok = "OK" if not fails else "FAIL"
            fail_s = "; ".join(fails[:3])
            if len(fails) > 3:
                fail_s += f" …(+{len(fails) - 3})"
            lines.append(
                f"{tag:<{wtag}} {pr:>7.1%}  {mean_ok:5s}  {fail_s}"
            )
        lines.append("")

    lines.append("해석")
    lines.append("- pass_rate: 샘플마다 모든 지표가 동시에 통과한 비율(현재 소표본에선 대부분 0%).")
    lines.append("- mean_ok: 배치 평균만 본 참고치. 엄격 게이트(curve MAE<=5 등)는 평균으로도 달성 어려움.")
    lines.append("- 조합별 상대 비교는 experiments/ablation_eval_summary.tsv 와 함께 본다.")
    text = "\n".join(lines) + "\n"

    print(text, end="")
    if args.out:
        op = args.out.resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(text, encoding="utf-8")
        print(f"[OK] wrote {op}", file=sys.stderr)


if __name__ == "__main__":
    main()
