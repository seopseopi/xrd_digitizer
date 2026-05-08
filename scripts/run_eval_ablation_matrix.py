#!/usr/bin/env python3
"""
플래그 조합별 phase-b (동일 max_samples) 연속 실행. 결과는 summarize_eval_reports.py 로 요약.

NOTE: 여기 생성되는 태그(eval_ablation_*) 산출물은 진단·ablation 참조(B1 성격)이며,
모델 도입 공식 baseline(B0 = dist/xrd_digitizer_model_v1_3)이 아니다.

예:
  python3 scripts/run_eval_ablation_matrix.py --repo-root . --max-samples 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# (tag, extra phase-b args)
ABLATION_MATRIX: list[tuple[str, list[str]]] = [
    ("eval_ablation_01_default", []),
    ("eval_ablation_02_ridge", ["--use-ridge-candidates"]),
    ("eval_ablation_03_m4", ["--axis-mask-margin", "4"]),
    ("eval_ablation_04_peak1", ["--peak-single-pass"]),
    ("eval_ablation_05_ridge_m4", ["--use-ridge-candidates", "--axis-mask-margin", "4"]),
    ("eval_ablation_06_ridge_peak1", ["--use-ridge-candidates", "--peak-single-pass"]),
    ("eval_ablation_07_m4_peak1", ["--axis-mask-margin", "4", "--peak-single-pass"]),
    (
        "eval_ablation_08_ridge_m4_peak1",
        ["--use-ridge-candidates", "--axis-mask-margin", "4", "--peak-single-pass"],
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--max-samples", type=int, default=10)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    plan = repo / "scripts" / "run_eval_plan.py"

    for tag, extra in ABLATION_MATRIX:
        cmd = [
            sys.executable,
            str(plan),
            "--repo-root",
            str(repo),
            "phase-b",
            "--max-samples",
            str(args.max_samples),
            "--tag",
            tag,
            *extra,
        ]
        print(f"\n>>> {' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True)

    tags = [t for t, _ in ABLATION_MATRIX]
    summ = [
        sys.executable,
        str(repo / "scripts" / "summarize_eval_reports.py"),
        "--repo-root",
        str(repo),
        "--tags",
        *tags,
        "--out",
        str(repo / "experiments" / "ablation_eval_summary.tsv"),
    ]
    print(f"\n>>> {' '.join(summ)}\n", flush=True)
    subprocess.run(summ, check=True)


if __name__ == "__main__":
    main()
