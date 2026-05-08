#!/usr/bin/env python3
"""Run the candidate preserve generalization panel from a validated manifest.

This script intentionally runs only three fixed global-oracle variants:
baseline, filter_preserve, and final_preserve. It does not enable continuity
preserve, dedupe experiments, source balance, risk selector, DP cost probes, or
training.

The script must not infer paths from selected_samples.csv or domain conventions.
It uses only per-row input_image/manual_json/gt_json from the manifest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


VARIANTS: Dict[str, List[str]] = {
    "baseline": [],
    "filter_preserve": [
        "--candidate-filter-enable-evidence-aware-preserve",
        "--candidate-filter-preserve-bins",
        "6",
        "--candidate-filter-preserve-per-bin",
        "1",
        "--candidate-filter-preserve-max-upper-frac",
        "0.5",
    ],
    "final_preserve": [
        "--candidate-filter-enable-evidence-aware-preserve",
        "--candidate-filter-preserve-bins",
        "6",
        "--candidate-filter-preserve-per-bin",
        "1",
        "--candidate-filter-preserve-max-upper-frac",
        "0.5",
        "--candidate-final-enable-evidence-aware-preserve",
        "--candidate-final-evidence-preserve-slots",
        "2",
    ],
}


def _resolve_path(value: str) -> str:
    s = str(value).strip()
    if not s:
        return s
    s = s.replace("\\", "/")
    marker = "xrd_digitizer_v1/"
    if marker in s:
        s = s.split(marker, 1)[1]
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def _build_command(
    *,
    sample_id: str,
    domain: str,
    variant: str,
    image_path: str,
    manual_inputs_path: str,
    gt_path: str,
    out_root: Path,
    pipeline: str,
) -> tuple[List[str], Path, Path, str]:
    run_key = f"{domain}_{sample_id}"
    variant_root = out_root / run_key / variant
    result_json = variant_root / f"{sample_id}_result.json"
    debug_dir = variant_root / f"debug_{sample_id}_global"
    cmd = [
        sys.executable,
        str(ROOT / "runner" / "run_local.py"),
        "--image_path",
        image_path,
        "--manual_inputs_path",
        manual_inputs_path,
        "--output_json_path",
        str(result_json),
        "--debug_dir",
        str(debug_dir),
        "--pipeline",
        pipeline,
        "--oracle-rerank-gt",
        gt_path,
        "--oracle-rerank-sigma",
        "8",
        "--dump-candidates-json",
        "--debug-filter-removal-reasons",
        "--debug-final-selection-reasons",
    ]
    cmd.extend(VARIANTS[variant])
    return cmd, result_json, debug_dir, run_key


def _assert_exists(label: str, path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_one(
    *,
    sample_id: str,
    domain: str,
    variant: str,
    image_path: str,
    manual_inputs_path: str,
    gt_path: str,
    out_root: Path,
    pipeline: str,
    resume: bool,
    dry_run: bool,
) -> None:
    _assert_exists("input_image", image_path)
    _assert_exists("manual_json", manual_inputs_path)
    _assert_exists("gt_json", gt_path)
    cmd, result_json, debug_dir, run_key = _build_command(
        sample_id=sample_id,
        domain=domain,
        variant=variant,
        image_path=image_path,
        manual_inputs_path=manual_inputs_path,
        gt_path=gt_path,
        out_root=out_root,
        pipeline=pipeline,
    )
    if resume and result_json.is_file() and (debug_dir / "debug.json").is_file():
        print(f"[skip] {run_key}/{variant}")
        return
    print(f"[plan] {run_key}/{variant}")
    print(f"  input_image={image_path}")
    print(f"  manual_json={manual_inputs_path}")
    print(f"  gt_json={gt_path}")
    print(f"  output_dir={result_json.parent}")
    print("  command=" + " ".join(cmd))
    if dry_run:
        return
    result_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_key}/{variant}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _parse_variants(value: str) -> Sequence[str]:
    parts = [x.strip() for x in str(value).split(",") if x.strip()]
    if not parts:
        raise ValueError("--variants is empty")
    unknown = [p for p in parts if p not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; allowed={sorted(VARIANTS)}")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--pipeline", default="v1_2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--variants",
        default="baseline,filter_preserve,final_preserve",
        help="comma-separated variants (e.g. baseline,final_preserve)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="optional guardrail for max manifest rows",
    )
    args = ap.parse_args()

    panel = pd.read_csv(args.manifest)
    needed = {"sample_id", "domain", "input_image", "gt_json", "manual_json"}
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if "validation_status" in panel.columns:
        bad = panel[~panel["validation_status"].astype(str).eq("PASS")]
        if len(bad):
            raise ValueError("manifest contains non-PASS validation rows; refusing to run")
    if args.max_rows is not None and len(panel) > int(args.max_rows):
        raise ValueError(
            f"panel has {len(panel)} rows; refusing to run more than {int(args.max_rows)} samples"
        )

    selected_variants = _parse_variants(str(args.variants))
    out_root = args.out_root
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for _, row in panel.iterrows():
        sample_id = str(row["sample_id"])
        domain = str(row["domain"])
        image_path = _resolve_path(str(row["input_image"]))
        manual_inputs_path = _resolve_path(str(row["manual_json"]))
        gt_path = _resolve_path(str(row["gt_json"]))
        for label in selected_variants:
            total += 1
            _run_one(
                sample_id=sample_id,
                domain=domain,
                variant=label,
                image_path=image_path,
                manual_inputs_path=manual_inputs_path,
                gt_path=gt_path,
                out_root=out_root,
                pipeline=str(args.pipeline),
                resume=bool(args.resume),
                dry_run=bool(args.dry_run),
            )
    print(f"total_planned_runs={total}")


if __name__ == "__main__":
    main()
