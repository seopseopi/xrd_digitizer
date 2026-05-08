#!/usr/bin/env python3
"""Run 1x vs ROI-2x deterministic diagnostic on a fixed small sample set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Explicitly fixed diagnostic samples (no full-51 run in this script).
FIXED_SAMPLE_KEYS: List[Tuple[str, str]] = [
    ("pattern_72296", "styled"),  # representative failure
    ("pattern_1915", "clean"),    # thin/weak candidate tendency
    ("pattern_1499", "clean"),    # clean medium difficulty
]


def _resolve_path(value: str) -> str:
    s = str(value).strip().replace("\\", "/")
    marker = "xrd_digitizer_v1/"
    if marker in s:
        s = s.split(marker, 1)[1]
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def _load_fixed_manifest(source_manifest: Path) -> pd.DataFrame:
    df = pd.read_csv(source_manifest)
    needed = {"sample_id", "domain", "input_image", "manual_json", "gt_json"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    out_rows = []
    for sid, dom in FIXED_SAMPLE_KEYS:
        sel = df[(df["sample_id"].astype(str) == sid) & (df["domain"].astype(str) == dom)]
        if len(sel) == 0:
            raise ValueError(f"missing fixed sample in manifest: sample_id={sid}, domain={dom}")
        out_rows.append(sel.iloc[0].to_dict())
    out = pd.DataFrame(out_rows)
    out = out[["sample_id", "domain", "input_image", "manual_json", "gt_json"]].copy()
    return out


def _build_cmd(
    *,
    image_path: str,
    manual_json: str,
    gt_json: str,
    result_json: Path,
    debug_dir: Path,
    roi_upscale_factor: int,
    roi_upscale_method: str,
    pipeline: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "runner" / "run_local.py"),
        "--image_path",
        image_path,
        "--manual_inputs_path",
        manual_json,
        "--output_json_path",
        str(result_json),
        "--debug_dir",
        str(debug_dir),
        "--pipeline",
        pipeline,
        "--dump-candidates-json",
        "--debug-filter-removal-reasons",
        "--debug-final-selection-reasons",
        # final_preserve ON as requested default diagnostic baseline
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
        "--roi-upscale-factor",
        str(int(roi_upscale_factor)),
        "--roi-upscale-method",
        str(roi_upscale_method),
    ]
    # Keep GT path in run plan/output manifest for explicitness.
    # For this diagnostic we do not enable oracle rerank; analysis compares against GT offline.
    _ = gt_json
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "_candidate_patch_full" / "full_manifest_validated.csv",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "_roi_upscale_diag",
    )
    ap.add_argument("--pipeline", type=str, default="v1_2")
    ap.add_argument("--roi-upscale-method", type=str, default="lanczos", choices=["lanczos", "bicubic"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    diag_manifest = _load_fixed_manifest(args.source_manifest)
    out_root = args.out_root
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    diag_manifest_path = out_root / "diag_manifest.csv"
    diag_manifest.to_csv(diag_manifest_path, index=False)
    print(f"[saved] {diag_manifest_path}")

    variants = [
        ("baseline_1x", 1),
        ("roi_upscale_2x", 2),
    ]

    total = 0
    for _, row in diag_manifest.iterrows():
        sid = str(row["sample_id"])
        dom = str(row["domain"])
        run_key = f"{dom}_{sid}"
        image_path = _resolve_path(str(row["input_image"]))
        manual_json = _resolve_path(str(row["manual_json"]))
        gt_json = _resolve_path(str(row["gt_json"]))
        for variant, factor in variants:
            total += 1
            variant_root = runs_root / run_key / variant
            result_json = variant_root / f"{sid}_result.json"
            debug_dir = variant_root / f"debug_{sid}_global"
            if args.resume and result_json.is_file() and (debug_dir / "debug.json").is_file():
                print(f"[skip] {run_key}/{variant}")
                continue
            cmd = _build_cmd(
                image_path=image_path,
                manual_json=manual_json,
                gt_json=gt_json,
                result_json=result_json,
                debug_dir=debug_dir,
                roi_upscale_factor=factor,
                roi_upscale_method=str(args.roi_upscale_method),
                pipeline=str(args.pipeline),
            )
            print(f"[plan] {run_key}/{variant} factor={factor}")
            print(f"  input_image={image_path}")
            print(f"  manual_json={manual_json}")
            print(f"  gt_json={gt_json}")
            print(f"  out={variant_root}")
            if args.dry_run:
                print("  command=" + " ".join(cmd))
                continue
            variant_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, cwd=str(ROOT), check=True)

    print(f"total_planned_runs={total}")
    print("note=variants are fixed to baseline_1x and roi_upscale_2x only")


if __name__ == "__main__":
    main()
