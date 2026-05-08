#!/usr/bin/env python3
"""
동일 샘플에 대해 v1_1(운영) / v2_experimental / 아카이브 번들 을 실행하고
GT 기준 eval(gate + 주요 main 지표)를 한 표로 출력합니다.

사용 예 (저장소 루트에서):
  python3 scripts/compare_v1_exp_bundle.py \\
    --sample_ids pattern_1499 pattern_72296 pattern_11832 \\
    --variant real_v4

선택:
  --tune_json path/to.json   v2_experimental 에만 전달
  --mi_dir outputs           _mi_{sample_id}.json 위치 (기본 outputs)
  --out_root outputs/compare_triple
  --skip_bundle              번들 실행 생략
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _print_row(cols: list[str], widths: list[int]) -> None:
    parts = [c.ljust(w)[:w] for c, w in zip(cols, widths)]
    print(" | ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description="v1_1 vs v2_experimental vs archive bundle + eval")
    ap.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="xrd_digitizer_v1 루트",
    )
    ap.add_argument("--sample_ids", nargs="+", required=True)
    ap.add_argument("--variant", default="real_v4", help="파일 접미사, 예: real_v4")
    ap.add_argument("--mi_dir", default="outputs", help="_mi_{id}.json 이 있는 상대/절대 경로")
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--gate", default="real_like", choices=["clean", "styled", "real_like"])
    ap.add_argument("--tune_json", default=None)
    ap.add_argument("--skip_bundle", action="store_true")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    mi_dir = Path(args.mi_dir)
    if not mi_dir.is_absolute():
        mi_dir = repo / mi_dir

    out_root = (args.out_root or (repo / "outputs" / "compare_triple")).resolve()
    bundle_root = repo / "experiments" / "archive" / "xrd_calibrate_v1_bundle"

    sys.path.insert(0, str(repo))
    from eval.report import evaluate_single  # noqa: E402
    from runner.run_local import run_single  # noqa: E402

    widths = [18, 8, 5, 10, 10, 10, 10]

    print()
    _print_row(["sample_id", "engine", "pass", "y_mae_px", "peak_rec", "maj_x", "conf"], widths)
    _print_row(["-" * w for w in widths], widths)

    for sid in args.sample_ids:
        img = repo / "data" / "rendered_real_like" / f"{sid}_{args.variant}.png"
        mi = mi_dir / f"_mi_{sid}.json"
        gt = repo / "data" / "gt" / f"{sid}_gt.json"

        if not img.exists():
            print(f"[skip] missing image: {img}", file=sys.stderr)
            continue
        if not mi.exists():
            print(f"[skip] missing manual inputs: {mi}", file=sys.stderr)
            continue
        if not gt.exists():
            print(f"[skip] missing gt: {gt}", file=sys.stderr)
            continue

        runs: list[tuple[str, Path, Path]] = []

        # v1
        d1 = out_root / "v1" / f"debug_{sid}"
        r1 = out_root / "v1" / f"{sid}_result.json"
        d1.mkdir(parents=True, exist_ok=True)
        run_single(
            str(img),
            str(mi),
            str(r1),
            str(d1),
            pipeline="v1_1",
        )
        runs.append(("v1", r1, d1 / "debug.json"))

        # v2
        d2 = out_root / "v2" / f"debug_{sid}"
        r2 = out_root / "v2" / f"{sid}_result.json"
        d2.mkdir(parents=True, exist_ok=True)
        run_single(
            str(img),
            str(mi),
            str(r2),
            str(d2),
            pipeline="v2_experimental",
            tune_json=args.tune_json,
            allow_experimental_v2=True,
        )
        runs.append(("v2", r2, d2 / "debug.json"))

        if not args.skip_bundle:
            if not bundle_root.is_dir():
                print(f"[skip bundle] missing: {bundle_root}", file=sys.stderr)
            else:
                db = out_root / "bundle" / f"debug_{sid}"
                rb = out_root / "bundle" / f"{sid}_result.json"
                db.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(bundle_root / "runner" / "run_local.py"),
                    "--image_path",
                    str(img),
                    "--manual_inputs_path",
                    str(mi),
                    "--output_json_path",
                    str(rb),
                    "--debug_dir",
                    str(db),
                ]
                subprocess.run(cmd, cwd=str(bundle_root), check=True)
                runs.append(("bundle", rb, db / "debug.json"))

        for name, rp, dp in runs:
            if not dp.exists():
                print(f"[eval skip] {sid} {name}: no {dp}", file=sys.stderr)
                continue
            ev = evaluate_single(str(rp), str(dp), str(gt), args.gate)
            main = ev["metrics"]["main"]
            passed = ev["gate"]["passed"]
            conf = float(json.loads(Path(rp).read_text(encoding="utf-8")).get("confidence", 0.0))
            _print_row(
                [
                    sid,
                    name,
                    "Y" if passed else "N",
                    f"{main.get('curve_y_mae_px', 0):.2f}",
                    f"{main.get('peak_recall', 0):.3f}",
                    f"{main.get('major_peak_x_error', 0):.2f}",
                    f"{conf:.3f}",
                ],
                widths,
            )

    print()
    print(f"gate={args.gate}  outputs -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
