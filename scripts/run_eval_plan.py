#!/usr/bin/env python3
"""
Phase A/B 평가 실행기 (v1.1 기준).

  Phase A: clean / styled / real 이미지 각 1장 스모크 (GT에서 MI 자동 생성)
  Phase B: 매니페스트 일부 배치 실행 후 clean / styled / real_like 게이트로 eval 리포트 3종

레거시 매니페스트의 Windows 경로(c:\\xrd_digitizer_v1\\...)는 저장소 루트로 치환합니다.

예 (--repo-root 는 서브커맨드 앞에 둔다):
  python3 scripts/run_eval_plan.py --repo-root . phase-a
  python3 scripts/run_eval_plan.py --repo-root . phase-b --max-samples 50 --tag eval_20260502
  python3 scripts/run_eval_plan.py --repo-root . phase-b --max-samples 10 --tag eval_ridge \\
    --use-ridge-candidates
  python3 scripts/run_eval_plan.py --repo-root . phase-b --max-samples 10 --tag eval_margin4 \\
    --axis-mask-margin 4
  python3 scripts/run_eval_plan.py --repo-root . phase-b --max-samples 10 --tag eval_peak1pass \\
    --peak-single-pass
  python3 scripts/run_eval_plan.py --repo-root . phase-b --eval-only \\
    --output-dir outputs/runs/eval_20260502/clean --manifest data/manifests/clean_manifest.csv \\
    --gate-type clean --gate-level development \\
    --report-json outputs/runs/eval_20260502/report_clean.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# 매니페스트에 남아 있을 수 있는 Windows 절대 경로 접두부(대소문자 무시)
_LEGACY_PREFIX = "c:/xrd_digitizer_v1"


def _norm_one(val: str, root: Path) -> str:
    if not isinstance(val, str) or not val.strip():
        return val
    s = val.strip().replace("\\", "/")
    low = s.lower()
    leg = _LEGACY_PREFIX.lower()
    if low.startswith(leg):
        rel = s[len(leg) :].lstrip("/")
        return str((root / rel).resolve())
    return val


def normalize_manifest_df(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = df.copy()
    pathish = {"gt_path", "source_json_path"}
    for c in out.columns:
        if c.endswith("_path") or c in pathish:
            out[c] = out[c].map(lambda v, r=root: _norm_one(str(v), r))
    return out


def _write_resolved(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _phase_a(repo: Path, sample_id: str, out: Path) -> None:
    sys.path.insert(0, str(repo))
    from runner.run_local import run_single

    gt = repo / "data" / "gt" / f"{sample_id}_gt.json"
    if not gt.is_file():
        raise FileNotFoundError(gt)
    gtd = json.loads(gt.read_text(encoding="utf-8"))
    am = gtd["axis_metadata"]
    pb = gtd["plot_box"]
    x0, y0, x1, y1 = pb
    pcp = gtd.get("pixel_curve_path", [])
    if pcp and len(pcp) > 2:
        mid = pcp[len(pcp) // 2]
        csp = [int(mid[0]), int(mid[1])]
    else:
        csp = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
    mi = {
        "plot_box": pb,
        "x_axis_points": [[x0, y1], [x1, y1]],
        "x_axis_values": [am["x_min"], am["x_max"]],
        "y_axis_points": [[x0, y1], [x0, y0]],
        "y_axis_values": [am["y_min"], am["y_max"]],
        "color_sample_point": csp,
        "legend_ignore_boxes": [],
        "perspective_corners": None,
        "color_resample_points": [],
    }
    out.mkdir(parents=True, exist_ok=True)

    runs = [
        ("clean", repo / "data" / "rendered_clean" / f"{sample_id}_clean_v1.png"),
        ("styled", repo / "data" / "rendered_styled" / f"{sample_id}_styled_v5.png"),
        ("real", repo / "data" / "rendered_real_like" / f"{sample_id}_real_v4.png"),
    ]
    for tag, img in runs:
        if not img.is_file():
            print(f"[skip] missing {img}")
            continue
        mi_path = out / f"_mi_smoke_{tag}.json"
        mi_path.write_text(json.dumps(mi, ensure_ascii=False, indent=2), encoding="utf-8")
        rj = out / f"smoke_{tag}_result.json"
        dd = out / f"debug_smoke_{tag}"
        run_single(str(img), str(mi_path), str(rj), str(dd), pipeline="v1_1")
        dbg = json.loads((dd / "debug.json").read_text(encoding="utf-8"))
        print(f"[OK] {tag} pipeline_version={dbg.get('pipeline_version')}")


def _run_batch(
    repo: Path,
    manifest: Path,
    out_dir: Path,
    max_samples: int,
    pipeline: str,
    *,
    axis_mask_margin: int = 15,
    use_ridge_candidates: bool = False,
    peak_single_pass: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(repo / "runner" / "batch_run.py"),
        "--manifest_csv",
        str(manifest),
        "--output_dir",
        str(out_dir),
        "--max_samples",
        str(max_samples),
        "--pipeline",
        pipeline,
    ]
    if pipeline == "v2_experimental":
        cmd.append("--allow_experimental_v2")
    if axis_mask_margin != 15:
        cmd.extend(["--axis-mask-margin", str(axis_mask_margin)])
    if use_ridge_candidates:
        cmd.append("--use-ridge-candidates")
    if peak_single_pass:
        cmd.append("--peak-single-pass")
    subprocess.run(cmd, check=True)


def _run_eval(
    repo: Path,
    manifest: Path,
    out_dir: Path,
    gate_type: str,
    report_json: Path,
    max_samples: int,
    gate_level: str = "development",
) -> None:
    cmd = [
        sys.executable,
        str(repo / "eval" / "report.py"),
        "--manifest_csv",
        str(manifest),
        "--output_dir",
        str(out_dir),
        "--gate_type",
        gate_type,
        "--gate_level",
        gate_level,
        "--report_json",
        str(report_json),
    ]
    if max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])
    subprocess.run(cmd, check=True)


def cmd_phase_a(ap: argparse.Namespace) -> None:
    repo = Path(ap.repo_root).resolve()
    out = Path(ap.out_dir).resolve() if ap.out_dir else repo / "outputs" / "runs" / "phase_a_smoke"
    _phase_a(repo, ap.sample_id, out)
    print(f"Phase A done -> {out}")


def cmd_phase_b(ap: argparse.Namespace) -> None:
    repo = Path(ap.repo_root).resolve()
    tag = ap.tag or "eval_plan"
    base = Path(ap.base_dir).resolve() if ap.base_dir else repo / "outputs" / "runs" / tag
    max_samples = ap.max_samples
    pipeline = ap.pipeline

    if ap.eval_only:
        if not ap.output_dir or not ap.manifest or not ap.report_json or not ap.gate_type:
            raise SystemExit("--eval-only requires --output-dir --manifest --gate-type --report-json")
        man = normalize_manifest_df(pd.read_csv(ap.manifest), repo).reset_index(drop=True)
        tmp = Path(ap.output_dir).resolve() / "_manifest_resolved.csv"
        _write_resolved(man, tmp)
        _run_eval(
            repo,
            tmp,
            Path(ap.output_dir).resolve(),
            ap.gate_type,
            Path(ap.report_json).resolve(),
            max_samples,
            gate_level=ap.gate_level,
        )
        print("Eval-only done.")
        return

    jobs = [
        (
            "clean",
            repo / "data" / "manifests" / "clean_manifest.csv",
            base / "clean",
            "clean",
            base / "report_clean.json",
        ),
        (
            "styled",
            repo / "data" / "manifests" / "styled_manifest.csv",
            base / "styled",
            "styled",
            base / "report_styled.json",
        ),
        (
            "real",
            repo / "data" / "manifests" / "real_manifest.csv",
            base / "real",
            "real_like",
            base / "report_real_like.json",
        ),
    ]

    for name, man_path, out_dir, gate_type, report_path in jobs:
        if not man_path.is_file():
            print(f"[skip] missing manifest {man_path}")
            continue
        df = pd.read_csv(man_path)
        df = normalize_manifest_df(df, repo)
        if name == "styled" and "variant_id" in df.columns:
            df = df[df["variant_id"] == "styled_v5"].reset_index(drop=True)
        if name == "real" and "variant_id" in df.columns:
            df = df[df["variant_id"] == "real_v4"].reset_index(drop=True)
        df = df.reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = out_dir / "_manifest_resolved.csv"
        _write_resolved(df, resolved)

        if not ap.skip_batch:
            _run_batch(
                repo,
                resolved,
                out_dir,
                max_samples,
                pipeline,
                axis_mask_margin=ap.axis_mask_margin,
                use_ridge_candidates=ap.use_ridge_candidates,
                peak_single_pass=ap.peak_single_pass,
            )
        _run_eval(
            repo, resolved, out_dir, gate_type, report_path, max_samples,
            gate_level=ap.gate_level,
        )
        print(f"[done] {name}: batch_out={out_dir} report={report_path}")

    print(f"Phase B complete under {base}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase A/B eval plan runner")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("phase-a", help="스모크 3장 (clean/styled/real)")
    pa.add_argument("--sample-id", default="pattern_74680")
    pa.add_argument("--out-dir", type=Path, default=None)
    pa.set_defaults(func=cmd_phase_a)

    pb = sub.add_parser("phase-b", help="소표본 배치 + eval 3종 (또는 --eval-only)")
    pb.add_argument("--tag", type=str, default=None, help="outputs/runs/<tag> 하위에 clean|styled|real")
    pb.add_argument("--base-dir", type=Path, default=None, help="tag 대신 직접 베이스 디렉터리")
    pb.add_argument("--max-samples", type=int, default=50)
    pb.add_argument("--pipeline", type=str, default="v1_1", choices=["v1_1", "v1", "v1.1", "v2_experimental"])
    pb.add_argument(
        "--axis-mask-margin",
        type=int,
        default=15,
        metavar="PX",
        help="v1_1 배치: mask_axis_lines margin (기본 15, 로드맵 1c)",
    )
    pb.add_argument(
        "--use-ridge-candidates",
        action="store_true",
        help="v1_1 배치: 세로 능선 후보 가산 켜기 (로드맵 2b)",
    )
    pb.add_argument(
        "--peak-single-pass",
        action="store_true",
        help="v1_1 배치: 피크 prominence 2패스·NMS 끄기 (로드맵 3a 기본은 2패스)",
    )
    pb.add_argument("--skip-batch", action="store_true", help="이미 result/debug 있으면 eval만")
    pb.add_argument("--eval-only", action="store_true")
    pb.add_argument("--output-dir", type=Path, default=None)
    pb.add_argument("--manifest", type=Path, default=None)
    pb.add_argument("--gate-type", choices=["clean", "styled", "real_like"], default=None)
    pb.add_argument(
        "--gate-level",
        choices=["mvp", "development", "strict"],
        default="development",
        help="eval/report.py 게이트 임계 세트 (기본 development)",
    )
    pb.add_argument("--report-json", type=Path, default=None)
    pb.set_defaults(func=cmd_phase_b)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
