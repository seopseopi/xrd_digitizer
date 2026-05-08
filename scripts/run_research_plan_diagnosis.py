#!/usr/bin/env python3
"""
연구 계획서 산출물 일괄 생성: 매니페스트 경로 정규화, 무결성 JSON, split 누출,
eval/report 다중 게이트 실행, 단계별 진단 MD 초안.

NOTE: `--batch-output-base` 로 묶는 배치 산출물은 진단 참조(B1)일 수 있다.
모델 도입 공식 baseline은 dist/xrd_digitizer_model_v1_3 (B0) 이며, 통합 판정은 ml.model_integration_compare 를 사용한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

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


def _integrity_report(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    img_col = None
    for c in ("image_path", "styled_image_path", "real_image_path", "clean_image_path"):
        if c in df.columns:
            img_col = c
            break

    if img_col:
        miss = df[~df[img_col].apply(lambda p: Path(str(p)).exists())]
        if len(miss):
            errors.append(f"missing_images_{img_col}: {len(miss)}")
    else:
        warnings.append("no_image_column")

    if "gt_path" in df.columns:
        miss = df[~df["gt_path"].apply(lambda p: Path(str(p)).exists())]
        if len(miss):
            errors.append(f"missing_gt: {len(miss)}")
    else:
        errors.append("missing_gt_path_column")

    if "source_json_path" in df.columns:
        miss = df[~df["source_json_path"].apply(lambda p: Path(str(p)).exists())]
        if len(miss):
            errors.append(f"missing_source_json: {len(miss)}")

    if "sample_id" in df.columns:
        dup = int(df["sample_id"].duplicated().sum())
        if dup:
            errors.append(f"duplicate_sample_id: {dup}")

    # plot_box vs image bounds (best-effort)
    bad_box = 0
    if img_col and "gt_path" in df.columns:
        import json as _json

        for _, row in df.head(200).iterrows():  # cap scan
            gp = Path(str(row["gt_path"]))
            if not gp.is_file():
                continue
            try:
                gt = _json.loads(gp.read_text(encoding="utf-8"))
                pb = gt.get("plot_box")
                ip = Path(str(row[img_col]))
                if pb and len(pb) >= 4 and ip.is_file():
                    # skip PIL if unavailable — optional
                    try:
                        from PIL import Image

                        w, h = Image.open(ip).size
                        x0, y0, x1, y1 = pb[:4]
                        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                            bad_box += 1
                    except Exception:
                        pass
            except Exception:
                pass
        if bad_box:
            warnings.append(f"plot_box_outside_image_estimated: {bad_box} (sampled)")

    return {
        "manifest_label": label,
        "rows": len(df),
        "errors": errors,
        "warnings": warnings,
        "ok": len(errors) == 0,
    }


def _split_leakage(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"manifest_label": label, "issues": []}
    if "split" not in df.columns:
        out["issues"].append("no_split_column")
        return out

    # 같은 source_json이 여러 split에 등장?
    if "source_json_path" in df.columns:
        src_to_splits: Dict[str, set] = defaultdict(set)
        for _, row in df.iterrows():
            src_to_splits[str(row["source_json_path"])].add(str(row["split"]))
        leaked_src = [s for s, sp in src_to_splits.items() if len(sp) > 1]
        if leaked_src:
            out["issues"].append({"type": "source_json_multi_split", "count": len(leaked_src)})

    if "family_id" in df.columns:
        fam_to_splits: Dict[str, set] = defaultdict(set)
        for _, row in df.iterrows():
            fid = row["family_id"]
            if pd.isna(fid) or str(fid).strip() == "":
                continue
            fam_to_splits[str(fid)].add(str(row["split"]))
        leaked_fam = [f for f, sp in fam_to_splits.items() if len(sp) > 1]
        if leaked_fam:
            out["issues"].append({"type": "family_id_multi_split", "count": len(leaked_fam)})
    else:
        out["notes"] = "family_id_column_absent_eval_manifest"

    split_counts = df["split"].value_counts().to_dict()
    out["split_counts"] = {str(k): int(v) for k, v in split_counts.items()}
    return out


def _run_eval(
    repo: Path,
    manifest: Path,
    output_dir: Path,
    gate_type: str,
    gate_level: str,
    report_json: Path,
    max_samples: int,
) -> None:
    cmd = [
        sys.executable,
        str(repo / "eval" / "report.py"),
        "--manifest_csv",
        str(manifest),
        "--output_dir",
        str(output_dir),
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


def _stage_md_from_reports(reports: List[Tuple[str, Path]], out_dir: Path) -> None:
    """reports: (name, path_to_json)"""
    counts: Dict[str, int] = defaultdict(int)
    samples_by_label: Dict[str, List[str]] = defaultdict(list)
    for name, path in reports:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for s in data.get("samples", []):
            sid = s.get("sample_id", "?")
            for fl in s.get("failure_labels", []):
                counts[fl] += 1
                if len(samples_by_label[fl]) < 15:
                    samples_by_label[fl].append(f"{sid}({name})")

    lines = [
        "# Stage-aligned failure hints (from eval failure taxonomy)",
        "",
        "라벨은 `eval/gates.label_failures` 규칙 기반이며, 파이프라인 단계와 1:1은 아님.",
        "",
        "## Taxonomy counts (aggregated over loaded reports)",
        "",
    ]
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## 예시 sample_id (라벨당 최대 15개)")
    lines.append("")
    for k in sorted(samples_by_label.keys()):
        lines.append(f"### {k}")
        lines.append(", ".join(samples_by_label[k]))
        lines.append("")

    (out_dir / "failure_taxonomy_from_eval.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Stub stage files pointing to interpretation
    stub = (
        "본 파일은 배치 eval 리포트의 failure taxonomy·debug 지표를 근거로 단계 가설을 적는다.\n"
        "실제 ROI/마스크/후보/DP 덤프는 각 샘플의 `debug_<sample_id>/debug.json`을 연다.\n"
    )
    for fname in (
        "preprocess_failure_report.md",
        "candidate_failure_report.md",
        "dp_failure_report.md",
        "recovery_failure_report.md",
        "postprocess_failure_report.md",
        "numeric_export_failure_report.md",
    ):
        p = out_dir / fname
        if not p.exists():
            p.write_text(f"# {fname.replace('_', ' ')[:-3]}\n\n{stub}", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument(
        "--batch-output-base",
        type=Path,
        default=None,
        help="배치 결과 루트 (clean/, styled/, real/ 각각 _manifest_resolved.csv 포함)",
    )
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_root = repo / "research_plan_outputs"
    ds_out = out_root / "02_dataset"
    eval_out = out_root / "eval_reports"
    stage_out = out_root / "03_stage_diagnosis"
    ds_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)
    stage_out.mkdir(parents=True, exist_ok=True)

    jobs_manifest = [
        ("clean", repo / "data" / "manifests" / "clean_manifest.csv"),
        ("styled", repo / "data" / "manifests" / "styled_manifest.csv"),
        ("real", repo / "data" / "manifests" / "real_manifest.csv"),
    ]

    integrity_all: Dict[str, Any] = {"manifests": []}
    leakage_all: Dict[str, Any] = {"manifests": []}

    for label, mpath in jobs_manifest:
        if not mpath.is_file():
            integrity_all["manifests"].append({"label": label, "error": "missing_file"})
            continue
        df = pd.read_csv(mpath)
        df = normalize_manifest_df(df, repo)
        if label == "styled" and "variant_id" in df.columns:
            df = df[df["variant_id"] == "styled_v5"].reset_index(drop=True)
        if label == "real" and "variant_id" in df.columns:
            df = df[df["variant_id"] == "real_v4"].reset_index(drop=True)
        df = df.reset_index(drop=True)

        resolved_path = ds_out / f"manifest_{label}_resolved.csv"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved_path, index=False)

        ir = _integrity_report(df, label)
        ir["resolved_csv"] = str(resolved_path)
        integrity_all["manifests"].append(ir)

        lr = _split_leakage(df, label)
        lr["resolved_csv"] = str(resolved_path)
        leakage_all["manifests"].append(lr)

    (ds_out / "dataset_integrity_report.json").write_text(
        json.dumps(integrity_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ds_out / "split_leakage_report.json").write_text(
        json.dumps(leakage_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Distribution summary via existing script
    dist_txt = ds_out / "distribution_summary.txt"
    meta_csv = repo / "data" / "metadata" / "all_samples.csv"
    dev_csv = repo / "data" / "metadata" / "dev_subset.csv"
    if meta_csv.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "summarize_dataset_distribution.py"),
                "--input_csv",
                str(meta_csv),
                "--subset_csv",
                str(dev_csv) if dev_csv.is_file() else str(meta_csv),
                "--output_txt",
                str(dist_txt),
            ],
            check=True,
        )

    if args.skip_eval or args.batch_output_base is None:
        print("[skip-eval] batch-output-base not set or skip-eval")
        _stage_md_from_reports([], stage_out)
        return

    base = args.batch_output_base.resolve()
    gate_levels = ("mvp", "development", "strict")
    eval_jobs = [
        ("clean", base / "clean", "clean"),
        ("styled", base / "styled", "styled"),
        ("real", base / "real", "real_like"),
    ]

    report_paths: List[Tuple[str, Path]] = []
    for name, subdir, gate_type in eval_jobs:
        man = subdir / "_manifest_resolved.csv"
        if not man.is_file():
            print(f"[warn] missing {man}")
            continue
        for lv in gate_levels:
            rjson = eval_out / f"report_{name}_{lv}.json"
            _run_eval(repo, man, subdir, gate_type, lv, rjson, args.max_samples)
            report_paths.append((f"{name}_{lv}", rjson))

    _stage_md_from_reports(report_paths, stage_out)

    # Ablation TSV summarize if present
    ab_tsv = repo / "experiments" / "ablation_eval_summary.tsv"
    ab_out = out_root / "04_ablation"
    ab_out.mkdir(parents=True, exist_ok=True)
    if ab_tsv.is_file():
        import shutil

        shutil.copy(ab_tsv, ab_out / "ablation_matrix.tsv")
        summary_lines = ["# Ablation summary", "", f"원본: `{ab_tsv}`", ""]
        try:
            adf = pd.read_csv(ab_tsv, sep="\t")
            summary_lines.append(adf.to_string(index=False))
        except Exception as exc:
            summary_lines.append(f"(표 파싱 실패: {exc})")
        (ab_out / "ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[DONE] outputs under {out_root}")


if __name__ == "__main__":
    main()
