#!/usr/bin/env python3
"""각 스터디 out_root의 selective arm debug.json에서 risk 적용 폭을 집계한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _taxonomy_map(results_csv: Path) -> Dict[tuple, str]:
    if not results_csv.is_file():
        return {}
    rdf = pd.read_csv(results_csv)
    sel = rdf[rdf["arm"] == "selective_oracle"]
    out: Dict[tuple, str] = {}
    for _, row in sel.iterrows():
        k = (str(row["sample_id"]), str(row["domain"]))
        out[k] = str(row.get("taxonomy_prior", "") or "")
    return out


def _parse_run_folder(folder_name: str) -> tuple[str, str]:
    # clean_pattern_123 / styled_pattern_123 / real_like_pattern_123
    if folder_name.startswith("real_like_"):
        return "real_like", folder_name[len("real_like_") :]
    if folder_name.startswith("clean_"):
        return "clean", folder_name[len("clean_") :]
    if folder_name.startswith("styled_"):
        return "styled", folder_name[len("styled_") :]
    return "?", folder_name


def _inspect_debug(path: Path, tax_prior: str) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    ma = d.get("model_assist") or {}
    ora = ma.get("oracle_rerank") or {}
    summ = ora.get("selective_oracle_score_summary") or {}
    sel_top = ma.get("selective_oracle") or {}
    rd_top = ma.get("risk_detector") or {}

    risk_cols_used = summ.get("risk_columns_used")
    if risk_cols_used is None:
        risk_cols_used = sel_top.get("risk_columns_used")
    if risk_cols_used is None:
        risk_cols_used = ora.get("risk_columns_used")

    rs_ora = ora.get("risk_summary") or {}
    roi_w = (d.get("candidate_stats") or {}).get("total_columns")
    if roi_w is None:
        tr = d.get("trace") or {}
        diag = tr.get("diagnostics") or {}
        roi_w = diag.get("ok_columns")
    if roi_w is None:
        roi_w = rs_ora.get("total_column_count")
    if roi_w is None:
        roi_w = rd_top.get("total_column_count")

    rc = int(risk_cols_used) if risk_cols_used is not None else None
    rw = int(roi_w) if roi_w is not None else None

    ratio_official = rs_ora.get("risk_ratio")
    if ratio_official is None:
        ratio_official = rd_top.get("risk_ratio")
    if ratio_official is not None:
        ratio = float(ratio_official)
    elif rw and rw > 0 and rc is not None:
        ratio = float(rc) / float(rw)
    else:
        ratio = float("nan")

    style_skip = bool(rs_ora.get("style_policy_skip", False))

    detail = ora.get("risk_segments_detail") or []
    seg_cnt = len(detail)
    s0_start = detail[0].get("segment_start_x") if detail else ""
    s0_end = detail[0].get("segment_end_x") if detail else ""

    applied = sel_top.get("applied_candidate_count")
    if applied is None:
        applied = summ.get("applied_candidate_count")
    preserved = sel_top.get("preserved_rule_candidate_count")
    if preserved is None:
        preserved = summ.get("preserved_rule_candidate_count")

    tr = d.get("trace") or {}
    tv = tr.get("valid_ratio")
    tv_count = tr.get("valid_count")
    tv_total = tr.get("total_count")

    return {
        "taxonomy_prior": tax_prior,
        "risk_columns_used": rc,
        "roi_width": rw,
        "risk_ratio": ratio,
        "risk_ratio_source": "risk_summary" if rs_ora.get("risk_ratio") is not None else (
            "risk_detector" if rd_top.get("risk_ratio") is not None else "columns/roi"
        ),
        "style_policy_skip": style_skip,
        "segment_count": seg_cnt,
        "first_segment_start": s0_start,
        "first_segment_end": s0_end,
        "applied_candidate_count": applied,
        "preserved_rule_candidate_count": preserved,
        "trace_valid_ratio": tv,
        "trace_valid_count": tv_count,
        "trace_total_count": tv_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="selective arm debug.json risk 비율 검사")
    ap.add_argument("out_roots", type=str, nargs="*", help="스터디 출력 디렉터리들")
    ap.add_argument("--root", dest="root_opts", action="append", default=[], help="스터디 출력 디렉터리(반복 가능)")
    args = ap.parse_args()
    out_roots = [*args.out_roots, *args.root_opts]
    if not out_roots:
        raise SystemExit("at least one out_root is required (positional or --root)")

    hdr = (
        "out_root\tsample_id\tdomain\ttaxonomy_prior\tflags\t"
        "risk_columns_used\troi_width\trisk_ratio\trisk_ratio_src\tstyle_policy_skip\tsegment_count\t"
        "first_segment_start\tfirst_segment_end\tapplied_candidate_count\t"
        "preserved_rule_candidate_count\ttrace_valid_ratio\ttrace_valid_count\ttrace_total_count"
    )
    print(hdr)

    global_lt1 = 0
    global_le08 = 0
    total_sel = 0

    for root_s in out_roots:
        root = Path(root_s).expanduser().resolve()
        results_csv = root / "selective_oracle_rerank_results.csv"
        tax_map = _taxonomy_map(results_csv)
        paths = sorted(root.glob("runs/*/selective/*/debug.json"))

        root_lt1 = 0
        root_le08 = 0

        for p in paths:
            run_key = p.parent.parent.parent.name
            dom, sid = _parse_run_folder(run_key)
            tax_prior = tax_map.get((sid, dom), "")
            row = _inspect_debug(p, tax_prior)
            rr = row["risk_ratio"]
            total_sel += 1
            if rr < 1.0 - 1e-12:
                root_lt1 += 1
                global_lt1 += 1
            flags = ""
            # style_policy_skip으로 risk=0이면 부분 oracle 후보가 아님
            if (not row["style_policy_skip"]) and (1e-12 < rr <= 0.8 + 1e-12):
                root_le08 += 1
                global_le08 += 1
                flags = "PRIORITY_partial_risk_rr<=0.8"
            elif row["style_policy_skip"] and rr < 1e-12:
                flags = "STYLE_OR_DOMAIN_SKIP_risk0"

            print(
                f"{root.name}\t{sid}\t{dom}\t{tax_prior}\t{flags}\t"
                f"{row['risk_columns_used']}\t{row['roi_width']}\t{row['risk_ratio']:.6g}\t"
                f"{row['risk_ratio_source']}\t{row['style_policy_skip']}\t{row['segment_count']}\t"
                f"{row['first_segment_start']}\t{row['first_segment_end']}\t"
                f"{row['applied_candidate_count']}\t{row['preserved_rule_candidate_count']}\t"
                f"{row['trace_valid_ratio']}\t{row['trace_valid_count']}\t{row['trace_total_count']}",
            )

        print(f"# --- {root.name}: selective_n={len(paths)}, risk_ratio<1: {root_lt1}, risk_ratio<=0.8: {root_le08}")

        mp = root / "selective_oracle_metrics.csv"
        if mp.is_file():
            mdf = pd.read_csv(mp)
            if "risk_ratio" in mdf.columns:
                rr = pd.to_numeric(mdf["risk_ratio"], errors="coerce")
                n01 = int(((rr > 0) & (rr < 1)).sum())
                n0 = int((rr <= 1e-12).sum())
                n1 = int((rr >= 1.0 - 1e-12).sum())
                print(
                    f"# --- {root.name} selective_oracle_metrics.csv: "
                    f"risk_ratio min={float(rr.min()):.6g} mean={float(rr.mean()):.6g} max={float(rr.max()):.6g}"
                )
                print(f"# --- 0<risk_ratio<1: {n01}, risk_ratio==0: {n0}, risk_ratio>=~1: {n1}")

    print("\n=== 판정 요약 ===")
    print(f"전체 selective 디버그 수: {total_sel}")
    print(f"risk_ratio < 1.0 샘플 수(중복 허용·루트 합산): {global_lt1}")
    print(f"risk_ratio <= 0.8 우선 분석 후보 수: {global_le08}")
    if global_lt1 == 0 and total_sel > 0:
        print(
            "분류: 이 배치에서는 risk_ratio가 모두 1.0에 근접 — "
            "risk segmentation이 ROI 전폭으로 과대 적용된 패턴과 일치할 수 있음.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
