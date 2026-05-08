#!/usr/bin/env python3
"""Selective oracle이 rule 대비 no-op인 원인을 artifacts(debug.json/result.json)로 추적한다."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run_dir(root: Path, domain: str, sample_id: str) -> Path:
    return root / "runs" / f"{domain}_{sample_id}"


def _debug_path(run_dir: Path, arm: str, sample_id: str) -> Path:
    suf = {"rule": "rule", "global_oracle": "global", "selective_oracle": "selective"}[arm]
    return run_dir / suf / f"debug_{sample_id}_{suf}" / "debug.json"


def _result_path(run_dir: Path, arm: str, sample_id: str) -> Path:
    suf = {"rule": "rule", "global_oracle": "global", "selective_oracle": "selective"}[arm]
    return run_dir / suf / f"{sample_id}_result.json"


def _result_metrics(row: pd.Series) -> Dict[str, Any]:
    return {
        "curve_y_mae_px": float(row["curve_y_mae_px"]),
        "peak_f1": float(row["peak_f1"]),
        "trace_valid_ratio": float(row["trace_valid_ratio"]),
    }


def _intensity_compare(a: List[float], b: List[float], rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[int, int, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return -1, -1, float("nan")
    eq = np.isclose(aa, bb, rtol=rtol, atol=atol)
    n = int(eq.size)
    n_same = int(eq.sum())
    frac = float(n_same / max(n, 1))
    return n_same, n, frac


def _classify_meta_only(
    *,
    n_segments: int,
    ts_rule: Optional[float],
    ts_sel: Optional[float],
    ts_glo: Optional[float],
    intens_same_sel_rule: float,
    intens_same_sel_global: float,
) -> str:
    if n_segments == 0:
        return "SEGMENT_EMPTY_OR_MISSING"
    if intens_same_sel_rule >= 1.0 - 1e-9:
        if ts_sel is not None and ts_rule is not None and abs(ts_sel - ts_rule) < 1e-6:
            if ts_glo is not None and abs(ts_glo - ts_rule) > 1e-3:
                return (
                    "SELECTIVE_DP_IDENTICAL_TO_RULE_BUT_GLOBAL_DP_DIFFERS "
                    "(메타 trace_score: rule≈selective≠global; 최종 intensity rule≈selective)"
                )
            return "SELECTIVE_equals_RULE_AND_GLOBAL_SIMILAR_META"
        return "FINAL_CURVE_MATCH_RULE_BUT_DP_META_MISMATCH_CHECK"
    return "SEGMENTS_PRESENT_FINAL_MISMATCH_UNEXPECTED"


def _print_section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def _print_candidate_compare_dump(data: Dict[str, Any]) -> None:
    """selective_candidate_field_compare.json (XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON 산출) 해석 출력."""

    _print_section("A. candidate field 구조 요약 (rule / selective / global)")
    for key in ("fc_rule_summary", "fc_sel_summary", "fc_global_summary"):
        lab = key.replace("fc_", "").replace("_summary", "")
        print(f"  [{lab}] {json.dumps(data.get(key), ensure_ascii=False)}")

    _print_section("B. confidence hash 비교")
    sha = data.get("confidence_sha256") or {}
    print(f"  hash(fc_rule confidence vector) = {sha.get('fc_rule')}")
    print(f"  hash(fc_sel confidence vector)  = {sha.get('fc_sel')}")
    print(f"  hash(fc_global confidence vector)= {sha.get('fc_global')}")
    print(f"  fc_sel == fc_rule (sha256): {data.get('fc_sel_equals_fc_rule_sha')}")
    print(f"  fc_sel == fc_global (sha256): {data.get('fc_sel_equals_fc_global_sha')}")

    ysha = data.get("candidate_y_keys_sha256") or {}
    print(f"\n  (후보 y 집합) hash(fc_rule) = {ysha.get('fc_rule')}")
    print(f"  (후보 y 집합) hash(fc_sel)  = {ysha.get('fc_sel')}")
    print(f"  (후보 y 집합) hash(fc_global)= {ysha.get('fc_global')}")
    print(f"  fc_sel y-keys == fc_rule: {data.get('fc_sel_equals_fc_rule_y_keys_sha')}")
    print(f"  fc_sel y-keys == fc_global: {data.get('fc_sel_equals_fc_global_y_keys_sha')}")

    _print_section("C. column-level confidence 차이 집계")
    dc = data.get("confidence_diff_counts") or {}
    dm = data.get("confidence_diff_abs_max") or {}
    dma = data.get("confidence_diff_mean_abs") or {}
    print(f"  rule vs selective — 다른 candidate 수: {dc.get('rule_vs_selective_diff_cells')}")
    print(f"  global vs selective — 다른 candidate 수: {dc.get('global_vs_selective_diff_cells')}")
    print(f"  rule vs global — 다른 candidate 수: {dc.get('rule_vs_global_diff_cells')}")
    print(f"  abs 차이 max (rs/sg/rg): {dm.get('rule_vs_selective')}, {dm.get('global_vs_selective')}, {dm.get('rule_vs_global')}")
    print(f"  abs 차이 mean (차이 난 셀만): {dma.get('rule_vs_selective')}, {dma.get('global_vs_selective')}, {dma.get('rule_vs_global')}")

    _print_section("D. selective vs global — 첫 confidence 차이 column 상세")
    print(f"  first differing column index: {data.get('first_col_where_selective_differs_from_global')}")
    detail = data.get("first_diff_column_detail")
    if detail:
        print(f"  column={detail.get('column')}")
        print(f"  candidate y list: {detail.get('y_list')}")
        print(f"  rule confidence list: {detail.get('rule_confidence_list')}")
        print(f"  selective confidence list: {detail.get('selective_confidence_list')}")
        print(f"  global confidence list: {detail.get('global_confidence_list')}")
        print(f"  rule top y: {detail.get('rule_top_y')}")
        print(f"  selective top y: {detail.get('selective_top_y')}")
        print(f"  global top y: {detail.get('global_top_y')}")
    else:
        print("  (전역적으로 selective confidence가 global과 동일하면 상세 없음)")

    _print_section("E. dp_trace 직전 입력 검증 (selective)")
    dpi = data.get("dp_trace_input") or {}
    print(f"  fc_sel python id(): {dpi.get('fc_sel_python_id')}")
    print(f"  fc_sel type: {dpi.get('fc_sel_type')}")
    print(f"  fc_sel column 수: {dpi.get('fc_sel_num_columns')}")
    print(f"  fc_sel confidence sha256 (dp_trace 직전): {dpi.get('fc_sel_confidence_sha256_pre_dp_trace')}")

    _print_section("F. dp_trace 결과 및 apex 후 최종 path 비교")
    pre = data.get("dp_trace_pre_apex") or {}
    post = data.get("dp_trace_post_apex") or {}
    print(f"  trace_score rule_dp: {pre.get('rule_trace_score')}")
    print(f"  trace_score selective_dp: {pre.get('selective_trace_score')}")
    print(f"  trace_score global_dp: {pre.get('global_trace_score')}")
    print(f"  dp path sha256 (pre-apex) rule/sel/global: {pre.get('rule_path_sha256')} | {pre.get('selective_path_sha256')} | {pre.get('global_path_sha256')}")
    print(f"  selective path == rule (pre-apex): {pre.get('selective_path_equals_rule')}")
    print(f"  selective path == global (pre-apex): {pre.get('selective_path_equals_global')}")
    print(f"  final y sha256 (post-apex) rule/sel/global: {post.get('rule_final_y_sha256')} | {post.get('selective_final_y_sha256')} | {post.get('global_final_y_sha256')}")
    print(f"  selective final y == rule: {post.get('selective_final_y_equals_rule')}")
    print(f"  selective final y == global: {post.get('selective_final_y_equals_global')}")

    _print_section("원인 분류 (confidence/path 해시 기준)")
    print(_classify_fc_dp_case(data))


def _classify_fc_dp_case(data: Dict[str, Any]) -> str:
    """덤프 JSON만으로 CASE 1~5 분류 (사용자 제공 기준)."""

    same_rule = bool(data.get("fc_sel_equals_fc_rule_sha"))
    same_global = bool(data.get("fc_sel_equals_fc_global_sha"))
    y_same_rule = bool(data.get("fc_sel_equals_fc_rule_y_keys_sha", True))
    y_same_global = bool(data.get("fc_sel_equals_fc_global_y_keys_sha", True))

    if same_rule:
        summ = data.get("selective_apply_summary_returned") or {}
        rc = summ.get("risk_columns_used")
        if rc is not None and int(rc) == 0:
            return (
                "fc_sel == fc_rule (정상 범주): selective_apply summary 상 risk_columns_used=0 — "
                "위험열이 없어 oracle confidence가 적용되지 않고 rule 후보가 그대로 dp_trace에 들어감."
            )
        note = ""
        if not y_same_global:
            note = (
                " [부가] fc_global만 (column,y) 집합 해시가 다름 — global oracle 경로에서 후보 구조가 달라졌을 수 있음."
            )
        return (
            "CASE 1 (히스토리): fc_sel confidence 해시 == fc_rule — "
            "과거 버그(score가 호출부 fc_sel에 미반영) 시나리오 또는 risk 미적용·동일 후보 결과일 수 있음."
            + note
        )

    if not y_same_rule:
        return (
            "CASE 5 (rule 대비): selective 후보 (column,y) 집합이 rule과 다름 — "
            "confidence 외 후보 필터/프루닝 차이 가능."
        )

    if not y_same_global:
        return (
            "CASE 5 (global 대비): selective와 global의 후보 (column,y) 집합이 다름 — "
            "global oracle 적용 경로에서 후보 구조가 바뀐 경우."
        )
    pre = data.get("dp_trace_pre_apex") or {}
    post = data.get("dp_trace_post_apex") or {}

    def _ts(name: str) -> float:
        v = pre.get(name)
        return float(v) if v is not None else float("nan")

    ts_r, ts_s, ts_g = _ts("rule_trace_score"), _ts("selective_trace_score"), _ts("global_trace_score")

    if same_global:
        path_same_g = bool(pre.get("selective_path_equals_global"))
        ts_close = abs(ts_s - ts_g) < 1e-6
        fy_same_g = bool(post.get("selective_final_y_equals_global"))
        if not (ts_close and path_same_g and fy_same_g):
            return (
                "CASE 2: fc_sel == fc_global (confidence 해시)인데 dp_trace/apex 결과가 global과 다름 — "
                "동일 입력에 대한 비결정성·버그·또는 덤프 시점과 실제 DP 호출 객체 불일치 가능."
            )
        return (
            "검증 통과: fc_sel confidence == fc_global 이고 dp_trace·apex 최종 path도 global과 일치 "
            "(risk가 전 열이면 selective 점수장이 global과 동일해질 수 있음)."
        )

    dp_same_rule = abs(ts_s - ts_r) < 1e-6 and bool(pre.get("selective_path_equals_rule"))
    fy_same_rule = bool(post.get("selective_final_y_equals_rule"))

    if dp_same_rule and fy_same_rule and abs(ts_g - ts_r) > 1e-3:
        return (
            "CASE 3: fc_sel은 rule/global 양쪽과 confidence가 다르지만 dp_trace 결과는 rule과 동일 — "
            "oracle 선택적 가중이 transition/smoothness 비용에 비해 약하거나 국소적이라 DP 최적해가 변하지 않음."
        )

    if not dp_same_rule and fy_same_rule:
        return (
            "CASE 4 후보: DP 단계에서 selective≠rule trace/path 이지만 apex 후 최종 path가 rule과 동일 — "
            "후처리 경로는 배제했다면 apex 또는 후속 단계 정렬 문제 가능 (관측상 드묾)."
        )

    return (
        "분류 보류: fc_sel은 rule/global 모두와 confidence가 다르고, "
        "DP 결과도 rule/global과 완전 동일하지 않음 — 위 CASE에 해당하지 않거나 추가 조건 필요."
    )


def _resolve_workspace(workspace: Path) -> Path:
    return workspace.expanduser().resolve()


def _resolve_input_path(raw: str, workspace: Path) -> Path:
    """CSV 절대경로가 다른 동기화 폴더명을 가리킬 때 workspace 기준으로 폴백."""
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()
    parts = p.parts
    try:
        idx = parts.index("xrd_digitizer_v1")
        rel = Path(*parts[idx + 1 :])
        cand = workspace / rel
        if cand.is_file():
            return cand.resolve()
    except ValueError:
        pass
    if not p.is_absolute():
        cand = workspace / p
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(raw)


def _run_one_dump_from_csv(
    *,
    workspace: Path,
    samples_csv: Path,
    sample_id: str,
    domain: str,
    taxonomy_prior: str,
    dump_out: Path,
) -> None:
    if not samples_csv.is_file():
        raise FileNotFoundError(samples_csv)
    df = pd.read_csv(samples_csv)
    mask = (df["sample_id"].astype(str) == sample_id) & (df["domain"].astype(str) == domain)
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError(f"CSV에 없음: sample_id={sample_id!r} domain={domain!r} ({samples_csv})")
    row = sub.iloc[0]
    tax_csv = str(row.get("taxonomy_prior") or row.get("proxy_failure_labels") or "")
    if taxonomy_prior and tax_csv and taxonomy_prior != tax_csv:
        print(f"[WARN] taxonomy_prior CLI={taxonomy_prior!r} CSV={tax_csv!r} — CSV 값으로 실행합니다.")

    def _cell_path(col: str) -> Path:
        return _resolve_input_path(str(row[col]), workspace)

    image_path = _cell_path("image_path")
    gt_path = _cell_path("gt_path")
    manual_path = _cell_path("manual_inputs_path")
    for label, pth in [("image", image_path), ("gt", gt_path), ("manual", manual_path)]:
        if not pth.is_file():
            raise FileNotFoundError(f"{label}: {pth}")

    dump_out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON"] = str(dump_out.resolve())

    tax_use = tax_csv or taxonomy_prior
    cmd = [
        sys.executable,
        "-m",
        "runner.run_local",
        "--image_path",
        str(image_path),
        "--manual_inputs_path",
        str(manual_path),
        "--output_json_path",
        str((dump_out.parent / f"{sample_id}_dump_run_result.json").resolve()),
        "--debug_dir",
        str((dump_out.parent / f"{sample_id}_dump_run_debug").resolve()),
        "--selective-oracle-rerank-gt",
        str(gt_path),
        "--run-domain",
        domain,
    ]
    if tax_use:
        cmd.extend(["--selective-oracle-taxonomy-prior", tax_use])

    print("\n=== run-one-dump (단일 샘플 run_local) ===")
    print(" ".join(cmd))
    print(f"XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON={env['XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON']}")
    subprocess.run(cmd, cwd=str(workspace), env=env, check=True)
    if not dump_out.is_file():
        raise RuntimeError(f"덤프 파일이 생성되지 않음: {dump_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug selective oracle vs rule no-op")
    ap.add_argument("--root", type=str, required=True, help="배치 출력 루트 (예: outputs/_sel_study_full)")
    ap.add_argument("--sample-id", type=str, required=True, dest="sample_id")
    ap.add_argument("--domain", type=str, required=True, choices=["clean", "styled", "real_like"])
    ap.add_argument("--taxonomy-prior", type=str, required=True, dest="taxonomy_prior")
    ap.add_argument(
        "--compare-json",
        type=str,
        default=None,
        metavar="PATH",
        help="XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON 로 저장된 selective_candidate_field_compare.json",
    )
    ap.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="프로젝트 루트 (--run-one-dump 시 CSV 경로 해석 및 run_local cwd)",
    )
    ap.add_argument(
        "--samples-csv",
        type=str,
        default="outputs/_sel_study_full/selected_samples.csv",
        help="--run-one-dump 시 행 조회용 CSV",
    )
    ap.add_argument(
        "--run-one-dump",
        action="store_true",
        help="CSV에서 단일 샘플 경로를 읽어 run_local 1회 실행 후 덤프 생성",
    )
    ap.add_argument(
        "--dump-json-out",
        type=str,
        default="outputs/_debug_selective_cmp/selective_candidate_field_compare.json",
        metavar="PATH",
        help="--run-one-dump 시 저장할 비교 JSON 경로",
    )
    args = ap.parse_args()

    workspace = _resolve_workspace(Path(args.workspace))
    root = Path(args.root)
    if not root.is_absolute():
        root = workspace / root
    root = root.resolve()

    sid = args.sample_id
    dom = args.domain
    tax_expected = args.taxonomy_prior

    compare_path = Path(args.compare_json).expanduser() if args.compare_json else None
    dump_default = workspace / args.dump_json_out
    if args.run_one_dump:
        _run_one_dump_from_csv(
            workspace=workspace,
            samples_csv=(workspace / args.samples_csv).resolve(),
            sample_id=sid,
            domain=dom,
            taxonomy_prior=tax_expected,
            dump_out=dump_default.resolve(),
        )
        if compare_path is None:
            compare_path = dump_default.resolve()

    results_csv = root / "selective_oracle_rerank_results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(results_csv)

    rdf = pd.read_csv(results_csv)
    mask = (rdf["sample_id"] == sid) & (rdf["domain"] == dom)
    sub = rdf.loc[mask]
    if sub.empty:
        raise ValueError(f"{sid} / {dom} not found in results CSV")

    tax_actual = str(sub.iloc[0]["taxonomy_prior"])
    if tax_actual != tax_expected:
        print(f"[WARN] taxonomy_prior 불일치: CLI={tax_expected!r} CSV={tax_actual!r} (CSV 기준으로 계속)")

    rows_by_arm = {str(a): sub.loc[sub["arm"] == a].iloc[0] for a in sub["arm"].unique().tolist()}
    needed = {"rule", "global_oracle", "selective_oracle"}
    missing_arm = sorted(needed - set(rows_by_arm))
    if missing_arm:
        raise ValueError(f"missing arms in CSV: {missing_arm}")

    run_dir = _run_dir(root, dom, sid)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    print(f"\n=== sample={sid} domain={dom} ===")
    print(f"taxonomy_prior(CSV)={tax_actual}")

    # --- per-arm outputs ---
    for arm in ("rule", "global_oracle", "selective_oracle"):
        dbg_p = _debug_path(run_dir, arm, sid)
        res_p = _result_path(run_dir, arm, sid)
        print(f"\n[{arm}]")
        print(f"  debug.json: {dbg_p}")
        print(f"  result.json: {res_p}")
        m = _result_metrics(rows_by_arm[arm])
        print(f"  curve_y_mae_px={m['curve_y_mae_px']:.6g} peak_f1={m['peak_f1']:.6g} trace_valid_ratio={m['trace_valid_ratio']:.6g}")

    # --- selective-only meta ---
    sel_dbg = _load_json(_debug_path(run_dir, "selective_oracle", sid))
    ma = sel_dbg.get("model_assist") or {}
    ora = ma.get("oracle_rerank") or {}
    rd = ma.get("risk_detector") or {}
    sel_o = ma.get("selective_oracle") or {}

    risk_segments_top = int(rd.get("risk_segments", 0))
    detail = ora.get("risk_segments_detail") or []
    seg_count = len(detail)

    ts_rule = ora.get("trace_score_rule_dp")
    ts_glo = ora.get("trace_score_global_oracle_dp")
    ts_sel = ora.get("trace_score_selective_oracle_dp")

    print("\n--- selective debug meta ---")
    print(f"risk_detector.risk_segments={risk_segments_top} risk_segments_detail.len={seg_count}")
    if detail:
        print("risk_segments_detail (first 5):")
        for seg in detail[:5]:
            print(
                f"  start={seg.get('segment_start_x')} end={seg.get('segment_end_x')} "
                f"len={seg.get('segment_len')} reasons={seg.get('risk_reasons')}",
            )

    print(f"selective_oracle applied_candidate_count={sel_o.get('applied_candidate_count')}")
    print(f"selective_oracle preserved_rule_candidate_count={sel_o.get('preserved_rule_candidate_count')}")
    print(f"oracle_rerank selective_oracle_score_summary.risk_columns_used={ora.get('selective_oracle_score_summary', {}).get('risk_columns_used')}")
    print(f"trace_score_rule_dp={ts_rule}")
    print(f"trace_score_global_oracle_dp={ts_glo}")
    print(f"trace_score_selective_oracle_dp={ts_sel}")

    # --- intensity path proxy ---
    ir = _load_json(_result_path(run_dir, "rule", sid)).get("intensities") or []
    ig = _load_json(_result_path(run_dir, "global_oracle", sid)).get("intensities") or []
    is_ = _load_json(_result_path(run_dir, "selective_oracle", sid)).get("intensities") or []

    nsr, ntot, fsr = _intensity_compare(ir, is_)
    nsg, _, fsg = _intensity_compare(is_, ig)
    nrg, _, frg = _intensity_compare(ir, ig)

    print("\n--- final curve proxy (result intensities column-aligned) ---")
    print(f"n_columns rule={len(ir)} global={len(ig)} selective={len(is_)}")
    print(f"same_frac selective_vs_rule={fsr:.6f} ({nsr}/{ntot})")
    print(f"same_frac selective_vs_global={fsg:.6f}")
    print(f"same_frac rule_vs_global={frg:.6f}")

    rg_diff = np.asarray(ir, dtype=np.float64) != np.asarray(ig, dtype=np.float64)
    if rg_diff.size:
        n_diff_rg = int(rg_diff.sum())
        print(f"positions rule!=global: {n_diff_rg} / {rg_diff.size}")

    verdict = _classify_meta_only(
        n_segments=max(seg_count, risk_segments_top),
        ts_rule=float(ts_rule) if ts_rule is not None else None,
        ts_sel=float(ts_sel) if ts_sel is not None else None,
        ts_glo=float(ts_glo) if ts_glo is not None else None,
        intens_same_sel_rule=fsr,
        intens_same_sel_global=fsg,
    )

    print("\n=== CLASSIFICATION (배치 메타·intensity) ===")
    print(verdict)

    if compare_path is not None:
        cmp_p = compare_path.expanduser()
        if not cmp_p.is_absolute():
            cmp_p = workspace / cmp_p
        cmp_p = cmp_p.resolve()
        if not cmp_p.is_file():
            raise FileNotFoundError(f"--compare-json 파일 없음: {cmp_p}")
        _print_candidate_compare_dump(_load_json(cmp_p))
    else:
        print("\n(팁) 후보 confidence 비교는 XRD_DUMP_SELECTIVE_ORACLE_CMP_JSON 로 덤프 후 --compare-json PATH 로 확인.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
