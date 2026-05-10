"""
§9: 배치 실행 – manifest 기반으로 여러 이미지를 순차 처리.

이미 처리된 샘플은 건너뛰어 중단 후 재개가 가능하다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contrast_aux_settings import ContrastAuxSettings
from core.model_assist_settings import ModelAssistSettings
from core.oracle_rerank_settings import OracleRerankSettings
from core.selective_oracle_settings import SelectiveOracleSettings
from core.sharp_peak_settings import SharpPeakPreserveSettings
from runner.run_local import run_single


def _ensure_manual_inputs_for_gt(gt_path: str, output_path: str) -> str:
    """GT JSON에서 manual_inputs JSON을 자동 생성 (개발/평가용)."""
    with Path(gt_path).open("r", encoding="utf-8") as f:
        gt = json.load(f)
    am = gt["axis_metadata"]
    pb = gt["plot_box"]
    x0, y0, x1, y1 = pb

    pcp = gt.get("pixel_curve_path", [])
    if pcp and len(pcp) > 2:
        mid_pt = pcp[len(pcp) // 2]
        csp = [int(mid_pt[0]), int(mid_pt[1])]
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
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(mi, f, ensure_ascii=False, indent=2)
    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="§9: Batch run engine on manifest")
    parser.add_argument("--manifest_csv", type=str, required=True,
                        help="clean_manifest or styled_manifest CSV")
    parser.add_argument("--output_dir", type=str, default=r"c:\xrd_digitizer_v1\outputs")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose output JSON already exists")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="v1_1",
        choices=["v1_1", "v1", "v1.1", "v1_2", "v1.2", "v2", "v2_experimental"],
        help="엔진 파이프라인 (run_local과 동일; 기본 v1.1 = v1_1, v1_2=v1.2 스냅샷 태그)",
    )
    parser.add_argument("--tune_json", type=str, default=None, help="v2 파라미터 JSON 경로")
    parser.add_argument("--allow_experimental_v2", action="store_true",
                        help="v2_experimental 실행 잠금 해제 플래그")
    parser.add_argument(
        "--axis-mask-margin",
        type=int,
        default=15,
        metavar="PX",
        help="v1_1: mask_axis_lines margin (기본 15)",
    )
    parser.add_argument(
        "--mask-b-mag-percentile",
        type=float,
        default=50.0,
        help="mask_b Sobel 분위수 (기본 50)",
    )
    parser.add_argument(
        "--mask-b-thr-clip-lo",
        type=float,
        default=10.0,
        help="mask_b 임계 하한 (기본 10)",
    )
    parser.add_argument(
        "--mask-b-thr-clip-hi",
        type=float,
        default=40.0,
        help="mask_b 임계 상한 (기본 40)",
    )
    parser.add_argument(
        "--use-ridge-candidates",
        action="store_true",
        help="v1_1: 후보 신뢰도에 세로 능선 응답 가산",
    )
    parser.add_argument(
        "--peak-single-pass",
        action="store_true",
        help="v1_1: 피크 prominence 2패스·NMS 끄기",
    )
    parser.add_argument(
        "--use-contrast-aux",
        action="store_true",
        help="v1_1: contrast_aux_v1 후보 confidence 보조",
    )
    parser.add_argument("--contrast-aux-weight", type=float, default=0.25, metavar="W")
    parser.add_argument("--contrast-aux-min-base-conf", type=float, default=0.15, metavar="T")
    parser.add_argument("--contrast-aux-bg-kernel-ratio", type=float, default=0.035, metavar="R")
    parser.add_argument("--contrast-aux-border-suppress-px", type=int, default=8, metavar="PX")
    parser.add_argument("--curvature-blend-strength", type=float, default=0.32, metavar="S")
    parser.add_argument("--peak-loose-prominence-factor", type=float, default=None, metavar="F")
    parser.add_argument(
        "--use-sharp-peak-preserve",
        action="store_true",
        help="v1_1: sharp peak preserve 후처리 경로",
    )
    parser.add_argument("--curve-smooth-window", type=int, default=9, metavar="W")
    parser.add_argument("--peak-smooth-window", type=int, default=5, metavar="W")
    parser.add_argument("--peak-preserve-radius", type=int, default=3, metavar="R")
    parser.add_argument("--peak-blend-raw-weight", type=float, default=0.75, metavar="A")
    parser.add_argument("--global-prom-ratio", type=float, default=0.015, metavar="G")
    parser.add_argument("--local-prom-window", type=int, default=61, metavar="W")
    parser.add_argument("--local-prom-ratio", type=float, default=0.12, metavar="L")
    parser.add_argument("--local-noise-k", type=float, default=3.0, metavar="K")
    parser.add_argument(
        "--no-dp-candidate-bridge",
        action="store_true",
        help="v1_1: DP 전 브리지 후보 확장 끔",
    )
    parser.add_argument(
        "--no-dp-column-apex-pull",
        action="store_true",
        help="v1_1: DP 후 열별 apex pull 끔",
    )
    parser.add_argument(
        "--dump-candidates-json",
        action="store_true",
        help="debug 디렉터리에 raw/filtered/final candidates JSON 저장",
    )
    parser.add_argument("--model-assist", action="store_true")
    parser.add_argument("--model-assist-ckpt", type=str, default=None)
    parser.add_argument("--model-assist-lambda", type=float, default=0.25)
    parser.add_argument("--model-assist-device", type=str, default="cpu")
    parser.add_argument("--model-assist-patch-size", type=int, default=33)
    parser.add_argument("--model-assist-fallback-vr-margin", type=float, default=0.0)
    parser.add_argument("--model-assist-fallback-ts-margin", type=float, default=0.0)
    parser.add_argument("--peak-apex-roi-refine", action="store_true")
    parser.add_argument("--peak-apex-roi-radius", type=int, default=5)
    parser.add_argument(
        "--oracle-rerank-gt",
        type=str,
        default=None,
        help="단일 GT JSON 경로(모든 샘플에 동일 적용; 단일 샘플 배치용)",
    )
    parser.add_argument(
        "--oracle-rerank-from-manifest",
        action="store_true",
        help="각 행의 gt_path 로 oracle 재랭크 (매니페스트 평가 권장)",
    )
    parser.add_argument("--oracle-rerank-sigma", type=float, default=8.0)
    parser.add_argument(
        "--selective-oracle-rerank-gt",
        type=str,
        default=None,
        metavar="PATH",
        help="모든 샘플에 동일 GT로 selective oracle (run_local과 동일 플래그 의미)",
    )
    parser.add_argument(
        "--selective-oracle-from-manifest",
        action="store_true",
        help="각 행의 gt_path로 selective oracle (oracle-rerank-from-manifest와 동일 패턴)",
    )
    parser.add_argument("--selective-oracle-sigma", type=float, default=8.0)
    parser.add_argument(
        "--run-domain",
        type=str,
        default="clean",
        choices=["clean", "styled", "real_like"],
        help="selective oracle 도메인 태그 (run_local --run-domain과 동일)",
    )
    parser.add_argument(
        "--selective-oracle-taxonomy-prior",
        type=str,
        default=None,
        metavar="LABELS",
        help="run_local --selective-oracle-taxonomy-prior와 동일",
    )
    parser.add_argument(
        "--selective-oracle-allow-styled-real",
        action="store_true",
        help="run_local --selective-oracle-allow-styled-real와 동일",
    )
    parser.add_argument(
        "--selective-oracle-risk-features-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="run_local --selective-oracle-risk-features-csv와 동일",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    img_col = None
    # real/styled 매니페스트에도 clean_image_path가 있으므로 도메인 전용 열을 먼저 본다.
    for candidate in [
        "real_image_path",
        "styled_image_path",
        "clean_image_path",
        "image_path",
    ]:
        if candidate in df.columns:
            img_col = candidate
            break
    if img_col is None:
        raise ValueError("manifest must have an image path column")
    if "gt_path" not in df.columns:
        raise ValueError("manifest must have 'gt_path' column")
    df["image_path"] = df[img_col]

    caf = ContrastAuxSettings(
        use_contrast_aux=bool(args.use_contrast_aux),
        contrast_aux_weight=float(args.contrast_aux_weight),
        contrast_aux_min_base_conf=float(args.contrast_aux_min_base_conf),
        contrast_aux_bg_kernel_ratio=float(args.contrast_aux_bg_kernel_ratio),
        contrast_aux_border_suppress_px=int(args.contrast_aux_border_suppress_px),
    )

    sps = SharpPeakPreserveSettings(
        use_sharp_peak_preserve=bool(args.use_sharp_peak_preserve),
        curve_smooth_window=int(args.curve_smooth_window),
        peak_smooth_window=int(args.peak_smooth_window),
        peak_preserve_radius=int(args.peak_preserve_radius),
        peak_blend_raw_weight=float(args.peak_blend_raw_weight),
        global_prom_ratio=float(args.global_prom_ratio),
        local_prom_window=int(args.local_prom_window),
        local_prom_ratio=float(args.local_prom_ratio),
        local_noise_k=float(args.local_noise_k),
    )

    mas = ModelAssistSettings(
        enabled=bool(args.model_assist),
        model_ckpt_path=args.model_assist_ckpt,
        lambda_model=float(args.model_assist_lambda),
        device=str(args.model_assist_device),
        patch_size=int(args.model_assist_patch_size),
        fallback_valid_ratio_margin=float(args.model_assist_fallback_vr_margin),
        fallback_trace_score_margin=float(args.model_assist_fallback_ts_margin),
    )
    samples = df.head(args.max_samples)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0
    t_total = time.perf_counter()

    for _, row in samples.iterrows():
        sid = str(row["sample_id"])
        image_path = str(row["image_path"])
        gt_path = str(row["gt_path"])

        oracle_gt_one = None
        if args.oracle_rerank_from_manifest:
            oracle_gt_one = gt_path
        elif args.oracle_rerank_gt:
            oracle_gt_one = args.oracle_rerank_gt

        selective_gt_one = None
        if args.selective_oracle_from_manifest:
            selective_gt_one = gt_path
        elif args.selective_oracle_rerank_gt:
            selective_gt_one = args.selective_oracle_rerank_gt

        selective_active = bool(selective_gt_one)
        orac = OracleRerankSettings(
            enabled=bool(oracle_gt_one) and not selective_active,
            gt_json_path=oracle_gt_one if (bool(oracle_gt_one) and not selective_active) else None,
            sigma_px=float(args.oracle_rerank_sigma),
        )
        run_dom = str(args.run_domain)
        if "domain" in row.index and row.get("domain") is not None and str(row["domain"]).strip():
            d = str(row["domain"]).strip()
            if d in ("clean", "styled", "real_like"):
                run_dom = d
        s_orac = SelectiveOracleSettings(
            enabled=selective_active,
            gt_json_path=selective_gt_one,
            sigma_px=float(args.selective_oracle_sigma),
            run_domain=run_dom,
            taxonomy_prior=args.selective_oracle_taxonomy_prior,
            allow_styled_real_selective=bool(args.selective_oracle_allow_styled_real),
            risk_features_csv_path=args.selective_oracle_risk_features_csv,
        )

        result_json = str(out_root / f"{sid}_result.json")
        debug_dir = str(out_root / f"debug_{sid}")

        if args.resume and Path(result_json).is_file():
            skip += 1
            continue

        mi_path = str(out_root / f"_mi_{sid}.json")
        try:
            mi_path = _ensure_manual_inputs_for_gt(gt_path, mi_path)
            run_single(
                image_path,
                mi_path,
                result_json,
                debug_dir,
                pipeline=args.pipeline,
                tune_json=args.tune_json,
                allow_experimental_v2=args.allow_experimental_v2,
                axis_mask_margin=args.axis_mask_margin,
                mask_b_mag_percentile=float(args.mask_b_mag_percentile),
                mask_b_thr_clip_lo=float(args.mask_b_thr_clip_lo),
                mask_b_thr_clip_hi=float(args.mask_b_thr_clip_hi),
                use_ridge_candidates=args.use_ridge_candidates,
                peak_two_pass=not args.peak_single_pass,
                contrast_aux_settings=caf,
                curvature_blend_strength=float(args.curvature_blend_strength),
                loose_peak_prominence_factor=args.peak_loose_prominence_factor,
                sharp_peak_settings=sps,
                use_dp_candidate_bridge=not args.no_dp_candidate_bridge,
                use_dp_column_apex_pull=not args.no_dp_column_apex_pull,
                dump_candidates_json=bool(args.dump_candidates_json),
                model_assist_settings=mas,
                oracle_rerank_settings=orac,
                selective_oracle_settings=s_orac,
                use_peak_apex_roi_refine=bool(args.peak_apex_roi_refine),
                peak_apex_roi_radius=int(args.peak_apex_roi_radius),
            )
            ok += 1
            print(f"[OK] {sid}")
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {sid}: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t_total
    print(f"\n[DONE] ok={ok}, skip={skip}, fail={fail}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
