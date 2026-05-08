#!/usr/bin/env python3
"""
07_components_overlay · 08_candidate_map_raw 등 디버그 PNG와 debug.json 을 이용해
열 방향 argmax(프록시) 분포·상단 밴드 지배 여부를 진단한다.

주의: DP 가 사용하는 `comp_score_map` 은 기본 디버그 PNG로 저장되지 않는다.
      존재하는 `candidate_map_final` / `candidate_map_raw` / `components_overlay` 를
      grayscale 로 읽어 **프록시 argmax** 로 분석한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column


def _norm_sid(sample_id: str) -> str:
    s = sample_id.strip()
    return s[: -len("_result")] if s.endswith("_result") else s


def _resolve_debug_dir(root: Path, sample_id: str, domain: str, arm: str) -> Path:
    sid = _norm_sid(sample_id)
    dom = str(domain)
    arm_dir = {"global_oracle": "global", "rule": "rule", "selective_oracle": "selective"}.get(arm, arm)
    base = root.expanduser().resolve()
    return base / "runs" / f"{dom}_{sid}" / arm_dir / f"debug_{sid}_{arm_dir}"


def _glob_one(d: Path, pattern: str) -> Optional[Path]:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def _column_argmax_y(score: np.ndarray) -> Tuple[np.ndarray, int, int]:
    h, w = score.shape
    ys = np.argmax(score, axis=0).astype(np.int64)
    return ys, h, w


def _longest_near_constant_run(seq: np.ndarray, tol: int = 2) -> int:
    if seq.size == 0:
        return 0
    best = 1
    start = 0
    for i in range(1, seq.size):
        window = seq[start : i + 1]
        if int(np.max(window) - np.min(window)) <= tol:
            best = max(best, i - start + 1)
        else:
            start = i
    return int(best)


def _upper_band_frac(ys: np.ndarray, h: int, frac: float = 0.2) -> float:
    thr = float(frac) * float(h)
    return float(np.mean(ys < thr)) if ys.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="debug 디렉터리 또는 그 부모")
    ap.add_argument("--root", default=None)
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--arm", default="global_oracle", help="global_oracle | rule | selective_oracle")
    ap.add_argument(
        "--proxy",
        default="candidate_map_final",
        choices=["candidate_map_final", "candidate_map_raw", "components_overlay"],
    )
    ap.add_argument("--upper-band-frac", type=float, default=0.2)
    args = ap.parse_args()

    if args.run_dir:
        rd = Path(args.run_dir).expanduser().resolve()
        dbg_dir = rd if (rd / "debug.json").is_file() else rd
        if not (dbg_dir / "debug.json").is_file():
            hit = _glob_one(rd, "**/debug.json")
            dbg_dir = hit.parent if hit else rd
    else:
        if not args.root or not args.sample_id or not args.domain:
            ap.error("either --run-dir or (--root --sample-id --domain) required")
        dbg_dir = _resolve_debug_dir(Path(args.root), str(args.sample_id), str(args.domain), str(args.arm))

    dj = dbg_dir / "debug.json"
    if not dj.is_file():
        raise FileNotFoundError(dj)

    d = json.loads(dj.read_text(encoding="utf-8"))
    plot_box = tuple(int(x) for x in d["plot_box"])
    x0, y0, x1, y1 = plot_box
    roi_w, roi_h = int(x1 - x0), int(y1 - y0)

    ma = d.get("model_assist") or {}
    orac = ma.get("oracle_rerank") or {}
    gt_path = orac.get("gt_json_path")
    summ = orac.get("oracle_score_summary") or {}
    mean_oracle = summ.get("mean_oracle_dist_px") if isinstance(summ, dict) else None

    cand_stats = d.get("candidate_stats") or {}
    total_candidates = cand_stats.get("final_candidates_total")
    total_columns = cand_stats.get("total_columns")

    proxy_key = str(args.proxy)
    proxy_png = _glob_one(dbg_dir, f"*{proxy_key}.png")
    cand_raw_png = _glob_one(dbg_dir, "*candidate_map_raw.png")
    comp_png = _glob_one(dbg_dir, "*components_overlay.png")

    trace_block = d.get("trace") or {}
    path = trace_block.get("path")

    print(f"debug_dir={dbg_dir}")
    print(f"proxy_png ({proxy_key})={proxy_png}")
    print(f"candidate_map_raw_png={cand_raw_png}")
    print(f"components_overlay_png={comp_png}")
    print(f"roi_w={roi_w} roi_h={roi_h}")
    print(f"candidate_stats.final_candidates_total={total_candidates}")
    print(f"candidate_stats.total_columns={total_columns}")
    print(f"oracle.mean_oracle_dist_px={mean_oracle}")

    gt_y_by_col: dict = {}
    if gt_path and Path(str(gt_path)).is_file():
        gt = _load_gt_json(str(gt_path))
        gt_y_by_col, _ = build_gt_y_roi_per_column(gt, plot_box, roi_h, roi_w)

    if proxy_png is None or not proxy_png.is_file():
        print("\n[SKIP] proxy PNG 없음 — argmax/밴드 분석 불가.")
        if path is None:
            print("[NOTE] trace.path 없음. 최신 runner는 debug.json.trace.path에 DP 경로를 저장한다.")
        return

    score = _load_gray(proxy_png)
    ys, h, w = _column_argmax_y(score)
    if (h, w) != (roi_h, roi_w):
        print(f"[WARN] proxy_shape={score.shape} expected_roi=({roi_h},{roi_w}) 불일치")

    ub = float(args.upper_band_frac)
    print(f"\nproxy score map: {proxy_png}")
    print(f"proxy_height_width={h},{w}")
    print(
        f"argmax_y min/mean/max/p50="
        f"{float(np.min(ys)):.4g},{float(np.mean(ys)):.4g},{float(np.max(ys)):.4g},{float(np.percentile(ys, 50)):.4g}"
    )
    print(f"fraction_argmax_in_upper_{ub:.2f}={_upper_band_frac(ys, h, ub):.6g}")
    print(f"longest_near_horizontal_run_tol2px={_longest_near_constant_run(ys, tol=2)}")

    if isinstance(path, list) and len(path) == int(w):
        py = np.array([np.nan if p is None else float(p) for p in path], dtype=np.float64)
        finite = np.isfinite(py)
        if finite.any():
            print(f"path_vs_argmax_agreement_tol1.5px={float(np.mean(finite & (np.abs(py - ys) <= 1.5))):.6g}")
            print(f"global_final_path_upper_band_frac={_upper_band_frac(py[finite].astype(np.int64), h, ub):.6g}")
        else:
            print("path_vs_argmax: (no finite path entries)")
    else:
        print("[NOTE] trace.path 부재 또는 길이 불일치 — DP 재실행 후 확인.")

    if gt_y_by_col:
        dists_a = []
        dists_p = []
        for col in range(min(int(w), int(roi_w))):
            gty = gt_y_by_col.get(col)
            if gty is None:
                continue
            dists_a.append(abs(float(gty) - float(ys[col])))
            if isinstance(path, list) and col < len(path) and path[col] is not None:
                dists_p.append(abs(float(gty) - float(path[col])))
        if dists_a:
            print(f"mean_abs_gt_minus_argmax_y_proxy={float(np.mean(dists_a)):.6g}")
        if dists_p:
            print(f"mean_abs_gt_minus_final_path_y={float(np.mean(dists_p)):.6g}")
    else:
        print("GT 거리: gt_json 없거나 매핑 실패")


if __name__ == "__main__":
    main()
