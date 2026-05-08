#!/usr/bin/env python3
"""candidate_map_raw / candidate_map_final 상단 밴드(argmax) 원인 추적용 진단 스크립트.

PNG에서 열별 argmax y를 구하고, (선택) 후보 JSON·GT로 신뢰도·GT 근접성을 비교한다.
후보 JSON은 `runner/run_local.py --dump-candidates-json` 시 같은 debug 디렉터리에 저장된다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace.oracle_rerank import _load_gt_json, build_gt_y_roi_per_column


def _norm_cands(fc: Mapping[Any, Any]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for k, v in fc.items():
        try:
            ck = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[ck] = list(v)
    return out


def _glob_one(run_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        raise FileNotFoundError(f"ambiguous {pattern!r}: {[str(h) for h in hits]}")
    return hits[0]


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("L")
    return np.asarray(img, dtype=np.float64)


def argmax_y_per_column(img: np.ndarray) -> np.ndarray:
    """shape (H,W) → 길이 W 의 각 열 argmax row."""
    return np.argmax(img, axis=0).astype(np.int32)


def upper_band_mask(y: np.ndarray, h: int, upper_frac: float) -> np.ndarray:
    bound = int(math.ceil(float(h) * float(upper_frac)))
    bound = max(1, min(h, bound))
    return y < bound


def top1_per_column(cands_by_col: Dict[int, List[dict]]) -> Tuple[np.ndarray, np.ndarray]:
    """columns 0..W-1 에 대해 top1 y, conf (없으면 nan)."""
    if not cands_by_col:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    w = max(cands_by_col.keys()) + 1
    ys = np.full(w, np.nan, dtype=np.float64)
    cs = np.full(w, np.nan, dtype=np.float64)
    for c, lst in cands_by_col.items():
        if c < 0 or c >= w or not lst:
            continue
        sorted_lst = sorted(lst, key=lambda d: -float(d.get("confidence", 0.0)))
        top = sorted_lst[0]
        ys[c] = float(top.get("y", np.nan))
        cs[c] = float(top.get("confidence", np.nan))
    return ys, cs


def max_conf_near_gt(
    cands_by_col: Dict[int, List[dict]],
    gt_by_col: Dict[int, float],
    thr: float,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """각 열: GT와 거리<=thr 인 후보 중 최대 confidence (없으면 nan)."""
    w = max(gt_by_col.keys(), default=-1) + 1
    if w <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    cs = np.full(w, np.nan, dtype=np.float64)
    ys = np.full(w, np.nan, dtype=np.float64)
    for col, gty in gt_by_col.items():
        if col < 0 or col >= w:
            continue
        best_c = float("-inf")
        best_y = float("nan")
        for c in cands_by_col.get(col, []):
            try:
                yy = float(c.get("y", float("nan")))
                cf = float(c.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(yy) or not (0 <= yy < h):
                continue
            if abs(yy - float(gty)) <= thr:
                if cf > best_c:
                    best_c = cf
                    best_y = yy
        if best_c > float("-inf"):
            cs[col] = best_c
            ys[col] = best_y
    return ys, cs


def mean_finite(a: np.ndarray) -> Optional[float]:
    m = a[np.isfinite(a)]
    if m.size == 0:
        return None
    return float(np.mean(m))


def _cand_stats(cands_by_col: Dict[int, List[dict]], w_eff: int) -> Tuple[int, float, int, int]:
    counts = [len(cands_by_col.get(c, [])) for c in range(w_eff)]
    total = int(sum(counts))
    if not counts:
        return total, 0.0, 0, 0
    return total, float(np.mean(counts)), int(min(counts)), int(max(counts))


def _nearest_dist_stats(cands_by_col: Dict[int, List[dict]], gt_by_col: Dict[int, float], w_eff: int) -> Dict[str, Optional[float]]:
    dists: List[float] = []
    for col in range(w_eff):
        gty = gt_by_col.get(col)
        if gty is None:
            continue
        cands = cands_by_col.get(col, [])
        if not cands:
            continue
        dd = min(abs(float(c.get("y", 0.0)) - float(gty)) for c in cands)
        dists.append(float(dd))
    if not dists:
        return {
            "mean_nearest_candidate_gt_dist_px": None,
            "median_nearest_candidate_gt_dist_px": None,
            "p90_nearest_candidate_gt_dist_px": None,
            "candidate_gt_near_recall_px3": None,
            "candidate_gt_near_recall_px5": None,
            "candidate_gt_near_recall_px10": None,
        }
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "mean_nearest_candidate_gt_dist_px": float(np.mean(arr)),
        "median_nearest_candidate_gt_dist_px": float(np.median(arr)),
        "p90_nearest_candidate_gt_dist_px": float(np.percentile(arr, 90)),
        "candidate_gt_near_recall_px3": float(np.mean(arr <= 3.0)),
        "candidate_gt_near_recall_px5": float(np.mean(arr <= 5.0)),
        "candidate_gt_near_recall_px10": float(np.mean(arr <= 10.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="candidate map upper-band 진단")
    ap.add_argument("--run-dir", type=str, required=True, help="debug_pattern_* 디렉터리 (debug.json 포함)")
    ap.add_argument("--gt-json", type=str, default="", help="GT JSON (미지정 시 GT·후보 대비 통계 생략)")
    ap.add_argument("--upper-frac", type=float, default=0.2, help="ROI 높이 상단 비율 (기본 0.2)")
    ap.add_argument("--gt-near-px", type=float, default=5.0, help="GT 근접 후보 거리(px)")
    ap.add_argument("--raw-candidates-json", type=str, default="", help="명시적 raw 후보 JSON 경로")
    ap.add_argument("--final-candidates-json", type=str, default="", help="명시적 final 후보 JSON 경로")
    ap.add_argument("--max-column-lines", type=int, default=24, help="열별 샘플 출력 상한")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dbg_path = run_dir / "debug.json"
    if not dbg_path.is_file():
        raise SystemExit(f"missing debug.json: {dbg_path}")

    raw_png = _glob_one(run_dir, "*_candidate_map_raw.png")
    fin_png = _glob_one(run_dir, "*_candidate_map_final.png")
    if raw_png is None or fin_png is None:
        raise SystemExit(f"need *_candidate_map_raw.png / *_candidate_map_final.png under {run_dir}")

    dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
    plot_box_t = tuple(int(x) for x in dbg.get("plot_box", [0, 0, 0, 0]))
    cand_stats = dbg.get("candidate_stats") or {}
    roi_w = int(cand_stats.get("total_columns", 0))
    tr_path = (dbg.get("trace") or {}).get("path")
    if not isinstance(tr_path, list) or len(tr_path) == 0:
        raise SystemExit("debug.trace.path missing or empty")

    raw_img = _load_gray(raw_png)
    fin_img = _load_gray(fin_png)
    h0, w0 = raw_img.shape[:2]
    h1, w1 = fin_img.shape[:2]
    if (h0, w0) != (h1, w1):
        raise SystemExit(f"shape mismatch raw {raw_img.shape} vs final {fin_img.shape}")
    h, w_map = h0, w0
    if roi_w <= 0:
        roi_w = w_map
    if roi_w != w_map:
        print(f"# warn: candidate_stats.total_columns={roi_w} vs map W={w_map} (map 기준으로 진행)")

    w_eff = min(int(roi_w), w_map)
    raw_am = argmax_y_per_column(raw_img[:, :w_eff])
    fin_am = argmax_y_per_column(fin_img[:, :w_eff])
    ub_raw = upper_band_mask(raw_am, h, args.upper_frac)
    ub_fin = upper_band_mask(fin_am, h, args.upper_frac)
    ratio_raw = float(np.mean(ub_raw))
    ratio_fin = float(np.mean(ub_fin))
    moved_to_ub = int(np.sum(~ub_raw & ub_fin))
    moved_from_ub = int(np.sum(ub_raw & ~ub_fin))

    path_y = np.asarray([float(tr_path[c]) if c < len(tr_path) else float("nan") for c in range(w_eff)], dtype=np.float64)
    ub_path = upper_band_mask(path_y.astype(np.int32), h, args.upper_frac)
    path_ub_frac = float(np.mean(np.isfinite(path_y) & ub_path))

    print(f"run_dir={run_dir}")
    print(f"candidate_map_raw={raw_png.name}")
    print(f"candidate_map_final={fin_png.name}")
    print(f"roi_h={h} roi_w_eff={w_eff} upper_frac={args.upper_frac}")
    print(f"candidate_map_raw_argmax_upper{int(args.upper_frac*100)}_ratio={ratio_raw:.6f}")
    print(f"candidate_map_final_argmax_upper{int(args.upper_frac*100)}_ratio={ratio_fin:.6f}")
    print(f"columns_raw_not_ub_final_ub={moved_to_ub}")
    print(f"columns_raw_ub_final_not_ub={moved_from_ub}")
    print(f"final_path_upper_band_fraction={path_ub_frac:.6f}")
    oracle_s = ((dbg.get("model_assist") or {}).get("oracle_rerank") or {}).get("oracle_score_summary") or {}
    if "mean_oracle_dist_px" in oracle_s:
        print(f"mean_oracle_dist_px={oracle_s.get('mean_oracle_dist_px')}")

    # 후보 JSON
    raw_json_path = Path(args.raw_candidates_json).expanduser() if args.raw_candidates_json else None
    fin_json_path = Path(args.final_candidates_json).expanduser() if args.final_candidates_json else None
    if raw_json_path is None or not raw_json_path.is_file():
        hit = _glob_one(run_dir, "*_raw_candidates.json")
        raw_json_path = hit
    if fin_json_path is None or not fin_json_path.is_file():
        hit = _glob_one(run_dir, "*_final_candidates.json")
        fin_json_path = hit

    raw_cands: Dict[int, List[dict]] = {}
    fin_cands: Dict[int, List[dict]] = {}
    if raw_json_path is not None and raw_json_path.is_file():
        raw_cands = _norm_cands(json.loads(raw_json_path.read_text(encoding="utf-8")))
        print(f"raw_candidates_json={raw_json_path}")
    else:
        print("raw_candidates_json=MISSING (PNG-only 상단비율만 확정; 신뢰도 비교는 생략)")
    if fin_json_path is not None and fin_json_path.is_file():
        fin_cands = _norm_cands(json.loads(fin_json_path.read_text(encoding="utf-8")))
        print(f"final_candidates_json={fin_json_path}")
    else:
        print("final_candidates_json=MISSING")

    gt_by_col: Dict[int, float] = {}
    gt_path_str = str(args.gt_json).strip()
    if gt_path_str:
        gt = _load_gt_json(gt_path_str)
        gt_by_col, gt_meta = build_gt_y_roi_per_column(gt, plot_box_t, h, w_eff)
        print(f"gt_json={gt_path_str} gt_columns_mapped={gt_meta.get('columns_mapped')}")

    thr = float(args.gt_near_px)
    png_gt_diag: Dict[str, Any] = {}

    # GT만 있는 경우: map argmax / DP path vs GT 거리 (후보 JSON 없을 때 프록시)
    if gt_by_col and not fin_cands:
        ub_bound = int(math.ceil(float(h) * float(args.upper_frac)))
        ub_bound = max(1, min(h, ub_bound))
        dist_raw: List[float] = []
        dist_fin: List[float] = []
        dist_path: List[float] = []
        raw_near = fin_near_am = path_near = 0
        raw_ub_hit = fin_ub_hit = path_ub_hit = 0
        for col in range(w_eff):
            gty = gt_by_col.get(col)
            if gty is None:
                continue
            gy = float(gty)
            ry = float(raw_am[col])
            fy = float(fin_am[col])
            py = float(path_y[col]) if col < path_y.size else float("nan")
            dist_raw.append(abs(ry - gy))
            dist_fin.append(abs(fy - gy))
            if math.isfinite(py):
                dist_path.append(abs(py - gy))
            if abs(ry - gy) <= thr:
                raw_near += 1
            if abs(fy - gy) <= thr:
                fin_near_am += 1
            if math.isfinite(py) and abs(py - gy) <= thr:
                path_near += 1
            if ry < ub_bound:
                raw_ub_hit += 1
            if fy < ub_bound:
                fin_ub_hit += 1
            if math.isfinite(py) and py < ub_bound:
                path_ub_hit += 1
        def _summ(dx: List[float]) -> str:
            if not dx:
                return "n=0"
            arr = np.asarray(dx, dtype=np.float64)
            return (
                f"n={len(dx)} mean={float(np.mean(arr)):.4f} median={float(np.median(arr)):.4f} "
                f"p90={float(np.percentile(arr, 90)):.4f}"
            )

        print("\n# PNG/GT 프록시 (후보 JSON 없음): 열별 |raw_am-GT|, |final_am-GT|, |path-GT|")
        print(f"  raw_am_vs_gt_px: {_summ(dist_raw)}")
        print(f"  final_am_vs_gt_px: {_summ(dist_fin)}")
        print(f"  path_vs_gt_px: {_summ(dist_path)}")
        print(f"  columns_argmax_raw_within_gt_near_px{thr:g}: {raw_near}/{w_eff}")
        print(f"  columns_argmax_final_within_gt_near_px{thr:g}: {fin_near_am}/{w_eff}")
        print(f"  columns_path_within_gt_near_px{thr:g}: {path_near}/{w_eff}")
        print(f"  columns_raw_argmax_in_upper_band: {raw_ub_hit}/{w_eff}")
        print(f"  columns_final_argmax_in_upper_band: {fin_ub_hit}/{w_eff}")
        print(f"  columns_path_y_in_upper_band: {path_ub_hit}/{w_eff}")
        png_gt_diag.update(
            {
                "raw_near": raw_near,
                "final_near_am": fin_near_am,
                "path_near": path_near,
                "fin_ub_hit": fin_ub_hit,
            }
        )

    def emit_column_sample(stage: str, ys: np.ndarray, cs: np.ndarray) -> None:
        n = min(int(args.max_column_lines), w_eff)
        print(f"# sample columns ({stage}, first {n}) col | y | conf")
        for col in range(n):
            yy = ys[col] if col < ys.size else float("nan")
            cc = cs[col] if col < cs.size else float("nan")
            print(f"  {col:4d} | {yy:8.2f} | {cc:8.5f}")

    if raw_cands:
        ry, rc = top1_per_column(raw_cands)
        emit_column_sample("raw top1", ry[:w_eff], rc[:w_eff])
    if fin_cands:
        fy, fc = top1_per_column(fin_cands)
        emit_column_sample("final top1", fy[:w_eff], fc[:w_eff])

    pruned_gt_near = 0
    mean_diff: Optional[float] = None
    gt_near_exists = 0
    path_gt_near_pick = 0
    final_path_upper_band_columns = 0
    gt_near_exists_but_path_upper = 0

    if gt_by_col and fin_cands:
        raw_total, raw_mean, raw_min, raw_max = _cand_stats(raw_cands, w_eff)
        fin_total, fin_mean, fin_min, fin_max = _cand_stats(fin_cands, w_eff)
        print(f"total_columns={w_eff}")
        print(f"total_candidates_raw={raw_total}")
        print(f"total_candidates_final={fin_total}")
        print(f"candidates_per_column_raw_min_mean_max={raw_min},{raw_mean:.4f},{raw_max}")
        print(f"candidates_per_column_final_min_mean_max={fin_min},{fin_mean:.4f},{fin_max}")

        nearest_raw = _nearest_dist_stats(raw_cands, gt_by_col, w_eff)
        nearest_fin = _nearest_dist_stats(fin_cands, gt_by_col, w_eff)
        for k, v in nearest_fin.items():
            print(f"{k}={v}")
        if nearest_raw.get("candidate_gt_near_recall_px5") is not None:
            print(f"raw_candidate_gt_near_recall_px5={nearest_raw.get('candidate_gt_near_recall_px5')}")

        fc_near_y, fc_near_c = max_conf_near_gt(fin_cands, gt_by_col, thr, h)
        fy, fc_top = top1_per_column(fin_cands)

        # 상단 밴드 내 후보들 중 최대 신뢰도 (열 단위)
        ub_bound = int(math.ceil(float(h) * float(args.upper_frac)))
        ub_bound = max(1, min(h, ub_bound))

        def max_conf_in_band(col: int, cands: List[dict], y_hi: int) -> Tuple[float, float]:
            best_c = float("-inf")
            best_y = float("nan")
            for c in cands:
                try:
                    yy = float(c.get("y", float("nan")))
                    cf = float(c.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(yy) and 0 <= yy < y_hi and cf > best_c:
                    best_c = cf
                    best_y = yy
            if best_c == float("-inf"):
                return float("nan"), float("nan")
            return best_y, best_c

        ub_conf_list: List[float] = []
        gt_near_conf_list: List[float] = []
        diff_list: List[float] = []
        path_ub_cols = 0
        path_ub_despite_gt_near = 0

        raw_near_exists = 0
        fin_near_exists = 0

        if raw_cands:
            _, rc_raw_near = max_conf_near_gt(raw_cands, gt_by_col, thr, h)
            _, rc_fin_near = max_conf_near_gt(fin_cands, gt_by_col, thr, h)
            for col in range(w_eff):
                if math.isfinite(rc_raw_near[col]) if col < rc_raw_near.size else False:
                    raw_near_exists += 1
                if math.isfinite(rc_fin_near[col]) if col < rc_fin_near.size else False:
                    fin_near_exists += 1
                r_ok = col < rc_raw_near.size and math.isfinite(rc_raw_near[col])
                f_ok = col < rc_fin_near.size and math.isfinite(rc_fin_near[col])
                if r_ok and not f_ok:
                    pruned_gt_near += 1

        for col in range(w_eff):
            gty = gt_by_col.get(col)
            if gty is None:
                continue
            lst = fin_cands.get(col, [])
            if not lst:
                continue
            _, ubc = max_conf_in_band(col, lst, ub_bound)
            gn_c = fc_near_c[col] if col < fc_near_c.size else float("nan")

            if math.isfinite(ubc):
                ub_conf_list.append(ubc)
            if math.isfinite(gn_c):
                gt_near_exists += 1
                gt_near_conf_list.append(gn_c)
                if math.isfinite(ubc):
                    diff_list.append(ubc - gn_c)

            py = path_y[col] if col < path_y.size else float("nan")
            if math.isfinite(py) and py < ub_bound:
                path_ub_cols += 1
                if math.isfinite(gn_c) and abs(py - float(gty)) > thr:
                    path_ub_despite_gt_near += 1

        mean_ub = mean_finite(np.asarray(ub_conf_list, dtype=np.float64))
        mean_gt = mean_finite(np.asarray(gt_near_conf_list, dtype=np.float64))
        mean_diff = mean_finite(np.asarray(diff_list, dtype=np.float64))

        # GT-near 존재 열 수 (3/5/10)
        _, fin_near3 = max_conf_near_gt(fin_cands, gt_by_col, 3.0, h)
        _, fin_near5 = max_conf_near_gt(fin_cands, gt_by_col, 5.0, h)
        _, fin_near10 = max_conf_near_gt(fin_cands, gt_by_col, 10.0, h)
        gt_near3_cols = int(np.sum(np.isfinite(fin_near3)))
        gt_near5_cols = int(np.sum(np.isfinite(fin_near5)))
        gt_near10_cols = int(np.sum(np.isfinite(fin_near10)))
        print(f"columns_with_gt_near_candidate_px3={gt_near3_cols}")
        print(f"columns_with_gt_near_candidate_px5={gt_near5_cols}")
        print(f"columns_with_gt_near_candidate_px10={gt_near10_cols}")

        print(f"gt_near_px={thr}")
        print(f"columns_with_gt_near_candidate_final={gt_near_exists}")
        print(f"mean_upper_band_max_conf={mean_ub}")
        print(f"mean_gt_near_max_conf={mean_gt}")
        print(f"mean_upper_band_conf_minus_gt_near_conf={mean_diff}")
        print(f"final_path_upper_band_columns={path_ub_cols}")
        print(f"gt_near_exists_but_path_upper_not_near_gt={path_ub_despite_gt_near}")
        final_path_upper_band_columns = int(path_ub_cols)
        gt_near_exists_but_path_upper = int(path_ub_despite_gt_near)

        if raw_cands:
            print(f"columns_gt_near_raw={raw_near_exists}")
            print(f"columns_gt_near_final={fin_near_exists}")
            print(f"columns_gt_near_raw_but_not_final_pruned={pruned_gt_near}")

        # DP 후보 vs top1 비교 (신뢰도 역전)
        better_near_than_top1 = 0
        top1_not_near = 0
        for col in range(w_eff):
            gty = gt_by_col.get(col)
            if gty is None:
                continue
            gn_c = fc_near_c[col] if col < fc_near_c.size else float("nan")
            t_c = fc_top[col] if col < fc_top.size else float("nan")
            t_y = fy[col] if col < fy.size else float("nan")
            if not math.isfinite(gn_c):
                continue
            if math.isfinite(t_y) and abs(t_y - float(gty)) > thr:
                top1_not_near += 1
            if math.isfinite(t_c) and gn_c > t_c + 1e-9:
                better_near_than_top1 += 1
            py = path_y[col] if col < path_y.size else float("nan")
            if math.isfinite(py) and abs(py - float(gty)) <= thr:
                path_gt_near_pick += 1
        print(f"columns_gt_near_non_top1_higher_conf_than_top1={better_near_than_top1}")
        print(f"columns_top1_not_gt_near={top1_not_near}")
        print(f"final_path_gt_near_pick_columns_px{int(thr)}={path_gt_near_pick}")

        # raw vs final 강화/소실 비교
        raw_to_final_upper_strengthened = 0
        final_upper_top1_cols = 0
        for col in range(w_eff):
            gty = gt_by_col.get(col)
            if gty is None:
                continue
            raw_lst = raw_cands.get(col, [])
            fin_lst = fin_cands.get(col, [])
            raw_top = max(raw_lst, key=lambda c: float(c.get("confidence", 0.0))) if raw_lst else None
            fin_top = max(fin_lst, key=lambda c: float(c.get("confidence", 0.0))) if fin_lst else None
            if fin_top is not None and float(fin_top.get("y", 0.0)) < ub_bound:
                final_upper_top1_cols += 1
            if raw_top is None or fin_top is None:
                continue
            raw_is_upper = float(raw_top.get("y", 0.0)) < ub_bound
            fin_is_upper = float(fin_top.get("y", 0.0)) < ub_bound
            raw_conf = float(raw_top.get("confidence", 0.0))
            fin_conf = float(fin_top.get("confidence", 0.0))
            if fin_is_upper and ((not raw_is_upper) or (fin_conf > raw_conf + 1e-9)):
                raw_to_final_upper_strengthened += 1
        print(f"final_top1_upper_band_columns={final_upper_top1_cols}")
        print(f"raw_to_final_upper_band_strengthened_columns={raw_to_final_upper_strengthened}")

    # 판정 힌트
    print("\n=== 판정 힌트 ===")
    if not gt_by_col:
        print("(GT 없음) PNG 상단비율·raw→final 이동만 참고.")
    elif not fin_cands:
        print(
            "후보 JSON 없음 → 위 PNG/GT 프록시 참고; "
            "신뢰도·진짜 후보 pruning 여부는 `--dump-candidates-json` 재실행 권장.",
        )
        rn = int(png_gt_diag.get("raw_near", -1))
        fn = int(png_gt_diag.get("final_near_am", -1))
        fu = int(png_gt_diag.get("fin_ub_hit", -1))
        if rn >= 0 and fn >= 0 and rn > fn + max(10, int(0.05 * w_eff)):
            print(
                "맵 argmax 기준 raw 대비 final에서 GT 근접 열 감소 → "
                "PRUNING_REMOVES_GT_NEAR_CANDIDATE (후보 단위 확인 필요) 신호.",
            )
        if fu >= 0 and fu > int(0.85 * w_eff) and fn >= 0 and fn < int(0.25 * w_eff):
            print(
                "final 맵 argmax 상단 고착 + GT와 불일치 다수 → "
                "GT_NEAR_CANDIDATE_MISSING 또는 MAP_CONFIDENCE_ARTIFACT 가능.",
            )
    elif pruned_gt_near > max(5, int(0.02 * w_eff)):
        print("PRUNING_REMOVES_GT_NEAR_CANDIDATE 가능성: raw 대비 final에서 GT 근접 후보 소실 열이 많음.")
    elif mean_diff is not None and mean_diff > 0.02:
        print("UPPER_BAND_CONFIDENCE_DOMINATES_GT_NEAR 가능성: 상단밴드 max conf가 GT 근접 max conf보다 평균적으로 큼.")
    elif mean_diff is not None and mean_diff <= 0 and path_ub_frac > 0.5:
        print("DP_COST_PREFERS_UPPER_BAND 가능성: GT 근접 conf가 상단과 비슷·더 나은데도 path가 상단에 많음 (후속 DP 코스트 확인).")
    elif gt_near_exists < max(5, int(0.02 * w_eff)):
        print("GT_NEAR_CANDIDATE_MISSING 가능성: final 후보 중 GT 근접이 거의 없음.")
    else:
        print("복합 또는 추가 분석 필요.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
