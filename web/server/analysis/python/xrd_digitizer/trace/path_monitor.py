from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _odd(v: int, *, min_value: int = 5) -> int:
    x = max(int(min_value), int(v))
    if x % 2 == 0:
        x += 1
    return x


def _as_float_array(y: Sequence[Optional[float]]) -> np.ndarray:
    return np.asarray([float(v) if v is not None else np.nan for v in y], dtype=np.float64)


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    w = _odd(window)
    n = int(x.size)
    out = np.full(n, np.nan, dtype=np.float64)
    r = w // 2
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def _rolling_iqr(x: np.ndarray, window: int) -> np.ndarray:
    w = _odd(window)
    n = int(x.size)
    out = np.full(n, np.nan, dtype=np.float64)
    r = w // 2
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 4:
            q25 = float(np.percentile(seg, 25))
            q75 = float(np.percentile(seg, 75))
            out[i] = q75 - q25
        elif seg.size:
            out[i] = float(np.std(seg))
    return out


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig < 1e-6:
        sig = float(np.std(x))
    return float(sig) if np.isfinite(sig) and sig > 1e-9 else 1.0


def _runs_from_mask(mask: np.ndarray, *, min_run: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    n = int(mask.size)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j < n and bool(mask[j]):
            j += 1
        if (j - i) >= int(min_run):
            runs.append((int(i), int(j - 1)))
        i = j
    return runs


def _clusters_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    clusters: List[Tuple[int, int]] = []
    n = int(mask.size)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j < n and bool(mask[j]):
            j += 1
        clusters.append((int(i), int(j - 1)))
        i = j
    return clusters


def top1_y_from_final_candidates(final_cands: Dict[int, List[Dict[str, Any]]], n: int) -> np.ndarray:
    y = np.full(int(n), np.nan, dtype=np.float64)
    for col, cands in final_cands.items():
        if col < 0 or col >= n:
            continue
        if not cands:
            continue
        best = None
        best_conf = -1e9
        for c in cands:
            conf = float(c.get("confidence", 0.0))
            if conf > best_conf:
                best_conf = conf
                best = c
        if best is not None and "y" in best:
            y[int(col)] = float(best["y"])
    return y


def selected_source_from_final_candidates(
    final_cands: Dict[int, List[Dict[str, Any]]], selected_y: np.ndarray
) -> List[Optional[str]]:
    n = int(selected_y.size)
    out: List[Optional[str]] = [None] * n
    for col in range(n):
        if col not in final_cands:
            continue
        yv = selected_y[col]
        if not np.isfinite(yv):
            continue
        yy = int(round(float(yv)))
        for c in final_cands[col]:
            if int(c.get("y", -999999)) == yy:
                out[col] = str(c.get("source")) if c.get("source") is not None else None
                break
    return out


def compute_selected_path_monitor(
    selected_y: Sequence[Optional[float]] | np.ndarray,
    *,
    top1_y: Optional[Sequence[Optional[float]] | np.ndarray] = None,
    selected_source: Optional[Sequence[Optional[str]]] = None,
    roi_h: Optional[int] = None,
    window: int = 51,
    min_run: int = 12,
) -> Dict[str, Any]:
    """Runtime-safe diagnostics only.

    Must NOT use source_numeric/GT/expected_y. Only uses selected_y (+ optional top1/source labels).
    """
    w = _odd(int(window), min_value=9)
    min_run = int(max(3, min_run))

    y = selected_y if isinstance(selected_y, np.ndarray) else _as_float_array(selected_y)
    n = int(y.size)
    warnings: List[str] = []

    if n == 0:
        return {
            "enabled": True,
            "runtime_safe": True,
            "uses_source_numeric": False,
            "uses_gt": False,
            "point_count": 0,
            "warnings": ["empty selected_y"],
        }

    # --- 1) bottom_branch_persistence_proxy ---
    med = _rolling_median(y, w)
    iqr = _rolling_iqr(y, w)
    # "bottom branch" proxy:
    # - local: y much larger than local center (toward large-y)
    # - global: y is in bottom band of ROI (if roi_h is known)
    thr_local = med + 1.0 * np.nan_to_num(iqr, nan=0.0)
    bottom_mask = np.isfinite(y) & np.isfinite(thr_local) & (y > thr_local)
    if roi_h is not None and int(roi_h) > 0:
        # runtime-safe absolute band; avoids missing long, stable bottom lock-in when local IQR is wide.
        bottom_mask |= np.isfinite(y) & (y >= 0.85 * float(int(roi_h)))
    bottom_runs = _runs_from_mask(bottom_mask, min_run=min_run)
    longest_bottom = max([b - a + 1 for a, b in bottom_runs], default=0)
    bottom_score = float(longest_bottom / max(1, n))

    bottom_obj = {
        "window": int(w),
        "min_run": int(min_run),
        "bottom_band_threshold_y": (0.85 * float(int(roi_h))) if roi_h is not None else None,
        "longest_bottom_branch_run_len": int(longest_bottom),
        "bottom_branch_run_count": int(len(bottom_runs)),
        "bottom_branch_run_ranges": [[int(a), int(b)] for a, b in bottom_runs],
        "bottom_branch_score": float(bottom_score),
    }

    # --- 2) local_trend_deviation_run ---
    trend = med  # median trend for robustness/speed
    resid = y - trend
    # rolling robust sigma via rolling MAD approximation (use global if window too sparse)
    sig_global = _robust_sigma(resid[np.isfinite(resid)])
    # local sigma: approximate with rolling IQR of residual / 1.349
    resid_iqr = _rolling_iqr(resid, w)
    sig_local = np.nan_to_num(resid_iqr / 1.349, nan=sig_global, posinf=sig_global, neginf=sig_global)
    sig_local = np.clip(sig_local, 1.0, None)
    z = resid / sig_local
    dev_mask = np.isfinite(z) & (np.abs(z) >= 3.0)
    dev_runs = _runs_from_mask(dev_mask, min_run=min_run)
    longest_dev = max([b - a + 1 for a, b in dev_runs], default=0)
    max_z = float(np.nanmax(np.abs(z))) if np.any(np.isfinite(z)) else None

    dev_obj = {
        "window": int(w),
        "min_run": int(min_run),
        "z_threshold": 3.0,
        "longest_trend_deviation_run_len": int(longest_dev),
        "trend_deviation_run_count": int(len(dev_runs)),
        "trend_deviation_run_ranges": [[int(a), int(b)] for a, b in dev_runs],
        "max_trend_deviation_z": max_z,
    }

    # --- 3) slope_jump_cluster_score ---
    dy = np.diff(y)
    ddy = np.diff(dy)
    dy_abs = np.abs(dy[np.isfinite(dy)])
    ddy_abs = np.abs(ddy[np.isfinite(ddy)])
    dy_max = float(np.nanmax(np.abs(dy))) if np.any(np.isfinite(dy)) else None
    ddy_max = float(np.nanmax(np.abs(ddy))) if np.any(np.isfinite(ddy)) else None
    dy_thr = float(np.median(dy_abs) + 3.0 * (1.4826 * np.median(np.abs(dy_abs - np.median(dy_abs))))) if dy_abs.size else 0.0
    ddy_thr = float(np.median(ddy_abs) + 3.0 * (1.4826 * np.median(np.abs(ddy_abs - np.median(ddy_abs))))) if ddy_abs.size else 0.0
    if not np.isfinite(dy_thr) or dy_thr < 1.0:
        dy_thr = 8.0
    if not np.isfinite(ddy_thr) or ddy_thr < 1.0:
        ddy_thr = 12.0

    jump_mask = np.zeros(n, dtype=bool)
    if dy.size:
        jump_mask[1:] |= np.isfinite(dy) & (np.abs(dy) >= dy_thr)
    if ddy.size:
        jump_mask[2:] |= np.isfinite(ddy) & (np.abs(ddy) >= ddy_thr)
    clusters = _clusters_from_mask(jump_mask)
    cluster_lens = [b - a + 1 for a, b in clusters]
    slope_jump_obj = {
        "dy_abs_max": dy_max,
        "ddy_abs_max": ddy_max,
        "dy_threshold": float(dy_thr),
        "ddy_threshold": float(ddy_thr),
        "slope_jump_count": int(np.sum(jump_mask)),
        "slope_jump_cluster_count": int(len(clusters)),
        "slope_jump_cluster_ranges": [[int(a), int(b)] for a, b in clusters],
        "slope_jump_cluster_score": float((max(cluster_lens) if cluster_lens else 0) / max(1, n)),
    }

    # --- 4) top1_selected_divergence_rate ---
    top1_obj: Dict[str, Any] = {"available": False}
    if top1_y is not None:
        t = top1_y if isinstance(top1_y, np.ndarray) else _as_float_array(top1_y)
        if int(t.size) != n:
            warnings.append(f"top1_y size mismatch: {int(t.size)} != {n}")
        else:
            diff = np.abs(t - y)
            m = np.isfinite(diff)
            mean_abs = float(np.mean(diff[m])) if np.any(m) else None
            rate20 = float(np.mean(diff[m] >= 20.0)) if np.any(m) else None
            rate50 = float(np.mean(diff[m] >= 50.0)) if np.any(m) else None
            div_mask = m & (diff >= 50.0)
            div_runs = _runs_from_mask(div_mask, min_run=min_run)
            top1_obj = {
                "available": True,
                "mean_abs_diff": mean_abs,
                "divergence_rate_20px": rate20,
                "divergence_rate_50px": rate50,
                "divergence_ranges_50px": [[int(a), int(b)] for a, b in div_runs],
                "min_run": int(min_run),
            }

    # --- 5) selected_source_dependency_ratio ---
    src_obj: Dict[str, Any] = {"available": False}
    if selected_source is not None:
        if len(selected_source) != n:
            warnings.append(f"selected_source size mismatch: {len(selected_source)} != {n}")
        else:
            counts: Dict[str, int] = {}
            total = 0
            exp_mask = np.zeros(n, dtype=bool)
            exp_sources = {"frag_reentry_keep", "shell_retained", "frag_bridge"}
            for i, s in enumerate(selected_source):
                if s is None:
                    continue
                ss = str(s)
                counts[ss] = counts.get(ss, 0) + 1
                total += 1
                if ss in exp_sources:
                    exp_mask[i] = True
            ratios = {k: float(v / max(total, 1)) for k, v in counts.items()}
            exp_ratio = float(np.sum(exp_mask) / max(total, 1))
            exp_runs = _runs_from_mask(exp_mask, min_run=min_run)
            longest_exp = max([b - a + 1 for a, b in exp_runs], default=0)
            src_obj = {
                "available": True,
                "selected_source_counts": counts,
                "selected_source_ratios": ratios,
                "experimental_source_ratio": exp_ratio,
                "longest_experimental_source_run_len": int(longest_exp),
                "experimental_source_run_ranges": [[int(a), int(b)] for a, b in exp_runs],
                "experimental_sources": sorted(list(exp_sources)),
            }

    # common header
    out: Dict[str, Any] = {
        "enabled": True,
        "runtime_safe": True,
        "uses_source_numeric": False,
        "uses_gt": False,
        "point_count": int(n),
        "roi_h": int(roi_h) if roi_h is not None else None,
        "bottom_branch_persistence_proxy": bottom_obj,
        "local_trend_deviation_run": dev_obj,
        "slope_jump_cluster_score": slope_jump_obj,
        "top1_selected_divergence_rate": top1_obj,
        "selected_source_dependency_ratio": src_obj,
        "warnings": warnings,
    }
    return out
