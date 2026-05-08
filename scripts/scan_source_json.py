"""
Step 3: 원본 JSON 전수 조사 → all_samples.csv 생성.

xrd_digitizer_v1_master_spec.md §3 준수.
- source_root 아래 모든 *.json을 스캔
- 16개 컬럼 고정 (§3.3)
- shape diversity feature 계산 (§3.6)
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# §3.2: JSON key 고정
X_KEY = "two_theta_values"
Y_KEY = "intensities"

# §3.6: Savitzky-Golay parameters
SG_WINDOW = 7
SG_POLYORDER = 2

OUTPUT_COLUMNS: List[str] = [
    "sample_id",
    "source_json_path",
    "num_points",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "y_dynamic_range",
    "dynamic_range_log",
    "peak_count_est",
    "peak_height_ratio",
    "mean_peak_spacing_norm",
    "tail_energy_ratio",
    "fwhm_mean_est",
    "family_id_raw",
    "is_valid",
    "invalid_reason",
]


def _empty_row_schema() -> Dict[str, Any]:
    return {
        "sample_id": "",
        "source_json_path": "",
        "num_points": 0,
        "x_min": np.nan,
        "x_max": np.nan,
        "y_min": np.nan,
        "y_max": np.nan,
        "y_dynamic_range": np.nan,
        "dynamic_range_log": np.nan,
        "peak_count_est": 0,
        "peak_height_ratio": 0.0,
        "mean_peak_spacing_norm": 0.0,
        "tail_energy_ratio": 0.0,
        "fwhm_mean_est": np.nan,
        "family_id_raw": "",
        "is_valid": False,
        "invalid_reason": "",
    }


def _join_reasons(reasons: Iterable[str]) -> str:
    return ";".join([r for r in reasons if r])


def make_sample_id_with_collision_guard(
    json_path: str, used_ids: Optional[set] = None
) -> str:
    """§3.4: 기본 stem, 충돌 시 4자리 해시 추가."""
    stem = Path(json_path).stem
    if used_ids is None:
        return stem
    if stem not in used_ids:
        return stem
    digest = hashlib.sha1(str(json_path).encode("utf-8")).hexdigest()[:4]
    candidate = f"{stem}_{digest}"
    while candidate in used_ids:
        digest = hashlib.sha1(f"{json_path}:{digest}".encode("utf-8")).hexdigest()[:4]
        candidate = f"{stem}_{digest}"
    return candidate


def load_json(path: str) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_xy(x: Any, y: Any) -> Tuple[bool, str]:
    """§3.5: invalid 규칙 검증."""
    reasons: List[str] = []

    if x is None:
        reasons.append("missing_x")
    if y is None:
        reasons.append("missing_y")
    if reasons:
        return False, _join_reasons(reasons)

    if not isinstance(x, (list, tuple, np.ndarray)):
        reasons.append("x_not_array_like")
    if not isinstance(y, (list, tuple, np.ndarray)):
        reasons.append("y_not_array_like")
    if reasons:
        return False, _join_reasons(reasons)

    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
    except (ValueError, TypeError):
        reasons.append("non_numeric_conversion_failed")
        return False, _join_reasons(reasons)

    if x_arr.ndim != 1:
        reasons.append(f"x_ndim_not_1:ndim={x_arr.ndim}")
    if y_arr.ndim != 1:
        reasons.append(f"y_ndim_not_1:ndim={y_arr.ndim}")
    if reasons:
        return False, _join_reasons(reasons)

    n = len(x_arr)
    m = len(y_arr)
    if n != m:
        reasons.append(f"length_mismatch:x={n},y={m}")
        return False, _join_reasons(reasons)

    if n < 50:
        reasons.append("too_few_points(<50)")

    if n >= 2 and not np.all(np.diff(x_arr) > 0):
        reasons.append("x_not_strictly_increasing")

    if not np.all(np.isfinite(x_arr)):
        reasons.append("x_contains_nan_or_inf")
    if not np.all(np.isfinite(y_arr)):
        reasons.append("y_contains_nan_or_inf")

    if n > 0 and np.all(np.isfinite(y_arr)):
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        if float(y_max - y_min) <= 0.0:
            reasons.append("y_dynamic_range<=0")

    return (len(reasons) == 0), _join_reasons(reasons)


def estimate_peaks(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """§3.6: SG smoothing + prominence 기반 peak 추정."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dyn_range = float(y_max - y_min)

    if len(y) < SG_WINDOW:
        y_smooth = y.astype(float, copy=False)
        residual = y - y_smooth
        sigma_local = float(np.std(residual)) if len(residual) > 1 else 0.0
        return {
            "peak_indices": np.array([], dtype=int),
            "peak_x": np.array([], dtype=float),
            "peak_y": np.array([], dtype=float),
            "peak_count_est": 0,
            "prominence_used": float("nan"),
            "sigma_local": sigma_local,
            "y_smooth": y_smooth,
        }

    y_smooth = savgol_filter(y, window_length=SG_WINDOW, polyorder=SG_POLYORDER, mode="interp")
    residual = y - y_smooth
    sigma_local = float(np.std(residual)) if len(residual) > 1 else 0.0

    local_noise_floor = max(3.0 * sigma_local, 0.01 * dyn_range)
    prominence_used = max(0.07 * dyn_range, local_noise_floor)

    peaks, _ = find_peaks(y_smooth, prominence=max(prominence_used, 1e-12))
    peak_y = y[peaks] if len(peaks) else np.array([], dtype=float)
    peak_x = x[peaks] if len(peaks) else np.array([], dtype=float)

    return {
        "peak_indices": np.asarray(peaks, dtype=int),
        "peak_x": np.asarray(peak_x, dtype=float),
        "peak_y": np.asarray(peak_y, dtype=float),
        "peak_count_est": int(len(peaks)),
        "prominence_used": float(prominence_used),
        "sigma_local": float(sigma_local),
        "y_smooth": np.asarray(y_smooth, dtype=float),
    }


def _linear_x_at_y(x: np.ndarray, y: np.ndarray, idx0: int, idx1: int, target_y: float) -> float:
    x0, y0 = float(x[idx0]), float(y[idx0])
    x1, y1 = float(x[idx1]), float(y[idx1])
    dy = y1 - y0
    if np.isclose(dy, 0.0):
        return float(x0)
    t = (target_y - y0) / dy
    return float(x0 + t * (x1 - x0))


def estimate_fwhm_mean_simple(x: np.ndarray, y: np.ndarray, peak_idx: int) -> Optional[float]:
    """§3.6 fwhm_mean_est: 각 peak의 half-max width 추정."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3 or peak_idx < 1 or peak_idx > n - 2:
        return None

    peak_y = float(y[peak_idx])
    baseline = float(min(y[0], y[-1]))
    half = baseline + 0.5 * (peak_y - baseline)
    if not np.isfinite(half) or peak_y <= baseline:
        return None

    left = peak_idx
    while left > 0 and y[left] > half:
        left -= 1
    if left <= 0 or y[left] > half:
        return None

    right = peak_idx
    while right < n - 1 and y[right] > half:
        right += 1
    if right >= n - 1 or y[right] > half:
        return None

    x_left = _linear_x_at_y(x, y, left, left + 1, half)
    x_right = _linear_x_at_y(x, y, right - 1, right, half)
    width = float(x_right - x_left)
    if width <= 0.0 or not np.isfinite(width):
        return None
    return width


def compute_shape_features(x: np.ndarray, y: np.ndarray, peak_info: Dict[str, Any]) -> Dict[str, Any]:
    """§3.6: shape diversity features 전체 계산."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dyn = float(y_max - y_min)
    dynamic_range_log = float(np.log10(dyn + 1.0)) if np.isfinite(dyn) else float("nan")

    peak_indices = np.asarray(peak_info.get("peak_indices", []), dtype=int)
    peak_y = np.asarray(peak_info.get("peak_y", []), dtype=float)

    if peak_y.size == 0:
        peak_height_ratio = 0.0
    else:
        mean_peak_height = float(np.mean(peak_y))
        max_peak_height = float(np.max(peak_y))
        if np.isclose(mean_peak_height, 0.0) or not np.isfinite(mean_peak_height):
            peak_height_ratio = 0.0
        else:
            peak_height_ratio = float(max_peak_height / mean_peak_height)

    if peak_indices.size < 2:
        mean_peak_spacing_norm = 0.0
    else:
        peak_x_sorted = np.sort(x[peak_indices])
        x_span = float(np.max(x) - np.min(x))
        if x_span <= 0.0 or not np.isfinite(x_span):
            mean_peak_spacing_norm = 0.0
        else:
            mean_peak_spacing_norm = float(np.mean(np.diff(peak_x_sorted)) / x_span)

    total = float(np.sum(y))
    if not np.isfinite(total) or np.isclose(total, 0.0):
        tail_energy_ratio = 0.0
    else:
        x_min_v = float(np.min(x))
        x_max_v = float(np.max(x))
        if not np.isfinite(x_min_v) or not np.isfinite(x_max_v) or x_max_v <= x_min_v:
            tail_energy_ratio = 0.0
        else:
            cutoff = x_min_v + 0.8 * (x_max_v - x_min_v)
            tail_mask = x >= cutoff
            tail_energy_ratio = float(np.sum(y[tail_mask]) / total)

    fwhm_mean_est = float("nan")
    try:
        y_smooth = np.asarray(peak_info.get("y_smooth", y), dtype=float)
        peak_idx_arr = np.asarray(peak_info.get("peak_indices", []), dtype=int)
        if peak_idx_arr.size > 0:
            heights = y_smooth[peak_idx_arr]
            order = np.argsort(heights)[::-1]
            top = order[: min(5, len(order))]
            widths: List[float] = []
            for k in top:
                idx = int(peak_idx_arr[int(k)])
                w = estimate_fwhm_mean_simple(x, y, idx)
                if w is not None and np.isfinite(w) and w > 0.0:
                    widths.append(w)
            if widths:
                fwhm_mean_est = float(np.mean(widths))
    except Exception:
        fwhm_mean_est = float("nan")

    return {
        "dynamic_range_log": dynamic_range_log,
        "peak_height_ratio": float(peak_height_ratio),
        "mean_peak_spacing_norm": float(mean_peak_spacing_norm),
        "tail_energy_ratio": float(tail_energy_ratio),
        "fwhm_mean_est": float(fwhm_mean_est) if np.isfinite(fwhm_mean_est) else np.nan,
    }


def build_row(path: str, data: dict, used_sample_ids: Optional[set] = None) -> dict:
    """§3.7: 한 JSON 파일에 대한 메타데이터 행 생성."""
    row = _empty_row_schema()
    row["source_json_path"] = str(path)
    row["sample_id"] = Path(path).stem
    row["family_id_raw"] = row["sample_id"]

    reasons: List[str] = []

    try:
        x_obj = data.get(X_KEY, None)
        y_obj = data.get(Y_KEY, None)
        if X_KEY not in data:
            reasons.append(f"missing_key:{X_KEY}")
        if Y_KEY not in data:
            reasons.append(f"missing_key:{Y_KEY}")
        if reasons:
            row["is_valid"] = False
            row["invalid_reason"] = _join_reasons(reasons)
            return row

        ok, why = validate_xy(x_obj, y_obj)
        if not ok:
            try:
                x_dbg = np.asarray(data[X_KEY], dtype=float)
                y_dbg = np.asarray(data[Y_KEY], dtype=float)
                row["num_points"] = int(min(len(x_dbg), len(y_dbg)))
                if len(x_dbg) > 0:
                    row["x_min"] = float(np.min(x_dbg))
                    row["x_max"] = float(np.max(x_dbg))
                if len(y_dbg) > 0:
                    row["y_min"] = float(np.min(y_dbg))
                    row["y_max"] = float(np.max(y_dbg))
                    dyn_dbg = float(row["y_max"] - row["y_min"])
                    row["y_dynamic_range"] = dyn_dbg
                    if np.isfinite(dyn_dbg):
                        row["dynamic_range_log"] = float(np.log10(dyn_dbg + 1.0))
            except Exception:
                pass
            row["is_valid"] = False
            row["invalid_reason"] = why
            return row

        x = np.asarray(data[X_KEY], dtype=float)
        y = np.asarray(data[Y_KEY], dtype=float)

        row["num_points"] = int(len(x))
        row["x_min"] = float(np.min(x))
        row["x_max"] = float(np.max(x))
        row["y_min"] = float(np.min(y))
        row["y_max"] = float(np.max(y))
        dyn = float(row["y_max"] - row["y_min"])
        row["y_dynamic_range"] = dyn
        row["dynamic_range_log"] = float(np.log10(dyn + 1.0))

        peak_info = estimate_peaks(x, y)
        row["peak_count_est"] = int(peak_info["peak_count_est"])

        feats = compute_shape_features(x, y, peak_info)
        row["dynamic_range_log"] = float(feats["dynamic_range_log"])
        row["peak_height_ratio"] = float(feats["peak_height_ratio"])
        row["mean_peak_spacing_norm"] = float(feats["mean_peak_spacing_norm"])
        row["tail_energy_ratio"] = float(feats["tail_energy_ratio"])
        row["fwhm_mean_est"] = feats["fwhm_mean_est"]

        row["is_valid"] = True
        row["invalid_reason"] = ""
        return row

    except Exception as exc:
        row["is_valid"] = False
        row["invalid_reason"] = f"exception:{type(exc).__name__}:{str(exc)}"
        return row


def scan_all_json_files(source_root: str) -> pd.DataFrame:
    """source_root 아래 모든 *.json 파일을 스캔하여 DataFrame으로 반환."""
    raw_path = Path(source_root)
    json_files = sorted(raw_path.glob("*.json"))
    total = len(json_files)
    print(f"[INFO] Found {total} JSON files in {source_root}")

    records: List[Dict[str, Any]] = []
    used_ids: set = set()

    for i, file_path in enumerate(json_files):
        path_str = str(file_path)

        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  scanning {i + 1}/{total} ...")

        base_row = _empty_row_schema()
        base_row["source_json_path"] = path_str
        base_row["sample_id"] = file_path.stem
        base_row["family_id_raw"] = file_path.stem

        try:
            obj = load_json(path_str)
        except json.JSONDecodeError as exc:
            row = dict(base_row)
            row["is_valid"] = False
            row["invalid_reason"] = f"json_decode_error:{exc.msg}"
            records.append(row)
            continue
        except Exception as exc:
            row = dict(base_row)
            row["is_valid"] = False
            row["invalid_reason"] = f"exception:{type(exc).__name__}:{str(exc)}"
            records.append(row)
            continue

        row = build_row(path_str, obj, used_sample_ids=used_ids)
        final_id = make_sample_id_with_collision_guard(path_str, used_ids)
        row["sample_id"] = final_id
        row["family_id_raw"] = final_id
        used_ids.add(final_id)
        records.append(row)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: scan all source JSON → all_samples.csv")
    parser.add_argument(
        "--source_root",
        type=str,
        default=r"c:\xrd_digitizer_v1\data\source_json",
        help="원본 JSON 폴더 경로",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"c:\xrd_digitizer_v1\data\metadata\all_samples.csv",
        help="출력 CSV 경로",
    )
    args = parser.parse_args()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = scan_all_json_files(args.source_root)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    valid_count = int(df["is_valid"].sum())
    invalid_count = len(df) - valid_count
    print(f"\n[DONE] Saved: {args.output_csv}")
    print(f"  Total: {len(df)}, Valid: {valid_count}, Invalid: {invalid_count}")
    print(df.head())


if __name__ == "__main__":
    main()
