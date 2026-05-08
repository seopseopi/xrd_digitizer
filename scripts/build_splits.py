"""
Step 8: train/val/test split CSV 생성 (family 단위 leakage 방지).

xrd_digitizer_v1_master_spec.md §8 준수.
- surrogate family_id: pc_{low|mid|high}__pp_{q1}_{q2}_{q3}__dr_{low|mid|high}
- 80/10/10 비율
- 같은 family가 둘 이상의 split에 없어야 함
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

X_KEY = "two_theta_values"
Y_KEY = "intensities"
DEFAULT_SEED = 42
_split_seed = DEFAULT_SEED


def load_subset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"sample_id", "source_json_path", "peak_count_est", "tail_energy_ratio", "dynamic_range_log"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input_csv missing columns: {missing}")
    return df


def _bin_low_mid_high(series: pd.Series, n_bins: int = 3) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    if int(v.notna().sum()) == 0:
        return pd.Series(["mid"] * len(series), index=series.index, dtype=object)
    cats = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    out: List[str] = []
    for x in cats.tolist():
        if x is None or (isinstance(x, float) and np.isnan(x)):
            out.append("mid")
        elif int(x) <= 0:
            out.append("low")
        elif int(x) == 1:
            out.append("mid")
        else:
            out.append("high")
    return pd.Series(out, index=series.index, dtype=object)


def _quantize_005(x_norm: float) -> str:
    if not np.isfinite(x_norm):
        return "0.00"
    q = round(float(x_norm) / 0.05) * 0.05
    return f"{float(np.clip(q, 0.0, 1.0)):.2f}"


def _major_peak_position_signature(source_json_path: str, x_min: float, x_max: float) -> str:
    """§8.2: major peak x 정규화 → 상위 3개 0.05 양자화."""
    p = Path(source_json_path)
    if not p.is_file():
        return "0.00_0.00_0.00"
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        x = np.asarray(data.get(X_KEY), dtype=float)
        y = np.asarray(data.get(Y_KEY), dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
            return "0.00_0.00_0.00"
        span = float(np.max(x) - np.min(x))
        if span <= 0:
            return "0.00_0.00_0.00"
        dyn = float(np.max(y) - np.min(y))
        prom = max(1e-9, 0.05 * dyn) if np.isfinite(dyn) else 1e-9
        peaks, _ = find_peaks(y, prominence=prom)
        if peaks.size == 0:
            return "0.00_0.00_0.00"
        order = np.argsort(y[peaks])[::-1][:3]
        top_idx = peaks[order]
        lo = float(x_min) if np.isfinite(x_min) else float(np.min(x))
        hi = float(x_max) if np.isfinite(x_max) else float(np.max(x))
        sp = hi - lo
        if sp <= 0:
            sp = span
            lo = float(np.min(x))
        norms = np.clip((x[top_idx].astype(float) - lo) / sp, 0.0, 1.0)
        vals = [_quantize_005(float(t)) for t in norms.tolist()]
        while len(vals) < 3:
            vals.append("0.00")
        return "_".join(vals[:3])
    except Exception:
        return "0.00_0.00_0.00"


def assign_family_id(df: pd.DataFrame) -> pd.DataFrame:
    """§8.2: surrogate family_id 생성."""
    out = df.copy()
    if "family_id_raw" not in out.columns:
        out["family_id_raw"] = ""

    pc_bin = _bin_low_mid_high(out["peak_count_est"])
    dr_bin = _bin_low_mid_high(out["dynamic_range_log"])

    x_mins = out["x_min"].to_numpy(dtype=float) if "x_min" in out.columns else np.full(len(out), np.nan)
    x_maxs = out["x_max"].to_numpy(dtype=float) if "x_max" in out.columns else np.full(len(out), np.nan)

    fam_ids: List[str] = []
    for pos in range(len(out)):
        row = out.iloc[pos]
        raw = str(row.get("family_id_raw", "") or "").strip()
        if raw and raw.lower() not in ("nan", "none", ""):
            fam_ids.append(raw)
            continue
        xm = float(x_mins[pos]) if np.isfinite(x_mins[pos]) else float("nan")
        xM = float(x_maxs[pos]) if np.isfinite(x_maxs[pos]) else float("nan")
        pp = _major_peak_position_signature(str(row["source_json_path"]), xm, xM)
        fam_ids.append(f"pc_{pc_bin.iloc[pos]}__pp_{pp}__dr_{dr_bin.iloc[pos]}")

    out["family_id"] = fam_ids
    return out


def group_split_by_family(df: pd.DataFrame, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, pd.DataFrame]:
    """§8.3: family 단위 greedy split (leakage 방지)."""
    rng = np.random.default_rng(int(_split_seed))
    n = len(df)
    if n == 0:
        return {"train": df.iloc[0:0].copy(), "val": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}

    r0, r1, r2 = [r / sum(ratios) for r in ratios]
    n_train, n_val = int(n * r0), int(n * r1)
    n_test = n - n_train - n_val
    targets = {"train": n_train, "val": n_val, "test": n_test}

    groups = [(fid, g.copy()) for fid, g in df.groupby("family_id", sort=False)]
    perm = rng.permutation(len(groups))
    groups = [groups[int(i)] for i in perm]

    counts = {"train": 0, "val": 0, "test": 0}
    parts: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for _fid, sub in groups:
        k = len(sub)
        slack = {s: targets[s] - counts[s] for s in ("train", "val", "test")}
        eligible = [s for s in slack if slack[s] >= k]
        s_name = max(eligible, key=lambda s: (slack[s], targets[s])) if eligible else max(("train", "val", "test"), key=lambda s: targets[s] - counts[s])
        counts[s_name] += k
        parts[s_name].append(sub)

    out: Dict[str, pd.DataFrame] = {}
    for s in ("train", "val", "test"):
        out[s] = pd.concat(parts[s], axis=0).copy() if parts[s] else df.iloc[0:0].copy()
        out[s]["split"] = s
    return out


def write_split_csv(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("sample_id", "source_json_path", "family_id", "peak_count_est", "tail_energy_ratio", "dynamic_range_log", "debug_split", "split") if c in df.columns]
    df.loc[:, cols].to_csv(out_path, index=False, encoding="utf-8")


def _assert_family_leakage_free(parts: Dict[str, pd.DataFrame]) -> None:
    f_train = set(parts["train"]["family_id"].astype(str).unique()) if len(parts["train"]) else set()
    f_val = set(parts["val"]["family_id"].astype(str).unique()) if len(parts["val"]) else set()
    f_test = set(parts["test"]["family_id"].astype(str).unique()) if len(parts["test"]) else set()
    for a, b, na, nb in [(f_train, f_val, "train", "val"), (f_train, f_test, "train", "test"), (f_val, f_test, "val", "test")]:
        overlap = a & b
        if overlap:
            raise RuntimeError(f"family leakage {na}∩{nb}: {overlap}")


def main() -> None:
    global _split_seed
    parser = argparse.ArgumentParser(description="Step 8: build train/val/test split CSVs")
    parser.add_argument("--input_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\dev_subset.csv")
    parser.add_argument("--train_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\split_train.csv")
    parser.add_argument("--val_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\split_val.csv")
    parser.add_argument("--test_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\split_test.csv")
    parser.add_argument("--metadata_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\all_samples.csv")
    parser.add_argument("--no_metadata_merge", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args, _ = parser.parse_known_args()

    _split_seed = int(args.seed)

    df = load_subset(args.input_csv)
    if not args.no_metadata_merge and Path(args.metadata_csv).is_file():
        meta = pd.read_csv(args.metadata_csv)
        need = {"sample_id", "family_id_raw", "x_min", "x_max"}
        if need.issubset(meta.columns):
            df = df.merge(meta.loc[:, list(need)].drop_duplicates(subset=["sample_id"]), on="sample_id", how="left")
        else:
            df["family_id_raw"] = ""
            df["x_min"] = np.nan
            df["x_max"] = np.nan
    else:
        if "family_id_raw" not in df.columns:
            df["family_id_raw"] = ""
        if "x_min" not in df.columns:
            df["x_min"] = np.nan
        if "x_max" not in df.columns:
            df["x_max"] = np.nan

    df = assign_family_id(df)
    parts = group_split_by_family(df, ratios=(0.8, 0.1, 0.1))
    _assert_family_leakage_free(parts)

    write_split_csv(parts["train"], args.train_csv)
    write_split_csv(parts["val"], args.val_csv)
    write_split_csv(parts["test"], args.test_csv)

    print(f"[DONE] train={len(parts['train'])}, val={len(parts['val'])}, test={len(parts['test'])}")
    print(f"  -> {args.train_csv}")
    print(f"  -> {args.val_csv}")
    print(f"  -> {args.test_csv}")


if __name__ == "__main__":
    main()
