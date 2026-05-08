"""
Step 4: all_samples.csv에서 stratified 300개 dev subset 추출.

xrd_digitizer_v1_master_spec.md §4 준수.
- 27-bin stratified sampling (peak_count × tail_energy × dynamic_range)
- debug/validation/holdout 각 100개
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS: List[str] = [
    "sample_id",
    "source_json_path",
    "is_valid",
    "peak_count_est",
    "tail_energy_ratio",
    "dynamic_range_log",
    "peak_height_ratio",
    "mean_peak_spacing_norm",
    "family_id_raw",
]

# §4.3: 3구간 binning
N_BINS = 3
RANDOM_SEED = 42


def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in metadata CSV: {missing}")
    return df


def bin_column(values: pd.Series, n_bins: int = 3) -> pd.Series:
    """§4.3: quantile binning → 0/1/2 (low/mid/high)."""
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() == 0:
        return pd.Series(np.nan, index=v.index, dtype=float)
    cats = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    return cats.astype("float")


def build_strat_bins(df: pd.DataFrame) -> pd.DataFrame:
    """§4.3: 3개 feature × 3구간 = 최대 27 bin 생성."""
    out = df.copy()
    out["bin_peak"] = bin_column(out["peak_count_est"], n_bins=N_BINS)
    out["bin_tail"] = bin_column(out["tail_energy_ratio"], n_bins=N_BINS)
    out["bin_dr"] = bin_column(out["dynamic_range_log"], n_bins=N_BINS)

    def _key_row(r: pd.Series) -> str:
        parts = []
        for c in ("bin_peak", "bin_tail", "bin_dr"):
            val = r.get(c, np.nan)
            if pd.isna(val):
                parts.append("NA")
            else:
                parts.append(str(int(val)))
        return "__".join(parts)

    out["strat_bin_key"] = out.apply(_key_row, axis=1)
    return out


def _quantize(x: float, decimals: int) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.round(float(x), decimals))


def _pick_from_bin_greedy_diverse(
    bin_df: pd.DataFrame,
    k: int,
    family_counts_global: Dict[str, int],
) -> pd.DataFrame:
    """§4.4: bin 내 diversity 우선 greedy 선택."""
    if k <= 0 or bin_df.empty:
        return bin_df.iloc[0:0]

    rows = bin_df.copy()
    rows = rows.sort_values(
        by=["family_id_raw", "peak_height_ratio", "mean_peak_spacing_norm", "sample_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    selected_idx: List[int] = []
    picked_ph: set = set()
    picked_ms: set = set()
    fam_counts = dict(family_counts_global)

    def score_row(r: pd.Series) -> Tuple:
        fam = str(r["family_id_raw"])
        phq = _quantize(float(r["peak_height_ratio"]), 3)
        msq = _quantize(float(r["mean_peak_spacing_norm"]), 4)
        ph_bonus = 1.0 if np.isfinite(phq) and phq not in picked_ph else 0.0
        ms_bonus = 1.0 if np.isfinite(msq) and msq not in picked_ms else 0.0
        fam_penalty = float(fam_counts.get(fam, 0))
        ph = float(r["peak_height_ratio"]) if np.isfinite(r["peak_height_ratio"]) else float("-inf")
        ms = float(r["mean_peak_spacing_norm"]) if np.isfinite(r["mean_peak_spacing_norm"]) else float("-inf")
        return (ph_bonus, ms_bonus, -fam_penalty, ph, ms, str(r["sample_id"]))

    remaining = rows
    for _ in range(min(k, len(rows))):
        best_i = None
        best_s = None
        for i, r in remaining.iterrows():
            s = score_row(r)
            if best_s is None or s > best_s:
                best_s = s
                best_i = i

        assert best_i is not None
        r = remaining.loc[best_i]
        phq = _quantize(float(r["peak_height_ratio"]), 3)
        msq = _quantize(float(r["mean_peak_spacing_norm"]), 4)
        if np.isfinite(phq):
            picked_ph.add(phq)
        if np.isfinite(msq):
            picked_ms.add(msq)

        fam = str(r["family_id_raw"])
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

        selected_idx.append(best_i)
        remaining = remaining.drop(index=best_i)

    return rows.loc[selected_idx]


def sample_balanced_subset(df: pd.DataFrame, total_n: int = 300) -> pd.DataFrame:
    """§4.3: bin별 균등 추출 후 diversity greedy 선택."""
    if total_n <= 0:
        return df.iloc[0:0]

    work = df.copy()
    if work["is_valid"].dtype == object:
        work["is_valid"] = work["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    work = work[work["is_valid"] == True].copy()  # noqa: E712
    if work.empty:
        raise ValueError("No valid rows after filtering is_valid==True")

    work = work[work["strat_bin_key"].notna()].copy()
    work = work[~work["strat_bin_key"].astype(str).str.contains("NA", regex=False)].copy()
    if work.empty:
        raise ValueError("No rows with complete stratification bins")

    fam_global = work["family_id_raw"].astype(str).value_counts().to_dict()
    bin_keys = sorted(work["strat_bin_key"].unique().tolist())
    counts = {k: int((work["strat_bin_key"] == k).sum()) for k in bin_keys}
    nonempty_bins = [k for k, c in counts.items() if c > 0]
    b = len(nonempty_bins)
    if b == 0:
        raise ValueError("No nonempty bins")

    base = total_n // b
    rem = total_n - base * b
    targets: Dict[str, int] = {k: base for k in nonempty_bins}
    for k in nonempty_bins[:rem]:
        targets[k] += 1

    for k in nonempty_bins:
        if targets[k] > counts[k]:
            targets[k] = counts[k]

    deficit = total_n - sum(targets.values())
    rr = 0
    while deficit > 0:
        progressed = False
        for _ in range(len(nonempty_bins)):
            k = nonempty_bins[rr % len(nonempty_bins)]
            rr += 1
            spare = counts[k] - targets[k]
            if spare > 0:
                targets[k] += 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
        if not progressed:
            break

    picked_parts: List[pd.DataFrame] = []
    for k in nonempty_bins:
        t = int(targets.get(k, 0))
        if t <= 0:
            continue
        bin_df = work[work["strat_bin_key"] == k].copy()
        picked = _pick_from_bin_greedy_diverse(bin_df, k=t, family_counts_global=fam_global)
        for fam in picked["family_id_raw"].astype(str).tolist():
            fam_global[fam] = fam_global.get(fam, 0) + 1
        picked_parts.append(picked)

    picked = pd.concat(picked_parts, axis=0) if picked_parts else work.iloc[0:0]

    if len(picked) < total_n:
        remaining = work.drop(index=picked.index, errors="ignore")
        need = total_n - len(picked)
        extra = _pick_from_bin_greedy_diverse(remaining, k=need, family_counts_global=fam_global)
        picked = pd.concat([picked, extra], axis=0)

    if len(picked) > total_n:
        picked = picked.sort_values(by=["strat_bin_key", "sample_id"], kind="mergesort").head(total_n)

    return picked


def split_debug_val_holdout(df: pd.DataFrame, debug_n: int = 100, val_n: int = 100, holdout_n: int = 100) -> Dict[str, pd.DataFrame]:
    """§4.4: debug/validation/holdout 3분할."""
    total_n = debug_n + val_n + holdout_n
    n = len(df)
    if n < total_n:
        raise ValueError(f"Picked subset size {n} < required {total_n}")

    rng = np.random.default_rng(RANDOM_SEED)
    quotas = {"debug": debug_n, "validation": val_n, "holdout": holdout_n}
    remaining = {k: int(v) for k, v in quotas.items()}
    split_of_index: Dict[int, str] = {}

    def pick_split_round_robin(start: int) -> str:
        order = ["debug", "validation", "holdout"]
        for step in range(len(order)):
            name = order[(start + step) % len(order)]
            if remaining[name] > 0:
                return name
        raise RuntimeError(f"No split has remaining quota: {remaining}")

    start = 0
    for _, bin_df in df.groupby("strat_bin_key", sort=True):
        idx = bin_df.index.to_numpy().copy()
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            i_int = int(i)
            name = pick_split_round_robin(start + j)
            split_of_index[i_int] = name
            remaining[name] -= 1
        start += len(idx)

    if any(v != 0 for v in remaining.values()):
        raise ValueError(f"Split assignment did not consume quotas exactly: {remaining}")

    labeled = df.copy()
    labeled["debug_split"] = labeled.index.map(lambda i: split_of_index[int(i)])

    return {
        "debug": labeled[labeled["debug_split"] == "debug"].copy(),
        "validation": labeled[labeled["debug_split"] == "validation"].copy(),
        "holdout": labeled[labeled["debug_split"] == "holdout"].copy(),
        "labeled": labeled.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: build stratified dev subset (300 samples)")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"c:\xrd_digitizer_v1\data\metadata\all_samples.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"c:\xrd_digitizer_v1\data\metadata\dev_subset.csv",
    )
    parser.add_argument("--total_n", type=int, default=300)
    args = parser.parse_args()

    debug_n = args.total_n // 3
    val_n = args.total_n // 3
    holdout_n = args.total_n - debug_n - val_n

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.input_csv)
    df = build_strat_bins(df)
    picked = sample_balanced_subset(df, total_n=args.total_n)
    parts = split_debug_val_holdout(picked, debug_n=debug_n, val_n=val_n, holdout_n=holdout_n)
    labeled = parts["labeled"]

    out = labeled.loc[
        :,
        [
            "sample_id",
            "source_json_path",
            "debug_split",
            "peak_count_est",
            "tail_energy_ratio",
            "dynamic_range_log",
        ],
    ].copy()
    out.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"[DONE] Wrote dev subset: {args.output_csv}")
    print(f"  Total: {len(out)}")
    print(out["debug_split"].value_counts().to_string())


if __name__ == "__main__":
    main()
