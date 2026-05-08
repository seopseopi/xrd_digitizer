#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

_LEGACY_PREFIX = "c:/xrd_digitizer_v1"


def _norm_one(val: str, root: Path) -> str:
    s = str(val).strip().replace("\\", "/")
    if not s:
        return s
    if s.lower().startswith(_LEGACY_PREFIX):
        rel = s[len(_LEGACY_PREFIX) :].lstrip("/")
        return str((root / rel).resolve())
    return s


def _normalize_manifest(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.endswith("_path") or c in {"gt_path", "source_json_path"}:
            out[c] = out[c].map(lambda v, r=root: _norm_one(v, r))
    return out


def _choose_image_path(row: pd.Series) -> str:
    for k in ("image_path", "real_image_path", "styled_image_path", "clean_image_path"):
        if k in row and isinstance(row[k], str) and row[k]:
            return row[k]
    return ""


def _find_debug_json(run_dir: Path, sample_id: str) -> Optional[Path]:
    matches = list(run_dir.glob(f"**/debug_{sample_id}/debug.json"))
    return matches[0] if matches else None


def _extract_patch(roi_gray: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    h, w = roi_gray.shape
    r = patch_size // 2
    x0, x1 = x - r, x + r + 1
    y0, y1 = y - r, y + r + 1
    out = np.zeros((patch_size, patch_size), dtype=np.float32)

    sx0, sx1 = max(0, x0), min(w, x1)
    sy0, sy1 = max(0, y0), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = roi_gray[sy0:sy1, sx0:sx1]
    return out


def _roi_axis_proximity(roi_h: int, roi_w: int, x: int, y: int, radius: float = 18.0) -> float:
    d = float(min(x, roi_w - 1 - x, y, roi_h - 1 - y))
    if d >= radius:
        return 0.0
    return 1.0 - d / radius


def _build_records_for_sample(
    row: pd.Series,
    run_dir: Path,
    output_patch_dir: Path,
    patch_size: int,
    positive_px: int,
    negative_px: int,
) -> List[Dict[str, Any]]:
    sid = str(row["sample_id"])
    gt_path = Path(str(row["gt_path"]))
    img_path = Path(_choose_image_path(row))
    dbg_path = _find_debug_json(run_dir, sid)
    if not gt_path.is_file() or not img_path.is_file() or dbg_path is None:
        return []

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
    plot_box = gt.get("plot_box", [0, 0, 0, 0])
    x0, y0, x1, y1 = [int(v) for v in plot_box]
    if x1 <= x0 or y1 <= y0:
        return []

    # 학습용 candidate는 debug dump가 있어야 정확하다.
    final_candidates = dbg.get("final_candidates")
    if not isinstance(final_candidates, dict):
        return []

    gt_curve_by_x = gt.get("pixel_curve_by_x", {})
    if not isinstance(gt_curve_by_x, dict):
        return []

    trace_path = dbg.get("trace", {}).get("path", [])
    trace_map: Dict[int, int] = {}
    if isinstance(trace_path, list):
        for c, y in enumerate(trace_path):
            if y is not None:
                trace_map[int(c)] = int(y)

    img = Image.open(str(img_path)).convert("L")
    roi = np.asarray(img.crop((x0, y0, x1, y1)), dtype=np.float32) / 255.0
    roi_h, roi_w = roi.shape

    records: List[Dict[str, Any]] = []
    for col_str, cands in final_candidates.items():
        try:
            col = int(col_str)
        except (TypeError, ValueError):
            continue
        x_abs = x0 + col
        gt_y_abs = gt_curve_by_x.get(str(x_abs))
        if gt_y_abs is None:
            continue
        gt_y_roi = int(gt_y_abs) - y0
        if gt_y_roi < 0 or gt_y_roi >= roi_h:
            continue

        for i, cand in enumerate(cands or []):
            cy = int(cand.get("y", -1))
            if cy < 0 or cy >= roi_h:
                continue
            dist = abs(cy - gt_y_roi)
            if dist <= positive_px:
                label = 1
            elif dist >= negative_px:
                label = 0
            else:
                continue

            patch = _extract_patch(roi, col, cy, patch_size)
            cand_map = np.zeros_like(patch, dtype=np.float32)
            cand_map[patch_size // 2, patch_size // 2] = 1.0
            axis_prox = np.full_like(patch, _roi_axis_proximity(roi_h, roi_w, col, cy), dtype=np.float32)
            patch_3ch = np.stack([patch, cand_map, axis_prox], axis=0)

            patch_name = f"{sid}_x{col:04d}_i{i:02d}.npy"
            patch_path = output_patch_dir / patch_name
            np.save(str(patch_path), patch_3ch)

            hard_neg = None
            if label == 0:
                if _roi_axis_proximity(roi_h, roi_w, col, cy) > 0.4:
                    hard_neg = "axis_or_border"
                elif trace_map.get(col) == cy:
                    hard_neg = "dp_selected_far_from_gt"
                else:
                    hard_neg = "generic_far"

            records.append(
                {
                    "sample_id": sid,
                    "x": col,
                    "candidate_y": cy,
                    "gt_y": gt_y_roi,
                    "distance_px": int(dist),
                    "label": int(label),
                    "hard_negative_type": hard_neg,
                    "style_group": str(row.get("variant_type", "unknown")),
                    "rule_confidence": float(cand.get("confidence", 0.0)),
                    "dp_selected": bool(trace_map.get(col) == cy),
                    "patch_path": str(patch_path),
                    "channels": ["roi_gray", "candidate_center", "axis_proximity"],
                }
            )
    return records


def _split_name(raw: str) -> str:
    x = str(raw).strip().lower()
    if x in {"debug", "train"}:
        return "train"
    if x in {"validation", "val"}:
        return "val"
    if x in {"holdout", "test"}:
        return "test"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build candidate rerank dataset from debug candidate dumps")
    ap.add_argument("--manifest_csv", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--patch_size", type=int, default=33)
    ap.add_argument("--positive_px", type=int, default=2)
    ap.add_argument("--negative_px", type=int, default=8)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    manifest = _normalize_manifest(pd.read_csv(args.manifest_csv), repo)
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.output_dir).resolve()
    patch_dir = out / "patches"
    out.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    skipped = 0

    for _, row in manifest.iterrows():
        recs = _build_records_for_sample(
            row, run_dir, patch_dir, args.patch_size, args.positive_px, args.negative_px
        )
        if not recs:
            skipped += 1
            continue
        sname = _split_name(row.get("split", "train"))
        buckets[sname].extend(recs)

    for split, recs in buckets.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "run_dir": str(run_dir),
        "output_dir": str(out),
        "counts": {k: len(v) for k, v in buckets.items()},
        "skipped_samples": int(skipped),
        "note": "debug.json에 final_candidates가 없으면 해당 샘플은 자동 skip",
    }
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
