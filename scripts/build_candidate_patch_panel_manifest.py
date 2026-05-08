#!/usr/bin/env python3
"""Build a row-explicit canonical manifest for candidate patch panel runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PANEL = ROOT / "outputs" / "_candidate_patch_panel" / "panel_samples.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "_candidate_patch_panel" / "panel_manifest.csv"

CANONICAL_MANUAL_ROOTS = {
    "clean": [
        ROOT / "outputs" / "0504" / "research_diag" / "baseline_v12_default" / "clean",
        ROOT / "outputs" / "0503" / "runs" / "eval_current_v11" / "clean",
    ],
    "styled": [
        ROOT / "outputs" / "0503" / "runs" / "eval_current_v11" / "styled",
    ],
    "real_like": [
        ROOT / "outputs" / "0503" / "runs" / "eval_current_v11" / "real",
    ],
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _input_image_for(sample_id: str, domain: str) -> Path:
    if domain == "clean":
        return ROOT / "data" / "rendered_clean" / f"{sample_id}_clean_v1.png"
    if domain == "styled":
        return ROOT / "data" / "rendered_styled" / f"{sample_id}_styled_v5.png"
    if domain == "real_like":
        return ROOT / "data" / "rendered_real_like" / f"{sample_id}_real_v4.png"
    return ROOT / "__invalid_domain__"


def _gt_for(sample_id: str) -> Path:
    return ROOT / "data" / "gt" / f"{sample_id}_gt.json"


def _find_canonical_manual(sample_id: str, domain: str) -> Tuple[Optional[Path], str]:
    roots = CANONICAL_MANUAL_ROOTS.get(domain, [])
    for root in roots:
        if not root.is_dir():
            continue
        matches = []
        matches.extend(sorted(root.glob(f"_mi_{sample_id}.json")))
        matches.extend(sorted(root.glob(f"_mi_pattern_{sample_id.split('pattern_')[-1]}.json")))
        uniq: List[Path] = []
        seen = set()
        for p in matches:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(p)
        if len(uniq) == 1:
            return uniq[0], f"canonical_manual_root={uniq[0].parent.relative_to(ROOT)}"
        if len(uniq) > 1:
            return None, "canonical_manual_ambiguous:" + "|".join(_rel(p) for p in uniq)
    return None, "canonical_manual_missing"


def _load_plot_box(manual_json: Optional[Path]) -> Tuple[Optional[List[int]], Optional[int], Optional[int], str]:
    if manual_json is None or not manual_json.is_file():
        return None, None, None, "manual_missing"
    try:
        obj = json.loads(manual_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, None, f"manual_json_load_failed:{type(exc).__name__}"
    pb = obj.get("plot_box")
    if not isinstance(pb, list) or len(pb) != 4:
        return None, None, None, "manual_plot_box_missing"
    try:
        x0, y0, x1, y1 = [int(v) for v in pb]
    except (TypeError, ValueError):
        return None, None, None, "manual_plot_box_invalid"
    return [x0, y0, x1, y1], int(x1 - x0), int(y1 - y0), "ok"


def _image_size(path: Path) -> Tuple[Optional[int], Optional[int], str]:
    if not path.is_file():
        return None, None, "image_missing"
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height), "ok"
    except Exception as exc:
        return None, None, f"image_load_failed:{type(exc).__name__}"


def _candidate_rows(panel_csv: Path) -> pd.DataFrame:
    if panel_csv.is_file():
        df = pd.read_csv(panel_csv)
        needed = {"sample_id", "domain", "failure_type"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{panel_csv} missing columns: {sorted(missing)}")
        return df.copy()
    raise FileNotFoundError(f"panel seed CSV not found: {panel_csv}")


def build(panel_csv: Path, out_csv: Path) -> pd.DataFrame:
    seed = _candidate_rows(panel_csv)
    rows: List[Dict[str, Any]] = []
    for _, src in seed.iterrows():
        sample_id = str(src["sample_id"])
        domain = str(src["domain"])
        failure_type = str(src.get("failure_type", ""))
        input_image = _input_image_for(sample_id, domain)
        manual_json, manual_note = _find_canonical_manual(sample_id, domain)
        gt_json = _gt_for(sample_id)
        iw, ih, image_note = _image_size(input_image)
        plot_box, roi_w, roi_h, plot_note = _load_plot_box(manual_json)

        exclude_reasons: List[str] = []
        if image_note != "ok":
            exclude_reasons.append(image_note)
        if manual_json is None:
            exclude_reasons.append(manual_note)
        elif plot_note != "ok":
            exclude_reasons.append(plot_note)
        if not gt_json.is_file():
            exclude_reasons.append("gt_missing")
        if domain not in {"clean", "styled", "real_like"}:
            exclude_reasons.append("unsupported_domain")

        rows.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "failure_type": failure_type,
                "input_image": _rel(input_image),
                "manual_json": "" if manual_json is None else _rel(manual_json),
                "gt_json": _rel(gt_json),
                "expected_image_width": iw,
                "expected_image_height": ih,
                "roi_width": roi_w,
                "roi_height": roi_h,
                "plot_box": "" if plot_box is None else json.dumps(plot_box),
                "source_note": ";".join(
                    [
                        "built_from=" + _rel(panel_csv),
                        manual_note,
                        f"image_rule={domain}",
                    ]
                ),
                "include": not exclude_reasons,
                "exclude_reason": ";".join(exclude_reasons),
                "taxonomy_prior": str(src.get("taxonomy_prior", "")),
                "rule_curve_y_mae_px": src.get("rule_curve_y_mae_px", ""),
                "global_curve_y_mae_px": src.get("global_curve_y_mae_px", ""),
                "peak_f1": src.get("peak_f1", ""),
                "candidate_gt_near_recall_px5": src.get("candidate_gt_near_recall_px5", ""),
            }
        )
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    df = build(args.panel_csv, args.out)
    print(f"wrote {args.out}")
    print(f"included={int(df['include'].sum())} excluded={int((~df['include']).sum())}")
    if "exclude_reason" in df:
        reasons = df.loc[~df["include"], "exclude_reason"].value_counts().to_dict()
        print("excluded_reason_counts=", reasons)


if __name__ == "__main__":
    main()
