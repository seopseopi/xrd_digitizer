#!/usr/bin/env python3
"""Check run-level input/manual/gt pair integrity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SID_RE = re.compile(r"(pattern_\d+)")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sid_from_text(text: str) -> Optional[str]:
    m = SID_RE.search(str(text))
    return m.group(1) if m else None


def _domain_from_path(path: str) -> Optional[str]:
    s = str(path).replace("\\", "/")
    if "/rendered_clean/" in s or "_clean_" in Path(s).name:
        return "clean"
    if "/rendered_styled/" in s or "_styled_" in Path(s).name:
        return "styled"
    if "/rendered_real_like/" in s or "_real_" in Path(s).name:
        return "real_like"
    return None


def _manual_domain_from_json(manual: Dict[str, Any]) -> Optional[str]:
    for k in ("domain", "run_domain", "data_domain"):
        v = manual.get(k)
        if isinstance(v, str) and v.strip():
            x = v.strip()
            if x in {"clean", "styled", "real_like"}:
                return x
    return None


def _resolve_proj_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    s = str(raw).strip().replace("\\", "/")
    marker = "xrd_digitizer_v1/"
    if marker in s:
        s = s.split(marker, 1)[1]
    p = Path(s)
    if p.is_absolute():
        return p
    return ROOT / p


def _resolve_run_dirs(run_dir: Optional[Path], root: Optional[Path]) -> List[Path]:
    if run_dir:
        return [run_dir.resolve()]
    if not root:
        raise ValueError("either --run-dir or --root is required")
    out: List[Path] = []
    for dbg in root.rglob("debug.json"):
        # .../<run_key>/<variant>/debug_<sid>_global/debug.json
        if dbg.parent.name.startswith("debug_") and dbg.parent.parent.is_dir():
            out.append(dbg.parent.parent.resolve())
    return sorted(set(out))


def _lookup_manifest_paths(run_variant_dir: Path, sample_id: Optional[str], domain: Optional[str]) -> Dict[str, Optional[str]]:
    if not sample_id or not domain:
        return {"input_image": None, "manual_json": None, "gt_json": None}
    candidates = [
        ROOT / "outputs" / "_roi_upscale_diag" / "diag_manifest.csv",
        ROOT / "outputs" / "_candidate_patch_full" / "full_manifest_validated.csv",
        ROOT / "outputs" / "_candidate_patch_panel" / "panel_manifest_validated.csv",
        ROOT / "outputs" / "_candidate_patch_panel" / "panel_samples.csv",
    ]
    for csv_path in candidates:
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "sample_id" not in df.columns or "domain" not in df.columns:
            continue
        sel = df[(df["sample_id"].astype(str) == str(sample_id)) & (df["domain"].astype(str) == str(domain))]
        if len(sel) == 0:
            continue
        row = sel.iloc[0]
        return {
            "input_image": str(row.get("input_image")) if "input_image" in row.index else None,
            "manual_json": str(row.get("manual_json")) if "manual_json" in row.index else None,
            "gt_json": str(row.get("gt_json")) if "gt_json" in row.index else None,
        }
    return {"input_image": None, "manual_json": None, "gt_json": None}


def _plot_box_warning(plot_box: List[int], w: int, h: int) -> Optional[str]:
    if not isinstance(plot_box, list) or len(plot_box) != 4:
        return None
    x0, y0, x1, y1 = [int(v) for v in plot_box]
    roi_w = x1 - x0
    roi_h = y1 - y0
    warns: List[str] = []
    if x0 <= 1 or y0 <= 1 or x1 >= (w - 1) or y1 >= (h - 1):
        warns.append("plot_box_touches_boundary")
    if w > 0 and h > 0:
        frac = (max(0, roi_w) * max(0, roi_h)) / float(w * h)
        if frac < 0.15:
            warns.append("plot_box_too_small_ratio")
    return ";".join(warns) if warns else None


def _check_one(run_variant_dir: Path) -> Dict[str, Any]:
    debug_candidates = list(run_variant_dir.glob("debug_*_global/debug.json"))
    result_candidates = list(run_variant_dir.glob("*_result.json"))
    if not debug_candidates:
        return {
            "run_dir": str(run_variant_dir),
            "pair_status": "FAIL",
            "fail_reason": "DEBUG_METADATA_MISSING:debug.json_not_found",
        }
    debug_json = debug_candidates[0]
    dbg = _read_json(debug_json)
    meta = dbg.get("run_metadata", {}) if isinstance(dbg.get("run_metadata"), dict) else {}
    input_image = meta.get("input_image")
    manual_json = meta.get("manual_json")
    gt_json = meta.get("gt_json")

    run_key = run_variant_dir.parent.name  # clean_pattern_xxx
    run_domain = run_key.split("_pattern_")[0] if "_pattern_" in run_key else None
    run_sample_id = _sid_from_text(run_key) or _sid_from_text(run_variant_dir.name)
    if run_sample_id is None and result_candidates:
        run_sample_id = _sid_from_text(result_candidates[0].name)
    if not input_image or not manual_json or not gt_json:
        from_manifest = _lookup_manifest_paths(run_variant_dir, run_sample_id, run_domain)
        input_image = input_image or from_manifest.get("input_image")
        manual_json = manual_json or from_manifest.get("manual_json")
        gt_json = gt_json or from_manifest.get("gt_json")

    image_w = dbg.get("image_size", [None, None])[0] if isinstance(dbg.get("image_size"), list) else None
    image_h = dbg.get("image_size", [None, None])[1] if isinstance(dbg.get("image_size"), list) else None
    plot_box = dbg.get("plot_box")
    roi_box = plot_box
    roi_width = None
    roi_height = None
    if isinstance(plot_box, list) and len(plot_box) == 4:
        try:
            roi_width = int(plot_box[2]) - int(plot_box[0])
            roi_height = int(plot_box[3]) - int(plot_box[1])
        except Exception:
            roi_width = None
            roi_height = None

    input_sid = _sid_from_text(input_image or "")
    manual_sid = _sid_from_text(manual_json or "")
    gt_sid = _sid_from_text(gt_json or "")
    input_dom = _domain_from_path(input_image or "")
    manual_dom = _domain_from_path(manual_json or "")

    manual_inner_sid = None
    manual_inner_dom = None
    manual_inner_plot_box = None
    manual_axis_info = None
    manual_inner_input_image = None
    manual_metadata_ambiguous = False
    if manual_json and Path(str(manual_json)).is_file():
        mj = _read_json(Path(str(manual_json)))
        manual_inner_sid = _sid_from_text(str(mj.get("sample_id", ""))) or _sid_from_text(str(mj.get("pattern_id", "")))
        manual_inner_dom = _manual_domain_from_json(mj)
        manual_inner_input_image = str(mj.get("input_image", "") or mj.get("image_path", "")).strip() or None
        manual_inner_plot_box = mj.get("plot_box")
        manual_axis_info = {
            "x_axis_points": mj.get("x_axis_points"),
            "x_axis_values": mj.get("x_axis_values"),
            "y_axis_points": mj.get("y_axis_points"),
            "y_axis_values": mj.get("y_axis_values"),
        }
        if not (manual_inner_sid and manual_inner_dom and manual_inner_input_image):
            manual_metadata_ambiguous = True
        if manual_inner_dom:
            manual_dom = manual_inner_dom
        if manual_inner_sid:
            manual_sid = manual_inner_sid

    reasons: List[str] = []
    if not input_image or not manual_json or not gt_json:
        reasons.append("DEBUG_METADATA_MISSING:run_metadata_paths_absent")
    if run_sample_id and input_sid and run_sample_id != input_sid:
        reasons.append("run_vs_input_sample_mismatch")
    if run_sample_id and manual_sid and run_sample_id != manual_sid:
        reasons.append("run_vs_manual_sample_mismatch")
    if run_sample_id and gt_sid and run_sample_id != gt_sid:
        reasons.append("run_vs_gt_sample_mismatch")
    if input_sid and manual_sid and input_sid != manual_sid:
        reasons.append("input_manual_sample_mismatch")
    if input_sid and gt_sid and input_sid != gt_sid:
        reasons.append("input_gt_sample_mismatch")
    if run_domain and input_dom and run_domain != input_dom:
        reasons.append("run_vs_input_domain_mismatch")
    if run_domain and manual_dom and run_domain != manual_dom:
        reasons.append("run_vs_manual_domain_mismatch")
    if input_dom and manual_dom and input_dom != manual_dom:
        reasons.append("input_manual_domain_mismatch")
    if manual_inner_input_image:
        manual_img_sid = _sid_from_text(manual_inner_input_image)
        manual_img_dom = _domain_from_path(manual_inner_input_image)
        if run_sample_id and manual_img_sid and run_sample_id != manual_img_sid:
            reasons.append("manual_internal_input_image_sample_mismatch")
        if run_domain and manual_img_dom and run_domain != manual_img_dom:
            reasons.append("manual_internal_input_image_domain_mismatch")

    if manual_inner_plot_box and image_w and image_h:
        try:
            x0, y0, x1, y1 = [int(v) for v in manual_inner_plot_box]
            if x0 < 0 or y0 < 0 or x1 > int(image_w) or y1 > int(image_h):
                reasons.append("manual_plot_box_outside_image")
        except Exception:
            reasons.append("manual_plot_box_invalid")

    warn = None
    warnings: List[str] = []
    if plot_box and image_w and image_h:
        warn = _plot_box_warning(plot_box, int(image_w), int(image_h))
        if warn:
            warnings.extend([x for x in str(warn).split(";") if x])
    if manual_metadata_ambiguous:
        warnings.append("MANUAL_JSON_AMBIGUOUS")

    if reasons:
        status = "FAIL"
    elif manual_metadata_ambiguous:
        status = "PASS_WITH_AMBIGUOUS_MANUAL"
    else:
        status = "PASS"
    return {
        "run_dir": str(run_variant_dir),
        "sample_id": run_sample_id,
        "domain": run_domain,
        "input_image": input_image,
        "manual_json": manual_json,
        "gt_json": gt_json,
        "plot_box": plot_box,
        "roi_box": roi_box,
        "image_width": image_w,
        "image_height": image_h,
        "roi_width": roi_width,
        "roi_height": roi_height,
        "manual_plot_box": manual_inner_plot_box,
        "manual_internal_sample_id": manual_inner_sid,
        "manual_internal_domain": manual_inner_dom,
        "manual_internal_input_image": manual_inner_input_image,
        "manual_axis_calibration": json.dumps(manual_axis_info, ensure_ascii=False) if manual_axis_info else None,
        "input_sample_id_from_filename": input_sid,
        "manual_sample_id_from_filename": manual_sid,
        "gt_sample_id_from_filename": gt_sid,
        "input_domain_from_path": input_dom,
        "manual_domain_from_filename_or_metadata": manual_dom,
        "pair_status": status,
        "fail_reason": ";".join(reasons),
        "warning": ";".join(warnings) if warnings else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    run_dirs = _resolve_run_dirs(args.run_dir, args.root)
    rows = [_check_one(rd) for rd in run_dirs]
    df = pd.DataFrame(rows)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[saved] {args.out_csv}")
    print(df.to_string(index=False))
    if any(str(v) == "FAIL" for v in df.get("pair_status", [])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
