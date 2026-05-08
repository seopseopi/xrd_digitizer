#!/usr/bin/env python3
"""Validate candidate patch panel manifest before any panel execution."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SID_RE = re.compile(r"(pattern_\d+)")


def _resolve(value: Any) -> Path:
    s = "" if value is None else str(value).strip()
    s = s.replace("\\", "/")
    marker = "xrd_digitizer_v1/"
    if marker in s:
        s = s.split(marker, 1)[1]
    p = Path(s)
    if p.is_absolute():
        return p
    return ROOT / p


def _load_json(path: Path) -> Tuple[Dict[str, Any], str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:
        return {}, f"json_load_failed:{type(exc).__name__}"


def _domain_matches_path(domain: str, image_path: Path) -> bool:
    s = str(image_path).replace("\\", "/")
    name = image_path.name
    if domain == "clean":
        return "data/rendered_clean/" in s and name.endswith("_clean_v1.png")
    if domain == "styled":
        return "data/rendered_styled/" in s and "_styled_" in name
    if domain == "real_like":
        return "data/rendered_real_like/" in s and "_real_" in name
    return False


def _sample_matches(sample_id: str, *paths: Path) -> bool:
    return all(sample_id in p.name for p in paths)


def _extract_sid(text: str) -> str:
    m = SID_RE.search(str(text))
    return m.group(1) if m else ""


def _infer_domain_from_path(path: Path) -> str:
    s = str(path).replace("\\", "/")
    if "/rendered_clean/" in s or "_clean_" in path.name:
        return "clean"
    if "/rendered_styled/" in s or "_styled_" in path.name:
        return "styled"
    if "/rendered_real_like/" in s or "_real_" in path.name:
        return "real_like"
    return ""


def _validate_row(row: pd.Series) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id", ""))
    domain = str(row.get("domain", ""))
    input_image = _resolve(row.get("input_image", ""))
    manual_json = _resolve(row.get("manual_json", ""))
    gt_json = _resolve(row.get("gt_json", ""))
    reasons: List[str] = []
    warnings: List[str] = []

    include_raw = row.get("include", True)
    include = bool(include_raw) and str(include_raw).lower() not in {"false", "0", "nan"}
    pre_raw = row.get("exclude_reason", "")
    pre_exclude = "" if pd.isna(pre_raw) else str(pre_raw)
    if not include or pre_exclude:
        reasons.append(pre_exclude or "pre_excluded")

    if not input_image.is_file():
        reasons.append("input_image_missing")
    if not manual_json.is_file():
        reasons.append("manual_json_missing")
    if not gt_json.is_file():
        reasons.append("gt_json_missing")
    if input_image.is_file() and not _domain_matches_path(domain, input_image):
        reasons.append("domain_input_image_mismatch")
    if input_image.is_file() and manual_json.is_file() and gt_json.is_file():
        if not _sample_matches(sample_id, input_image, manual_json, gt_json):
            reasons.append("sample_id_filename_mismatch")
        sid_input = _extract_sid(input_image.name)
        sid_manual = _extract_sid(manual_json.name)
        sid_gt = _extract_sid(gt_json.name)
        if sid_input and sid_input != sample_id:
            reasons.append("sample_id_input_filename_mismatch")
        if sid_manual and sid_manual != sample_id:
            reasons.append("sample_id_manual_filename_mismatch")
        if sid_gt and sid_gt != sample_id:
            reasons.append("sample_id_gt_filename_mismatch")
        if sid_input and sid_manual and sid_input != sid_manual:
            reasons.append("sample_id_input_manual_mismatch")
        if sid_input and sid_gt and sid_input != sid_gt:
            reasons.append("sample_id_input_gt_mismatch")
        dom_input = _infer_domain_from_path(input_image)
        dom_manual = _infer_domain_from_path(manual_json)
        if dom_input and dom_input != domain:
            reasons.append("domain_input_filename_mismatch")
        if dom_manual and dom_manual != domain:
            reasons.append("domain_manual_filename_mismatch")
        if dom_input and dom_manual and dom_input != dom_manual:
            reasons.append("domain_input_manual_mismatch")

    image_w = None
    image_h = None
    if input_image.is_file():
        try:
            with Image.open(input_image) as im:
                image_w, image_h = int(im.width), int(im.height)
        except Exception as exc:
            reasons.append(f"input_image_load_failed:{type(exc).__name__}")

    plot_box = None
    roi_w = None
    roi_h = None
    if manual_json.is_file():
        manual, err = _load_json(manual_json)
        if err:
            reasons.append("manual_" + err)
        else:
            pb = manual.get("plot_box")
            msid = _extract_sid(str(manual.get("sample_id", ""))) or _extract_sid(str(manual.get("pattern_id", "")))
            if msid and msid != sample_id:
                reasons.append("manual_internal_sample_id_mismatch")
            if not msid:
                warnings.append("MANUAL_JSON_AMBIGUOUS:missing_sample_id")
            mdom = str(manual.get("domain", "") or manual.get("run_domain", "")).strip()
            if mdom and mdom in {"clean", "styled", "real_like"} and mdom != domain:
                reasons.append("manual_internal_domain_mismatch")
            if not mdom:
                warnings.append("MANUAL_JSON_AMBIGUOUS:missing_domain")
            mimg = str(manual.get("input_image", "") or manual.get("image_path", "")).strip()
            if mimg:
                msid_from_path = _extract_sid(mimg)
                if msid_from_path and msid_from_path != sample_id:
                    reasons.append("manual_internal_image_sample_mismatch")
                mdom_from_path = _infer_domain_from_path(Path(mimg))
                if mdom_from_path and mdom_from_path != domain:
                    reasons.append("manual_internal_image_domain_mismatch")
            else:
                warnings.append("MANUAL_JSON_AMBIGUOUS:missing_input_image")
            if not isinstance(pb, list) or len(pb) != 4:
                reasons.append("manual_plot_box_missing")
            else:
                try:
                    x0, y0, x1, y1 = [int(v) for v in pb]
                    plot_box = [x0, y0, x1, y1]
                    roi_w = int(x1 - x0)
                    roi_h = int(y1 - y0)
                    if roi_w <= 0 or roi_h <= 0:
                        reasons.append("plot_box_non_positive_roi")
                    if image_w is not None and image_h is not None:
                        if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
                            reasons.append("plot_box_outside_image")
                except (TypeError, ValueError):
                    reasons.append("manual_plot_box_invalid")

    if gt_json.is_file():
        gt, err = _load_json(gt_json)
        if err:
            reasons.append("gt_" + err)
        else:
            gt_sample = str(gt.get("sample_id", ""))
            if gt_sample and gt_sample != sample_id:
                reasons.append("gt_sample_id_mismatch")

    return {
        "validation_status": "FAIL" if reasons else "PASS",
        "validation_reasons": ";".join(r for r in reasons if r),
        "validation_warnings": ";".join(warnings),
        "actual_image_width": image_w,
        "actual_image_height": image_h,
        "validated_plot_box": "" if plot_box is None else json.dumps(plot_box),
        "validated_roi_width": roi_w,
        "validated_roi_height": roi_h,
    }


def validate(manifest: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    required = {
        "sample_id",
        "domain",
        "failure_type",
        "input_image",
        "manual_json",
        "gt_json",
        "expected_image_width",
        "expected_image_height",
        "roi_width",
        "roi_height",
        "plot_box",
        "source_note",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    duplicates = df.duplicated(["sample_id", "domain"], keep=False)
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        vr = _validate_row(row)
        if bool(duplicates.iloc[idx]):
            vr["validation_status"] = "FAIL"
            base = str(vr.get("validation_reasons", ""))
            vr["validation_reasons"] = ";".join(x for x in [base, "duplicate_sample_domain"] if x)
        rows.append({**row.to_dict(), **vr})
    full = pd.DataFrame(rows)
    valid = full[full["validation_status"].eq("PASS")].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out_csv, index=False)
    full.to_csv(out_csv.with_name(out_csv.stem + "_all_rows.csv"), index=False)
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "_candidate_patch_panel" / "panel_manifest_validated.csv",
    )
    args = ap.parse_args()
    full = validate(args.manifest, args.out)
    included = int(full["validation_status"].eq("PASS").sum())
    excluded = int(len(full) - included)
    reasons = Counter()
    for txt in full.loc[~full["validation_status"].eq("PASS"), "validation_reasons"].fillna(""):
        for part in str(txt).split(";"):
            if part:
                reasons[part] += 1
    status = "PASS" if included > 0 and excluded == 0 else "FAIL"
    print(f"STATUS={status}")
    print(f"included_rows={included}")
    print(f"excluded_rows={excluded}")
    print("excluded_reason_counts=" + json.dumps(dict(sorted(reasons.items())), ensure_ascii=False))
    print(f"validated_csv={args.out}")
    if status != "PASS":
        bad_cols = ["sample_id", "domain", "validation_reasons"]
        print(full.loc[~full["validation_status"].eq("PASS"), bad_cols].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
