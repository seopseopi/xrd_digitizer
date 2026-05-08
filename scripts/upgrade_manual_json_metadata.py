#!/usr/bin/env python3
"""Upgrade manual input JSON files with explicit metadata (manual_input_v2)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
MANUAL_RE = re.compile(r"^(pattern_\d+)_(clean|styled|real_like)_manual\.json$")


def _parse_roots(text: str) -> List[Path]:
    roots: List[Path] = []
    for token in [x.strip() for x in text.split(",") if x.strip()]:
        p = Path(token)
        roots.append(p if p.is_absolute() else (ROOT / p))
    return roots


def _canonical_image_rel(sample_id: str, domain: str) -> str:
    if domain == "clean":
        return f"data/rendered_clean/{sample_id}_clean_v1.png"
    if domain == "styled":
        return f"data/rendered_styled/{sample_id}_styled_v5.png"
    if domain == "real_like":
        return f"data/rendered_real_like/{sample_id}_real_v4.png"
    raise ValueError(f"unsupported domain: {domain}")


def _resolve_existing_image(rel_path: str, roots: List[Path]) -> Optional[str]:
    # Prefer canonical relative path if it exists from project root.
    proj_abs = ROOT / rel_path
    if proj_abs.is_file():
        return rel_path
    # Fallback: scan provided image roots by filename.
    name = Path(rel_path).name
    for base in roots:
        cand = base / name
        if cand.is_file():
            try:
                return str(cand.relative_to(ROOT))
            except ValueError:
                return str(cand)
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


CORE_KEYS = (
    "plot_box",
    "x_axis_points",
    "x_axis_values",
    "y_axis_points",
    "y_axis_values",
    "color_sample_point",
    "legend_ignore_boxes",
    "perspective_corners",
    "color_resample_points",
)


def _core_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: data.get(k) for k in CORE_KEYS}


def _already_has_expected_metadata(
    data: Dict[str, Any],
    *,
    sample_id: str,
    domain: str,
    img_ref: str,
    gt_rel: str,
) -> bool:
    return (
        data.get("schema_version") == "manual_input_v2"
        and str(data.get("sample_id", "")).strip() == sample_id
        and str(data.get("domain", "")).strip() == domain
        and str(data.get("input_image", "")).strip().replace("\\", "/") == img_ref.replace("\\", "/")
        and str(data.get("gt_json", "")).strip().replace("\\", "/") == gt_rel.replace("\\", "/")
    )


def _upgrade_one(
    manual_path: Path,
    image_roots: List[Path],
    gt_root: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    name = manual_path.name
    m = MANUAL_RE.match(name)
    row: Dict[str, Any] = {
        "manual_json": str(manual_path),
        "status": "SKIP",
        "sample_id": "",
        "domain": "",
        "input_image": "",
        "gt_json": "",
        "reason": "",
    }
    if not m:
        row["status"] = "SKIP_UNSUPPORTED_FILENAME"
        row["reason"] = "filename_not_supported"
        return row

    sample_id, domain = m.group(1), m.group(2)
    img_rel = _canonical_image_rel(sample_id, domain)
    gt_rel = f"data/gt/{sample_id}_gt.json"
    img_ref = _resolve_existing_image(img_rel, image_roots)
    gt_abs = (ROOT / gt_rel) if not Path(gt_rel).is_absolute() else Path(gt_rel)

    row["sample_id"] = sample_id
    row["domain"] = domain
    row["input_image"] = img_ref or img_rel
    row["gt_json"] = gt_rel

    missing: List[str] = []
    if img_ref is None:
        missing.append("input_image_not_found")
    if not gt_abs.is_file():
        missing.append("gt_json_not_found")
    if missing:
        row["status"] = "SKIP_MISSING_DEPS"
        row["reason"] = ";".join(missing)
        return row

    assert img_ref is not None
    data = _load_json(manual_path)
    before_core = _core_snapshot(data)

    if _already_has_expected_metadata(data, sample_id=sample_id, domain=domain, img_ref=img_ref, gt_rel=gt_rel):
        row["status"] = "ALREADY_OK"
        row["reason"] = "already_manual_input_v2"
        return row

    probe = dict(data)
    probe["schema_version"] = "manual_input_v2"
    probe["sample_id"] = sample_id
    probe["domain"] = domain
    probe["input_image"] = img_ref
    probe["gt_json"] = gt_rel
    probe["created_from"] = f"upgrade_manual_json_metadata.py@{datetime.now(timezone.utc).isoformat()}"

    after_core = _core_snapshot(probe)
    if before_core != after_core:
        row["status"] = "SKIP_INTERNAL_ERROR"
        row["reason"] = "core_fields_changed_unexpectedly"
        return row

    if dry_run:
        row["status"] = "DRY_RUN_UPDATE"
        row["reason"] = "metadata_will_be_added"
        return row

    bak_path = manual_path.with_suffix(manual_path.suffix + ".bak")
    if not bak_path.exists():
        bak_path.write_text(manual_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(manual_path, probe)
    row["status"] = "UPDATED"
    row["reason"] = "metadata_added"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-root", required=True, type=Path)
    ap.add_argument(
        "--image-root",
        required=True,
        type=str,
        help="comma-separated roots, e.g. data/rendered_clean,data/rendered_styled,data/rendered_real_like",
    )
    ap.add_argument("--gt-root", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-sample-id", type=str, default=None)
    ap.add_argument("--only-domain", type=str, choices=["clean", "styled", "real_like"], default=None)
    ap.add_argument("--report-csv", type=Path, default=None)
    args = ap.parse_args()

    manual_root = args.manual_root if args.manual_root.is_absolute() else (ROOT / args.manual_root)
    image_roots = _parse_roots(args.image_root)
    gt_root = args.gt_root if args.gt_root.is_absolute() else (ROOT / args.gt_root)
    if not gt_root.exists():
        raise SystemExit(f"gt-root not found: {gt_root}")
    if not manual_root.exists():
        raise SystemExit(f"manual-root not found: {manual_root}")

    all_manuals = sorted(manual_root.glob("*.json"))
    if args.only_sample_id:
        all_manuals = [p for p in all_manuals if args.only_sample_id in p.name]
    if args.only_domain:
        all_manuals = [p for p in all_manuals if f"_{args.only_domain}_" in p.name]

    rows: List[Dict[str, Any]] = []
    for p in all_manuals:
        rows.append(_upgrade_one(p, image_roots, gt_root, bool(args.dry_run)))

    report = (
        args.report_csv
        if args.report_csv
        else (manual_root / "metadata_upgrade_report.csv")
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["manual_json", "status", "sample_id", "domain", "input_image", "gt_json", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {report}")
    for r in rows:
        print(
            f"{r['status']:>14} | {r['sample_id']:<14} | {r['domain']:<9} | {Path(r['manual_json']).name} | {r['reason']}"
        )


if __name__ == "__main__":
    main()
