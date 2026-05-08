#!/usr/bin/env python3
"""Build a canonical 30-item test set with image, source JSON, GT, synthetic MI, and metadata.

This script creates a self-contained test data folder under data/test_canonical_30.
It does not run digitization, tune models, or modify the original source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "test_canonical_30"

CLEAN_IDS = [
    "pattern_11832",
    "pattern_72296",
    "pattern_60890",
    "pattern_74680",
    "pattern_83520",
    "pattern_69527",
    "pattern_69750",
    "pattern_72074",
    "pattern_73604",
    "pattern_74346",
    "pattern_74347",
    "pattern_74381",
    "pattern_74392",
    "pattern_74902",
    "pattern_75013",
    "pattern_75068",
    "pattern_75079",
    "pattern_75375",
    "pattern_80400",
    "pattern_80510",
]

STYLED_IDS = [
    "pattern_72296",
    "pattern_74680",
    "pattern_60890",
    "pattern_69527",
    "pattern_69750",
]

REAL_LIKE_IDS = [
    "pattern_60890",
    "pattern_72296",
    "pattern_74680",
    "pattern_69527",
    "pattern_83398",
]

MANIFEST_FIELDS = [
    "test_id",
    "sample_id",
    "domain",
    "test_group",
    "input_image",
    "source_numeric_json",
    "gt_json",
    "mi_json",
    "metadata_json",
    "preview_full_plotbox",
    "preview_roi_crop",
    "render_variant",
    "mi_source",
    "mi_role",
    "image_width",
    "image_height",
    "plot_box",
    "roi_width",
    "roi_height",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "sha256_input_image",
    "sha256_source_numeric_json",
    "sha256_gt_json",
    "sha256_mi_json",
    "pair_status",
    "selection_reason",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def image_path_for(sample_id: str, domain: str) -> Tuple[Path, str]:
    if domain == "clean":
        return ROOT / "data" / "rendered_clean" / f"{sample_id}_clean_v1.png", "clean_v1"
    if domain == "styled":
        return ROOT / "data" / "rendered_styled" / f"{sample_id}_styled_v5.png", "styled_v5"
    if domain == "real_like":
        return ROOT / "data" / "rendered_real_like" / f"{sample_id}_real_v4.png", "real_v4"
    raise ValueError(f"Unsupported domain: {domain}")


def gt_path_for(sample_id: str) -> Path:
    return ROOT / "data" / "gt" / f"{sample_id}_gt.json"


def source_path_for(sample_id: str) -> Path:
    return ROOT / "data" / "source_json" / f"{sample_id}.json"


def synthetic_mi_from_gt(gt: Dict[str, Any]) -> Dict[str, Any]:
    am = gt["axis_metadata"]
    x0, y0, x1, y1 = [int(v) for v in gt["plot_box"]]
    pixel_curve = gt.get("pixel_curve_path") or []
    if len(pixel_curve) > 2:
        mid = pixel_curve[len(pixel_curve) // 2]
        color_sample_point = [int(round(mid[0])), int(round(mid[1]))]
    else:
        color_sample_point = [int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))]

    return {
        "schema_version": "manual_input_v2",
        "mi_source": "synthetic_from_gt",
        "mi_role": "calibration_input",
        "plot_box": [x0, y0, x1, y1],
        "x_axis_points": [[x0, y1], [x1, y1]],
        "x_axis_values": [float(am["x_min"]), float(am["x_max"])],
        "y_axis_points": [[x0, y1], [x0, y0]],
        "y_axis_values": [float(am["y_min"]), float(am["y_max"])],
        "color_sample_point": color_sample_point,
        "legend_ignore_boxes": [],
        "perspective_corners": None,
        "color_resample_points": [],
    }


def render_previews(image_path: Path, plot_box: List[int], full_out: Path, roi_out: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        x0, y0, x1, y1 = [int(v) for v in plot_box]

        full = img.copy()
        draw = ImageDraw.Draw(full)
        for offset in range(3):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=(255, 0, 0))
        full_out.parent.mkdir(parents=True, exist_ok=True)
        full.save(full_out)

        crop = img.crop((x0, y0, x1, y1))
        roi_out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(roi_out)

    return w, h


def test_group(sample_id: str, domain: str) -> Tuple[str, str]:
    if sample_id == "pattern_72296":
        return "known_failure_anchor", "includes pattern_72296 DP/candidate failure anchor"
    if sample_id == "pattern_11832":
        return "clean_smoke_anchor", "replaces pattern_1915 as clean smoke anchor"
    if sample_id == "pattern_60890":
        return "cross_domain_anchor", "includes styled/real_like cross-domain anchor"
    if domain == "clean":
        return "clean_coverage", "clean canonical coverage sample"
    if domain == "styled":
        return "styled_coverage", "styled_v5 canonical coverage sample"
    return "real_like_coverage", "real_v4 canonical coverage sample"


def selected_rows() -> Iterable[Tuple[str, str]]:
    for sid in CLEAN_IDS:
        yield sid, "clean"
    for sid in STYLED_IDS:
        yield sid, "styled"
    for sid in REAL_LIKE_IDS:
        yield sid, "real_like"


def build(out_root: Path, overwrite: bool = False) -> Dict[str, Any]:
    if out_root.exists() and not overwrite:
        raise FileExistsError(f"{out_root} already exists; pass --overwrite to rebuild")
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for sample_id, domain in selected_rows():
        test_id = f"{domain}_{sample_id}"
        item_dir = out_root / domain / sample_id
        image_src, render_variant = image_path_for(sample_id, domain)
        gt_src = gt_path_for(sample_id)
        source_src = source_path_for(sample_id)

        try:
            ensure_file(image_src, "input_image")
            ensure_file(gt_src, "gt_json")
            ensure_file(source_src, "source_numeric_json")

            input_dst = item_dir / "input.png"
            gt_dst = item_dir / "gt.json"
            source_dst = item_dir / "source_numeric.json"
            mi_dst = item_dir / "mi.json"
            metadata_dst = item_dir / "metadata.json"
            preview_full = item_dir / "preview_full_plotbox.png"
            preview_roi = item_dir / "preview_roi_crop.png"

            item_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_src, input_dst)
            shutil.copy2(gt_src, gt_dst)
            shutil.copy2(source_src, source_dst)

            gt = read_json(gt_src)
            mi = synthetic_mi_from_gt(gt)
            mi.update(
                {
                    "sample_id": sample_id,
                    "domain": domain,
                    "input_image": rel(input_dst),
                    "gt_json": rel(gt_dst),
                    "source_numeric_json": rel(source_dst),
                    "created_from": f"build_test_canonical_30.py@{datetime.now(timezone.utc).isoformat()}",
                }
            )
            write_json(mi_dst, mi)

            image_w, image_h = render_previews(input_dst, mi["plot_box"], preview_full, preview_roi)
            x0, y0, x1, y1 = [int(v) for v in mi["plot_box"]]
            am = gt["axis_metadata"]
            group, reason = test_group(sample_id, domain)

            hashes = {
                "input_image": sha256_file(input_dst),
                "source_numeric_json": sha256_file(source_dst),
                "gt_json": sha256_file(gt_dst),
                "mi_json": sha256_file(mi_dst),
            }
            metadata = {
                "schema_version": "canonical_test_item_v1",
                "test_id": test_id,
                "sample_id": sample_id,
                "domain": domain,
                "test_group": group,
                "render_variant": render_variant,
                "mi_source": "synthetic_from_gt",
                "mi_role": "calibration_input",
                "source_paths": {
                    "input_image": rel(image_src),
                    "source_numeric_json": rel(source_src),
                    "gt_json": rel(gt_src),
                },
                "canonical_paths": {
                    "input_image": rel(input_dst),
                    "source_numeric_json": rel(source_dst),
                    "gt_json": rel(gt_dst),
                    "mi_json": rel(mi_dst),
                    "preview_full_plotbox": rel(preview_full),
                    "preview_roi_crop": rel(preview_roi),
                },
                "image_size": [image_w, image_h],
                "plot_box": mi["plot_box"],
                "roi_size": [x1 - x0, y1 - y0],
                "axis_metadata": am,
                "sha256": hashes,
                "pair_status": "PASS",
                "selection_reason": reason,
                "created_from": "build_test_canonical_30.py",
            }
            write_json(metadata_dst, metadata)

            manifest_rows.append(
                {
                    "test_id": test_id,
                    "sample_id": sample_id,
                    "domain": domain,
                    "test_group": group,
                    "input_image": rel(input_dst),
                    "source_numeric_json": rel(source_dst),
                    "gt_json": rel(gt_dst),
                    "mi_json": rel(mi_dst),
                    "metadata_json": rel(metadata_dst),
                    "preview_full_plotbox": rel(preview_full),
                    "preview_roi_crop": rel(preview_roi),
                    "render_variant": render_variant,
                    "mi_source": "synthetic_from_gt",
                    "mi_role": "calibration_input",
                    "image_width": image_w,
                    "image_height": image_h,
                    "plot_box": json.dumps(mi["plot_box"]),
                    "roi_width": x1 - x0,
                    "roi_height": y1 - y0,
                    "x_min": float(am["x_min"]),
                    "x_max": float(am["x_max"]),
                    "y_min": float(am["y_min"]),
                    "y_max": float(am["y_max"]),
                    "sha256_input_image": hashes["input_image"],
                    "sha256_source_numeric_json": hashes["source_numeric_json"],
                    "sha256_gt_json": hashes["gt_json"],
                    "sha256_mi_json": hashes["mi_json"],
                    "pair_status": "PASS",
                    "selection_reason": reason,
                }
            )
        except Exception as exc:  # keep building report if a row is bad
            failures.append(f"{test_id}: {exc}")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    domain_counts: Dict[str, int] = {}
    for row in manifest_rows:
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1

    summary = {
        "final_decision": "CANONICAL_TEST_30_PASS" if len(manifest_rows) == 30 and not failures else "CANONICAL_TEST_30_FAIL",
        "out_root": rel(out_root),
        "manifest": rel(manifest_path),
        "total_rows": len(manifest_rows),
        "domain_counts": domain_counts,
        "mi_source": "synthetic_from_gt",
        "pair_status_counts": {"PASS": len(manifest_rows), "FAIL": len(failures)},
        "failures": failures,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_json = out_root / "summary.json"
    write_json(summary_json, summary)

    summary_md = out_root / "README.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Canonical Test 30",
                "",
                "This folder is a self-contained canonical test set for deterministic XRD digitizer experiments.",
                "",
                f"- Decision: `{summary['final_decision']}`",
                f"- Total rows: {summary['total_rows']}",
                f"- Domain counts: {json.dumps(domain_counts, ensure_ascii=False, sort_keys=True)}",
                "- MI source: `synthetic_from_gt`",
                "- MI role: `calibration_input`",
                "- No original source files were modified.",
                "",
                "Each item contains:",
                "- `input.png`",
                "- `source_numeric.json`",
                "- `gt.json`",
                "- `mi.json`",
                "- `metadata.json`",
                "- `preview_full_plotbox.png`",
                "- `preview_roi_crop.png`",
                "",
                "Use `manifest.csv` as the only execution source for tests.",
                "",
                "Failures:",
                *(f"- {x}" for x in failures),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/test_canonical_30")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build(Path(args.out_root), overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["final_decision"] != "CANONICAL_TEST_30_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
