"""
§23.7: 데이터셋 무결성 검사.
- image 존재, gt 존재, source json 존재, sample_id 중복 여부
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset integrity from manifest CSV")
    parser.add_argument("--manifest_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    errors: list[str] = []

    img_col = None
    for c in ("image_path", "styled_image_path", "real_image_path"):
        if c in df.columns:
            img_col = c
            break

    if img_col:
        missing_img = df[~df[img_col].apply(lambda p: Path(str(p)).exists())]
        if len(missing_img):
            errors.append(f"Missing images ({img_col}): {len(missing_img)}")

    if "gt_path" in df.columns:
        missing_gt = df[~df["gt_path"].apply(lambda p: Path(str(p)).exists())]
        if len(missing_gt):
            errors.append(f"Missing GT files: {len(missing_gt)}")

    if "source_json_path" in df.columns:
        missing_src = df[~df["source_json_path"].apply(lambda p: Path(str(p)).exists())]
        if len(missing_src):
            errors.append(f"Missing source JSON: {len(missing_src)}")

    if "sample_id" in df.columns:
        dupes = df["sample_id"].duplicated().sum()
        if dupes > 0:
            errors.append(f"Duplicate sample_ids: {dupes}")

    if errors:
        print("[WARN] Integrity issues found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[OK] All integrity checks passed.")


if __name__ == "__main__":
    main()
