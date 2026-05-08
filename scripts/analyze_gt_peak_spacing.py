#!/usr/bin/env python3
"""
로드맵 3b: GT 의 peak_indices(축 따라 샘플 인덱스) 간 최소 간격 분포를 내어
detect_peaks 의 min_peak_distance = max(3, round(0.004 * num_points)) 와 비교한다.

실행:
  python3 scripts/analyze_gt_peak_spacing.py --repo-root .
  python3 scripts/analyze_gt_peak_spacing.py --repo-root . --out experiments/gt_peak_spacing_stats.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _formula_distance(num_pts: int) -> int:
    return max(3, round(0.004 * num_pts))


def main() -> None:
    ap = argparse.ArgumentParser(description="GT peak index spacing statistics")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None, help="요약 텍스트 저장 경로")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    gt_dir = root / "data" / "gt"
    files = sorted(gt_dir.glob("*_gt.json"))

    rows: list[tuple[str, int, int, int]] = []
    # sample_id, n_pts, n_peaks, min_neighbor_gap (sorted consecutive along axis)

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        xv = data.get("x_values")
        peaks = data.get("peak_indices") or []
        if not isinstance(xv, list) or len(xv) < 10:
            continue
        n_pts = len(xv)
        if len(peaks) < 2:
            continue
        peaks_sorted = sorted(int(p) for p in peaks)
        gaps = [peaks_sorted[i + 1] - peaks_sorted[i] for i in range(len(peaks_sorted) - 1)]
        min_gap = min(gaps)
        sid = data.get("sample_id") or fp.stem.replace("_gt", "")
        rows.append((str(sid), n_pts, len(peaks_sorted), min_gap))

    if not rows:
        print("No GT samples with >=2 peaks.", file=sys.stderr)
        sys.exit(1)

    min_gaps = sorted(r[3] for r in rows)
    n_pts_list = [r[1] for r in rows]

    def pct(arr: list[int], q: float) -> float:
        if not arr:
            return 0.0
        k = (len(arr) - 1) * q
        f = int(k)
        c = min(f + 1, len(arr) - 1)
        return float(arr[f] + (arr[c] - arr[f]) * (k - f))

    below_formula = 0
    for sid, n_pts, _np, mgap in rows:
        if mgap < _formula_distance(n_pts):
            below_formula += 1

    lines = []
    lines.append("GT peak spacing (peak_indices along trace)")
    lines.append(f"samples_with_>=2_peaks: {len(rows)}  (of {len(files)} gt json)")
    lines.append("")
    lines.append("min_neighbor_gap_along_index (closest peak pair on sorted indices)")
    lines.append(f"  min={min_gaps[0]}  p10={pct(min_gaps, 0.10):.1f}  p50={pct(min_gaps, 0.50):.1f}  p90={pct(min_gaps, 0.90):.1f}  max={min_gaps[-1]}")
    lines.append("")
    lines.append("trace length x_values (num_points)")
    np_sorted = sorted(n_pts_list)
    lines.append(f"  min={np_sorted[0]}  p50={pct(np_sorted, 0.50):.0f}  max={np_sorted[-1]}")
    lines.append("")
    lines.append("detect_peaks uses min_peak_distance = max(3, round(0.004 * num_points)) on VALID trace length.")
    lines.append(f"fraction of GT charts where min_gap < formula_distance(N): {below_formula / len(rows):.3f}")
    lines.append("")
    lines.append("권장 (문서용)")
    lines.append("- 근접 피크가 많은 도메인은 scipy distance만으로 부족할 수 있어 3a NMS·2패스가 보조.")
    lines.append("- N이 매우 작을 때 formula가 3에 붙으므로, GT 최소간격이 3~4인 샘플은 커널/스무딩과 함께 검토.")
    text = "\n".join(lines) + "\n"

    print(text, end="")
    if args.out:
        out_p = args.out.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
        print(f"[OK] wrote {out_p}", file=sys.stderr)


if __name__ == "__main__":
    main()
