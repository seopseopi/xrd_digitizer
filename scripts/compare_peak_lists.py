#!/usr/bin/env python3
"""pattern_*_result.json 에서 peaks_numeric_curve 길이 출력 (로드맵 1단계 확인).

  python scripts/compare_peak_lists.py outputs/runs/760_미세피크v4/pattern_760_result.json

이미지 기반 peak_positions_2theta 는 debug_pattern_*/debug.json → calibration.peak_positions_2theta
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: compare_peak_lists.py <pattern_*_result.json>", file=sys.stderr)
        return 1
    p = Path(sys.argv[1])
    if not p.is_file():
        print(f"Not found: {p}", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    n = len(data.get("peaks_numeric_curve", []))
    print(f"peaks_numeric_curve: {n} entries")
    if n and data.get("peaks_numeric_curve"):
        ex = data["peaks_numeric_curve"][0]
        print("  example keys:", list(ex.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
