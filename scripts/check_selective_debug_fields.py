#!/usr/bin/env python3
"""Selective oracle 실행으로 생성된 debug.json만 검증한다(일반 rule 실행은 건너뜀)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _trace_valid_count(trace: dict) -> Optional[int]:
    if "valid_count" in trace:
        return int(trace["valid_count"])
    d = trace.get("diagnostics") or {}
    if "ok_columns" in d:
        return int(d["ok_columns"])
    return None


def _trace_total_count(trace: dict) -> Optional[int]:
    if "total_count" in trace:
        return int(trace["total_count"])
    d = trace.get("diagnostics") or {}
    for k in ("ok_columns", "starvation_columns", "path_choice_columns"):
        if k not in d:
            return None
    return int(d["ok_columns"] + d["starvation_columns"] + d["path_choice_columns"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Selective oracle 결과 트리에서 debug.json 필수 필드 검증",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="outputs/_sel_oracle_smoke",
        help="검색 루트(하위 rglob debug.json)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("debug.json"))
    if not files:
        print(f"No debug.json found under: {root}", file=sys.stderr)
        return 1

    ok = 0
    skipped = 0
    for path in files:
        try:
            debug = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[FAIL] {path}: cannot read json: {e}", file=sys.stderr)
            return 1

        model_assist = debug.get("model_assist") or {}
        if "selective_oracle" not in model_assist:
            skipped += 1
            continue

        trace = debug.get("trace") or {}
        missing: list[str] = []
        if "valid_ratio" not in trace:
            missing.append("trace.valid_ratio")
        if _trace_valid_count(trace) is None:
            missing.append("trace.valid_count (또는 diagnostics.ok_columns)")
        if _trace_total_count(trace) is None:
            missing.append("trace.total_count (또는 diagnostics 열 합)")

        if missing:
            print(f"[FAIL] {path}: missing {', '.join(missing)}", file=sys.stderr)
            return 1

        vc = _trace_valid_count(trace)
        tc = _trace_total_count(trace)
        print(
            f"[OK] {path} valid_ratio={trace.get('valid_ratio')} "
            f"valid_count={vc} total_count={tc}",
        )
        ok += 1

    if ok == 0:
        print(
            f"No selective-oracle debug.json under {root} "
            f"(skipped non-selective: {skipped})",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: checked {ok} selective debug.json (skipped {skipped} non-selective)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
