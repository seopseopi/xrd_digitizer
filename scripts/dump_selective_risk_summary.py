#!/usr/bin/env python3
"""selective arm debug.json 의 risk_summary 단계별 필드 덤프."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _norm_sid(sample_id: str) -> str:
    s = sample_id.strip()
    return s[: -len("_result")] if s.endswith("_result") else s


def _risk_summary_from_debug(d: Dict[str, Any]) -> Dict[str, Any]:
    ma = d.get("model_assist") or {}
    # selective 실행 시 oracle_rerank 메타에 risk_summary 가 붙음
    orac = ma.get("oracle_rerank") or {}
    rs = orac.get("risk_summary")
    if isinstance(rs, dict):
        return rs
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--domain", required=True)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    sid = _norm_sid(args.sample_id)
    dom = str(args.domain)
    dbg = root / "runs" / f"{dom}_{sid}" / "selective" / f"debug_{sid}_selective" / "debug.json"
    if not dbg.is_file():
        raise FileNotFoundError(dbg)

    d = json.loads(dbg.read_text(encoding="utf-8"))
    rs = _risk_summary_from_debug(d)
    print(f"path={dbg}\n")
    keys = [
        "risk_column_count_raw",
        "risk_column_count_after_taxonomy_prior",
        "risk_column_count_after_dilation",
        "risk_column_count_after_merge",
        "risk_ratio_raw",
        "risk_ratio_after_taxonomy_prior",
        "risk_ratio_after_dilation",
        "risk_ratio_after_merge",
        "taxonomy_prior_labels",
        "taxonomy_prior",
        "taxonomy_prior_disabled_for_risk",
        "raw_risk_reason_counts",
        "raw_risk_reason_ratios",
        "risk_ratio",
        "total_column_count",
    ]
    for k in keys:
        print(f"{k}: {rs.get(k)!r}")


if __name__ == "__main__":
    main()
