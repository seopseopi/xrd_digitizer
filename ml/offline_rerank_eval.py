#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _topk_has_positive(cands: List[Dict[str, Any]], key: str, k: int) -> bool:
    ordered = sorted(cands, key=lambda r: float(r.get(key, 0.0)), reverse=True)
    return any(int(r.get("label", 0)) == 1 for r in ordered[:k])


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline re-rank evaluation (candidate-level)")
    ap.add_argument("--pred_jsonl", type=str, required=True, help="infer_candidate_reranker output")
    ap.add_argument("--lambda_model", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    rows = _load_jsonl(Path(args.pred_jsonl))
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r.get("sample_id")), int(r.get("x", -1)))
        if key[1] < 0:
            continue
        rr = dict(r)
        rr["score_before"] = float(rr.get("rule_confidence", 0.0))
        rr["score_after"] = float(rr.get("rule_confidence", 0.0)) + args.lambda_model * float(rr.get("model_score_delta", 0.0))
        groups[key].append(rr)

    n_cols = len(groups)
    before_hit = 0
    after_hit = 0
    hard_neg_total = 0
    hard_neg_demoted = 0
    for _k, cands in groups.items():
        if _topk_has_positive(cands, "score_before", args.topk):
            before_hit += 1
        if _topk_has_positive(cands, "score_after", args.topk):
            after_hit += 1

        top_before = sorted(cands, key=lambda r: r["score_before"], reverse=True)[:1]
        top_after = sorted(cands, key=lambda r: r["score_after"], reverse=True)[:1]
        if top_before:
            hb = top_before[0].get("hard_negative_type")
            if hb:
                hard_neg_total += 1
                ha = top_after[0].get("hard_negative_type")
                if not ha:
                    hard_neg_demoted += 1

    out = {
        "num_columns": n_cols,
        "topk": args.topk,
        "lambda_model": args.lambda_model,
        "topk_recall_before": before_hit / max(1, n_cols),
        "topk_recall_after": after_hit / max(1, n_cols),
        "delta_topk_recall": (after_hit - before_hit) / max(1, n_cols),
        "hard_negative_suppression_rate": hard_neg_demoted / max(1, hard_neg_total),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
