#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ml.data.candidate_patch_dataset import CandidatePatchDataset
from ml.models.candidate_reranker_cnn import SmallCandidateCNN


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer candidate reranker scores")
    ap.add_argument("--model_ckpt", type=str, required=True)
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ds = CandidatePatchDataset(args.input_jsonl)
    ckpt = torch.load(args.model_ckpt, map_location=args.device)
    model = SmallCandidateCNN(in_channels=int(ckpt.get("in_channels", 3))).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for arr, _y, row in ds:
            x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(args.device)
            score = float(model(x).squeeze().cpu().item())
            r = dict(row)
            r["model_score_delta"] = score
            out_rows.append(r)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
