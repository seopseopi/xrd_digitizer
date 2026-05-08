from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class CandidatePatchDataset:
    def __init__(self, jsonl_path: str) -> None:
        self.path = Path(jsonl_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        row = self.rows[idx]
        arr = np.load(row["patch_path"]).astype(np.float32)
        y = float(row["label"])
        return arr, y, row


def split_label_stats(ds: CandidatePatchDataset) -> Dict[str, int]:
    pos = sum(1 for r in ds.rows if int(r.get("label", 0)) == 1)
    neg = len(ds.rows) - pos
    return {"total": len(ds.rows), "positive": pos, "negative": neg}
