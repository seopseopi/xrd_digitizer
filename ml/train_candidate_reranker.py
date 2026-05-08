#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ml.data.candidate_patch_dataset import CandidatePatchDataset, split_label_stats
from ml.models.candidate_reranker_cnn import SmallCandidateCNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class _TorchPatchDataset(Dataset):
    def __init__(self, base: CandidatePatchDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr, y, _ = self.base[idx]
        return torch.from_numpy(arr), torch.tensor(y, dtype=torch.float32)


def _build_loader(jsonl_path: str, batch_size: int, shuffle: bool) -> DataLoader:
    ds = CandidatePatchDataset(jsonl_path)
    return DataLoader(_TorchPatchDataset(ds), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _run_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer | None, device: str, pos_weight: float) -> Dict[str, float]:
    train_mode = optim is not None
    model.train(train_mode)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    total_loss = 0.0
    n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.float32)
        raw = model(xb)
        logits = torch.atanh(torch.clamp(raw, -0.999, 0.999))
        loss = loss_fn(logits, yb)

        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += float(loss.item()) * len(xb)
        n += len(xb)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((pred == yb).sum().item())

    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train candidate reranker CNN")
    ap.add_argument("--data_dir", type=str, required=True, help="contains train.jsonl/val.jsonl")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"
    if not train_jsonl.is_file() or not val_jsonl.is_file():
        raise FileNotFoundError("train.jsonl and val.jsonl are required")

    train_stats = split_label_stats(CandidatePatchDataset(str(train_jsonl)))
    pos_w = float(train_stats["negative"]) / max(float(train_stats["positive"]), 1.0)

    train_loader = _build_loader(str(train_jsonl), args.batch_size, True)
    val_loader = _build_loader(str(val_jsonl), args.batch_size, False)

    model = SmallCandidateCNN(in_channels=3).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_ckpt = out_dir / "candidate_reranker_v1.pt"

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, optim, args.device, pos_w)
        va = _run_epoch(model, val_loader, None, args.device, pos_w)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": 3,
                    "pos_weight": pos_w,
                },
                str(best_ckpt),
            )

    (out_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(
        json.dumps({"best_val_loss": best_val, "checkpoint": str(best_ckpt), "train_stats": train_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
