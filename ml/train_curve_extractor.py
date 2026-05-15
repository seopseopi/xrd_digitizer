"""
CurveExtractorNet 학습 스크립트.

사용법:
  cd xrd_digitizer_v1
  python -m ml.train_curve_extractor [--epochs 100] [--batch 8]

데이터:
  data/rendered_styled/*.png  + data/rendered_clean/*.png
  data/gt/*_gt.json (per_column_y_gt 필드)

학습 전략:
  - Encoder (ResNet-18) lr 10× 낮춤 (pretrained 보존)
  - Decoder lr 정상
  - CosineAnnealing + warmup
  - Augmentation: flip, brightness/contrast, color jitter, gaussian noise
  - Loss: Huber (outlier 강건) + smoothness penalty
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.curve_extractor import H_IN, W_IN, build_model

# ── 데이터셋 ──────────────────────────────────────────────────────────────────

class CurveDataset(Dataset):
    def __init__(self, samples: list[dict], augment: bool = True) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        x0, y0, x1, y1 = s['plot_box']
        roi_w, roi_h = x1 - x0, y1 - y0

        # ROI 크롭
        roi = img.crop((x0, y0, x1, y1))

        # GT → ROI-local 정규화 y
        col_gt: dict = s['per_column_y_gt']
        gt_arr = np.full(roi_w, np.nan, dtype=np.float32)
        for col_str, y_abs in col_gt.items():
            col = int(col_str) - x0
            if 0 <= col < roi_w:
                gt_arr[col] = float(np.clip((y_abs - y0) / roi_h, 0.0, 1.0))

        # NaN 보간
        nans = np.isnan(gt_arr)
        if nans.all():
            gt_arr[:] = 0.5
        elif nans.any():
            xs = np.arange(roi_w)
            gt_arr = np.interp(xs, xs[~nans], gt_arr[~nans]).astype(np.float32)

        if self.augment:
            roi, gt_arr = _augment(roi, gt_arr)

        # 리사이즈 + GT 보간
        roi = roi.resize((W_IN, H_IN), Image.BILINEAR)
        gt_resampled = np.interp(
            np.linspace(0, len(gt_arr) - 1, W_IN),
            np.arange(len(gt_arr)),
            gt_arr,
        ).astype(np.float32)

        img_t = torch.from_numpy(
            np.array(roi, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).contiguous()
        gt_t = torch.from_numpy(gt_resampled)
        return img_t, gt_t


def _augment(roi: Image.Image, gt: np.ndarray):
    """학습용 augmentation."""
    # 좌우 반전
    if random.random() < 0.5:
        roi = roi.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt[::-1].copy()

    # 밝기
    roi = ImageEnhance.Brightness(roi).enhance(random.uniform(0.75, 1.35))
    # 대비
    roi = ImageEnhance.Contrast(roi).enhance(random.uniform(0.75, 1.35))
    # 채도 (컬러 곡선 스타일 대응)
    roi = ImageEnhance.Color(roi).enhance(random.uniform(0.5, 1.5))

    # Gaussian blur (저해상도 이미지 시뮬레이션)
    if random.random() < 0.25:
        roi = roi.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    # Gaussian noise
    arr = np.array(roi, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(0, 8), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    roi = Image.fromarray(arr)

    return roi, gt


def build_samples(data_dir: Path) -> list[dict]:
    gt_dir     = data_dir / 'gt'
    styled_dir = data_dir / 'rendered_styled'
    clean_dir  = data_dir / 'rendered_clean'

    samples = []
    for gt_path in sorted(gt_dir.glob('*_gt.json')):
        try:
            gt = json.loads(gt_path.read_text())
        except Exception:
            continue

        pid   = gt_path.stem.replace('_gt', '')
        box   = gt.get('plot_box')
        col_gt = gt.get('per_column_y_gt')
        if not box or not col_gt:
            continue

        for v in ['v1', 'v3', 'v5']:
            p = styled_dir / f'{pid}_styled_{v}.png'
            if p.exists():
                samples.append({'img_path': str(p), 'plot_box': box,
                                 'per_column_y_gt': col_gt})

        for p in sorted(clean_dir.glob(f'{pid}_clean_*.png')):
            samples.append({'img_path': str(p), 'plot_box': box,
                             'per_column_y_gt': col_gt})
    return samples


# ── 학습 ─────────────────────────────────────────────────────────────────────

def make_optimizer(model, lr: float):
    """Encoder (pretrained) lr 10× 낮춤."""
    enc_params, dec_params = [], []
    enc_names = {'enc0', 'pool', 'enc1', 'enc2', 'enc3', 'enc4'}
    for name, param in model.named_parameters():
        if any(name.startswith(n) for n in enc_names):
            enc_params.append(param)
        else:
            dec_params.append(param)
    return optim.AdamW([
        {'params': enc_params, 'lr': lr / 10},
        {'params': dec_params, 'lr': lr},
    ], weight_decay=1e-4)


def warmup_cosine(optimizer, epoch: int, warmup: int, total: int) -> None:
    if epoch < warmup:
        scale = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (total - warmup)
        scale = 0.5 * (1 + np.cos(np.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * scale


def pixel_mae(pred: torch.Tensor, gt: torch.Tensor, roi_h: float = 690.0) -> float:
    """정규화 오차 → 픽셀 MAE."""
    return float((pred - gt).abs().mean().item() * roi_h)


def train(args: argparse.Namespace) -> None:
    data_dir = ROOT / 'data'
    all_samples = build_samples(data_dir)
    if not all_samples:
        print('[ERROR] 학습 데이터 없음.', file=sys.stderr)
        sys.exit(1)

    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * 0.15))
    val_s, train_s = all_samples[:n_val], all_samples[n_val:]

    print(f'학습: {len(train_s)}개  /  검증: {len(val_s)}개')

    train_loader = DataLoader(
        CurveDataset(train_s, augment=True),
        batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=False, persistent_workers=True,
    )
    val_loader = DataLoader(
        CurveDataset(val_s, augment=False),
        batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=False, persistent_workers=True,
    )

    device = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    model = build_model(pretrained=True).to(device)

    optimizer = make_optimizer(model, args.lr)
    # initial_lr 기록 (warmup용)
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    huber = nn.HuberLoss(delta=0.02)   # ~14px at 690px height

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_px = float('inf')

    for epoch in range(args.epochs):
        warmup_cosine(optimizer, epoch, warmup=5, total=args.epochs)
        cur_lr = optimizer.param_groups[1]['lr']   # decoder lr

        # ── 학습 ──────────────────────────────────────────────────────────
        model.train()
        train_px_sum = 0.0
        for imgs, gts in train_loader:
            imgs, gts = imgs.to(device), gts.to(device)
            preds = model(imgs)                          # B, W_IN

            loss = huber(preds, gts)
            # 연속성 페널티: 급격한 불연속 억제
            smooth = ((preds[:, 1:] - preds[:, :-1]) ** 2).mean()
            total  = loss + 0.005 * smooth

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_px_sum += pixel_mae(preds.detach(), gts)

        train_px = train_px_sum / len(train_loader)

        # ── 검증 ──────────────────────────────────────────────────────────
        model.eval()
        val_px_sum = 0.0
        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs, gts = imgs.to(device), gts.to(device)
                preds = model(imgs)
                val_px_sum += pixel_mae(preds, gts)
        val_px = val_px_sum / len(val_loader)

        ep1 = epoch + 1
        if ep1 % 10 == 0 or ep1 == 1:
            print(f'[{ep1:3d}/{args.epochs}]  '
                  f'train={train_px:.2f}px  val={val_px:.2f}px  '
                  f'lr={cur_lr:.2e}')

        if val_px < best_val_px:
            best_val_px = val_px
            torch.save({
                'epoch': ep1,
                'state_dict': model.state_dict(),
                'val_px': val_px,
                'H_IN': H_IN,
                'W_IN': W_IN,
            }, out_path)

    print(f'\n✓ 최적 모델 저장: {out_path}')
    print(f'  최적 검증 MAE: {best_val_px:.2f}px  (ROI 높이 690px 기준)')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int,   default=100)
    p.add_argument('--batch',  type=int,   default=8)
    p.add_argument('--lr',     type=float, default=3e-4)
    p.add_argument('--out',    type=str,   default='ml/weights/curve_extractor.pt')
    train(p.parse_args())
