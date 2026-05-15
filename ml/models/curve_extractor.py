"""
CurveExtractorNet — ResNet-18 backbone + FPN decoder + soft-argmax

구조:
  Encoder : ResNet-18 (ImageNet pretrained) — 에지/텍스처 사전지식 활용
  Decoder : FPN-style skip-connection 업샘플링 — 공간 해상도 복원
  Head    : 1채널 heatmap → column-wise soft-argmax → 서브픽셀 y좌표

입력:  (B, 3, H_IN, W_IN)
출력:  (B, W_IN)  — 정규화 y좌표 [0, 1], 0=top, 1=bottom

학습 데이터: 250개 (50 patterns × v1/v3/v5 + clean)
파라미터 수: ~14M (ResNet-18 backbone 포함)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# 학습/추론 고정 입력 해상도
H_IN = 256
W_IN = 512


# ── 유틸 블록 ─────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, p: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class _UpBlock(nn.Module):
    """Bilinear 업샘플 + Conv → skip connection 합산 후 정제."""

    def __init__(self, cin: int, skip_ch: int, cout: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            _ConvBnRelu(cin + skip_ch, cout),
            _ConvBnRelu(cout, cout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 크기 불일치 보정
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ── 메인 모델 ─────────────────────────────────────────────────────────────────

class CurveExtractorNet(nn.Module):
    """
    ResNet-18 encoder + FPN decoder + soft-argmax head.

    Soft-argmax: heatmap을 H 방향 확률분포로 보고 기댓값을 y좌표로 사용.
    - 완전 미분 가능 → end-to-end 학습
    - 서브픽셀 정밀도
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ── Encoder (ResNet-18) ──────────────────────────────────────────────
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = resnet18(weights=weights)

        # 각 스테이지 분리해서 중간 feature map 접근
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)   # /2,   64ch
        self.pool = bb.maxpool                                  # /4
        self.enc1 = bb.layer1   # /4,  64ch
        self.enc2 = bb.layer2   # /8,  128ch
        self.enc3 = bb.layer3   # /16, 256ch
        self.enc4 = bb.layer4   # /32, 512ch

        # ── Decoder (FPN-style) ─────────────────────────────────────────────
        self.dec4 = _UpBlock(512, 256, 256)    # /32 → /16
        self.dec3 = _UpBlock(256, 128, 128)    # /16 → /8
        self.dec2 = _UpBlock(128,  64,  64)    # /8  → /4
        self.dec1 = _UpBlock( 64,  64,  32)    # /4  → /2

        # ── Heatmap head ────────────────────────────────────────────────────
        # (B, 32, H/2, W/2) → upsample → (B, 1, H, W)
        self.heatmap_head = nn.Sequential(
            _ConvBnRelu(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 행 인덱스 버퍼 (soft-argmax용)
        self.register_buffer(
            'row_idx',
            torch.linspace(0.0, 1.0, H_IN).view(1, H_IN, 1),  # (1, H, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H_IN, W_IN)
        → (B, W_IN) 정규화 y 좌표 [0, 1]
        """
        # Encoder
        e0 = self.enc0(x)           # B, 64,  H/2, W/2
        p  = self.pool(e0)          # B, 64,  H/4, W/4
        e1 = self.enc1(p)           # B, 64,  H/4, W/4
        e2 = self.enc2(e1)          # B, 128, H/8, W/8
        e3 = self.enc3(e2)          # B, 256, H/16,W/16
        e4 = self.enc4(e3)          # B, 512, H/32,W/32

        # Decoder
        d = self.dec4(e4, e3)       # B, 256, H/16,W/16
        d = self.dec3(d,  e2)       # B, 128, H/8, W/8
        d = self.dec2(d,  e1)       # B, 64,  H/4, W/4
        d = self.dec1(d,  e0)       # B, 32,  H/2, W/2

        # Heatmap: (B, 1, H/2, W/2) → upsample → (B, 1, H, W)
        hm = self.heatmap_head(d)
        hm = F.interpolate(hm, size=(H_IN, W_IN), mode='bilinear', align_corners=False)
        hm = hm.squeeze(1)          # B, H, W

        # Soft-argmax: column-wise weighted mean
        w  = F.softmax(hm, dim=1)  # B, H, W  (확률 분포 along H)
        y  = (w * self.row_idx).sum(dim=1)   # B, W  — [0, 1]
        return y


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> CurveExtractorNet:
    return CurveExtractorNet(pretrained=pretrained)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = build_model(pretrained=False)
    n = count_params(m)
    print(f'파라미터: {n:,}  ({n/1e6:.1f}M)')
    x = torch.randn(2, 3, H_IN, W_IN)
    y = m(x)
    print(f'Input:  {tuple(x.shape)}')
    print(f'Output: {tuple(y.shape)}   range=[{y.min():.3f}, {y.max():.3f}]')
