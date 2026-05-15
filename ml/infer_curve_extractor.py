"""
CurveExtractorNet 추론 모듈.

파이프라인 통합:
  trace_path = infer_curve(roi_bgr, roi_h, roi_w, weights_path)
  → list[Optional[float]], 길이 roi_w, 각 원소는 ROI 픽셀 y 좌표 or None

사용 예:
  from ml.infer_curve_extractor import CurveExtractorInfer
  inf = CurveExtractorInfer()              # 기본 가중치 자동 탐색
  trace = inf.infer(roi_bgr_array)         # numpy H×W×3 (BGR or RGB)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.curve_extractor import H_IN, W_IN, build_model

_DEFAULT_WEIGHTS = ROOT / 'ml' / 'weights' / 'curve_extractor.pt'


class CurveExtractorInfer:
    """
    단일 인스턴스로 여러 이미지 추론.  모델은 생성 시 한 번만 로드.
    """

    def __init__(self, weights_path: Optional[Path | str] = None) -> None:
        wp = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not wp.exists():
            raise FileNotFoundError(f'가중치 파일 없음: {wp}')

        self.device = torch.device(
            'mps'  if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        ckpt = torch.load(wp, map_location=self.device, weights_only=False)
        self.model = build_model(pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    # ------------------------------------------------------------------

    def infer(
        self,
        roi: np.ndarray,                       # H×W×3, uint8, BGR or RGB — 순서 무관
        conf_threshold: float = 0.0,           # 미래 confidence 필터용 (현재 미사용)
    ) -> list[Optional[float]]:
        """
        roi: numpy ROI 이미지 (H, W, 3).  plot_box 크롭된 영역만 전달.
        반환: list[Optional[float]] 길이 W, ROI 픽셀 y 좌표 (float).

        반환값은 run_local.py 의 trace_path 와 동일한 포맷.
        None 없이 모든 열에 값을 채워 반환.
        """
        roi_h, roi_w = roi.shape[:2]

        # numpy → PIL (RGB) → resize to model input
        if roi.ndim == 3 and roi.shape[2] == 3:
            pil = Image.fromarray(roi[:, :, ::-1] if _is_bgr_likely(roi) else roi)
        else:
            pil = Image.fromarray(roi).convert('RGB')
        pil_resized = pil.resize((W_IN, H_IN), Image.BILINEAR)

        img_t = torch.from_numpy(
            np.array(pil_resized, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)   # 1, 3, H_IN, W_IN

        with torch.no_grad():
            y_norm = self.model(img_t).squeeze(0).cpu().numpy()   # (W_IN,)  [0, 1]

        # W_IN → roi_w 보간
        if roi_w != W_IN:
            y_norm = np.interp(
                np.linspace(0, W_IN - 1, roi_w),
                np.arange(W_IN),
                y_norm,
            ).astype(np.float32)

        # 정규화 → 픽셀
        y_pixel = (y_norm * (roi_h - 1)).tolist()
        return y_pixel


def _is_bgr_likely(arr: np.ndarray) -> bool:
    """OpenCV 로 읽은 BGR 배열은 채널 순서가 역전돼 있음.
    완벽한 판별은 불가능하므로 False(RGB) 기본 가정 — PIL과 호환."""
    return False


# ── 단독 실행 시 샘플 추론 ──────────────────────────────────────────────────

if __name__ == '__main__':
    import json

    gt_dir     = ROOT / 'data' / 'gt'
    styled_dir = ROOT / 'data' / 'rendered_styled'

    gt_path = next(gt_dir.glob('*_gt.json'), None)
    if gt_path is None:
        print('GT 파일 없음'); sys.exit(1)

    gt = json.loads(gt_path.read_text())
    pid = gt_path.stem.replace('_gt', '')
    box = gt['plot_box']
    img_path = styled_dir / f'{pid}_styled_v1.png'
    if not img_path.exists():
        print(f'{img_path} 없음'); sys.exit(1)

    from PIL import Image as _PIL
    img = np.array(_PIL.open(img_path).convert('RGB'))
    x0, y0, x1, y1 = box
    roi = img[y0:y1, x0:x1]

    inf = CurveExtractorInfer()
    trace = inf.infer(roi)
    print(f'ROI: {roi.shape}   trace 길이: {len(trace)}')
    print(f'y 범위: [{min(trace):.1f}, {max(trace):.1f}]  (ROI 픽셀)')

    # GT 비교
    col_gt = gt.get('per_column_y_gt', {})
    errs = []
    for col_str, y_abs in col_gt.items():
        col = int(col_str) - x0
        if 0 <= col < len(trace):
            y_gt_roi = float(y_abs) - y0
            errs.append(abs(trace[col] - y_gt_roi))
    if errs:
        print(f'GT MAE: {np.mean(errs):.2f}px  (n={len(errs)})')
