"""
XRD 그래프 이미지에서 축선 픽셀 좌표 + 곡선 색상을 자동 감지한다.

출력:
  {
    "success": bool,
    "calib_points": {
      "p1": {"x": int, "y": int},   # 원점 (y축 × x축 교점)
      "p2": {"x": int, "y": int},   # x축 끝점 (우측 경계)
      "p3": {"x": int, "y": int},   # y축 끝점 (상단 경계)
    },
    "curve_color": [R, G, B],       # ROI 내 주 곡선 색상
    "color_sample_point": [x, y],   # 해당 색상 픽셀 중 ROI 중앙에 가장 가까운 좌표
    "axis_values": {                 # OCR 성공 시 채워짐, 아니면 null
      "x_min": float | None,
      "x_max": float | None,
      "y_min": float | None,
      "y_max": float | None,
    } | None,
    "confidence": float,             # 0~1, 축선 감지 신뢰도
    "ocr_available": bool,
    "error": str | None,
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# 축선으로 인정할 최소 신뢰도
_MIN_AXIS_CONF = 0.15


# ── 그레이스케일 유틸 ──────────────────────────────────────────────────────────

def _to_gray(arr: np.ndarray) -> np.ndarray:
    """uint8 RGB → float32 grayscale."""
    return (arr[:, :, 0].astype(np.float32) * 0.299
            + arr[:, :, 1].astype(np.float32) * 0.587
            + arr[:, :, 2].astype(np.float32) * 0.114)


# ── 레이캐스팅 기반 축선 감지 ──────────────────────────────────────────────────
#
# 이미지 중심에서 상하좌우로 레이를 쏴,
# 배경과 다른 색이 행/열 전체에 걸쳐 이어진 첫 위치 = 축선 내부 경계.
#
# Sobel 대비 장점:
#   - 배경색 자동 감지 (모서리 샘플링) → 흰 배경 외에도 동작
#   - 스캔 방향이 데이터 영역 → 축 방향이므로 항상 내부 경계를 직접 반환
#   - 축선 두께·선명도 무관

_LINE_RATIO = 0.35   # 행/열 픽셀의 이 비율 이상이 비배경이면 축선으로 판단
_NONBG_THRESH = 20.0  # 배경 그레이스케일과의 최소 차이


def _sample_background(arr: np.ndarray, patch: int = 20) -> float:
    """4모서리 패치 중앙값으로 배경 그레이스케일 추정."""
    H, W = arr.shape[:2]
    p = max(1, min(patch, H // 8, W // 8))
    corners = np.concatenate([
        arr[:p,    :p   ].reshape(-1, 3),
        arr[:p,    W-p: ].reshape(-1, 3),
        arr[H-p:,  :p   ].reshape(-1, 3),
        arr[H-p:,  W-p: ].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0)
    return float(bg[0] * 0.299 + bg[1] * 0.587 + bg[2] * 0.114)


def _nonbg_ratio(gray: np.ndarray, idx: int, axis: str, bg_gray: float) -> float:
    """행(axis='h') 또는 열(axis='v')의 비배경 픽셀 비율."""
    strip = gray[idx, :] if axis == 'h' else gray[:, idx]
    return float(np.mean(np.abs(strip - bg_gray) > _NONBG_THRESH))


def _ray_find_line(
    gray: np.ndarray,
    start: int,
    end: int,
    axis: str,
    bg_gray: float,
) -> tuple[int, float]:
    """
    start → end 방향으로 행(axis='h') 또는 열(axis='v')을 스캔해 축선 내부 경계를 반환.

    1단계: 비배경 비율 >= _LINE_RATIO인 첫 위치 감지 (= 축선 외부 진입점)
    2단계: 그 위치에서 데이터 방향(역방향)으로 스캔 → 비율이 _LINE_RATIO 미만으로
           떨어지는 첫 위치 = 축선 밴드의 내부 경계 (데이터 영역 첫 픽셀)
           ※ _LINE_RATIO 기준을 재사용하므로 tick mark 등 낮은 배경 노이즈에 강함

    못 찾으면 (start, 0.0).
    """
    step = 1 if end > start else -1
    limit = gray.shape[0] if axis == 'h' else gray.shape[1]

    # 1단계: 축선 외부 진입점 탐색
    hit, hit_ratio = None, 0.0
    for i in range(start, end, step):
        r = _nonbg_ratio(gray, i, axis, bg_gray)
        if r >= _LINE_RATIO:
            hit, hit_ratio = i, r
            break
    if hit is None:
        return start, 0.0

    # 2단계: 역방향으로 축선 밴드를 통과 → 비율이 _LINE_RATIO 미만인 첫 위치
    backstep = -step
    inner = hit + backstep  # 기본값: 진입점 바로 앞 (1픽셀 후퇴)
    for j in range(0, 30):
        pos = hit + backstep * j
        if pos < 0 or pos >= limit:
            break
        if _nonbg_ratio(gray, pos, axis, bg_gray) < _LINE_RATIO:
            inner = pos
            break

    return inner, hit_ratio


def _detect_axes(arr: np.ndarray) -> dict[str, Any]:
    """
    이미지 중심에서 4방향 레이캐스팅으로 축선 내부 경계를 감지한다.
    스캔이 데이터 영역 → 축 방향이므로 반환값이 곧 내부 경계 픽셀.
    """
    H, W = arr.shape[:2]
    gray = _to_gray(arr)
    bg_gray = _sample_background(arr)

    cy, cx = H // 2, W // 2

    # _ray_find_line이 역방향 스캔으로 내부 경계를 직접 반환함 — 별도 오프셋 불필요
    x_row, x_conf = _ray_find_line(gray, cy, H,  'h', bg_gray)  # 중심 → 아래: x축
    t_row, t_conf = _ray_find_line(gray, cy, -1, 'h', bg_gray)  # 중심 → 위: 상단 경계
    y_col, y_conf = _ray_find_line(gray, cx, -1, 'v', bg_gray)  # 중심 → 왼쪽: y축
    r_col, r_conf = _ray_find_line(gray, cx, W,  'v', bg_gray)  # 중심 → 오른쪽: 우측 경계

    return {
        'x_axis_row': x_row, 'x_conf': float(min(x_conf, 1.0)),
        'y_axis_col': y_col, 'y_conf': float(min(y_conf, 1.0)),
        'right_col':  r_col, 'r_conf': float(min(r_conf, 1.0)),
        'top_row':    t_row, 't_conf': float(min(t_conf, 1.0)),
    }


# ── 곡선 색상 감지 ────────────────────────────────────────────────────────────

_BG_THRESH = 240   # R, G, B 모두 이 값 이상이면 흰 배경으로 간주
_QUANT = 32        # 색상 quantize 단위


def _detect_curve_color(
    arr: np.ndarray,
    x_row: int, y_col: int, r_col: int, t_row: int,
) -> tuple[list[int], list[int]] | None:
    """
    ROI 내부에서 주 곡선 색상을 감지한다.

    Returns
    -------
    (curve_color, color_sample_point) or None if detection fails
    curve_color        : [R, G, B]
    color_sample_point : [abs_x, abs_y]  — 이미지 전체 좌표
    """
    H, W = arr.shape[:2]

    # ROI crop: (t_row, y_col) ~ (x_row, r_col)
    roi_y0 = max(0, int(t_row))
    roi_y1 = min(H, int(x_row) + 1)
    roi_x0 = max(0, int(y_col))
    roi_x1 = min(W, int(r_col) + 1)
    if roi_y1 <= roi_y0 or roi_x1 <= roi_x0:
        return None

    roi = arr[roi_y0:roi_y1, roi_x0:roi_x1]  # (rH, rW, 3)

    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

    # 배경 제거: 각 채널이 모두 이미지 최대값의 92% 이상이면 배경
    bg_thresh = max(200, int(arr.max() * 0.92))
    not_bg = ~((r >= bg_thresh) & (g >= bg_thresh) & (b >= bg_thresh))

    # 축선/프레임 제거: 비배경 픽셀 중 하위 25%ile (가장 어두운 것 = 축선/프레임)
    # ※ 전체 ROI로 percentile 계산하면 90% 흰 배경 때문에 임계값이 배경 수준이 돼 버그 발생
    gray_roi = r.astype(np.float32) * 0.299 + g.astype(np.float32) * 0.587 + b.astype(np.float32) * 0.114
    non_bg_gray = gray_roi[not_bg]
    if non_bg_gray.size > 0:
        axis_thresh = float(np.percentile(non_bg_gray, 25))
    else:
        axis_thresh = 0.0
    not_axis = gray_roi > axis_thresh

    mask = not_bg & not_axis
    if mask.sum() == 0:
        return None

    pixels = roi[mask]  # (N, 3) uint8

    # 32단위 quantize → bin 인덱스
    q = (pixels.astype(np.int32) // _QUANT)
    bin_ids = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    unique, counts = np.unique(bin_ids, return_counts=True)
    best_bin = unique[counts.argmax()]

    qr = (best_bin // 10000)
    qg = (best_bin % 10000) // 100
    qb = best_bin % 100

    # bin 중심값 = 대표 색상
    curve_r = int(qr * _QUANT + _QUANT // 2)
    curve_g = int(qg * _QUANT + _QUANT // 2)
    curve_b = int(qb * _QUANT + _QUANT // 2)

    # 같은 bin에 속하는 픽셀 중 ROI 중앙에 가장 가까운 픽셀 좌표 찾기
    same_bin = (q[:, 0] == qr) & (q[:, 1] == qg) & (q[:, 2] == qb)
    ys, xs = np.where(mask)           # ROI-local 좌표
    cand_y = ys[same_bin]
    cand_x = xs[same_bin]

    cy = (roi_y1 - roi_y0) / 2.0
    cx = (roi_x1 - roi_x0) / 2.0
    dists = (cand_y - cy) ** 2 + (cand_x - cx) ** 2
    best_idx = int(dists.argmin())

    abs_x = int(cand_x[best_idx]) + roi_x0
    abs_y = int(cand_y[best_idx]) + roi_y0

    return [curve_r, curve_g, curve_b], [abs_x, abs_y]


# ── OCR (pytesseract 우선, EasyOCR 폴백) ─────────────────────────────────────

import re as _re


def _parse_ocr_texts(texts: list[str]) -> list[float]:
    """OCR 텍스트 리스트에서 숫자만 추출해 정렬."""
    nums: list[float] = []
    for text in texts:
        cleaned = text.replace(',', '').replace(' ', '').strip()
        cleaned = cleaned.replace('×10^', 'e').replace('x10^', 'e')
        # O/o → 0, l/I → 1 OCR 오인식 보정
        cleaned = cleaned.replace('O', '0').replace('o', '0')
        cleaned = cleaned.replace('l', '1').replace('I', '1')
        cleaned = _re.sub(r'[^\d.\-+e]', '', cleaned)
        # 소수점이 두 개 이상이면 첫 번째만 유지
        parts = cleaned.split('.')
        if len(parts) > 2:
            cleaned = parts[0] + '.' + ''.join(parts[1:])
        try:
            v = float(cleaned)
            if cleaned:
                nums.append(v)
        except ValueError:
            pass
    return sorted(nums)


def _upscale_for_ocr(region: np.ndarray, scale: int = 3) -> np.ndarray:
    """소수점·작은 문자 인식 향상을 위해 리전을 업스케일."""
    from PIL import Image as _PILImage
    img = _PILImage.fromarray(region)
    w, h = img.size
    img = img.resize((w * scale, h * scale), _PILImage.LANCZOS)
    return np.array(img)


def _filter_axis_nums(nums: list[float], is_x: bool) -> list[float]:
    """물리적으로 불가능하거나 소수점 탈락으로 인한 이상값 제거."""
    if len(nums) < 2:
        return nums
    if is_x:
        # 2θ는 0°~180° 범위
        filtered = [v for v in nums if -5.0 <= v <= 185.0]
        return filtered if len(filtered) >= 2 else nums
    else:
        # 소수점 탈락 탐지: 값이 모두 100 이상인데 인접 값 비율이 일정하면 1000으로 나눔
        positives = [v for v in nums if v > 0]
        if positives and min(positives) > 100:
            scaled = [v / 1000.0 for v in nums]
            return scaled
        if positives and min(positives) > 10:
            scaled = [v / 100.0 for v in nums]
            return scaled
        return nums


_TESSERACT_CANDIDATES = [
    'tesseract',
    '/opt/homebrew/bin/tesseract',
    '/opt/homebrew/Cellar/tesseract/5.5.2/bin/tesseract',
    '/usr/local/bin/tesseract',
]


def _find_tesseract_cmd() -> str | None:
    import shutil
    for c in _TESSERACT_CANDIDATES:
        if shutil.which(c) or (c.startswith('/') and __import__('os').path.isfile(c)):
            return c
    return None


def _read_region_tesseract(region: np.ndarray, is_x: bool = False) -> list[float]:
    """pytesseract로 숫자 읽기. 3× 업스케일 + PSM 11(분산 텍스트)."""
    if region.shape[0] < 8 or region.shape[1] < 8:
        return []
    import pytesseract  # type: ignore
    from PIL import Image as _PILImage
    cmd = _find_tesseract_cmd()
    if cmd is None:
        raise RuntimeError('tesseract not found')
    pytesseract.pytesseract.tesseract_cmd = cmd
    region_up = _upscale_for_ocr(region, scale=3)
    img_pil = _PILImage.fromarray(region_up)
    # PSM 11: 분산 텍스트 (축 레이블처럼 흩어진 숫자에 적합)
    cfg = '--psm 11 -c tessedit_char_whitelist=0123456789.-+eE'
    raw = pytesseract.image_to_string(img_pil, config=cfg)
    tokens = [t.strip() for t in raw.split() if t.strip()]
    nums = _parse_ocr_texts(tokens)
    return _filter_axis_nums(nums, is_x)


def _read_region_easyocr(reader, region: np.ndarray, is_x: bool = False) -> list[float]:
    """EasyOCR로 숫자 읽기. 3× 업스케일."""
    if region.shape[0] < 8 or region.shape[1] < 8:
        return []
    region_up = _upscale_for_ocr(region, scale=3)
    results = reader.readtext(region_up)
    texts = [text for (_, text, conf) in results if conf >= 0.2]
    nums = _parse_ocr_texts(texts)
    return _filter_axis_nums(nums, is_x)


def _try_ocr(arr: np.ndarray, x_axis_row: int, y_axis_col: int) -> tuple[dict | None, bool]:
    """축 숫자 OCR. pytesseract → EasyOCR 순서로 시도. 둘 다 없으면 (None, False)."""
    H, W = arr.shape[:2]
    pad = max(5, int(H * 0.015))

    # 프레임선 포함 방지: 좌우/상하 여백 제거
    h_margin = max(10, int(W * 0.03))
    v_margin = max(5, int(H * 0.02))

    # x축 레이블: x축선 아래쪽, 좌우 프레임 제외
    x_bot = min(H, x_axis_row + int(H * 0.13))
    x_region = arr[x_axis_row + pad: x_bot, h_margin: W - h_margin]

    # y축 레이블: y축선 왼쪽, 상하 프레임 제외
    y_region = arr[v_margin: H - v_margin, :max(1, y_axis_col - pad)]

    # ── pytesseract 시도 ──
    try:
        import pytesseract  # type: ignore  # noqa: F401
        x_nums = _read_region_tesseract(x_region, is_x=True)
        y_nums = _read_region_tesseract(y_region, is_x=False)
        axis_values = {
            'x_min': x_nums[0]  if len(x_nums) >= 2 else None,
            'x_max': x_nums[-1] if len(x_nums) >= 2 else None,
            'y_min': y_nums[0]  if len(y_nums) >= 2 else None,
            'y_max': y_nums[-1] if len(y_nums) >= 2 else None,
        }
        if any(v is not None for v in axis_values.values()):
            return axis_values, True
    except (ImportError, Exception):
        pass

    # ── EasyOCR 폴백 ──
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(['en'], verbose=False)
        x_nums = _read_region_easyocr(reader, x_region, is_x=True)
        y_nums = _read_region_easyocr(reader, y_region, is_x=False)
        axis_values = {
            'x_min': x_nums[0]  if len(x_nums) >= 2 else None,
            'x_max': x_nums[-1] if len(x_nums) >= 2 else None,
            'y_min': y_nums[0]  if len(y_nums) >= 2 else None,
            'y_max': y_nums[-1] if len(y_nums) >= 2 else None,
        }
        return axis_values, True
    except (ImportError, Exception):
        pass

    return None, False


# ── 메인 함수 ─────────────────────────────────────────────────────────────────

def auto_detect(image_path: str) -> dict[str, Any]:
    """
    XRD 이미지에서 3점 캘리브레이션 픽셀 좌표와 (선택적) 축값을 자동 감지한다.

    Parameters
    ----------
    image_path : str
        입력 이미지 경로 (PNG/JPG/TIFF)

    Returns
    -------
    dict
        성공: {"success": True, "calib_points": {...}, "axis_values": {...}|None,
                "confidence": float, "ocr_available": bool}
        실패: {"success": False, "error": str}
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {'success': False, 'error': f'이미지 열기 실패: {e}', 'ocr_available': False}

    arr = np.asarray(img, dtype=np.uint8)
    axes = _detect_axes(arr)

    x_row = axes['x_axis_row']
    y_col = axes['y_axis_col']
    r_col = axes['right_col']
    t_row = axes['top_row']

    axis_conf = (axes['x_conf'] + axes['y_conf']) / 2.0
    if axes['x_conf'] < _MIN_AXIS_CONF or axes['y_conf'] < _MIN_AXIS_CONF:
        return {
            'success': False,
            'error': '축선 감지 신뢰도가 낮습니다 (이미지가 흐리거나 축이 없을 수 있습니다).',
            'ocr_available': False,
            'confidence': round(axis_conf, 3),
        }

    calib_points = {
        'p1': {'x': int(y_col), 'y': int(x_row)},  # 원점
        'p2': {'x': int(r_col), 'y': int(x_row)},  # x축 끝
        'p3': {'x': int(y_col), 'y': int(t_row)},  # y축 끝
    }

    # 곡선 색상 감지
    color_result = _detect_curve_color(arr, x_row, y_col, r_col, t_row)
    curve_color       = color_result[0] if color_result else None
    color_sample_point = color_result[1] if color_result else None

    axis_values, ocr_ok = _try_ocr(arr, x_row, y_col)

    return {
        'success': True,
        'calib_points': calib_points,
        'curve_color': curve_color,
        'color_sample_point': color_sample_point,
        'axis_values': axis_values,
        'confidence': round(axis_conf, 3),
        'ocr_available': ocr_ok,
        'error': None,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python auto_detect.py <image_path>', file=sys.stderr)
        sys.exit(1)
    result = auto_detect(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
