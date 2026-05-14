"""
Simple pixel-based curve extraction.

깨끗한 이미지(흰/단색 배경 + 단일 색상의 얇은 곡선) 전용 trace 추출.
classical 파이프라인의 candidate building / DP trace 등을 우회한다.

각 ROI 컬럼에서 curve_rgb와 유클리드 거리가 max_dist 이하인 픽셀들의
darkness-weighted centroid y를 반환한다.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def extract_curve_simple(
    roi: np.ndarray,
    curve_rgb: Tuple[int, int, int],
    max_dist: float = 80.0,
    method: str = 'topmost',
) -> List[Optional[float]]:
    """
    각 컬럼에서 curve y를 반환.

    method:
      'topmost'                 - 기본. 임계값 이상 매칭인 가장 위(작은 y) 픽셀.
                                  peak 꼭대기 보존에 최적 — 선이 두꺼워도 정점에서 위쪽 가장자리를
                                  잡으므로 압축 없음. 이어서 topmost 클러스터 안에서 darkness-
                                  weighted centroid로 sub-pixel 보정.
      'argmin'                  - 가장 어두운(target에 가장 가까운) 픽셀.
                                  주의: 부드럽게 렌더링된 peak에서 선의 중심을 잡아 정점을 놓침.
      'centroid'                - 컬럼 전체 매칭 픽셀의 darkness-weighted centroid.
                                  굵은 곡선 / 노이즈 평균에 강함.

    Returns:
        list[Optional[float]] of length roi.shape[1].
        매칭 픽셀이 없는 컬럼은 None.
    """
    if roi.ndim != 3 or roi.shape[2] < 3:
        raise ValueError(f'roi must be HxWx3, got shape {roi.shape}')
    h, w = roi.shape[:2]
    target = np.asarray(curve_rgb, dtype=np.float32)

    rgb = roi[:, :, :3].astype(np.float32)
    diff = rgb - target[None, None, :]
    dist = np.linalg.norm(diff, axis=2)  # H x W
    mask = dist < float(max_dist)  # H x W
    has_match = mask.any(axis=0)

    # Frame/border 감지: 가로 (row) 및 세로 (col) 양쪽으로 적용.
    # plot_box 테두리, 격자선, y/x 축 line 등이 curve 색상과 비슷할 때 topmost가 그걸 잡는 걸 방지.
    # 임계값 0.95: 진짜 frame은 거의 전체 폭/높이에 걸쳐 있으므로 0.95에서도 안정적으로 감지.
    # 0.70은 wide diffuse peak이 있는 noisy 패턴의 실제 데이터 row 까지 nuke 해서 over-aggressive.
    need_copy = True
    row_match_ratio = mask.sum(axis=1).astype(np.float32) / max(1, w)
    frame_rows = row_match_ratio > 0.95
    if frame_rows.any():
        if need_copy:
            mask = mask.copy()
            need_copy = False
        mask[frame_rows, :] = False

    # 세로 frame (y축 line, 우측 frame): col 의 매칭 픽셀 비율이 0.95 초과면 axis line 으로 간주.
    # 이 column 은 trace 후보에서 제외 (None 반환).
    # 추가로 frame col 인접 ±1 (dilate) 도 함께 제외 — axis line 의 AA spread / label tick 이
    # 인접 컬럼에 흘러들면 매칭 비율 0.8~0.9 정도로 0.95 임계값을 넘지 못하지만 trace 가 frame
    # 위쪽 끝(y≈0)을 잡아서 spike 가 생긴다.
    # 세로 임계값은 0.80 (가로 0.95 보다 낮음): 진짜 XRD 곡선은 한 column 에서 chart 높이의
    # 80% 이상 차지 못한다 (sharp peak 도 1-3 px 폭). 0.80 이상은 axis line / AA spread.
    col_match_ratio = mask.sum(axis=0).astype(np.float32) / max(1, h)
    frame_cols = col_match_ratio > 0.80
    if frame_cols.any():
        # dilate by 1
        fc_dilated = frame_cols.copy()
        fc_dilated[1:]  |= frame_cols[:-1]
        fc_dilated[:-1] |= frame_cols[1:]
        if need_copy:
            mask = mask.copy()
            need_copy = False
        mask[:, fc_dilated] = False

    has_match = mask.any(axis=0)

    if method == 'argmin':
        argmin_y = np.argmin(dist, axis=0)
        return [float(argmin_y[c]) if has_match[c] else None for c in range(w)]

    if method == 'centroid':
        weight = np.where(mask, np.maximum(1.0, float(max_dist) - dist), 0.0)
        w_sum = weight.sum(axis=0)
        ys_full = np.arange(h, dtype=np.float32)
        y_weighted = (weight * ys_full[:, None]).sum(axis=0)
        trace_c = []
        for c in range(w):
            ws = float(w_sum[c])
            trace_c.append(float(y_weighted[c] / ws) if ws > 0.0 else None)
        return trace_c

    # 기본: topmost + sub-pixel AA edge 보간.
    # 클러스터 centroid는 darkness-weighted라 커브 body 쪽으로 끌려 apex를 1-2px 놓치는 문제가 있다.
    # 대신 AA 전환점(dist가 max_dist를 가로지르는 sub-pixel 위치)을 선형 보간으로 찾는다.
    #
    # 원리:
    #   y = top_y - 1: 마스크 밖 픽셀 (dist > max_dist, 배경)
    #   y = top_y    : 마스크 안 첫 픽셀 (dist < max_dist, AA edge)
    #   두 점 사이 dist 가 max_dist 와 교차하는 sub-pixel y → 진짜 curve 상단
    trace: List[Optional[float]] = []
    md = float(max_dist)
    for c in range(w):
        col_m = mask[:, c]
        if not col_m.any():
            trace.append(None)
            continue
        top_y = int(np.argmax(col_m))
        d_at = float(dist[top_y, c])

        if top_y > 0:
            d_above = float(dist[top_y - 1, c])
            denom = d_above - d_at
            if d_above > md > d_at and denom > 1e-6:
                # dist 가 max_dist 와 교차하는 sub-pixel y (top_y-1 ~ top_y 구간)
                frac = (d_above - md) / denom
                # frac: 0이면 d_above==md (교차점=top_y-1), 1이면 d_at==md (교차점=top_y)
                y_apex = float(top_y - 1) + float(frac)
            else:
                y_apex = float(top_y)
        else:
            y_apex = float(top_y)
        trace.append(y_apex)
    return trace


def sample_background_rgb(roi: np.ndarray, patch: int = 20) -> Tuple[int, int, int]:
    """ROI 4 모서리 패치의 중앙값으로 배경 RGB 추정."""
    h, w = roi.shape[:2]
    p = max(1, min(int(patch), h // 8, w // 8))
    corners = np.concatenate([
        roi[:p, :p, :3].reshape(-1, 3),
        roi[:p, w - p:, :3].reshape(-1, 3),
        roi[h - p:, :p, :3].reshape(-1, 3),
        roi[h - p:, w - p:, :3].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    med = np.median(corners, axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def adaptive_max_dist(curve_rgb: Tuple[int, int, int], bg_rgb: Tuple[int, int, int],
                      fraction: float = 0.6, floor: float = 80.0) -> float:
    """bg ↔ curve 거리의 fraction을 매칭 임계값으로 사용 (anti-alias 흡수).
    fraction=0.6은 sub-pixel AA(거리 절반 지점)도 포함."""
    diff = np.asarray(curve_rgb, dtype=np.float32) - np.asarray(bg_rgb, dtype=np.float32)
    d = float(np.linalg.norm(diff))
    return max(float(floor), d * float(fraction))


def sample_curve_rgb(
    roi: np.ndarray,
    color_sample_point: Tuple[int, int],
    plot_box: Tuple[int, int, int, int],
    patch_radius: int = 2,
    dark_search_radius: int = 30,
) -> Tuple[int, int, int]:
    """
    full-image 좌표 color_sample_point에서 ROI 픽셀 RGB를 샘플링.
    샘플 위치가 배경(밝음)이면 주변에서 가장 어두운 픽셀을 찾는다.
    """
    x0_pb, y0_pb = int(plot_box[0]), int(plot_box[1])
    h, w = roi.shape[:2]
    cx = max(0, min(w - 1, int(color_sample_point[0]) - x0_pb))
    cy = max(0, min(h - 1, int(color_sample_point[1]) - y0_pb))

    def _patch_mean(y: int, x: int) -> np.ndarray:
        ly, hy = max(0, y - patch_radius), min(h, y + patch_radius + 1)
        lx, hx = max(0, x - patch_radius), min(w, x + patch_radius + 1)
        return roi[ly:hy, lx:hx, :3].astype(np.float32).reshape(-1, 3).mean(axis=0)

    mean_rgb = _patch_mean(cy, cx)
    lum = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]

    if lum > 200.0:
        # 배경 가능성 — 주변에서 가장 어두운 픽셀로 재샘플
        ly = max(0, cy - dark_search_radius)
        hy = min(h, cy + dark_search_radius + 1)
        lx = max(0, cx - dark_search_radius)
        hx = min(w, cx + dark_search_radius + 1)
        patch = roi[ly:hy, lx:hx, :3].astype(np.float32)
        lums = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
        flat_idx = int(np.argmin(lums))
        py, px = divmod(flat_idx, lums.shape[1])
        mean_rgb = _patch_mean(ly + py, lx + px)

    return (int(round(mean_rgb[0])), int(round(mean_rgb[1])), int(round(mean_rgb[2])))
