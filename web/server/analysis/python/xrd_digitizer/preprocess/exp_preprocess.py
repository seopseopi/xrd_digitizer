"""
실험용 통합 전처리: color scores, ridge response, edge support, grid masks.
§6 수치 구현(k-means, Hessian ridge, Hough-like grid). (운영 v1.1 전처리와 별도)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, gaussian_filter, sobel, label as cc_label

try:
    from skimage.feature import canny
except ImportError:
    canny = None  # type: ignore


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float64) / 255.0
    mask = arr > 0.04045
    arr = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x /= 0.95047
    z /= 1.08883

    def _f(t):
        mask_t = t > 0.008856
        return np.where(mask_t, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return np.stack([L, a, b_ch], axis=-1)


def _kmeans_lab(pixels: np.ndarray, k: int, max_iter: int = 25, rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """pixels [N,3], returns centers [k,3], labels [N]"""
    rng = rng or np.random.default_rng(42)
    n = pixels.shape[0]
    if n == 0:
        return np.zeros((k, 3)), np.zeros(n, dtype=int)
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    centers = pixels[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d2 = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for ki in range(k):
            mask = labels == ki
            if mask.any():
                new_centers[ki] = pixels[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, labels


def _mahalanobis_diag(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 4.0)
    return np.sqrt(np.sum((x - mu) ** 2 / var, axis=-1))


def build_color_scores_experimental(
    roi_rgb: np.ndarray,
    color_sample_point: List[int],
    plot_box_abs: Tuple[int, int, int, int],
    legend_ignore_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Returns color_score [H,W] in [0,1] and meta dict.
    """
    h, w = roi_rgb.shape[:2]
    roi_lab = _rgb_to_lab(roi_rgb)
    x0_pb, y0_pb, _, _ = plot_box_abs
    cx = int(color_sample_point[0]) - x0_pb
    cy = int(color_sample_point[1]) - y0_pb
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    def patch_mean(px, py, r=2):
        y0, y1 = max(0, py - r), min(h, py + r + 1)
        x0, x1 = max(0, px - r), min(w, px + r + 1)
        return roi_lab[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    main_seed = patch_mean(cx, cy)
    if main_seed[0] > 85.0:
        patch = roi_lab[max(0, cy - 30) : min(h, cy + 31), max(0, cx - 30) : min(w, cx + 31), 0]
        iy, ix = np.unravel_index(np.argmin(patch), patch.shape)
        cy2, cx2 = max(0, cy - 30) + iy, max(0, cx - 30) + ix
        main_seed = patch_mean(cx2, cy2)

    seeds = [main_seed]
    for tx in (int(w * 0.2), int(w * 0.5), int(w * 0.8)):
        tx = max(0, min(w - 1, tx))
        col = roi_lab[:, tx, :]
        dists = np.linalg.norm(col - main_seed[None, :], axis=1)
        best_y = int(np.argmin(dists))
        seeds.append(patch_mean(tx, best_y))

    seed_pixels = np.array(seeds, dtype=np.float64)
    k_fg = min(3, max(1, len(seeds)))
    fg_centers, _ = _kmeans_lab(seed_pixels, k=k_fg)
    flat = roi_lab.reshape(-1, 3)
    fg_dists = []
    for ki in range(fg_centers.shape[0]):
        mu = fg_centers[ki]
        pts = seed_pixels if k_fg == 1 else seed_pixels  # use seed spread for var
        var = np.var(pts, axis=0) + 4.0
        var = np.maximum(var, 4.0)
        fg_dists.append(_mahalanobis_diag(flat, mu, var))
    fg_dist = np.min(np.stack(fg_dists, axis=1), axis=1).reshape(h, w)

    bg_candidates = []
    yy, xx = np.mgrid[0:h, 0:w]
    gx = np.gradient(roi_lab[..., 0], axis=(0, 1))
    gy = np.gradient(roi_lab[..., 1], axis=(0, 1))
    gz = np.gradient(roi_lab[..., 2], axis=(0, 1))
    grad_mag = np.sqrt(gx[0] ** 2 + gx[1] ** 2) + np.sqrt(gy[0] ** 2 + gy[1] ** 2) + np.sqrt(gz[0] ** 2 + gz[1] ** 2)
    grad_flat = grad_mag.reshape(-1)
    thresh = np.percentile(grad_flat, 35)
    mask_bg = grad_flat <= thresh
    dist_main = np.linalg.norm(flat - main_seed[None, :], axis=1)
    mask_bg &= dist_main >= 22.0
    if legend_ignore_mask is not None:
        mask_bg &= legend_ignore_mask.reshape(-1) < 0.5
    dil_r = max(8, round(0.012 * w))
    cyx, cxx = np.ogrid[-dil_r : dil_r + 1, -dil_r : dil_r + 1]
    disk = cyx**2 + cxx**2 <= dil_r**2
    seed_dil = np.zeros((h, w), dtype=bool)
    seed_dil[cy, cx] = True
    seed_dil = binary_dilation(seed_dil, structure=disk.astype(bool))
    mask_bg &= ~seed_dil.reshape(-1)

    bg_pixels = flat[mask_bg]
    if bg_pixels.shape[0] < 200:
        mu_bg = np.median(flat, axis=0)
        bg_dist = np.linalg.norm(flat - mu_bg[None, :], axis=1).reshape(h, w)
        color_raw = np.clip(np.exp(-fg_dist.reshape(-1) / 14.0) - 0.7 * np.exp(-bg_dist.reshape(-1) / 14.0), -1, 1).reshape(h, w)
        pct_map = np.ones((h, w)) * 0.85
        color_score = np.clip(pct_map * color_raw, 0.0, 1.0).astype(np.float64)
        return color_score, {"fallback_simple": True}

    sub = bg_pixels[: min(8000, bg_pixels.shape[0])]
    bg_centers, bg_labels = _kmeans_lab(sub, k=3)
    bg_dists = []
    for ki in range(3):
        pts = sub[bg_labels == ki]
        mu = bg_centers[ki]
        if pts.shape[0] < 2:
            var = np.var(sub, axis=0) + 4.0
        else:
            var = np.var(pts, axis=0) + 4.0
        var = np.maximum(var, 4.0)
        bg_dists.append(_mahalanobis_diag(flat, mu, var))
    bg_dist = np.min(np.stack(bg_dists, axis=1), axis=1).reshape(h, w)

    fg_score = np.exp(-fg_dist / 12.0)
    bg_score = np.exp(-bg_dist / 12.0)
    color_raw = fg_score - 0.70 * bg_score
    gray = np.mean(roi_rgb.astype(np.float64), axis=2)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)
    g95 = float(np.percentile(gmag, 95)) + 1e-9
    ls = np.clip(0.6 + 0.4 * (1.0 - np.clip(gmag / g95, 0.0, 1.0)), 0.6, 1.0)
    color_score = np.clip(ls * color_raw, 0.0, 1.0).astype(np.float64)

    meta = {"fallback_simple": False, "n_bg_pixels": int(bg_pixels.shape[0])}
    return color_score, meta


def _unused_hessian_eig_gray(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gaussian_filter(gray, sigma=1.0)
    gxx = ndimage.sobel(ndimage.sobel(g, axis=1), axis=1)
    gyy = ndimage.sobel(ndimage.sobel(g, axis=0), axis=0)
    gxy = ndimage.sobel(ndimage.sobel(g, axis=1), axis=0)
    trace = gxx + gyy
    det = gxx * gyy - gxy * gxy
    disc = np.sqrt(np.maximum(trace * trace - 4 * det, 0))
    lam1 = 0.5 * (trace - disc)
    lam2 = 0.5 * (trace + disc)
    order = np.abs(lam1) <= np.abs(lam2)
    l1 = np.where(order, lam1, lam2)
    l2 = np.where(order, lam2, lam1)
    return l1, l2, gxy * 0  # third return unused


def compute_ridge_score(gray: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale dark-line ridge response, fused by geometric mean.
    Returns ridge_score [H,W], ridge_theta [H,W] (orientation of lambda1 eigenvector, approximated).
    """
    h, w = gray.shape
    scales = [1.0 * s, 1.8 * s, 2.6 * s]
    resps = []
    for sig in scales:
        sig = max(0.5, float(sig))
        sm = gaussian_filter(gray, sigma=sig)
        gxx = ndimage.sobel(ndimage.sobel(sm, axis=1), axis=1)
        gyy = ndimage.sobel(ndimage.sobel(sm, axis=0), axis=0)
        gxy = ndimage.sobel(ndimage.sobel(sm, axis=1), axis=0)
        trace = gxx + gyy
        det = gxx * gyy - gxy * gxy
        disc = np.sqrt(np.maximum(trace * trace - 4 * det, 0))
        lam1 = 0.5 * (trace - disc)
        lam2 = 0.5 * (trace + disc)
        swap = np.abs(lam1) > np.abs(lam2)
        l1 = np.where(swap, lam2, lam1)
        l2 = np.where(swap, lam1, lam2)
        resp = np.zeros_like(l2)
        valid = l2 < 0
        rb = np.abs(l1) / (np.abs(l2) + 1e-6)
        S = np.sqrt(l1**2 + l2**2)
        r = np.exp(-(rb**2) / (2 * 0.7**2)) * (1.0 - np.exp(-(S**2) / (2 * 12.0**2)))
        resp = np.where(valid, r, 0.0)
        p99 = np.percentile(resp, 99)
        if p99 > 1e-9:
            resp = resp / p99
        resp = np.maximum(resp, 1e-4)
        resps.append(resp)
    ridge = np.exp(np.mean(np.log(np.stack(resps, axis=0)), axis=0))
    theta = np.zeros((h, w), dtype=np.float64)
    return ridge.astype(np.float64), theta


def compute_edge_support(gray: np.ndarray) -> np.ndarray:
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    p25, p75 = np.percentile(mag, 25), np.percentile(mag, 75)
    edges = ((mag >= p25) & (mag <= p75 * 2.5)).astype(np.float64)
    loc = ndimage.uniform_filter(edges, size=3)
    emax = loc.max() + 1e-9
    return np.clip(loc / emax, 0.0, 1.0)


def _hough_strong_grid(gray: np.ndarray, plot_w: int, plot_h: int, ridge_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    if canny is not None:
        edges = canny((gray / 255.0 if gray.max() > 1 else gray).astype(np.float64), sigma=1.0).astype(np.uint8) * 255
    else:
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        edges = (mag > np.percentile(mag, 75)).astype(np.uint8) * 255

    try:
        import cv2

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(30, plot_w // 25), minLineLength=max(plot_w // 2, 50), maxLineGap=12)
    except Exception:
        lines = None

    strong = np.zeros((h, w), dtype=np.float64)
    penalty = np.full((h, w), 0.35, dtype=np.float64)
    min_dim = min(plot_w, plot_h)
    thick = max(2, int(round(0.003 * min_dim)))

    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            length = math.hypot(x2 - x1, y2 - y1)
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            horiz = ang < 10 or ang > 170
            vert = abs(ang - 90) < 10
            ok_len = (horiz and length >= 0.75 * plot_w) or (vert and length >= 0.75 * plot_h)
            if not ok_len:
                continue
            # sample overlap with color/ridge (simplified)
            n = max(int(length), 1)
            xs = np.linspace(x1, x2, n).astype(int)
            ys = np.linspace(y1, y2, n).astype(int)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            rvals = ridge_score[ys, xs]
            if rvals.mean() > 0.22:
                continue
            for t in range(n):
                yy, xx = ys[t], xs[t]
                y0, y1 = max(0, yy - thick), min(h, yy + thick + 1)
                x0, x1_ = max(0, xx - thick), min(w, xx + thick + 1)
                strong[y0:y1, x0:x1_] = 1.0
                penalty[y0:y1, x0:x1_] = np.maximum(penalty[y0:y1, x0:x1_], 0.60)

    dil_r = max(1, int(round(0.0025 * min_dim)))
    if dil_r > 0:
        struct = np.ones((2 * dil_r + 1, 2 * dil_r + 1), dtype=bool)
        strong = binary_dilation(strong > 0.5, structure=struct).astype(np.float64)

    return strong, penalty


def build_combined_experimental(
    color_score: np.ndarray,
    ridge_score: np.ndarray,
    strong_grid_mask: np.ndarray,
    edge_support: np.ndarray,
    mask_a_thr: float = 0.18,
    mask_b_thr: float = 0.22,
    support_edge_thr: float = 0.20,
    support_ridge_thr: float = 0.16,
) -> np.ndarray:
    mask_a = color_score > mask_a_thr
    mask_b = ridge_score > mask_b_thr
    combined = (mask_a | mask_b) & (strong_grid_mask < 0.5)
    struct = np.ones((3, 3), dtype=bool)
    dil_a = binary_dilation(mask_a, structure=struct)
    support_mask = (edge_support > support_edge_thr) & dil_a
    extra = support_mask & (ridge_score > support_ridge_thr) & (strong_grid_mask < 0.5)
    combined = combined | extra
    combined = ndimage.binary_closing(combined, structure=struct)
    return combined.astype(np.uint8)


def run_preprocess_experimental(
    roi_rgb: np.ndarray,
    color_sample_point: List[int],
    plot_box_abs: Tuple[int, int, int, int],
    legend_ignore_boxes: Optional[List] = None,
    s: float = 1.0,
    params: Optional[dict] = None,
) -> dict:
    h, w = roi_rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in plot_box_abs]
    plot_w = x1 - x0 + 1
    plot_h = y1 - y0 + 1

    ign = np.zeros((h, w), dtype=np.float64)
    if legend_ignore_boxes:
        for box in legend_ignore_boxes:
            if hasattr(box, "x0"):
                xa, ya, xb, yb = int(box.x0 - x0), int(box.y0 - y0), int(box.x1 - x0), int(box.y1 - y0)
            else:
                xa, ya, xb, yb = int(box[0] - x0), int(box[1] - y0), int(box[2] - x0), int(box[3] - y0)
            xa, xb = max(0, xa), min(w, xb)
            ya, yb = max(0, ya), min(h, yb)
            ign[ya:yb, xa:xb] = 1.0

    color_score, cmeta = build_color_scores_experimental(roi_rgb, color_sample_point, plot_box_abs, ign)
    gray = np.mean(roi_rgb.astype(np.float64), axis=2)
    ridge_score, ridge_theta = compute_ridge_score(gray, s)
    edge_support = compute_edge_support(gray)
    strong_grid, grid_penalty = _hough_strong_grid(gray, plot_w, plot_h, ridge_score)

    pp = (params or {})
    combined = build_combined_experimental(
        color_score,
        ridge_score,
        strong_grid,
        edge_support,
        mask_a_thr=float(pp.get("mask_a_thr", 0.18)),
        mask_b_thr=float(pp.get("mask_b_thr", 0.22)),
        support_edge_thr=float(pp.get("support_edge_thr", 0.20)),
        support_ridge_thr=float(pp.get("support_ridge_thr", 0.16)),
    )

    area = plot_w * plot_h
    min_area = max(40, int(round(0.00006 * area)))
    labeled, nlab = cc_label(combined)
    for lab_id in range(1, nlab + 1):
        mask = labeled == lab_id
        if mask.sum() < min_area:
            rs = ridge_score[mask]
            if rs.size and np.percentile(rs, 95) < 0.42:
                combined[mask] = 0

    return {
        "color_score": color_score,
        "ridge_score": ridge_score,
        "ridge_theta": ridge_theta,
        "edge_support": edge_support,
        "strong_grid_mask": strong_grid,
        "grid_penalty_mask": grid_penalty,
        "combined_mask": combined,
        "color_meta": cmeta,
    }
