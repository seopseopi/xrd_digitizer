"""
Step 5: clean XRD 이미지 렌더링 + GT JSON + manifest 생성.

xrd_digitizer_v1_master_spec.md §5 준수.
- canvas 1200×900, plot_box [170,90,1120,780]
- 16px Noto Sans (fallback: DejaVu → system default)
- GT JSON 필드 12개 고정
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks, peak_prominences, savgol_filter

# JSON keys (§3.2)
X_KEY = "two_theta_values"
Y_KEY = "intensities"

# §5.2: Clean render constants
CANVAS_W, CANVAS_H = 1200, 900
PLOT_BOX = (170, 90, 1120, 780)

BG = (255, 255, 255)
AXIS_COLOR = (30, 30, 30)
AXIS_THICKNESS = 2
TICK_LEN = 6
XTICKS = 8
YTICKS = 6
CURVE_COLOR = (20, 20, 20)
CURVE_THICKNESS = 2
FONT_SIZE = 16
LABEL_MARGIN_PX = 8

HEADROOM_RATIO = 0.08


def _try_load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_subset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"sample_id", "source_json_path", "debug_split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"dev_subset.csv missing columns: {missing}")
    return df


def load_sample_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_x_tick(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def _format_y_tick(v: float) -> str:
    av = abs(float(v))
    if av >= 1000:
        return f"{int(round(float(v))):,}"
    if av >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.4f}".rstrip("0").rstrip(".")


def map_xy_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    plot_box: Tuple[int, int, int, int],
    y_map_min: float | None = None,
    y_map_max: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    x0, y0, x1, y1 = plot_box
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    span_x = x_max - x_min
    span_y = y_max - y_min
    if span_x <= 0 or span_y <= 0:
        raise ValueError("Degenerate axis span")

    px_w = float(x1 - x0)
    px_h = float(y1 - y0)

    y0m = float(y_min if y_map_min is None else y_map_min)
    y1m = float(y_max if y_map_max is None else y_map_max)
    span_y_map = float(y1m - y0m)
    if span_y_map <= 0:
        raise ValueError("Degenerate Y mapping span")

    px = x0 + (x - x_min) * (px_w / span_x)
    py = y1 - (y - y0m) * (px_h / span_y_map)
    return px, py, x_min, x_max, y_min, y_max


def _clamp_plot_xy(px: float, py: float, plot_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = plot_box
    return float(np.clip(px, x0, x1)), float(np.clip(py, y0, y1))


def _draw_aa_polyline(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], color, width: int) -> None:
    if len(pts) < 2:
        return
    draw.line(pts, fill=color, width=width, joint="curve")


def _linear_interpolate_y_at_x(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    if xq <= float(x[0]):
        x0_, x1_ = float(x[0]), float(x[1])
        y0_, y1_ = float(y[0]), float(y[1])
        t = (xq - x0_) / (x1_ - x0_)
        return y0_ + t * (y1_ - y0_)
    if xq >= float(x[-1]):
        x0_, x1_ = float(x[-2]), float(x[-1])
        y0_, y1_ = float(y[-2]), float(y[-1])
        t = (xq - x0_) / (x1_ - x0_)
        return y0_ + t * (y1_ - y0_)
    idx = int(np.searchsorted(x, xq, side="right"))
    x0_ = float(x[idx - 1])
    x1_ = float(x[idx])
    y0_ = float(y[idx - 1])
    y1_ = float(y[idx])
    t = (xq - x0_) / (x1_ - x0_)
    return y0_ + t * (y1_ - y0_)


def _build_per_column_y_gt(
    x: np.ndarray, y: np.ndarray,
    plot_box: Tuple[int, int, int, int],
    x_min: float, x_max: float, y_min: float, y_max: float,
    y_map_max: float,
) -> Dict[str, int]:
    """§5.7: plot_box 내부 정수 x column마다 y 1개 (선형 보간)."""
    x0, y0, x1, y1 = plot_box
    px_w = float(x1 - x0)
    px_h = float(y1 - y0)
    span_x = float(x_max - x_min)
    span_y = float(y_map_max - y_min)
    if span_y <= 0:
        raise ValueError("Degenerate y span for per-column GT")

    out: Dict[str, int] = {}
    for xc in range(int(x0), int(x1) + 1):
        xq = x_min + (float(xc - x0) * span_x / px_w)
        yq = _linear_interpolate_y_at_x(x, y, float(xq))
        py = float(y1) - (float(yq) - float(y_min)) * (px_h / span_y)
        out[str(int(xc))] = int(round(py))
    return out


def _estimate_peaks_for_gt(x: np.ndarray, y: np.ndarray):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dyn = float(y_max - y_min)
    if len(y) < 7:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    y_s = savgol_filter(y, window_length=7, polyorder=2, mode="interp")
    resid = y - y_s
    sigma = float(np.std(resid)) if len(resid) > 1 else 0.0
    local_floor = max(3.0 * sigma, 0.01 * dyn)
    prom = max(0.07 * dyn, local_floor)

    peaks, _ = find_peaks(y_s, prominence=max(prom, 1e-12))
    if peaks.size == 0:
        return peaks.astype(int), np.array([], dtype=float), np.array([], dtype=float)

    proms = peak_prominences(y_s, peaks, wlen=7)[0]
    return peaks.astype(int), proms.astype(float), y_s


def render_clean_image(sample_id: str, x: np.ndarray, y: np.ndarray, plot_box: Tuple[int, int, int, int]):
    font = _try_load_font(FONT_SIZE)
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)

    x0, y0, x1, y1 = plot_box
    _, _, x_min, x_max, y_min, y_max = map_xy_to_pixels(x, y, plot_box)

    y_range = float(y_max - y_min)
    y_plot_min = float(y_min)
    y_plot_max = float(y_max + HEADROOM_RATIO * y_range)

    px, py, _, _, _, _ = map_xy_to_pixels(x, y, plot_box, y_map_min=y_plot_min, y_map_max=y_plot_max)

    draw.rectangle([x0, y0, x1, y1], outline=AXIS_COLOR, width=AXIS_THICKNESS)

    x_axis_y = y1
    y_axis_x = x0
    draw.line([(x0, x_axis_y), (x1, x_axis_y)], fill=AXIS_COLOR, width=AXIS_THICKNESS)
    draw.line([(y_axis_x, y0), (y_axis_x, y1)], fill=AXIS_COLOR, width=AXIS_THICKNESS)

    xt = np.linspace(x_min, x_max, XTICKS)
    yt = np.linspace(y_min, y_max, YTICKS)

    for xv in xt:
        pxv = float(x0 + (xv - x_min) * ((x1 - x0) / (x_max - x_min)))
        draw.line([(pxv, x_axis_y), (pxv, x_axis_y + TICK_LEN)], fill=AXIS_COLOR, width=AXIS_THICKNESS)
        label = _format_x_tick(float(xv))
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        tx = pxv - tw / 2
        ty = x_axis_y + TICK_LEN + LABEL_MARGIN_PX
        draw.text((tx, ty), label, fill=AXIS_COLOR, font=font)

    for yv in yt:
        pyv = float(y1 - (yv - y_min) * ((y1 - y0) / (y_max - y_min)))
        draw.line([(y_axis_x - TICK_LEN, pyv), (y_axis_x, pyv)], fill=AXIS_COLOR, width=AXIS_THICKNESS)
        label = _format_y_tick(float(yv))
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        tx = y_axis_x - TICK_LEN - LABEL_MARGIN_PX - tw
        ty = pyv - th / 2
        draw.text((tx, ty), label, fill=AXIS_COLOR, font=font)

    pts: List[Tuple[float, float]] = []
    pixel_curve_path: List[List[int]] = []
    for i in range(len(x)):
        pxi, pyi = _clamp_plot_xy(float(px[i]), float(py[i]), plot_box)
        pts.append((pxi, pyi))
        pixel_curve_path.append([int(round(pxi)), int(round(pyi))])

    scale = 2
    overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    big = overlay.resize((CANVAS_W * scale, CANVAS_H * scale), resample=Image.Resampling.NEAREST)
    bdraw = ImageDraw.Draw(big)
    pts_s = [(px * scale, py * scale) for px, py in pts]
    _draw_aa_polyline(bdraw, pts_s, CURVE_COLOR + (255,), width=int(CURVE_THICKNESS * scale))
    overlay_sm = big.resize((CANVAS_W, CANVAS_H), resample=Image.Resampling.LANCZOS)

    a = overlay_sm.split()[-1]
    ink = Image.new("RGB", (CANVAS_W, CANVAS_H), CURVE_COLOR)
    img.paste(ink, (0, 0), mask=a)

    meta = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    return img, pixel_curve_path, meta


def build_gt(
    sample_id: str, source_json_path: str,
    x: np.ndarray, y: np.ndarray,
    plot_box: Tuple[int, int, int, int],
    pixel_curve_path: List[List[int]],
    axis_meta: Dict[str, Any],
) -> dict:
    """§5.5: GT JSON 필드 12개 고정."""
    x0, y0, x1, y1 = plot_box
    peaks, proms, y_s = _estimate_peaks_for_gt(x, y)

    peak_x_values = x[peaks].astype(float).tolist() if peaks.size else []
    peak_y_values = y[peaks].astype(float).tolist() if peaks.size else []
    peak_prominences_list = proms.astype(float).tolist() if peaks.size else []

    peak_pixel_points: List[List[int]] = []
    if peaks.size:
        y_min_m = float(axis_meta["y_min"])
        y_max_m = float(axis_meta["y_max"])
        y_range_m = float(y_max_m - y_min_m)
        y_plot_max_m = float(y_max_m + HEADROOM_RATIO * y_range_m)
        px_arr, py_arr, _, _, _, _ = map_xy_to_pixels(x, y, plot_box, y_map_min=y_min_m, y_map_max=y_plot_max_m)
        for idx in peaks.tolist():
            pxi, pyi = _clamp_plot_xy(float(px_arr[idx]), float(py_arr[idx]), plot_box)
            peak_pixel_points.append([int(round(pxi)), int(round(pyi))])

    major_peak_indices: List[int] = []
    if peaks.size and y_s is not None:
        num_peaks = len(peaks)
        n_major = max(3, int(np.ceil(0.1 * num_peaks)))
        n_major = min(n_major, 8, num_peaks)
        if proms.size > 0:
            order = np.argsort(proms)[::-1]
            major_peak_indices = [int(peaks[int(i)]) for i in order[:n_major]]
        else:
            order = np.argsort(y_s[peaks])[::-1]
            major_peak_indices = [int(peaks[int(i)]) for i in order[:n_major]]

    y_min_m = float(axis_meta["y_min"])
    y_max_m = float(axis_meta["y_max"])
    y_range_m = float(y_max_m - y_min_m)
    y_plot_max_m = float(y_max_m + HEADROOM_RATIO * y_range_m)

    per_column_y_gt = _build_per_column_y_gt(x, y, plot_box, float(axis_meta["x_min"]), float(axis_meta["x_max"]), y_min_m, y_max_m, y_plot_max_m)

    return {
        "sample_id": sample_id,
        "source_json_path": source_json_path,
        "x_values": x.astype(float).tolist(),
        "y_values": y.astype(float).tolist(),
        "plot_box": [int(x0), int(y0), int(x1), int(y1)],
        "pixel_curve_path": pixel_curve_path,
        "per_column_y_gt": per_column_y_gt,
        "axis_metadata": {
            "x_min": float(axis_meta["x_min"]),
            "x_max": float(axis_meta["x_max"]),
            "y_min": float(axis_meta["y_min"]),
            "y_max": float(axis_meta["y_max"]),
        },
        "peak_indices": peaks.astype(int).tolist(),
        "peak_x_values": peak_x_values,
        "peak_y_values": peak_y_values,
        "peak_prominences": peak_prominences_list,
        "peak_pixel_points": peak_pixel_points,
        "major_peak_indices": major_peak_indices,
        "render_variant": "clean_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: render clean images + GT JSON")
    parser.add_argument("--subset_csv", type=str, default=r"c:\xrd_digitizer_v1\data\metadata\dev_subset.csv")
    parser.add_argument("--output_dir", type=str, default=r"c:\xrd_digitizer_v1\data\rendered_clean")
    parser.add_argument("--gt_dir", type=str, default=r"c:\xrd_digitizer_v1\data\gt")
    parser.add_argument("--manifest_csv", type=str, default=r"c:\xrd_digitizer_v1\data\manifests\clean_manifest.csv")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    subset = load_subset(args.subset_csv).head(args.max_samples)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.gt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.manifest_csv).parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []

    for _, r in subset.iterrows():
        sample_id = str(r["sample_id"])
        src = str(r["source_json_path"])
        split = str(r["debug_split"])

        try:
            obj = load_sample_json(src)
            x = np.asarray(obj[X_KEY], dtype=float)
            y = np.asarray(obj[Y_KEY], dtype=float)
            if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
                raise ValueError("bad x/y arrays")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("non-finite values")
            if not np.all(np.diff(x) > 0):
                raise ValueError("x not strictly increasing")

            img, pixel_curve_path, axis_meta = render_clean_image(sample_id, x, y, PLOT_BOX)
            gt = build_gt(sample_id, src, x, y, PLOT_BOX, pixel_curve_path, axis_meta)

            img_path = str(Path(args.output_dir) / f"{sample_id}_clean_v1.png")
            gt_path = str(Path(args.gt_dir) / f"{sample_id}_gt.json")

            img.save(img_path, format="PNG")
            with Path(gt_path).open("w", encoding="utf-8") as f:
                json.dump(gt, f, ensure_ascii=False, indent=2)

            manifest_rows.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "gt_path": gt_path,
                "source_json_path": src,
                "variant_type": "clean",
                "variant_id": "clean_v1",
                "family_id": "",
                "split": split,
            })
            print(f"[OK] {sample_id}")

        except Exception as exc:
            print(f"[SKIP] {sample_id}: {type(exc).__name__}: {exc}")

    man_df = pd.DataFrame(manifest_rows, columns=[
        "sample_id", "image_path", "gt_path", "source_json_path",
        "variant_type", "variant_id", "family_id", "split",
    ])
    man_df.to_csv(args.manifest_csv, index=False, encoding="utf-8")
    print(f"\n[DONE] manifest: {args.manifest_csv} (rows={len(man_df)})")


if __name__ == "__main__":
    main()
