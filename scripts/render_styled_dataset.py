"""
Step 6: styled XRD 이미지 렌더링 (GT 변경 없이 스타일만 변경).

xrd_digitizer_v1_master_spec.md §6 준수.
- styled_v1~v5 variant 고정
- color palette §6.4
- legend/grid 규칙 §6.5-6.6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Canvas / mapping (match Step 5 §5.2)
CANVAS_W, CANVAS_H = 1200, 900
HEADROOM_RATIO = 0.08
TICK_LEN_DEFAULT = 6
LABEL_MARGIN_PX = 8

# §6.4 color palette
BLACK = (20, 20, 20)
DARK_NAVY = (25, 45, 90)
DARK_RED = (140, 35, 35)
DARK_GREEN = (35, 110, 60)
MEDIUM_GRAY = (90, 90, 90)
MUTED_BLUE = (90, 120, 170)
MUTED_GREEN = (90, 140, 110)
LIGHT_GRAY_GRID = (220, 220, 220)
LIGHT_GRAY_GRID_ALT = (230, 230, 235)
DARK_GRAY_TEXT = (70, 70, 70)
DARK_GRAY_CURVE = (80, 80, 80)
PURE_WHITE = (255, 255, 255)
BG_V4 = (245, 246, 248)


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _choice_index(seed: int, sample_id: str, variant_id: str, salt: str, n: int) -> int:
    h = hashlib.sha256(f"{seed}:{sample_id}:{variant_id}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % max(n, 1)


def load_clean_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "image_path", "gt_path", "source_json_path", "variant_type", "variant_id", "family_id", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"clean_manifest.csv missing columns: {missing}")
    return df


def load_gt_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_style_variant(variant_id: str) -> dict:
    """§6.3: variant별 스타일 파라미터."""
    v = variant_id.strip()
    if v == "styled_v1":
        return {"variant_id": v, "background": PURE_WHITE, "curve_color_choices": (BLACK, DARK_NAVY), "curve_thickness": 2.0, "font_preference": "noto", "font_size": 16, "grid": False, "legend": False, "axis_thickness": 2, "x_tick_count": 8, "y_tick_count": 6, "axis_color": BLACK}
    if v == "styled_v2":
        return {"variant_id": v, "background": PURE_WHITE, "curve_color_choices": (DARK_NAVY, DARK_RED, DARK_GREEN), "curve_thickness": 2.0, "font_preference": "dejavu", "font_size": 17, "grid": True, "grid_color_choices": (LIGHT_GRAY_GRID, LIGHT_GRAY_GRID_ALT), "grid_width": 1, "legend": False, "axis_thickness": 2, "x_tick_count": 8, "y_tick_count": 6, "axis_color": DARK_GRAY_TEXT}
    if v == "styled_v3":
        return {"variant_id": v, "background": PURE_WHITE, "curve_color_choices": (DARK_RED, DARK_NAVY), "curve_thickness": 2.5, "font_preference": "arial_like", "font_size": 16, "grid": False, "legend": True, "legend_width_frac_choices": (0.18, 0.20, 0.22, 0.25), "legend_height_frac_choices": (0.08, 0.09, 0.10, 0.11, 0.12), "legend_pos_choices": ("top_left", "top_right", "upper_center_right"), "legend_text_choices": ("Sample A", "XRD Pattern", "Measured", "Run 01", "Pattern-1"), "legend_padding": 8, "legend_line_len_choices": tuple(range(28, 37)), "axis_thickness": 2, "x_tick_count": 8, "y_tick_count": 6, "axis_color": DARK_GRAY_TEXT}
    if v == "styled_v4":
        return {"variant_id": v, "background": BG_V4, "curve_color_choices": (MEDIUM_GRAY, MUTED_BLUE, MUTED_GREEN), "curve_thickness": 1.5, "font_preference": "noto", "font_size": 16, "grid": True, "grid_color_choices": (LIGHT_GRAY_GRID_ALT, LIGHT_GRAY_GRID), "grid_width": 1, "legend": False, "axis_thickness": 2, "x_tick_count": 10, "y_tick_count": 8, "axis_color": DARK_GRAY_TEXT}
    if v == "styled_v5":
        return {"variant_id": v, "background": PURE_WHITE, "curve_color": DARK_GRAY_CURVE, "curve_thickness": 1.8, "font_preference": "dejavu", "font_size": 16, "grid": False, "grid_weak": True, "grid_color": LIGHT_GRAY_GRID_ALT, "grid_width": 1, "legend": False, "axis_thickness": 2, "x_tick_count": 8, "y_tick_count": 6, "axis_color": DARK_GRAY_TEXT, "tail_contrast_drop_frac": None}
    raise KeyError(f"Unknown styled variant_id: {variant_id!r}")


def _try_load_font(preference: str, size: int) -> ImageFont.ImageFont:
    win_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    noto = ["/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]
    dejavu = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    arial_like = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]

    if preference == "noto":
        order = noto + win_fonts + dejavu + arial_like
    elif preference == "dejavu":
        order = dejavu + win_fonts + noto + arial_like
    else:
        order = arial_like + win_fonts + noto + dejavu
    for fp in order:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


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


def build_tick_values(vmin: float, vmax: float, n_ticks: int) -> np.ndarray:
    return np.linspace(float(vmin), float(vmax), max(2, int(n_ticks)))


def map_xy_to_pixels(x, y, plot_box, x_min, x_max, y_min, y_max, y_map_min, y_map_max):
    x0, y0, x1, y1 = plot_box
    span_x = float(x_max - x_min)
    span_y_map = float(y_map_max - y_map_min)
    if span_x <= 0 or span_y_map <= 0:
        raise ValueError("Degenerate axis span")
    px_w = float(x1 - x0)
    px_h = float(y1 - y0)
    px = x0 + (x - x_min) * (px_w / span_x)
    py = y1 - (y - y_map_min) * (px_h / span_y_map)
    return px.astype(float), py.astype(float)


def _clamp_plot_xy(px, py, plot_box):
    x0, y0, x1, y1 = plot_box
    return float(np.clip(px, x0, x1)), float(np.clip(py, y0, y1))


def build_tick_pixel_positions(plot_box, x_min, x_max, y_min, y_max, n_x_ticks, n_y_ticks):
    x0, y0, x1, y1 = plot_box
    xv = build_tick_values(x_min, x_max, n_x_ticks)
    yv = build_tick_values(y_min, y_max, n_y_ticks)
    span_x = float(x_max - x_min)
    span_y = float(y_max - y_min)
    if span_x <= 0 or span_y <= 0:
        raise ValueError("Degenerate span for ticks")
    w = float(x1 - x0)
    h = float(y1 - y0)
    x_px = x0 + (xv - x_min) * (w / span_x)
    y_px = y1 - (yv - y_min) * (h / span_y)
    return xv, yv, x_px.astype(float), y_px.astype(float)


def draw_grid(draw, plot_box, x_tick_px, y_tick_px, color, width):
    x0, y0, x1, y1 = plot_box
    for xp in x_tick_px:
        draw.line([(float(xp), float(y0)), (float(xp), float(y1))], fill=color, width=width)
    for yp in y_tick_px:
        draw.line([(float(x0), float(yp)), (float(x1), float(yp))], fill=color, width=width)


def draw_axes_and_ticks(draw, plot_box, x_min, x_max, y_min, y_max, n_x, n_y, font, axis_color, axis_th, tick_len):
    x0, y0, x1, y1 = plot_box
    draw.line([(x0, float(y1)), (x1, float(y1))], fill=axis_color, width=axis_th)
    draw.line([(float(x0), y0), (float(x0), y1)], fill=axis_color, width=axis_th)
    xv, yv, x_px, y_px = build_tick_pixel_positions(plot_box, x_min, x_max, y_min, y_max, n_x, n_y)
    for pxv, xv_ in zip(x_px, xv):
        draw.line([(float(pxv), float(y1)), (float(pxv), float(y1) + float(tick_len))], fill=axis_color, width=axis_th)
        label = _format_x_tick(float(xv_))
        tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else draw.textbbox((0, 0), label, font=font)[2]
        draw.text((float(pxv) - float(tw) / 2, float(y1) + float(tick_len) + LABEL_MARGIN_PX), label, fill=axis_color, font=font)
    for pyv, yv_ in zip(y_px, yv):
        draw.line([(float(x0) - float(tick_len), float(pyv)), (float(x0), float(pyv))], fill=axis_color, width=axis_th)
        label = _format_y_tick(float(yv_))
        tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else draw.textbbox((0, 0), label, font=font)[2]
        th = draw.textbbox((0, 0), label, font=font)[3] - draw.textbbox((0, 0), label, font=font)[1]
        draw.text((float(x0) - float(tick_len) - LABEL_MARGIN_PX - float(tw), float(pyv) - float(th) / 2), label, fill=axis_color, font=font)


def draw_legend(draw, plot_box, variant_cfg, font, curve_color, legend_text, legend_pos, box_w, box_h, line_len):
    x0, y0, x1, y1 = plot_box
    pad = int(variant_cfg.get("legend_padding", 8))
    if legend_pos == "top_left":
        bx0 = float(x0) + pad
        bx1 = bx0 + box_w
        by0 = float(y0) + pad
        by1 = by0 + box_h
    elif legend_pos == "top_right":
        bx1 = float(x1) - pad
        bx0 = bx1 - box_w
        by0 = float(y0) + pad
        by1 = by0 + box_h
    else:
        pw = float(x1 - x0)
        bx0 = float(x0) + pw * 0.52
        bx1 = min(float(x1) - pad, bx0 + box_w)
        bx0 = bx1 - box_w
        by0 = float(y0) + pad
        by1 = by0 + box_h
    draw.rectangle([(bx0, by0), (bx1, by1)], outline=LIGHT_GRAY_GRID, width=1, fill=PURE_WHITE)
    line_y = by0 + pad + 4
    line_x0 = bx0 + pad
    line_x1 = line_x0 + float(line_len)
    draw.line([(line_x0, line_y), (line_x1, line_y)], fill=curve_color, width=max(2, int(round(variant_cfg["curve_thickness"]))))
    draw.text((line_x1 + pad, by0 + pad), legend_text, fill=DARK_GRAY_TEXT, font=font)


def _draw_aa_polyline(draw, pts, color, width):
    if len(pts) < 2:
        return
    draw.line(pts, fill=color, width=width, joint="curve")


def _resolve_curve_color(variant_cfg):
    if "curve_color" in variant_cfg:
        return tuple(variant_cfg["curve_color"])
    choices = variant_cfg.get("curve_color_choices") or (BLACK,)
    idx = _choice_index(int(variant_cfg["_seed"]), str(variant_cfg["_sample_id"]), str(variant_cfg["variant_id"]), "curve_color", len(choices))
    return tuple(choices[idx])


def apply_tail_contrast_drop(image, gt, variant_cfg, drop_frac):
    plot_box_t = tuple(int(v) for v in gt["plot_box"])
    am = gt["axis_metadata"]
    x_min = float(am["x_min"])
    x_max = float(am["x_max"])
    x0, y0, x1, y1 = plot_box_t
    tail_x0 = x_min + 0.8 * (x_max - x_min)
    span = float(x_max - x_min)
    if span <= 0:
        return image
    px_tail0 = int(round(x0 + (tail_x0 - x_min) * (float(x1 - x0) / span)))
    px_tail0 = max(int(x0), min(int(x1), px_tail0))
    arr = np.asarray(image, dtype=np.float32)
    bg = np.array(variant_cfg["background"], dtype=np.float32)
    x_start, x_end = max(0, px_tail0), min(arr.shape[1], int(x1) + 1)
    y_start, y_end = max(0, int(y0)), min(arr.shape[0], int(y1) + 1)
    slab = arr[y_start:y_end, x_start:x_end].copy()
    diff = slab - bg[None, None, :]
    dist = np.linalg.norm(diff, axis=2)
    rows = np.arange(y_start, y_end, dtype=np.int32)[:, None]
    mask = (rows < (int(y1) - 2)) & (dist > 12.0)
    slab = np.where(mask[..., None], bg[None, None, :] + diff * (1.0 - float(drop_frac)), slab)
    out = arr.copy()
    out[y_start:y_end, x_start:x_end] = slab
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def render_styled_from_gt(gt: dict, variant_cfg: Mapping[str, Any]) -> Image.Image:
    cfg: Dict[str, Any] = dict(variant_cfg)
    plot_box_t = tuple(int(v) for v in gt["plot_box"])
    am = gt["axis_metadata"]
    x_min, x_max = float(am["x_min"]), float(am["x_max"])
    y_min, y_max = float(am["y_min"]), float(am["y_max"])
    x = np.asarray(gt["x_values"], dtype=float)
    y = np.asarray(gt["y_values"], dtype=float)
    y_range = float(y_max - y_min)
    y_plot_min, y_plot_max = float(y_min), float(y_max + HEADROOM_RATIO * y_range)

    curve_color = _resolve_curve_color(cfg)
    bg = tuple(cfg["background"])
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    draw = ImageDraw.Draw(img)

    x0, y0, x1, y1 = plot_box_t
    n_x, n_y = int(cfg["x_tick_count"]), int(cfg["y_tick_count"])
    xv, yv, x_px, y_px = build_tick_pixel_positions(plot_box_t, x_min, x_max, y_min, y_max, n_x, n_y)

    if cfg.get("variant_id") == "styled_v5" and cfg.get("grid_weak") and not cfg.get("grid"):
        draw_grid(draw, plot_box_t, x_px, y_px, (242, 242, 244), 1)
    if cfg.get("grid"):
        gc_choices = cfg.get("grid_color_choices") or (LIGHT_GRAY_GRID,)
        gi = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "grid_color", len(gc_choices))
        draw_grid(draw, plot_box_t, x_px, y_px, gc_choices[gi], int(cfg.get("grid_width", 1)))

    axis_color = tuple(cfg.get("axis_color", BLACK))
    axis_th = int(cfg["axis_thickness"])
    draw.rectangle([x0, y0, x1, y1], outline=axis_color, width=axis_th)

    font = _try_load_font(str(cfg["font_preference"]), int(cfg["font_size"]))
    draw_axes_and_ticks(draw, plot_box_t, x_min, x_max, y_min, y_max, n_x, n_y, font, axis_color, axis_th, TICK_LEN_DEFAULT)

    px, py = map_xy_to_pixels(x, y, plot_box_t, x_min, x_max, y_min, y_max, y_plot_min, y_plot_max)
    pts = [_clamp_plot_xy(float(px[i]), float(py[i]), plot_box_t) for i in range(len(x))]

    scale = 2
    overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    big = overlay.resize((CANVAS_W * scale, CANVAS_H * scale), resample=Image.Resampling.NEAREST)
    bdraw = ImageDraw.Draw(big)
    pts_s = [(px * scale, py * scale) for px, py in pts]
    lw = max(1, int(round(float(cfg["curve_thickness"]) * scale)))
    _draw_aa_polyline(bdraw, pts_s, curve_color + (255,), width=lw)
    overlay_sm = big.resize((CANVAS_W, CANVAS_H), resample=Image.Resampling.LANCZOS)
    a = overlay_sm.split()[-1]
    ink = Image.new("RGB", (CANVAS_W, CANVAS_H), curve_color)
    img.paste(ink, (0, 0), mask=a)

    if cfg.get("legend"):
        wi = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "legend_wi", len(tuple(cfg["legend_width_frac_choices"])))
        hi = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "legend_hi", len(tuple(cfg["legend_height_frac_choices"])))
        box_w = (float(x1) - float(x0)) * float(cfg["legend_width_frac_choices"][wi])
        box_h = (float(y1) - float(y0)) * float(cfg["legend_height_frac_choices"][hi])
        pi = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "legend_pos", len(tuple(cfg["legend_pos_choices"])))
        ti = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "legend_text", len(tuple(cfg["legend_text_choices"])))
        li = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "legend_line", len(tuple(cfg["legend_line_len_choices"])))
        draw = ImageDraw.Draw(img)
        draw_legend(draw, plot_box_t, cfg, font, curve_color, cfg["legend_text_choices"][ti], cfg["legend_pos_choices"][pi], box_w, box_h, int(cfg["legend_line_len_choices"][li]))

    if cfg.get("variant_id") == "styled_v5":
        drop = cfg.get("tail_contrast_drop_frac")
        if drop is None:
            di = _choice_index(int(cfg["_seed"]), str(cfg["_sample_id"]), str(cfg["variant_id"]), "tail_drop", 11)
            drop = 0.10 + di * 0.01
        img = apply_tail_contrast_drop(img, gt, cfg, float(drop))
    return img


def _safe_filename_part(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s).strip())


def write_manifest_row(row: Dict[str, str], manifest_csv_path: str) -> None:
    path = Path(manifest_csv_path)
    ensure_dir(path.parent)
    columns = ["sample_id", "styled_image_path", "gt_path", "clean_image_path", "source_json_path", "variant_type", "variant_id", "family_id", "split"]
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: render styled PNGs from clean manifest + GT")
    parser.add_argument("--clean_manifest", type=str, default=r"c:\xrd_digitizer_v1\data\manifests\clean_manifest.csv")
    parser.add_argument("--output_dir", type=str, default=r"c:\xrd_digitizer_v1\data\rendered_styled")
    parser.add_argument("--manifest_csv", type=str, default=r"c:\xrd_digitizer_v1\data\manifests\styled_manifest.csv")
    parser.add_argument("--variants", nargs="+", default=["styled_v1", "styled_v2", "styled_v3", "styled_v4", "styled_v5"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    df = load_clean_manifest(args.clean_manifest).head(int(args.max_samples))
    ensure_dir(args.output_dir)
    ensure_dir(Path(args.manifest_csv).parent)

    if Path(args.manifest_csv).exists():
        Path(args.manifest_csv).unlink()

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        clean_path = str(row["image_path"])
        gt_path = str(row["gt_path"])
        src_path = str(row["source_json_path"])
        family_id = str(row.get("family_id", "") or "")
        split = str(row.get("split", "") or "")

        gt = load_gt_json(gt_path)
        for vid in args.variants:
            base_cfg = get_style_variant(vid)
            cfg = {**base_cfg, "_seed": int(args.seed), "_sample_id": sample_id}
            img = render_styled_from_gt(gt, cfg)

            out_name = f"{_safe_filename_part(sample_id)}_{_safe_filename_part(vid)}.png"
            styled_path = str(Path(args.output_dir) / out_name)
            img.save(styled_path, format="PNG")

            write_manifest_row({
                "sample_id": sample_id, "styled_image_path": styled_path,
                "gt_path": gt_path, "clean_image_path": clean_path,
                "source_json_path": src_path, "variant_type": "styled",
                "variant_id": vid, "family_id": family_id, "split": split,
            }, args.manifest_csv)
            print(f"[OK] {sample_id} {vid}")

    print(f"\n[DONE] styled manifest: {args.manifest_csv}")


if __name__ == "__main__":
    main()
