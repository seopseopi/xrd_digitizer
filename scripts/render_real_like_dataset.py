"""
Step 7: real-like XRD 이미지 렌더링 (styled 기반 품질 저하).

xrd_digitizer_v1_master_spec.md §7 준수.
- real_v1~v4 variant 고정
- JPEG/blur/perspective/tail drop 적용
- GT 변경 없음
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _choice_index(seed: int, sample_id: str, variant_id: str, salt: str, n: int) -> int:
    h = hashlib.sha256(f"{seed}:{sample_id}:{variant_id}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % max(n, 1)


def _safe_filename_part(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s).strip())


def _pick_float_in_range(seed: int, sample_id: str, variant_id: str, salt: str, lo: float, hi: float) -> float:
    h = hashlib.sha256(f"{seed}:{sample_id}:{variant_id}:{salt}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(2**64 - 1)
    return float(lo + u * (hi - lo))


def load_styled_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "styled_image_path", "gt_path", "clean_image_path", "source_json_path", "variant_type", "variant_id", "family_id", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"styled_manifest missing columns: {missing}")
    return df


def load_gt_plot_box(gt_path: str) -> Tuple[int, int, int, int]:
    with Path(gt_path).open("r", encoding="utf-8") as f:
        gt = json.load(f)
    pb = gt["plot_box"]
    return int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])


def get_real_variant(variant_id: str) -> dict:
    """§7.3: real-like variant 정의."""
    v = variant_id.strip()
    if v == "real_v1":
        return {"variant_id": v, "base_styled": ("styled_v2", "styled_v3"), "jpeg_quality": 75, "blur_sigma": 0.0, "brightness_delta_range": None, "contrast_scale_range": None, "border": False, "shadow": False, "perspective": False, "tail_drop": False}
    if v == "real_v2":
        return {"variant_id": v, "base_styled": ("styled_v4", "styled_v5"), "jpeg_quality": 55, "blur_sigma": 0.6, "brightness_delta_range": (-0.03, 0.03), "contrast_scale_range": (0.95, 1.0), "border": False, "shadow": False, "perspective": False, "tail_drop": False}
    if v == "real_v3":
        return {"variant_id": v, "base_styled": ("styled_v2", "styled_v3"), "jpeg_quality": 75, "blur_sigma_choices": (0.6, 1.0), "brightness_delta_range": (-0.05, 0.05), "contrast_scale_range": None, "border": True, "shadow": True, "perspective": False, "tail_drop": False}
    if v == "real_v4":
        return {"variant_id": v, "base_styled": ("styled_v5",), "jpeg_quality_choices": (75, 55), "blur_sigma": 0.6, "brightness_delta_range": None, "contrast_scale_range": None, "border": False, "shadow": False, "perspective": True, "tail_drop": True, "tail_ratio_range": (0.10, 0.20)}
    raise KeyError(f"Unknown real variant_id: {variant_id!r}")


def apply_jpeg_artifact(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


def apply_blur(image: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0.0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def _homography_from_4pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A: List[List[float]] = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, X * x, X * y, X])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, Y * x, Y * y, Y])
    amat = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(amat)
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    return H


def _bilinear_sample(ch, sx, sy, fill):
    h, w = ch.shape
    out = np.full_like(sx, fill, dtype=np.float64)
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    xi = np.floor(sx[valid]).astype(np.int32)
    yi = np.floor(sy[valid]).astype(np.int32)
    xf = sx[valid] - xi
    yf = sy[valid] - yi
    x1 = np.clip(xi + 1, 0, w - 1)
    y1 = np.clip(yi + 1, 0, h - 1)
    out[valid] = (1 - xf) * (1 - yf) * ch[yi, xi] + xf * (1 - yf) * ch[yi, x1] + (1 - xf) * yf * ch[y1, xi] + xf * yf * ch[y1, x1]
    return out


def apply_perspective(image: Image.Image, offset_ratio: float) -> Image.Image:
    """§7.8: 약한 원근 왜곡."""
    img = image.convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float64)
    head = arr.tobytes()[: min(12_000, arr.nbytes)]
    ent = hashlib.sha256(head + struct.pack("d", float(offset_ratio))).digest()

    lim = max(1.0, min(float(offset_ratio) * float(min(w, h)), 0.03 * float(min(w, h))))
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    deltas = np.zeros((4, 2), dtype=np.float64)
    for k in range(8):
        deltas[k // 2, k % 2] = (ent[k] / 255.0 * 2.0 - 1.0) * lim * 0.5
    dst = src + deltas
    H = _homography_from_4pts(src, dst)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return img
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    hom = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=0)
    src_h = Hi @ hom
    src_h /= np.clip(src_h[2:3, :], 1e-9, None)
    sx = src_h[0].reshape(h, w)
    sy = src_h[1].reshape(h, w)
    out = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        out[:, :, c] = _bilinear_sample(arr[:, :, c], sx, sy, 255.0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def apply_tail_drop(image: Image.Image, plot_box: Tuple[int, int, int, int], ratio: float) -> Image.Image:
    """§7.9: 오른쪽 20% 구간 대비 완화."""
    x0, y0, x1, y1 = plot_box
    px_tail0 = max(int(x0), min(int(x1), int(round(float(x0) + 0.8 * (float(x1) - float(x0))))))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    x_start, x_end = max(0, px_tail0), min(arr.shape[1], int(x1) + 1)
    y_start, y_end = max(0, int(y0)), min(arr.shape[0], int(y1) + 1)
    slab = arr[y_start:y_end, x_start:x_end].copy()
    diff = slab - bg[None, None, :]
    dist = np.linalg.norm(diff, axis=2)
    rows = np.arange(y_start, y_end, dtype=np.int32)[:, None]
    mask = (rows < (int(y1) - 2)) & (dist > 12.0)
    slab = np.where(mask[..., None], bg[None, None, :] + diff * (1.0 - float(ratio)), slab)
    out = arr.copy()
    out[y_start:y_end, x_start:x_end] = slab
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _apply_brightness_contrast(image, bright_factor, contrast_factor):
    im = image.convert("RGB")
    if abs(bright_factor - 1.0) > 1e-6:
        im = ImageEnhance.Brightness(im).enhance(float(bright_factor))
    if abs(contrast_factor - 1.0) > 1e-6:
        im = ImageEnhance.Contrast(im).enhance(float(contrast_factor))
    return im


def _apply_border_and_shadow(image, seed, sample_id, variant_id):
    """§7.7: 외곽 border + 약한 그림자."""
    w, h = image.size
    bi = _choice_index(seed, sample_id, variant_id, "border_px", 21)
    m = 10 + int(bi)
    nw, nh = w + 2 * m, h + 2 * m
    pad_extra = 6
    canvas = Image.new("RGB", (nw + pad_extra, nh + pad_extra), (255, 255, 255))
    sh = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([0, 0, nw - 1, nh - 1], fill=(40, 40, 40, 35))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=3.5))
    canvas.paste(sh, (m + 4, m + 4), sh)
    canvas.paste(image.convert("RGB"), (m, m))
    return canvas


def _pick_base_styled_row(styled_df, sample_id, bases, seed, real_variant_id):
    sub = styled_df[(styled_df["sample_id"].astype(str) == str(sample_id)) & (styled_df["variant_id"].astype(str).isin(list(bases)))]
    if sub.empty:
        return None
    k = _choice_index(seed, sample_id, real_variant_id, "base_pick", len(sub))
    return sub.iloc[int(k)]


def write_real_manifest_row(row: Dict[str, str], manifest_csv_path: str) -> None:
    path = Path(manifest_csv_path)
    ensure_dir(path.parent)
    columns = ["sample_id", "real_image_path", "styled_image_path", "base_styled_variant_id", "gt_path", "clean_image_path", "source_json_path", "variant_type", "variant_id", "family_id", "split"]
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in columns})


def _ordered_unique_sample_ids(df: pd.DataFrame) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for sid in df["sample_id"].astype(str).tolist():
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _process_one_real(styled_df, sample_id, real_cfg, seed, output_dir):
    rid = str(real_cfg["variant_id"])
    bases = tuple(real_cfg["base_styled"])
    row = _pick_base_styled_row(styled_df, sample_id, bases, seed, rid)
    if row is None:
        return None, None
    styled_path = str(row["styled_image_path"])
    if not Path(styled_path).is_file():
        return None, None
    img = Image.open(styled_path).convert("RGB")
    img.load()

    if "jpeg_quality_choices" in real_cfg:
        qi = _choice_index(seed, sample_id, rid, "jpeg_q", len(real_cfg["jpeg_quality_choices"]))
        q = int(real_cfg["jpeg_quality_choices"][qi])
    else:
        q = int(real_cfg["jpeg_quality"])
    img = apply_jpeg_artifact(img, q)

    if "blur_sigma_choices" in real_cfg:
        si = _choice_index(seed, sample_id, rid, "blur_s", len(real_cfg["blur_sigma_choices"]))
        sig = float(real_cfg["blur_sigma_choices"][si])
    else:
        sig = float(real_cfg["blur_sigma"])
    img = apply_blur(img, sig)

    br = real_cfg.get("brightness_delta_range")
    cr = real_cfg.get("contrast_scale_range")
    bf, cf = 1.0, 1.0
    if br is not None:
        bf = 1.0 + _pick_float_in_range(seed, sample_id, rid, "bright", float(br[0]), float(br[1]))
    if cr is not None:
        cf = _pick_float_in_range(seed, sample_id, rid, "contr", float(cr[0]), float(cr[1]))
    img = _apply_brightness_contrast(img, bf, cf)

    plot_box = load_gt_plot_box(str(row["gt_path"]))
    if real_cfg.get("tail_drop"):
        lo, hi = real_cfg["tail_ratio_range"]
        ratio = _pick_float_in_range(seed, sample_id, rid, "tailr", float(lo), float(hi))
        img = apply_tail_drop(img, plot_box, ratio)

    if real_cfg.get("border") or real_cfg.get("shadow"):
        img = _apply_border_and_shadow(img, seed, sample_id, rid)

    if real_cfg.get("perspective"):
        p_idx = _choice_index(seed, sample_id, rid, "persp_gate", 4)
        if p_idx == 0:  # §7.8: 확률 0.25
            off = _pick_float_in_range(seed, sample_id, rid, "poff", 0.01, 0.03)
            img = apply_perspective(img, off)

    out_name = f"{_safe_filename_part(sample_id)}_{rid}.png"
    out_path = Path(output_dir) / out_name
    img.save(str(out_path), format="PNG")
    return str(out_path), row


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: render real-like PNGs from styled manifest")
    parser.add_argument("--styled_manifest", type=str, default=r"c:\xrd_digitizer_v1\data\manifests\styled_manifest.csv")
    parser.add_argument("--output_dir", type=str, default=r"c:\xrd_digitizer_v1\data\rendered_real_like")
    parser.add_argument("--manifest_csv", type=str, default=r"c:\xrd_digitizer_v1\data\manifests\real_manifest.csv")
    parser.add_argument("--variants", nargs="+", default=["real_v1", "real_v2", "real_v3", "real_v4"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    styled_df = load_styled_manifest(args.styled_manifest)
    sample_ids = _ordered_unique_sample_ids(styled_df)[:int(args.max_samples)]
    ensure_dir(args.output_dir)
    ensure_dir(Path(args.manifest_csv).parent)

    if Path(args.manifest_csv).exists():
        Path(args.manifest_csv).unlink()

    for sid in sample_ids:
        for vid in args.variants:
            try:
                rcfg = get_real_variant(vid)
            except KeyError as e:
                print(f"[SKIP] {sid} {vid}: {e}")
                continue
            out_path, base_row = _process_one_real(styled_df, sid, rcfg, int(args.seed), args.output_dir)
            if not out_path or base_row is None:
                print(f"[SKIP] {sid} {vid}: missing styled base")
                continue
            write_real_manifest_row({
                "sample_id": sid, "real_image_path": out_path,
                "styled_image_path": str(base_row["styled_image_path"]),
                "base_styled_variant_id": str(base_row["variant_id"]),
                "gt_path": str(base_row["gt_path"]),
                "clean_image_path": str(base_row["clean_image_path"]),
                "source_json_path": str(base_row["source_json_path"]),
                "variant_type": "real_like", "variant_id": str(vid),
                "family_id": str(base_row.get("family_id", "") or ""),
                "split": str(base_row.get("split", "") or ""),
            }, args.manifest_csv)
            print(f"[OK] {sid} {vid}")

    print(f"\n[DONE] real manifest: {args.manifest_csv}")


if __name__ == "__main__":
    main()
