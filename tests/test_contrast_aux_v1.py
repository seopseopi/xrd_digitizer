"""contrast_aux_v1: 맵 생성·confidence 블렌드 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.contrast_aux_settings import ContrastAuxSettings  # noqa: E402
from preprocess.contrast_aux import (  # noqa: E402
    blend_confidence_with_contrast_aux,
    build_contrast_aux_map,
)


class TestContrastAuxV1(unittest.TestCase):
    def test_blend_skips_low_base_conf(self) -> None:
        cfg = ContrastAuxSettings(
            use_contrast_aux=True,
            contrast_aux_weight=0.25,
            contrast_aux_min_base_conf=0.15,
        )
        m = np.ones((10, 10), dtype=np.float32)
        f, b = blend_confidence_with_contrast_aux(0.10, 5, 5, m, cfg)
        self.assertEqual(f, 0.10)
        self.assertEqual(b, 1.0)

    def test_blend_mixes_when_base_high(self) -> None:
        cfg = ContrastAuxSettings(
            use_contrast_aux=True,
            contrast_aux_weight=0.25,
            contrast_aux_min_base_conf=0.15,
        )
        m = np.ones((10, 10), dtype=np.float32)
        f, _ = blend_confidence_with_contrast_aux(0.50, 5, 5, m, cfg)
        self.assertAlmostEqual(f, 0.75 * 0.5 + 0.25 * 1.0)

    def test_map_shape_and_range(self) -> None:
        cfg = ContrastAuxSettings(contrast_aux_border_suppress_px=0)
        img = np.full((64, 128, 3), 250, dtype=np.uint8)
        img[30:34, 60:68] = 20
        out = build_contrast_aux_map(img, (0, 0, 128, 64), None, cfg)
        self.assertEqual(out.shape, (64, 128))
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))


if __name__ == "__main__":
    unittest.main()
