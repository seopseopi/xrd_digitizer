"""blend_sg_toward_gapfill_on_high_curvature: 고곡률에서 SG→gap-fill 블렌드."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from trace.postprocess import blend_sg_toward_gapfill_on_high_curvature  # noqa: E402


class TestCurvatureBlend(unittest.TestCase):
    def test_moves_toward_gapfill_on_peak(self) -> None:
        n = 20
        v = np.ones(n, dtype=bool)
        yf = np.linspace(100, 100, n)
        yf[10] = 20.0
        # SG가 꼭짓점을 뭉개 40으로 가정
        ys = yf.copy()
        ys[10] = 45.0
        out = blend_sg_toward_gapfill_on_high_curvature(yf, ys, v, strength=0.5)
        self.assertLess(float(out[10]), float(ys[10]))
        self.assertLess(abs(float(out[10]) - float(yf[10])), abs(float(ys[10]) - float(yf[10])))

    def test_flat_unchanged(self) -> None:
        n = 10
        v = np.ones(n, dtype=bool)
        yf = np.full(n, 50.0)
        ys = np.full(n, 52.0)
        out = blend_sg_toward_gapfill_on_high_curvature(yf, ys, v, strength=0.5)
        self.assertTrue(np.allclose(out, ys))


if __name__ == "__main__":
    unittest.main()
