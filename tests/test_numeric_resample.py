"""numeric_export 균일 2θ 리샘플 (단계 1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibrate.numeric_export import resample_two_theta_uniform  # noqa: E402


class TestNumericResample(unittest.TestCase):
    def test_noop_when_n_small(self) -> None:
        tt = [0.0, 1.0, 2.0]
        yy = [10.0, 20.0, 10.0]
        a, b = resample_two_theta_uniform(tt, yy, 0.0, 2.0, 3)
        self.assertEqual(len(a), 3)

    def test_interpolates_length(self) -> None:
        tt = [0.0, 2.0]
        yy = [0.0, 100.0]
        a, b = resample_two_theta_uniform(tt, yy, 0.0, 2.0, 5)
        self.assertEqual(len(a), 5)
        self.assertEqual(len(b), 5)
        self.assertAlmostEqual(b[0], 0.0, places=5)
        self.assertAlmostEqual(b[-1], 100.0, places=5)

    def test_resample_peak_envelope_not_below_apex(self) -> None:
        """격자가 피크 2θ와 어긋날 때 선형만 쓰면 꼭짓점보다 낮아짐 → 구간 max로 보정."""
        tt = [0.0, 1.0, 2.0, 3.0, 4.0]
        yy = [0.0, 40.0, 100.0, 40.0, 0.0]
        _, b = resample_two_theta_uniform(tt, yy, 0.0, 4.0, 6)
        self.assertGreater(max(b), 95.0, "linear-only would stay ~76 near x=2.4")


if __name__ == "__main__":
    unittest.main()
