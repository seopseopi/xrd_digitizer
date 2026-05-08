"""numeric_curve_peaks: 수출 곡선 적응형 피크 검출 (분리 로드맵 1단계)."""



from __future__ import annotations



import sys

import unittest

from pathlib import Path



import numpy as np



_REPO = Path(__file__).resolve().parents[1]

if str(_REPO) not in sys.path:

    sys.path.insert(0, str(_REPO))



from calibrate.numeric_curve_peaks import (  # noqa: E402

    detect_peaks_on_numeric_curve,

    refine_peak_parabolic_tt_yy,

)





class TestNumericCurvePeaks(unittest.TestCase):

    def test_parabolic_refine_shifts_toward_true_vertex(self) -> None:

        tt = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        yy = np.array([100.0, 120.0, 115.0], dtype=np.float64)

        tr, yr = refine_peak_parabolic_tt_yy(tt, yy, 1)

        self.assertGreater(tr, 1.0)

        self.assertLess(tr, 2.0)

        self.assertGreater(yr, 120.0)



    def test_resolves_two_close_peaks(self) -> None:

        n = 600

        tt = np.linspace(1.0, 160.0, n).tolist()

        tta = np.asarray(tt)

        # 서로 떨어진 두 돌출 — 검출 안정

        y1 = 9000.0 / (1.0 + ((tta - 34.5) / 0.4) ** 2)

        y2 = 7000.0 / (1.0 + ((tta - 42.75) / 0.38) ** 2)

        yy = (1100.0 + y1 + y2 + np.random.default_rng(0).normal(0, 15.0, n)).tolist()

        peaks, prm = detect_peaks_on_numeric_curve(tt, yy)

        self.assertGreaterEqual(len(peaks), 2, prm)

        tts = sorted([p["two_theta"] for p in peaks])

        self.assertTrue(any(33.5 < t < 35.5 for t in tts))

        self.assertTrue(any(41.5 < t < 44.0 for t in tts))



    def test_resolves_adjacent_pair_when_curve_supports(self) -> None:

        """근접 쌍: 충분히 분리된 로렌츠 두 개 (약 1° 간격)."""

        n = 800

        tt = np.linspace(20.0, 35.0, n).tolist()

        tta = np.asarray(tt)

        y1 = 5000.0 / (1.0 + ((tta - 26.2) / 0.25) ** 2)

        y2 = 4500.0 / (1.0 + ((tta - 27.1) / 0.25) ** 2)

        yy = (900.0 + y1 + y2).tolist()

        peaks, _ = detect_peaks_on_numeric_curve(tt, yy)

        self.assertGreaterEqual(len(peaks), 2)



    def test_flat_curve_empty_or_few(self) -> None:

        tt = np.linspace(0.0, 10.0, 50).tolist()

        yy = [100.0] * 50

        peaks, prm = detect_peaks_on_numeric_curve(tt, yy)

        self.assertEqual(len(peaks), 0)

        self.assertEqual(prm.get("reason"), "flat")





if __name__ == "__main__":

    unittest.main()

