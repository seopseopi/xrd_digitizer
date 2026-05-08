"""restore_peaks_lowered_by_smoothing: SG에 눌린 작은 피크 꼭짓점 복원."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from trace.postprocess import (  # noqa: E402
    _repair_sg_vs_gapfilled,
    fidelity_thresholds_from_trace,
    repair_isolated_spike_down_y,
    restore_peaks_lowered_by_smoothing,
)


class TestPostprocessPeakRestore(unittest.TestCase):
    def test_restores_apex_when_sg_flattened(self) -> None:
        n = 15
        valid = np.ones(n, dtype=bool)
        y_filled = np.full(n, 52.0)
        y_filled[7] = 10.0
        y_filled[6] = y_filled[8] = 48.0
        y_smooth = y_filled.copy()
        y_smooth[7] = 28.0
        out = restore_peaks_lowered_by_smoothing(
            y_filled, y_smooth, valid, radius=5, min_prominence_px=0.85, min_lift_px=0.35,
        )
        self.assertAlmostEqual(float(out[7]), 10.0, places=5)

    def test_auto_thresholds_restore_same_apex(self) -> None:
        n = 15
        valid = np.ones(n, dtype=bool)
        y_filled = np.full(n, 52.0)
        y_filled[7] = 10.0
        y_filled[6] = y_filled[8] = 48.0
        y_smooth = y_filled.copy()
        y_smooth[7] = 28.0
        out = restore_peaks_lowered_by_smoothing(y_filled, y_smooth, valid)
        self.assertAlmostEqual(float(out[7]), 10.0, places=3)

    def test_sg_gapfill_envelope_uses_min_y(self) -> None:
        """강도↑ = y↓ 이므로 SG/DP 병합은 minimum(y)로 주피크 유지( maximum 시 강도 하락)."""
        v = np.ones(5, dtype=bool)
        yf = np.array([100.0, 50.0, 100.0, 100.0, 100.0])
        yg = np.array([100.0, 55.0, 100.0, 100.0, 100.0])
        out = _repair_sg_vs_gapfilled(yf, yg, v)
        self.assertAlmostEqual(float(out[1]), 50.0, places=5)

    def test_fidelity_thresholds_keys(self) -> None:
        n = 30
        valid = np.ones(n, dtype=bool)
        y = 40.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, n))
        th = fidelity_thresholds_from_trace(y, valid)
        for k in ("radius", "min_prominence_px", "min_lift_px", "spike_delta_px"):
            self.assertIn(k, th)
            self.assertGreater(th[k], 0.0)

    def test_repair_spike_auto_runs(self) -> None:
        n = 5
        valid = np.ones(n, dtype=bool)
        y = np.array([10.0, 10.0, 40.0, 10.0, 10.0])
        out = repair_isolated_spike_down_y(y, valid)
        self.assertLess(float(out[2]), 40.0)

    def test_skips_baseline_noise(self) -> None:
        n = 20
        valid = np.ones(n, dtype=bool)
        y_filled = 50.0 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n))
        y_smooth = y_filled + 0.5
        out = restore_peaks_lowered_by_smoothing(
            y_filled, y_smooth, valid, radius=4, min_prominence_px=2.0, min_lift_px=0.35,
        )
        self.assertTrue(np.allclose(out, y_smooth, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
