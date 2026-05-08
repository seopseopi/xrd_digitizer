"""eval metrics / gates / report smoke (dummy GT·result·debug)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.gates import GATES, check_gate, flatten_gate_thresholds  # noqa: E402
from eval.metrics import compute_all_metrics  # noqa: E402


def _dummy_gt() -> dict:
    return {
        "plot_box": [0, 0, 100, 100],
        "pixel_curve_path": [[10, 50], [11, 49], [12, 48]],
        "x_values": [0.0, 1.0, 2.0, 3.0],
        "y_values": [100.0, 200.0, 150.0, 100.0],
        "peaks": [{"two_theta": 1.0}, {"two_theta": 2.5}],
        "axis_metadata": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 300},
    }


def _dummy_result() -> dict:
    return {
        "two_theta_values": [0.0, 1.0, 2.0, 3.0],
        "intensities": [105.0, 195.0, 155.0, 98.0],
    }


def _dummy_debug() -> dict:
    return {
        "plot_box": [0, 0, 100, 100],
        "calibration": {
            "y_scale": 1.0,
            "y_offset": 0.0,
            "x_scale": 1.0,
            "x_offset": 0.0,
            "roundtrip_error": {"total_mean_error_px": 0.1},
            "peak_positions_2theta": [
                {"two_theta": 1.05, "is_major": True, "intensity": 200.0},
                {"two_theta": 2.4, "is_major": False, "intensity": 150.0},
            ],
        },
        "postprocess": {"gap_fill": {"gap_ranges": []}, "peak_list": []},
        "trace": {"valid_ratio": 1.0, "trace_score": 1.0, "diagnostics": {}, "blockwise": []},
        "candidate_stats": {
            "total_columns": 100,
            "raw_nonempty_columns": 95,
            "missing_column_ratio": 0.05,
        },
        "recovery": {"zones": []},
    }


class TestEvalMetricsGates(unittest.TestCase):
    def test_numeric_y_mae_norm_small_error(self) -> None:
        m = compute_all_metrics(_dummy_result(), _dummy_debug(), _dummy_gt())
        n = m["main"]["numeric_y_mae_norm"]
        self.assertLess(n, 0.2)
        self.assertNotEqual(n, 999.0)

    def test_major_peak_x_error_2theta_degrees(self) -> None:
        m = compute_all_metrics(_dummy_result(), _dummy_debug(), _dummy_gt())
        e = m["main"]["major_peak_x_error_2theta"]
        self.assertLess(e, 0.2)
        self.assertNotEqual(e, 999.0)

    def test_major_peak_y_error_in_debug_only(self) -> None:
        m = compute_all_metrics(_dummy_result(), _dummy_debug(), _dummy_gt())
        self.assertIn("major_peak_y_error", m["debug"])
        self.assertNotIn("major_peak_y_error", m["main"])

    def test_gate_result_includes_level(self) -> None:
        main = compute_all_metrics(_dummy_result(), _dummy_debug(), _dummy_gt())["main"]
        g = check_gate(main, "clean", "development")
        self.assertEqual(g["gate_level"], "development")
        self.assertIn("details", g)

    def test_flatten_strict(self) -> None:
        d = flatten_gate_thresholds("clean", "strict")
        self.assertIn("curve_y_mae_px", d)
        self.assertIn("numeric_y_mae_norm", d)
        self.assertNotIn("major_peak_y_error", d)

    def test_gates_nested_structure(self) -> None:
        self.assertEqual(set(GATES.keys()), {"mvp", "development", "strict"})
        self.assertIn("clean", GATES["mvp"])


if __name__ == "__main__":
    unittest.main()
