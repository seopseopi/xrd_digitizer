from __future__ import annotations

import numpy as np

from trace.candidates import (
    _bridge_confidence_batch,
    _candidate_confidence,
    _candidate_confidence_batch,
)


def test_candidate_confidence_batch_matches_scalar() -> None:
    rng = np.random.default_rng(42)
    color_dists = rng.uniform(0.0, 200.0, 1000)
    y_currs = rng.uniform(0.0, 2000.0, 1000)
    comp_scores = rng.normal(2.0, 5.0, 1000)
    axis_dists = rng.uniform(0.0, 100.0, 1000)

    for y_prev in (None, 123.4):
        for ridge_resps in (None, rng.uniform(-1.0, 2.0, 1000)):
            actual = _candidate_confidence_batch(
                color_dists,
                y_currs,
                y_prev,
                comp_scores,
                axis_dists,
                ridge_resps=ridge_resps,
            )
            expected = np.asarray(
                [
                    _candidate_confidence(
                        float(color_dists[i]),
                        float(y_currs[i]),
                        y_prev,
                        float(comp_scores[i]),
                        float(axis_dists[i]),
                        ridge_resp=None if ridge_resps is None else float(ridge_resps[i]),
                    )
                    for i in range(color_dists.shape[0])
                ],
                dtype=np.float64,
            )
            assert np.max(np.abs(actual - expected)) <= 1e-12


def test_bridge_confidence_batch_matches_formula() -> None:
    rng = np.random.default_rng(43)
    color_dists = rng.uniform(0.0, 200.0, 1000)
    comp_scores = rng.normal(2.0, 5.0, 1000)

    actual = _bridge_confidence_batch(color_dists, comp_scores)
    expected = np.clip(
        0.26
        + 0.42
        * np.exp(-color_dists / 20.0)
        * (1.0 / (1.0 + np.exp(-0.8 * (comp_scores - 2.0)))),
        0.22,
        0.52,
    )
    assert np.max(np.abs(actual - expected)) <= 1e-12
