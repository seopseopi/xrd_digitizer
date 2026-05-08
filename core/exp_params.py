from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def default_exp_params() -> Dict[str, Any]:
    return {
        "preprocess": {
            "mask_a_thr": 0.18,
            "mask_b_thr": 0.22,
            "support_edge_thr": 0.20,
            "support_ridge_thr": 0.16,
        },
        "candidates": {
            "ridge_fallback_thr": 0.18,
            "conf_min": 0.12,
            "comp_support_min": 0.10,
            "grid_penalty_high": 0.55,
            "grid_ridge_low": 0.15,
            "merge_dist_scale": 0.004,
        },
        "recovery": {
            "gain_min": 0.15,
            "coverage_tol": 0.03,
            "boundary_mul": 2.0,
        },
        "postprocess": {
            "smooth_k_thr": 1.5,
            "smooth_p_thr": 0.40,
            "nms_x_scale": 0.0045,
            "raw_prom_ratio": 0.85,
            "peak_prominence_scale": 1.0,
            "peak_noise_scale": 1.0,
        },
        "ml_rescue": {
            "enabled": False,
            "prob_thr": 0.35,
            "coverage_min": 0.60,
            "gain_min": 0.10,
        },
    }


def load_exp_params(path: str | None) -> Dict[str, Any]:
    p = default_exp_params()
    if not path:
        return p
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in j.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    return p
