"""
§13 파라미터 조정 순서 기반 자동 튜닝.

계획서 순서(고정):
  1) evaluator thresholds / band 폭에 해당하는 후처리·스무딩 관련
  2) preprocess (color / ridge / grid)
  3) candidate (conf / component_support)
  4) block difficulty — trace 내부 B는 코드 상수; 여기서는 후보 cap에 영향 주는 conf_min 등으로 근사
  5) dual-pass disagreement — recovery 경계 파라미터
  6) rescue gain / boundary
  7) smoothing gate
  8) peak NMS / prominence
  9) ml_rescue gate (활성화 플래그만)

각 단계는 이전 단계에서 확정된 best_params를 상속한 채 그 단계의 키만 탐색한다.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from core.exp_params import default_exp_params

# (stage_name, list of (dot_path, candidate_values))
V13_SEARCH_PLAN: List[Tuple[str, List[Tuple[str, List[float]]]]] = [
    (
        "1_evaluator_band_proxy",
        [
            ("postprocess.smooth_k_thr", [1.2, 1.5, 1.8]),
            ("postprocess.smooth_p_thr", [0.35, 0.40, 0.45]),
        ],
    ),
    (
        "2_preprocess",
        [
            ("preprocess.mask_a_thr", [0.14, 0.16, 0.18, 0.20]),
            ("preprocess.mask_b_thr", [0.20, 0.22, 0.24]),
            ("preprocess.support_ridge_thr", [0.14, 0.16, 0.18]),
        ],
    ),
    (
        "3_candidates",
        [
            ("candidates.conf_min", [0.08, 0.10, 0.12]),
            ("candidates.comp_support_min", [0.06, 0.08, 0.10]),
            ("candidates.ridge_fallback_thr", [0.16, 0.18, 0.20]),
        ],
    ),
    (
        "4_block_difficulty_proxy",
        [
            ("candidates.merge_dist_scale", [0.003, 0.004, 0.005]),
        ],
    ),
    (
        "5_dual_pass_proxy",
        [
            ("recovery.boundary_mul", [1.5, 2.0, 2.5]),
        ],
    ),
    (
        "6_rescue",
        [
            ("recovery.gain_min", [0.10, 0.12, 0.15]),
            ("recovery.coverage_tol", [0.04, 0.03, 0.02]),
        ],
    ),
    (
        "7_smoothing",
        [
            ("postprocess.smooth_k_thr", [1.1, 1.3, 1.5]),
        ],
    ),
    (
        "8_peak",
        [
            ("postprocess.nms_x_scale", [0.0035, 0.0045, 0.0055]),
            ("postprocess.peak_prominence_scale", [0.75, 0.85, 0.95]),
            ("postprocess.peak_noise_scale", [0.9, 1.0, 1.1]),
        ],
    ),
    (
        "9_ml_rescue",
        [
            ("ml_rescue.enabled", [False]),
        ],
    ),
]


def _set_path(params: Dict[str, Any], dot_path: str, value: Any) -> None:
    parts = dot_path.split(".")
    cur: Any = params
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _get_path(params: Dict[str, Any], dot_path: str) -> Any:
    parts = dot_path.split(".")
    cur: Any = params
    for p in parts:
        cur = cur[p]
    return cur


def run_sequential_v13(
    score_fn: Callable[[Dict[str, Any]], float],
    initial: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    score_fn(best_params) -> 낮을수록 좋음.
    반환: (best_params, log_entries)
    """
    best = deepcopy(initial or default_exp_params())
    log: List[Dict[str, Any]] = []

    for stage_name, keys_vals in V13_SEARCH_PLAN:
        for dot_path, candidates in keys_vals:
            best_score = score_fn(best)
            best_val = _get_path(best, dot_path)
            for v in candidates:
                trial = deepcopy(best)
                _set_path(trial, dot_path, v)
                sc = score_fn(trial)
                log.append({"stage": stage_name, "key": dot_path, "value": v, "score": sc})
                if sc < best_score:
                    best_score = sc
                    best_val = v
            best = deepcopy(best)
            _set_path(best, dot_path, best_val)

    return best, log


def save_params(path: Path, params: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
