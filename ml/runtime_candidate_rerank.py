"""
DP 직전 런타임 후보 재랭크 + 자동 fallback.

학습 데이터 빌더(`ml/data/build_candidate_rerank_dataset.py`)와 동일한 3채널 패치를 사용한다.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.model_assist_settings import ModelAssistSettings
from trace.candidates import dp_transition_window_width
from trace.dp_trace import dp_trace, refine_dp_path_column_apex_pull


def _roi_to_luma01(roi_rgb: np.ndarray) -> np.ndarray:
    if roi_rgb.ndim == 2:
        return roi_rgb.astype(np.float32) / 255.0
    rgb = roi_rgb[..., :3].astype(np.float32)
    return (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]) / 255.0


def _axis_proximity(roi_h: int, roi_w: int, x: int, y: int, radius: float = 18.0) -> float:
    d = float(min(x, roi_w - 1 - x, y, roi_h - 1 - y))
    if d >= radius:
        return 0.0
    return 1.0 - d / radius


def _extract_patch(roi_gray: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    h, w = roi_gray.shape
    r = patch_size // 2
    x0, x1 = x - r, x + r + 1
    y0, y1 = y - r, y + r + 1
    out = np.zeros((patch_size, patch_size), dtype=np.float32)
    sx0, sx1 = max(0, x0), min(w, x1)
    sy0, sy1 = max(0, y0), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = roi_gray[sy0:sy1, sx0:sx1]
    return out


def _valid_ratio(path: List[Optional[int]]) -> float:
    if not path:
        return 0.0
    return float(sum(1 for p in path if p is not None)) / float(len(path))


def _apply_apex_pull(
    path: List[Optional[int]],
    final_cands: Dict[int, List[dict]],
    roi_w: int,
    enabled: bool,
) -> List[Optional[int]]:
    if not enabled:
        return list(path)
    max_pull = max(120, 4 * int(dp_transition_window_width(roi_w)))
    return refine_dp_path_column_apex_pull(
        path,
        final_cands,
        conf_slack=0.22,
        max_upward_pull_px=max_pull,
    )


def _score_candidates_with_model(
    final_cands: Dict[int, List[dict]],
    roi_rgb: np.ndarray,
    settings: ModelAssistSettings,
) -> Tuple[bool, str, int]:
    """후보 dict에 rule_confidence·model_score_delta·confidence(결합) 기록. 성공 여부와 메시지 반환."""
    try:
        import torch
        from ml.models.candidate_reranker_cnn import SmallCandidateCNN
    except Exception as exc:
        return False, f"torch_or_model_import_failed:{type(exc).__name__}", 0

    ckpt_path = settings.model_ckpt_path
    if not ckpt_path:
        return False, "missing_checkpoint_path", 0

    try:
        ckpt = torch.load(ckpt_path, map_location=settings.device)
    except Exception as exc:
        return False, f"checkpoint_load_failed:{type(exc).__name__}", 0

    roi_gray = _roi_to_luma01(roi_rgb)
    roi_h, roi_w = roi_gray.shape
    ps = int(settings.patch_size)
    pairs: List[Tuple[int, dict]] = []
    batch_tensors: List[np.ndarray] = []

    for col in sorted(final_cands.keys()):
        for cand in final_cands.get(col, []) or []:
            y = int(cand.get("y", -1))
            if y < 0 or y >= roi_h:
                continue
            col_i = int(col)
            if col_i < 0 or col_i >= roi_w:
                continue
            patch = _extract_patch(roi_gray, col_i, y, ps)
            cmap = np.zeros_like(patch, dtype=np.float32)
            cmap[ps // 2, ps // 2] = 1.0
            ax = np.full_like(patch, _axis_proximity(roi_h, roi_w, col_i, y), dtype=np.float32)
            ch = np.stack([patch, cmap, ax], axis=0)
            pairs.append((col_i, cand))
            batch_tensors.append(ch)

    n = len(batch_tensors)
    if n == 0:
        return False, "no_candidates_to_score", 0

    in_ch = int(ckpt.get("in_channels", 3))
    model = SmallCandidateCNN(in_channels=in_ch).to(settings.device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as exc:
        return False, f"state_dict_load_failed:{type(exc).__name__}", 0
    model.eval()

    rule_vals: List[float] = []
    with torch.no_grad():
        xb = np.stack(batch_tensors, axis=0).astype(np.float32)
        xt = torch.from_numpy(xb).to(settings.device)
        scores = np.atleast_1d(model(xt).squeeze(-1).detach().cpu().numpy())

    lam = float(settings.lambda_model)
    for (_col, cand), sc in zip(pairs, scores):
        rule = float(cand.get("confidence", 0.0))
        cand["rule_confidence"] = rule
        delta = float(sc)
        cand["model_score_delta"] = delta
        cand["confidence"] = float(np.clip(rule + lam * delta, 0.0, 1.0))

    # 열별로 높은 confidence 우선 정렬 (apex pull·디버그 일관성)
    for col in final_cands.keys():
        final_cands[col].sort(key=lambda c: -float(c.get("confidence", 0.0)))

    return True, "ok", n


def run_dp_with_optional_model_assist(
    final_cands_orig: Dict[int, List[dict]],
    roi_rgb: np.ndarray,
    roi_w: int,
    roi_h: int,
    comp_score_map: np.ndarray,
    settings: ModelAssistSettings,
    *,
    use_dp_column_apex_pull: bool,
) -> Tuple[Dict[int, List[dict]], dict, dict]:
    """
    Returns:
      final_cands_active — 이후 recovery 등에 사용할 후보 맵
      trace_result — dp_trace 형식 + path 에 apex pull 반영
      model_assist_meta — 결과 JSON `model_assist` 에 넣을 요약
    """
    base_meta: Dict[str, Any] = {
        "enabled": bool(settings.enabled and settings.model_ckpt_path),
        "lambda_model": float(settings.lambda_model),
        "patch_size": int(settings.patch_size),
        "device": str(settings.device),
        "checkpoint": settings.model_ckpt_path,
        "dp_branch": "rule_only",
        "fallback_reason": None,
        "num_candidates_scored": 0,
        "trace_score_rule": None,
        "trace_score_model": None,
        "valid_ratio_rule": None,
        "valid_ratio_model": None,
        "fallback_valid_ratio_margin": float(settings.fallback_valid_ratio_margin),
        "fallback_trace_score_margin": float(settings.fallback_trace_score_margin),
    }

    fc_rule = copy.deepcopy(final_cands_orig)
    tr_rule = dp_trace(fc_rule, roi_w, roi_h, comp_score_map)
    path_rule = _apply_apex_pull(tr_rule["path"], fc_rule, roi_w, use_dp_column_apex_pull)
    tr_rule = {**tr_rule, "path": path_rule}
    base_meta["trace_score_rule"] = float(tr_rule["trace_score"])
    base_meta["valid_ratio_rule"] = float(_valid_ratio(path_rule))

    if not settings.enabled or not settings.model_ckpt_path:
        base_meta["fallback_reason"] = "disabled_or_no_checkpoint"
        return fc_rule, tr_rule, base_meta

    fc_model = copy.deepcopy(final_cands_orig)
    ok, reason, n_scored = _score_candidates_with_model(fc_model, roi_rgb, settings)
    base_meta["num_candidates_scored"] = int(n_scored)

    if not ok:
        base_meta["fallback_reason"] = reason
        return fc_rule, tr_rule, base_meta

    tr_model = dp_trace(fc_model, roi_w, roi_h, comp_score_map)
    path_model = _apply_apex_pull(tr_model["path"], fc_model, roi_w, use_dp_column_apex_pull)
    tr_model = {**tr_model, "path": path_model}

    ts_r = float(tr_rule["trace_score"])
    ts_m = float(tr_model["trace_score"])
    vr_r = float(_valid_ratio(path_rule))
    vr_m = float(_valid_ratio(path_model))

    base_meta["trace_score_model"] = ts_m
    base_meta["valid_ratio_model"] = vr_m

    margin_v = float(settings.fallback_valid_ratio_margin)
    margin_t = float(settings.fallback_trace_score_margin)

    use_model = (vr_m >= vr_r - margin_v) and (ts_m <= ts_r + margin_t)
    if use_model:
        base_meta["dp_branch"] = "model_assist"
        base_meta["fallback_reason"] = None
        return fc_model, tr_model, base_meta

    base_meta["dp_branch"] = "rule_only"
    base_meta["fallback_reason"] = "model_dp_worse_or_invalid_vs_rule"
    return fc_rule, tr_rule, base_meta
