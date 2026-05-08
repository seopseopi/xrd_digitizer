"""GT JSON 호환: peak_x_values 등 → 평가용 peaks 리스트 보강."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def normalize_gt_for_eval(gt: Dict[str, Any]) -> Dict[str, Any]:
    """peaks 키가 없으면 peak_x_values + axis_metadata로 peaks 생성."""
    if gt.get("peaks"):
        return gt
    xs = gt.get("peak_x_values") or []
    if not xs:
        return gt
    out = deepcopy(gt)
    out["peaks"] = [{"two_theta": float(x)} for x in xs]
    return out
