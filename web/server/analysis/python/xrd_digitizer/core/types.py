"""
§9.5: 엔진 입출력 타입 정의.

ManualInputs  – 사용자가 제공하는 수동 입력 (§9.3, §10)
RunResult     – 엔진 최종 출력 (§9.4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ManualInputs:
    """§9.3 필수 + 선택 입력."""

    plot_box: list
    x_axis_points: list
    x_axis_values: list
    y_axis_points: list
    y_axis_values: list
    color_sample_point: list
    legend_ignore_boxes: Optional[list] = None
    perspective_corners: Optional[list] = None
    color_resample_points: Optional[list] = None
    # None이면 기존과 동일(열당 1점). 양수면 x_range 구간을 균일 분할해 선형 보간(해상도 단계 1).
    export_resample_points: Optional[int] = None

    @property
    def click_count(self) -> int:
        """§10.4 클릭 예산 계산."""
        n = 6  # plot_box 1 + x축 2 + y축 2 + color 1
        if self.legend_ignore_boxes:
            n += len(self.legend_ignore_boxes)
        if self.perspective_corners:
            n += 4
        if self.color_resample_points:
            n += len(self.color_resample_points)
        return n

    @property
    def click_budget_status(self) -> str:
        """§10.4: 정상 6 / 정상최대 7 / 하드 11 / 하드최대 12 / >12 UX fail."""
        c = self.click_count
        if c <= 6:
            return "normal"
        if c <= 7:
            return "normal_max"
        if c <= 11:
            return "hard"
        if c <= 12:
            return "hard_max"
        return "ux_fail"


@dataclass
class RunResult:
    """§9.4 최종 출력 JSON 필드."""

    two_theta_values: list = field(default_factory=list)
    intensities: list = field(default_factory=list)
    x_range: list = field(default_factory=list)
    y_range: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    confidence: float = 0.0
    warnings: list = field(default_factory=list)
    used_manual_inputs: dict = field(default_factory=dict)
    # 분리 로드맵 1단계: 수출 곡선만으로 재검출한 피크(이미지 경로 피크와 별개)
    peaks_numeric_curve: list = field(default_factory=list)
    # 런타임 모델 보조(DP 직전 rerank)·fallback·피크 ROI 보정 요약 (없으면 JSON 에 미포함)
    model_assist: Optional[Dict[str, Any]] = None
    # 해상도/업스케일 진단용 (평가 지표와 분리; 없으면 미포함)
    resolution_diagnostics: Optional[Dict[str, Any]] = None
    # 평가/고해상도 export를 명시적으로 분리한 정식 출력 필드.
    export_points_eval: Optional[Dict[str, Any]] = None
    export_points_highres: Optional[Dict[str, Any]] = None
    export_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        out: Dict[str, Any] = {
            "two_theta_values": self.two_theta_values,
            "intensities": self.intensities,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "quality": self.quality,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "used_manual_inputs": self.used_manual_inputs,
            "peaks_numeric_curve": self.peaks_numeric_curve,
        }
        if self.model_assist is not None:
            out["model_assist"] = self.model_assist
        if self.resolution_diagnostics is not None:
            out["resolution_diagnostics"] = self.resolution_diagnostics
        if self.export_points_eval is not None:
            out["export_points_eval"] = self.export_points_eval
        if self.export_points_highres is not None:
            out["export_points_highres"] = self.export_points_highres
        if self.export_metadata is not None:
            out["export_metadata"] = self.export_metadata
        return out


PIPELINE_STAGES = [
    "preprocess",
    "color_segment",
    "mask_combine",
    "skeletonize",
    "trace_extract",
    "axis_map",
    "peak_detect",
    "postprocess",
]

DEBUG_OUTPUTS = [
    "overlay",
    "color_mask",
    "combined_mask",
    "skeleton",
    "candidate_map",
    "trace_path",
    "peaks_overlay",
    "debug.json",
]
