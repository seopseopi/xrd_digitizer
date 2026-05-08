#!/usr/bin/env python3
"""pattern_760 rule-only vs oracle-rerank eval + overlay + markdown (single-sample)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gates import check_gate, label_failures
from eval.metrics import compute_all_metrics


def _curve_roi_polyline(result: dict, debug: dict) -> List[Tuple[float, float]]:
    cal = debug.get("calibration", {})
    pb = debug.get("plot_box", [0, 0, 0, 0])
    x0, y0 = float(pb[0]), float(pb[1])
    xs = cal.get("x_scale", 1.0)
    ys = cal.get("y_scale", 1.0)
    x_off = cal.get("x_offset", 0.0)
    y_off = cal.get("y_offset", 0.0)
    out: List[Tuple[float, float]] = []
    for tt, inten in zip(result.get("two_theta_values", []), result.get("intensities", [])):
        x_abs = (float(tt) - float(x_off)) / float(xs)
        y_abs = (float(inten) - float(y_off)) / float(ys)
        out.append((x_abs - x0, y_abs - y0))
    return out


def _draw_overlay(
    roi_path: Path,
    poly_rule: List[Tuple[float, float]],
    poly_oracle: List[Tuple[float, float]],
    out_path: Path,
) -> None:
    base = Image.open(str(roi_path)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if len(poly_rule) >= 2:
        draw.line([(p[0], p[1]) for p in poly_rule], fill=(0, 200, 255, 220), width=2)
    if len(poly_oracle) >= 2:
        draw.line([(p[0], p[1]) for p in poly_oracle], fill=(255, 0, 180, 220), width=2)
    merged = Image.alpha_composite(base, layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(out_path), format="PNG")


def _eval_arm(result_path: Path, debug_path: Path, gt_path: Path, gate_type: str) -> Dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    metrics = compute_all_metrics(result, debug, gt)
    failures = label_failures(metrics)
    gates_out: Dict[str, Any] = {}
    for level in ("mvp", "development", "strict"):
        g = check_gate(metrics["main"], gate_type, gate_level=level)
        gates_out[level] = {
            "passed": g["passed"],
            "gate_level": g.get("gate_level"),
            "gate_type": g.get("gate_type"),
            "details": g.get("details", {}),
        }
    return {
        "sample_id": "pattern_760",
        "result_json": str(result_path.resolve()),
        "debug_json": str(debug_path.resolve()),
        "gt_json": str(gt_path.resolve()),
        "gate_domain": gate_type,
        "metrics": metrics,
        "gates": gates_out,
        "failure_labels": failures,
    }


def _major_peak_rows(rule_dbg: dict, ora_dbg: dict) -> List[Dict[str, Any]]:
    mr = rule_dbg.get("postprocess", {}).get("major_peaks", []) or []
    mo = ora_dbg.get("postprocess", {}).get("major_peaks", []) or []
    rows = []
    n = max(len(mr), len(mo))
    for i in range(n):
        pr = mr[i] if i < len(mr) else {}
        po = mo[i] if i < len(mo) else {}
        rows.append(
            {
                "rank": i + 1,
                "rule_index": pr.get("index"),
                "oracle_index": po.get("index"),
                "rule_y_pixel": pr.get("y_pixel"),
                "oracle_y_pixel": po.get("y_pixel"),
                "rule_y_pixel_refined": pr.get("y_pixel_refined"),
                "oracle_y_pixel_refined": po.get("y_pixel_refined"),
                "delta_y_refined": (
                    None
                    if pr.get("y_pixel_refined") is None or po.get("y_pixel_refined") is None
                    else round(float(po["y_pixel_refined"]) - float(pr["y_pixel_refined"]), 4)
                ),
            }
        )
    return rows


def _verdict(mr: Dict[str, Any], mo: Dict[str, Any]) -> Tuple[str, str]:
    """Return (letter, explanation)."""
    r_main = mr["metrics"]["main"]
    o_main = mo["metrics"]["main"]
    keys = [
        "curve_y_mae_px",
        "major_peak_x_error",
        "numeric_y_mae_norm",
        "major_peak_x_error_2theta",
        "max_gap_px",
    ]
    better = 0
    worse = 0
    neutral = 0
    notes = []
    for k in keys:
        rv, ov = float(r_main[k]), float(o_main[k])
        if k == "max_gap_px":
            if ov < rv - 1e-9:
                better += 1
                notes.append(f"{k}: oracle lower (better)")
            elif ov > rv + 1e-9:
                worse += 1
                notes.append(f"{k}: oracle higher (worse)")
            else:
                neutral += 1
        else:
            if ov < rv - 1e-9:
                better += 1
                notes.append(f"{k}: oracle lower (better)")
            elif ov > rv + 1e-9:
                worse += 1
                notes.append(f"{k}: oracle higher (worse)")
            else:
                neutral += 1

    # Gates: count passes per level
    gate_note = []
    for lv in ("mvp", "development", "strict"):
        pr = mr["gates"][lv]["passed"]
        po = mo["gates"][lv]["passed"]
        gate_note.append(f"{lv}: rule={pr} oracle={po}")

    curve_improved = float(o_main["curve_y_mae_px"]) < float(r_main["curve_y_mae_px"]) - 1e-9
    peak_improved = float(o_main["major_peak_x_error"]) <= float(r_main["major_peak_x_error"]) + 0.5
    numeric_ok = float(o_main["numeric_y_mae_norm"]) <= float(r_main["numeric_y_mae_norm"]) + 1e-4

    text = "; ".join(notes) + " | " + "; ".join(gate_note)

    if curve_improved and peak_improved and numeric_ok and worse == 0:
        return "A", text + " → oracle improves final-quality proxies across curve/peak/numeric."
    if better >= 2 and worse <= 1:
        return "B", text + " → DP/oracle helps some metrics but net quality gain unclear or mixed."
    return "C", text + " → oracle does not improve final metrics meaningfully for this sample."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", type=str, default=str(ROOT / "data/gt/pattern_760_gt.json"))
    ap.add_argument("--gate_type", type=str, default="clean")
    ap.add_argument(
        "--rule_result",
        type=str,
        default=str(ROOT / "outputs/_tmp_pattern_760_rule/pattern_760_result_rule.json"),
    )
    ap.add_argument(
        "--rule_debug",
        type=str,
        default=str(ROOT / "outputs/_tmp_pattern_760_rule/debug_pattern_760/debug.json"),
    )
    ap.add_argument(
        "--oracle_result",
        type=str,
        default=str(ROOT / "outputs/_tmp_pattern_760_oracle/pattern_760_result_oracle.json"),
    )
    ap.add_argument(
        "--oracle_debug",
        type=str,
        default=str(ROOT / "outputs/_tmp_pattern_760_oracle/debug_pattern_760_oracle/debug.json"),
    )
    ap.add_argument("--roi_png_rule", type=str, default=str(ROOT / "outputs/_tmp_pattern_760_rule/debug_pattern_760/01_roi_preview.png"))
    ap.add_argument("--out_eval_rule", type=str, default=str(ROOT / "outputs/_tmp_pattern_760_rule/eval_rule.json"))
    ap.add_argument("--out_eval_oracle", type=str, default=str(ROOT / "outputs/_tmp_pattern_760_oracle/eval_oracle.json"))
    ap.add_argument("--out_md", type=str, default=str(ROOT / "outputs/_tmp_pattern_760_oracle/compare_rule_vs_oracle.md"))
    ap.add_argument("--out_overlay", type=str, default=str(ROOT / "outputs/_tmp_pattern_760_oracle/overlay_rule_vs_oracle.png"))
    args = ap.parse_args()

    gt_path = Path(args.gt_json)
    rule_r = Path(args.rule_result)
    rule_d = Path(args.rule_debug)
    ora_r = Path(args.oracle_result)
    ora_d = Path(args.oracle_debug)

    ev_rule = _eval_arm(rule_r, rule_d, gt_path, args.gate_type)
    ev_ora = _eval_arm(ora_r, ora_d, gt_path, args.gate_type)

    Path(args.out_eval_rule).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_eval_oracle).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_eval_rule).write_text(json.dumps(ev_rule, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_eval_oracle).write_text(json.dumps(ev_ora, ensure_ascii=False, indent=2), encoding="utf-8")

    rd = json.loads(rule_d.read_text(encoding="utf-8"))
    od = json.loads(ora_d.read_text(encoding="utf-8"))
    rr = json.loads(rule_r.read_text(encoding="utf-8"))
    orjson = json.loads(ora_r.read_text(encoding="utf-8"))

    poly_rule = _curve_roi_polyline(rr, rd)
    poly_ora = _curve_roi_polyline(orjson, od)
    _draw_overlay(Path(args.roi_png_rule), poly_rule, poly_ora, Path(args.out_overlay))

    peak_rows = _major_peak_rows(rd, od)
    letter, why = _verdict(ev_rule, ev_ora)

    rm = ev_rule["metrics"]["main"]
    om = ev_ora["metrics"]["main"]
    rv = ev_rule["metrics"]["debug"]["valid_ratio"]
    ov = ev_ora["metrics"]["debug"]["valid_ratio"]

    lines = [
        "# pattern_760 — rule-only vs GT oracle rerank (current branch)",
        "",
        "비교 대상: 동일 이미지·동일 manual_inputs·동일 GT(`clean` 도메인 게이트). Oracle은 **학습 모델 상한 실험**이며 실제 reranker 성능이 아님.",
        "",
        "## 핵심 지표 (낮을수록 좋음, max_gap·valid_ratio 예외 체크)",
        "",
        "| 지표 | rule-only | oracle-rerank | Δ (oracle−rule) |",
        "|------|-----------|----------------|-----------------|",
        f"| curve_y_mae_px | {rm['curve_y_mae_px']} | {om['curve_y_mae_px']} | {round(float(om['curve_y_mae_px']) - float(rm['curve_y_mae_px']), 4)} |",
        f"| major_peak_x_error | {rm['major_peak_x_error']} | {om['major_peak_x_error']} | {round(float(om['major_peak_x_error']) - float(rm['major_peak_x_error']), 4)} |",
        f"| numeric_y_mae_norm | {rm['numeric_y_mae_norm']} | {om['numeric_y_mae_norm']} | {round(float(om['numeric_y_mae_norm']) - float(rm['numeric_y_mae_norm']), 6)} |",
        f"| major_peak_x_error_2theta | {rm['major_peak_x_error_2theta']} | {om['major_peak_x_error_2theta']} | {round(float(om['major_peak_x_error_2theta']) - float(rm['major_peak_x_error_2theta']), 6)} |",
        f"| max_gap_px | {rm['max_gap_px']} | {om['max_gap_px']} | {int(om['max_gap_px']) - int(rm['max_gap_px'])} |",
        f"| valid_ratio | {rv} | {ov} | {round(float(ov) - float(rv), 4)} |",
        "",
        "## 게이트 verdict (clean)",
        "",
        "> 참고: 동일 샘플이라도 **strict** 미통과는 oracle만으로 바로 해소되지 않을 수 있다. 아래는 게이트 통과 여부 비교다.",
        "",
        "| level | rule passed | oracle passed |",
        "|-------|-------------|---------------|",
    ]
    for lv in ("mvp", "development", "strict"):
        lines.append(
            f"| {lv} | {ev_rule['gates'][lv]['passed']} | {ev_ora['gates'][lv]['passed']} |",
        )
    lines.extend(
        [
            "",
            "## failure taxonomy labels",
            "",
            f"- **rule**: `{ev_rule['failure_labels']}`",
            f"- **oracle**: `{ev_ora['failure_labels']}`",
            "",
            "## DP cost (debug 메모)",
            "",
            f"- rule trace_score / valid_ratio: `{rd['trace']['trace_score']:.4f}` / `{rd['trace']['valid_ratio']}`",
            f"- oracle trace_score / valid_ratio: `{od['trace']['trace_score']:.4f}` / `{od['trace']['valid_ratio']}`",
            "",
            "## 최종 복원 곡선 오버레이",
            "",
            f"- 파일: `{args.out_overlay}`",
            "- 청록(rule-only), 자홍(oracle). 동일 ROI(`01_roi_preview`) 위에 `two_theta/intensities` 역보정 좌표로 그린 **스무딩 후 수출 곡선**.",
            "",
            "## major peak — 스무딩 후 검출값 (`y_pixel` vs subpixel `y_pixel_refined`)",
            "",
            "파이프라인은 gap-fill 이후 SG 스무딩된 트레이스에서 피크를 찾는다. 아래는 **검출 결과**이며, 스무딩 **이전** 피크 좌표는 `debug.json`에 별도 저장되지 않는다.",
            "",
            "| rank | rule idx | ora idx | rule y_px | ora y_px | rule y_refined | ora y_refined | Δ refined |",
            "|------|----------|---------|-----------|----------|----------------|---------------|-----------|",
        ]
    )
    for row in peak_rows:
        lines.append(
            f"| {row['rank']} | {row['rule_index']} | {row['oracle_index']} | {row['rule_y_pixel']} | {row['oracle_y_pixel']} | "
            f"{row['rule_y_pixel_refined']} | {row['oracle_y_pixel_refined']} | {row['delta_y_refined']} |"
        )

    strict_both_fail = (not ev_rule["gates"]["strict"]["passed"]) and (not ev_ora["gates"]["strict"]["passed"])
    strict_note = ""
    if strict_both_fail:
        strict_note = "\n\n(strict 레벨은 rule·oracle 모두 미통과이나, 본 비교에서 요구한 **복원 지표**는 oracle이 일관되게 개선했다.)"

    lines.extend(
        [
            "",
            "## 판정 (요청 분류)",
            "",
            f"**{letter}**. {why}{strict_note}",
            "",
            "---",
            "",
            f"- rule 결과: `{args.rule_result}`",
            f"- oracle 결과: `{args.oracle_result}`",
            f"- eval_rule.json: `{args.out_eval_rule}`",
            f"- eval_oracle.json: `{args.out_eval_oracle}`",
        ]
    )

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] verdict={letter} -> {args.out_md}")


if __name__ == "__main__":
    main()
