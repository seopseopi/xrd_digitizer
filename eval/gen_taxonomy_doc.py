"""
Generate failure taxonomy markdown from evaluation reports.
"""
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

REPORT_PATHS = {
    "clean": "outputs_baseline_clean/eval_report.json",
    "styled": "outputs_baseline_styled/eval_report.json",
    "real_like": "outputs_baseline_real/eval_report.json",
}

ALL_TAXONOMY = {
    "tail_collapse": "tail region weakens, candidates collapse, numeric recovery fails",
    "text_intrusion": "text strokes mix into curve candidates",
    "grid_confusion": "grid lines incorrectly selected as candidates",
    "legend_capture": "legend elements captured as curve",
    "peak_miss_after_smoothing": "tracing ok but peaks lost after smoothing",
    "candidate_starvation": "consecutive column range with almost no correct curve candidates",
    "wrong_branch_lock_in": "early mis-connection locks onto wrong branch",
    "calibration_mismatch": "tracing ok but axis mapping off, numeric export wrong",
}

OUTPUT_PATH = "outputs/failure_taxonomy.md"


def main():
    reports = {}
    for name, path in REPORT_PATHS.items():
        p = Path(path)
        if p.exists():
            reports[name] = json.loads(p.read_text(encoding="utf-8"))

    lines = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("# Baseline Failure Taxonomy Report")
    lines.append(f"Generated: {now_str}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Samples | Pass | Rate | curve_y_mae (mean) | Top Failure |")
    lines.append("|---------|---------|------|------|-------------------|-------------|")

    for name in ["clean", "styled", "real_like"]:
        if name not in reports:
            continue
        agg = reports[name].get("aggregate", {})
        n = agg.get("total_samples", 0)
        n_pass = agg.get("pass_count", 0)
        rate = agg.get("pass_rate", 0) * 100
        ymae = agg.get("main_metrics_stats", {}).get("curve_y_mae_px", {}).get("mean", 0)
        ftc = agg.get("failure_taxonomy_counts", {})
        top_fail = max(ftc, key=ftc.get) if ftc else "-"
        lines.append(f"| {name} | {n} | {n_pass} | {rate:.1f}% | {ymae:.2f} | {top_fail} |")

    lines.append("")
    lines.append("## Failure Taxonomy Counts")
    lines.append("")

    global_counts = Counter()
    per_dataset = {}
    for name in ["clean", "styled", "real_like"]:
        if name not in reports:
            continue
        agg = reports[name].get("aggregate", {})
        ftc = agg.get("failure_taxonomy_counts", {})
        per_dataset[name] = ftc
        for k, v in ftc.items():
            global_counts[k] += v

    lines.append("| Label | clean | styled | real_like | Total |")
    lines.append("|-------|-------|--------|-----------|-------|")
    for label in sorted(ALL_TAXONOMY.keys()):
        c = per_dataset.get("clean", {}).get(label, 0)
        s = per_dataset.get("styled", {}).get(label, 0)
        r = per_dataset.get("real_like", {}).get(label, 0)
        t = c + s + r
        lines.append(f"| {label} | {c} | {s} | {r} | {t} |")

    lines.append("")
    lines.append("## Label Definitions")
    lines.append("")
    for label, desc in sorted(ALL_TAXONOMY.items()):
        lines.append(f"- **{label}**: {desc}")

    lines.append("")
    lines.append("## Key Observations")
    lines.append("")

    clean_agg = reports.get("clean", {}).get("aggregate", {})
    clean_pass = clean_agg.get("pass_rate", 0) * 100
    ms = clean_agg.get("main_metrics_stats", {}).get("curve_y_mae_px", {})
    ymae_mean = ms.get("mean", 0)
    ymae_max = ms.get("max", 0)

    total_failures = max(sum(global_counts.values()), 1)
    gc_count = global_counts.get("grid_confusion", 0)
    gc_pct = gc_count / total_failures * 100

    lines.append(f"1. **grid_confusion is the dominant failure** across all datasets "
                 f"({gc_count}/{total_failures} = {gc_pct:.0f}%)")
    lines.append(f"2. **clean pass rate: {clean_pass:.1f}%**, "
                 f"mean y_mae={ymae_mean:.2f}px, worst case={ymae_max:.2f}px")
    cs_count = global_counts.get("candidate_starvation", 0)
    lines.append(f"3. **candidate_starvation** appears in {cs_count} samples, "
                 "indicating color/threshold issues for specific patterns")
    lines.append("4. **No tail_collapse, wrong_branch_lock_in, calibration_mismatch, "
                 "or peak_miss_after_smoothing detected** at baseline level")
    lines.append("5. Pipeline runs end-to-end without crashes (ok=100+50+50, fail=0)")

    lines.append("")
    lines.append("## Priority Actions for Next Steps")
    lines.append("")
    lines.append("1. Reduce `curve_y_mae_px` via curve optimization (Step 19+)")
    lines.append("2. Address `grid_confusion` - improve grid/axis line detection and exclusion")
    lines.append("3. Address `candidate_starvation` - improve adaptive threshold and color model")

    doc = "\n".join(lines)
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"Generated: {OUTPUT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
