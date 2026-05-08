from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEGACY_RUNS = ROOT / "experiments" / "archive" / "outputs_legacy_runs"
RUNS = ROOT / "outputs" / "runs"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    best_json = RUNS / "exp_autotune" / "best_params.json"
    if not best_json.exists():
        alt = LEGACY_RUNS / "v2_autotune" / "best_params.json"
        if not alt.exists():
            alt = LEGACY_RUNS / "exp_autotune" / "best_params.json"
        if alt.exists():
            best_json = alt
        else:
            raise FileNotFoundError(f"missing tuned params: {RUNS / 'exp_autotune' / 'best_params.json'}")

    targets = [
        ("clean", ROOT / "data/manifests/clean_manifest.csv", RUNS / "exp_tuned_clean", "clean"),
        ("styled", ROOT / "data/manifests/styled_manifest.csv", RUNS / "exp_tuned_styled", "styled"),
        ("real", ROOT / "data/manifests/real_manifest.csv", RUNS / "exp_tuned_real", "real-like"),
    ]
    for _, mf, out, _ in targets:
        run(
            [
                sys.executable,
                str(ROOT / "runner/batch_run.py"),
                "--manifest_csv",
                str(mf),
                "--output_dir",
                str(out),
                "--max_samples",
                "9999",
                "--pipeline",
                "v2_experimental",
                "--allow_experimental_v2",
                "--tune_json",
                str(best_json),
            ]
        )

    # per-domain pdf
    for _, mf, out, gt in targets:
        run(
            [
                sys.executable,
                str(ROOT / "scripts/exp_execution_report_pdf.py"),
                "--manifest_csv",
                str(mf),
                "--output_dir",
                str(out),
                "--gate_type",
                gt,
                "--pdf_out",
                str(out / "exp_tuned_report.pdf"),
            ]
        )

    # detailed overall report
    run([sys.executable, str(ROOT / "scripts/generate_exp_detailed_report.py")])
    print("tuned reports generated")


if __name__ == "__main__":
    main()
