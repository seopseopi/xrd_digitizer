from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from copy import deepcopy

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.exp_params import default_exp_params
from eval.gates_exp import check_gate_v2_band, check_gate_v2_strict
from eval.metrics import compute_all_metrics
from eval.metrics_exp import compute_metrics_v2, merge_main_with_v2


MANIFESTS = [
    ("clean", ROOT / "data/manifests/clean_manifest.csv", "clean"),
    ("styled", ROOT / "data/manifests/styled_manifest.csv", "styled"),
    ("real-like", ROOT / "data/manifests/real_manifest.csv", "real-like"),
]


def evaluate(out_dir: pathlib.Path, max_per_domain: int = 20) -> float:
    maes = []
    passes = 0
    total = 0
    for name, mf, gate_type in MANIFESTS:
        df = pd.read_csv(mf).head(max_per_domain)
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            rp = out_dir / f"{sid}_result.json"
            dp = out_dir / f"debug_{sid}" / "debug.json"
            if not rp.exists() or not dp.exists():
                continue
            res = json.loads(rp.read_text(encoding="utf-8"))
            dbg = json.loads(dp.read_text(encoding="utf-8"))
            gt = json.loads(pathlib.Path(str(row["gt_path"])).read_text(encoding="utf-8"))
            m = compute_all_metrics(res, dbg, gt)
            v2 = compute_metrics_v2(res, dbg, gt)
            mm = merge_main_with_v2(m["main"], v2)
            gs = check_gate_v2_strict(mm, v2, gate_type)["passed"]
            gb = check_gate_v2_band(mm, v2, gate_type)["passed"]
            passes += 1 if (gs and gb) else 0
            total += 1
            maes.append(v2["strict_curve_y_mae_px"])
    if not maes:
        return 1e9
    pass_penalty = 20.0 * (1.0 - (passes / max(total, 1)))
    return float(sum(maes) / len(maes) + pass_penalty)


def run_batch_all(out_dir: pathlib.Path, tune_json: pathlib.Path, max_samples: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mf, _ in MANIFESTS:
        sub_out = out_dir / name
        sub_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "runner/batch_run.py"),
            "--manifest_csv",
            str(mf),
            "--output_dir",
            str(sub_out),
            "--max_samples",
            str(max_samples),
            "--pipeline",
            "v2",
            "--tune_json",
            str(tune_json),
        ]
        subprocess.run(cmd, check=True)


def merge_domain_outputs(work_dir: pathlib.Path, merged_dir: pathlib.Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    for d in ("clean", "styled", "real-like"):
        src = work_dir / d
        if not src.exists():
            continue
        for p in src.glob("*"):
            target = merged_dir / p.name
            if p.is_dir():
                continue
            target.write_bytes(p.read_bytes())
        for dp in src.glob("debug_*"):
            td = merged_dir / dp.name
            td.mkdir(parents=True, exist_ok=True)
            for f in dp.glob("*"):
                (td / f.name).write_bytes(f.read_bytes())


def set_nested(p: dict, path: str, value: float) -> None:
    a, b = path.split(".", 1)
    p[a][b] = value


def main() -> None:
    best = default_exp_params()
    tune_root = ROOT / "outputs" / "runs" / "exp_autotune"
    tune_root.mkdir(parents=True, exist_ok=True)
    best_json = tune_root / "best_params.json"
    best_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    search_plan = [
        ("preprocess.mask_a_thr", [0.16, 0.18, 0.20]),
        ("preprocess.mask_b_thr", [0.20, 0.22, 0.24]),
        ("candidates.conf_min", [0.10, 0.12, 0.14]),
        ("candidates.comp_support_min", [0.08, 0.10, 0.12]),
        ("recovery.gain_min", [0.10, 0.15, 0.20]),
        ("recovery.boundary_mul", [1.5, 2.0, 2.5]),
        ("postprocess.smooth_k_thr", [1.2, 1.5, 1.8]),
        ("postprocess.nms_x_scale", [0.0035, 0.0045, 0.0055]),
    ]

    best_score = 1e9
    for key, cand_vals in search_plan:
        local_best = None
        local_score = 1e9
        for v in cand_vals:
            trial = deepcopy(best)
            set_nested(trial, key, v)
            trial_json = tune_root / "trial_params.json"
            trial_json.write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")
            trial_out = tune_root / "trial_run"
            run_batch_all(trial_out, trial_json, max_samples=8)
            merged = tune_root / "trial_merged"
            merge_domain_outputs(trial_out, merged)
            score = evaluate(merged, max_per_domain=8)
            if score < local_score:
                local_score = score
                local_best = v
        if local_best is not None:
            set_nested(best, key, local_best)
            best_score = local_score
            best_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[TUNE] {key} -> {local_best} (score={best_score:.4f})")

    print(f"[DONE] best score={best_score:.4f}")
    print(f"best params: {best_json}")


if __name__ == "__main__":
    main()
