# XRD Digitizer V1

XRD graph image digitization engine for reconstructing numeric JSON data from plotted XRD images.

The repository contains the core pipeline, calibration logic, preprocessing, tracing, evaluation scripts, ML helper modules, tests, and experiment notes. Local datasets and generated outputs are intentionally excluded from git.

## Repository Layout

- `core/` - shared config, IO, types, and pipeline settings
- `preprocess/` - ROI, perspective, masking, color, and morphology preprocessing
- `trace/` - curve candidate extraction, DP tracing, recovery, and postprocessing
- `calibrate/` - axis mapping and numeric export
- `peaks/` - peak detection and smoothing helpers
- `runner/` - local and batch pipeline runners
- `eval/` - metrics, gates, and reports
- `ml/` - candidate reranking data/model utilities
- `scripts/` - dataset generation, diagnostics, evaluation, and reporting tools
- `tests/` - focused unit tests
- `experiments/` - active notes and evaluation summaries

## Excluded Local Files

The following are not committed:

- `data/` - source datasets and ground truth files
- `outputs/` - generated run outputs and reports
- `dist/` - packaged artifacts
- `debug_artifacts/` - local debug outputs
- `experiments/archive/` - archived bundles
- virtual environments, Python caches, and OS/editor files

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Test

```bash
pytest
```

## Notes

Most evaluation and rendering scripts expect local dataset files under `data/`, which must be provided separately.
