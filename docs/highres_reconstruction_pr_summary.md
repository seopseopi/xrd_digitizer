# 2x Highres Reconstruction Pipeline and Canonical Evaluation

## A. Recommended Commit Title

```text
feat: add 2x highres ROI export and canonical source-numeric evaluation
```

Alternative:

```text
feat: establish 2x highres reconstruction pipeline with canonical evaluation
```

## B. Commit Body

```text
feat: add 2x highres ROI export and canonical source-numeric evaluation

- Add 2x highres ROI tracing/export path
- Add --final-export-mode eval_grid|highres
- Add export_points_eval, export_points_highres, export_metadata
- Keep 950-point eval_grid as legacy compatibility output
- Add canonical test set workflow based on data/test_canonical_30/manifest.csv
- Enforce manifest-based input_image + mi_json + gt_json + source_numeric_json execution
- Remove pattern_1915 from canonical tests and add clean_pattern_11832 as clean anchor
- Add highres-primary source_numeric evaluation policy
- Run canonical 3-domain and canonical 30 highres-primary evaluation
- Add source_numeric stage-wise error decomposition
- Add real_like_pattern_83398 catastrophic failure analysis
- Add candidate dump root-cause analysis for candidate explosion and false branch dominance
- Add isolated stabilization ablation reports

Results:
- ROI/tracing resolution increased from 950x690 to 1900x1380
- Raw trace points increased from 950 to 1900
- 3-domain highres evaluation improved 3/3 samples
- Canonical 30 improved 17/30 samples
- Mean normalized_y_mae improved from 0.04718 to 0.04542
- Shape correlation improved from 0.8481 to 0.9435
- Main remaining blocker is noisy real_like instability, especially real_like_pattern_83398

Current judgment:
HIGHRES_PRIMARY_EVAL_MIXED_NEEDS_FAILURE_ANALYSIS

Not changed:
- No canonical source files modified
- No mi/gt/source_numeric/metadata mutation
- No plot_box/calibration changes
- No candidate/DP/tracing scoring changes
- No threshold/margin tuning
```

## C. PR Summary

This update introduces the 2x highres ROI tracing/export pipeline and establishes a canonical `source_numeric.json`-based evaluation workflow.

The reconstruction path is no longer limited to the original 950-point eval grid. The pipeline can now trace on a 2x upscaled ROI and export 1900-point highres numeric outputs, while preserving the legacy 950-point eval output for compatibility.

This also adds the canonical test set workflow, highres-primary evaluation policy, source-numeric error decomposition, and real-like catastrophic failure analysis.

Current status:

```text
HIGHRES_PRIMARY_DIRECTION_PROMISING_WITH_REAL_LIKE_INSTABILITY
```

## D. Performance Changes

| Metric | Before | After |
| --- | ---: | ---: |
| ROI size | 950 x 690 | 1900 x 1380 |
| Raw trace points | 950 | 1900 |
| Legacy eval export | 950 points | 950 points |
| Highres export | N/A | 1900 points |
| 3-domain smoke improvement | N/A | 3/3 |
| Canonical 30 improved samples | N/A | 17/30 |
| Canonical 30 worsened samples | N/A | 13/30 |
| Mean normalized_y_mae | 0.04718 | 0.04542 |
| Median delta | N/A | -0.00048 |
| Shape correlation | 0.8481 | 0.9435 |
| x_monotonic fail | N/A | 0 |
| gap fail | N/A | 0 |

Representative 3-domain smoke results:

| Sample | Baseline norm MAE | Highres norm MAE |
| --- | ---: | ---: |
| clean_pattern_11832 | 0.073817 | 0.004689 |
| styled_pattern_72296 | 0.277043 | 0.035443 |
| real_like_pattern_60890 | 0.083236 | 0.032724 |

3-domain judgment:

```text
HIGHRES_PRIMARY_EVAL_PASS_CURRENT_BEST_3DOMAIN
```

Canonical 30 judgment:

```text
HIGHRES_PRIMARY_EVAL_MIXED_NEEDS_FAILURE_ANALYSIS
```

## E. Change List

- Added ROI 2x highres tracing/export path.
- Added `--final-export-mode eval_grid|highres`.
- Added `export_points_eval`.
- Added `export_points_highres`.
- Added `export_metadata`.
- Kept 950-point `eval_grid` output as legacy compatibility mode.
- Promoted 1900-point highres output as the primary `source_numeric.json` evaluation candidate.
- Added canonical test set workflow under `data/test_canonical_30`.
- Fixed execution to `data/test_canonical_30/manifest.csv`.
- Removed `pattern_1915` from canonical testing due to data issues.
- Added `clean_pattern_11832` as the clean smoke anchor.
- Added source-numeric stage-wise error decomposition.
- Added real-like catastrophic failure analysis for `real_like_pattern_83398`.
- Added candidate dump root-cause analysis.
- Added branch instability stabilization ablation reports.

## F. Timeline

1. Added 2x ROI upscale tracing and highres numeric export.
2. Preserved 950-point `eval_grid` output for legacy evaluator compatibility.
3. Added structured export fields to separate eval-grid and highres outputs.
4. Built the canonical 30-sample test workflow around `manifest.csv`.
5. Rejected filename-based path inference, image-only execution, and mi-only execution.
6. Observed that 2x could look worse under the legacy eval-grid metric.
7. Ran `source_numeric.json`-based decomposition on `clean_pattern_11832`.
8. Found that 2x raw/highres tracing improved strongly, while 1900-to-950 downscale could destroy the gain.
9. Reframed primary evaluation from 950-point `eval_grid` to 1900-point highres output.
10. Ran 3-domain smoke evaluation and canonical 30 evaluation.
11. Investigated `real_like_pattern_83398` as the largest catastrophic failure.
12. Added candidate dump analysis and isolated stabilization ablations.

## G. Failure Analysis Summary

The largest catastrophic failure was:

```text
real_like_pattern_83398
```

Canonical 30 result:

```text
baseline norm MAE: 0.03350
highres norm MAE: 0.41504
delta: +0.38155
```

Calibration was ruled out:

```text
roundtrip_error: 0.0px
calibration_confidence: 1.0
```

Export/downscale was not the main cause in this case. The failure was already present in candidate/tracing stages.

Main signals:

```text
raw_candidates_total: 47054 -> 126768
n_components: 2 -> 30
dy_abs_max: 136 -> 1259
long_jump_count: 4 -> 10
top1_far_from_trace_columns: 342
```

Failure taxonomy:

```text
HIGHRES_NOISE_AMPLIFICATION
CANDIDATE_EXPLOSION
LOCAL_OSCILLATION_INSTABILITY
BRANCH_SWITCH_INSTABILITY
```

Candidate dump root cause:

```text
CANDIDATE_DUMP_ROOTCAUSE_MIXED
```

Interpretation:

```text
Raw candidate burst occurs first.
Filtering removes some noise, but false branches still survive.
Final ranking then selects far-from-trace top-1 candidates in many columns.
This causes branch switching, oscillation, and catastrophic divergence.
```

## H. Current Limitations

2x highres is promising and improves shape fidelity overall, but it is not yet the fixed current-best baseline across canonical 30.

Known blocker:

```text
noisy real_like instability
```

Current label:

```text
HIGHRES_PRIMARY_EVAL_MIXED_NEEDS_FAILURE_ANALYSIS
```

The stronger interpretation is:

```text
HIGHRES_PRIMARY_DIRECTION_PROMISING_WITH_REAL_LIKE_INSTABILITY
```

## I. Not Changed

This update did not modify:

- canonical input files
- `mi.json`
- `gt.json`
- `source_numeric.json`
- `metadata.json`
- `plot_box`
- calibration
- candidate scoring formula
- DP/tracing scoring formula
- thresholds
- margins
- ROI 2x structure rollback
- highres export rollback

This work focused on export structure, canonical evaluation, source-numeric diagnosis, and failure analysis, not tuning.

## J. Next Steps

Immediate next step:

```text
candidate-stage runtime optimization
```

Observed runtime example:

```text
total runtime: 384s
candidates stage: 380.45s
```

Optimization target:

```text
trace/candidates.py
```

Plan:

1. Add stage-level timing.
2. Vectorize scalar numpy calls in `build_raw_candidates`.
3. Optimize `filter_candidates` neighbor/top1 cache only if result equivalence is preserved.
4. Vectorize bridge synth computation.
5. Optimize expanded deepcopy safely with pre-trim snapshot.
6. Verify result equivalence on canonical samples.

Required validation samples:

```text
clean_pattern_11832
styled_pattern_72296
real_like_pattern_83398
```

Goal:

```text
Reduce candidates-stage runtime without changing reconstruction results.
```
