# 모델 보조 도입 실행 가이드 (코드 반영본)

## 공식 baseline (B0)

- **B0** = `dist/xrd_digitizer_model_v1_3` (코드 단일 정의: `core/model_integration_baseline.py`)
- **M1 > B0** 가 입증될 때만 모델 통합을 논의한다. **M1 > B1(진단 산출물)** 만으로는 부족하다.
- B1 예시: `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1` → **diagnostic reference**

### 정책 업데이트 (oracle 배치 스터디 이후)

- **런타임 모델 통합(전역 candidate re-ranker 기본 적용)은 보류**한다. CNN 학습·오프라인 평가는 계속할 수 있으나, **적용 범위는 selective model assist로 한정**한다.
- **아키텍처 방향**: `ManualInputs → 전처리 → 후보 생성 → Risk Detector → Selective Candidate Re-ranker(위험 열만) → DP → Recovery → Postprocess → Numeric`. 비위험 열은 **rule confidence 유지**.
- **GT oracle 배치** (`outputs/oracle_rerank_batch_study/`)는 **current branch 기준 upper bound**이며, **B0 공식 성능 비교와 분리**해 기록한다. oracle 수치를 “학습 모델이 B0를 이겼다”는 근거로 쓰지 않는다.
- **근거·한계·16 vs 34 분석**: `model_integration_decision_updated.md`, `improved_vs_worsened_analysis.md`, `selective_model_assist_plan.md`, `risk_detector_feature_candidates.csv` 참고.
- **styled / real_like**: 이상치 서브셋에서 역행 신호가 있어 **모델 적용 기본 off**(또는 보수적 on + 강화 fallback).
- **strict**: oracle에서도 통과율 0% → **모델·oracle만으로 최종 품질 기준을 만족한다고 서술하지 않는다.**

통합 비교 리포트:

```bash
PYTHONPATH=. python3 -m ml.model_integration_compare \
  --manifest research_plan_outputs/model_eval/compare_manifest.filled.json \
  --out_json research_plan_outputs/model_eval/model_integration_verdict.json
```

`compare_manifest.example.json`을 복사해 B0·M1 리포트 경로를 채운다.

---

이번 반영은 계획서의 핵심을 유지하면서, 현재 저장소에서 바로 실행 가능한 최소 경로로 정리했다.

## 1) 후보 dump 생성 (필수)

학습 데이터 생성을 위해 `debug.json`에 후보를 남겨야 한다.

- `runner/run_local.py`, `runner/batch_run.py`에 `--dump-candidates-json` 추가됨.
- 켜면 `debug_dir` 안에 `raw_candidates`, `filtered_candidates`, `final_candidates` JSON이 함께 저장된다.

예시:

```bash
python3 runner/batch_run.py \
  --manifest_csv data/manifests/clean_manifest.csv \
  --output_dir outputs/model_data_dump/clean \
  --pipeline v1_1 \
  --max_samples 50 \
  --dump-candidates-json
```

## 2) Candidate Re-ranker 데이터셋 생성

```bash
python3 -m ml.data.build_candidate_rerank_dataset \
  --manifest_csv data/manifests/clean_manifest.csv \
  --run_dir outputs/model_data_dump/clean \
  --output_dir data/model_candidate_rerank/clean \
  --patch_size 33 \
  --positive_px 2 \
  --negative_px 8
```

주의:
- `debug.json`에 `final_candidates`가 없으면 해당 샘플은 자동 skip.
- 현재 빌더 기본 채널은 `roi_gray`, `candidate_center`, `axis_proximity` 3채널.

## 3) 학습

```bash
python3 -m ml.train_candidate_reranker \
  --data_dir data/model_candidate_rerank/clean \
  --output_dir outputs/model_ckpt/candidate_reranker_v1 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --device cpu
```

## 4) 추론 + 오프라인 재랭크 평가

```bash
python3 -m ml.infer_candidate_reranker \
  --model_ckpt outputs/model_ckpt/candidate_reranker_v1/candidate_reranker_v1.pt \
  --input_jsonl data/model_candidate_rerank/clean/val.jsonl \
  --output_jsonl outputs/model_ckpt/candidate_reranker_v1/val_pred.jsonl
```

```bash
python3 -m ml.offline_rerank_eval \
  --pred_jsonl outputs/model_ckpt/candidate_reranker_v1/val_pred.jsonl \
  --lambda_model 0.25 \
  --topk 3 \
  --out_json outputs/model_ckpt/candidate_reranker_v1/offline_eval_l025.json
```

## 5) 리포트 비교(통합 후)

도메인·게이트 레벨 전체를 묶은 **공식 판정**은 `ml.model_integration_compare` 만 사용한다.

단일 도메인 빠른 delta만 필요할 때(참고용):

```bash
PYTHONPATH=. python3 -m ml.dp_rerun_compare \
  --b0_report path/to/B0_report_one_domain.json \
  --m1_report path/to/M1_report_one_domain.json \
  --out_json outputs/runs/model_assist/partial_delta.json
```

## 6) 런타임 통합 (DP 직전 rerank + fallback + model_assist JSON)

구현 위치: `ml/runtime_candidate_rerank.py`, `runner/run_local.py`, `runner/batch_run.py`, `core/model_assist_settings.py`.

- **현재 구현**: `final_candidates` 확정 직후, ROI 패치로 `SmallCandidateCNN` 추론 → 각 후보에 `rule_confidence`, `model_score_delta` 기록 후 `confidence = clip(rule + λ·model, 0, 1)` 로 DP 비용에 반영하는 **전역 혼합**에 가깝다.
- **목표 구조(정책)**: 위 파이프라인 전 **Risk Detector**로 열/구간 마스크를 만든 뒤, **Selective Candidate Re-ranker**는 마스크 on 열에서만 confidence를 수정한다. 상세·fallback·도메인 정책은 `outputs/oracle_rerank_batch_study/selective_model_assist_plan.md`를 따른다.
- **fallback**: 규칙 전용 DP(`trace_score_rule`, `valid_ratio_rule`)와 모델 보조 DP를 비교해, valid_ratio·trace_score가 허용 마진 밖으로 악화되면 **규칙 경로로 자동 복귀** (`fallback_reason` 기록). 추가로 **numeric_y_mae_norm·max_gap·development 역행** 등 게이트 기반 fallback을 강화할 것(`selective_model_assist_plan.md` 참고).
- **결과 JSON**: 모든 실행에 `model_assist` 객체가 포함되며(`RunResult.model_assist`), `debug.json`에도 동일 요약이 들어간다.
- **peak apex ROI 보정**: `--peak-apex-roi-refine` 시 major peak에 대해 열별 밝기 프로파일로 `y_pixel_roi_refine` → `export_numeric`에서 `peak_positions_2theta` 우선 반영.

단일 이미지 예시:

```bash
python3 runner/run_local.py \
  --image_path path/to.png \
  --manual_inputs_path path/to_mi.json \
  --output_json_path out/result.json \
  --debug_dir out/debug \
  --model-assist \
  --model-assist-ckpt outputs/model_ckpt/candidate_reranker_v1/candidate_reranker_v1.pt \
  --model-assist-lambda 0.25 \
  --peak-apex-roi-refine \
  --peak-apex-roi-radius 5
```

배치는 `runner/batch_run.py`에 동일 플래그 이름으로 전달하면 된다.

### GT Oracle 재랭크 (학습 weight 불필요)

구현: `trace/oracle_rerank.py`, `core/oracle_rerank_settings.py`.

- GT의 `pixel_curve_path`(또는 `pixel_curve_by_x`)로 플롯 영역 각 열의 기대 y를 보간한 뒤, 후보마다 `confidence = exp(-(dist/sigma)^2)` 로 바꾸고 DP·apex pull을 다시 돌린다.
- **규칙 DP**와 **oracle DP**의 `trace_score`·`valid_ratio`를 `model_assist.oracle_rerank`에 같이 남기고, 후처리·recovery는 **oracle 경로**를 따른다(“후보만 맞게 고르면 개선되는가” 상한).
- 단일 이미지: `--oracle-rerank-gt path/to_gt.json` `[--oracle-rerank-sigma 8]`
- 배치: 샘플마다 GT가 다르면 `--oracle-rerank-from-manifest`(각 행 `gt_path`). 단일 GT만 쓸 때만 `--oracle-rerank-gt` 고정 경로.
- oracle을 켠 실행에서는 CNN `--model-assist`는 **적용하지 않는다**(경고 로그).

## 7) 오프라인 vs 런타임

- 오프라인: `ml.offline_rerank_eval` — 후보 단위 JSONL로 재현·분석.
- 런타임: 위 플래그 — 실제 엔진 한 번 실행 안에서 DP까지 반영 (평가 게이트에 사용).
