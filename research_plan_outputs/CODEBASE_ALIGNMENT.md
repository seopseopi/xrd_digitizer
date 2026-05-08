# 계획서 ↔ 코드베이스 정렬 노트

이 문서는 제출하신 연구·개발 실행 계획을 **현재 저장소의 실제 인터페이스**에 맞출 때 바꿔야 하는 부분만 요약한다. 계획의 목표·순서·해석 프레임워크는 그대로 두고, 명령줄·파일 경로·지표 키만 아래를 따르면 된다.

---

## 1. 배치 실행 (`runner/batch_run.py`)

- 매니페스트 인자명은 **`--manifest_csv`**이다. (`--manifest` 아님)
- 현재 워크스페이스에는 `data/eval_manifest.csv`가 없고, 평가용 후보는 예를 들어 아래가 있다.
  - `data/manifests/clean_manifest.csv`
  - `data/manifests/styled_manifest.csv`
  - `data/manifests/real_manifest.csv`
- 통합 평가 시에는 위 세 파일을 합친 매니페스트를 쓰거나, 이미 프로젝트에서 쓰는 eval용 CSV 경로를 그대로 사용한다.

**예시 (경로는 환경에 맞게 수정):**

```bash
python runner/batch_run.py \
  --manifest_csv data/manifests/clean_manifest.csv \
  --output_dir outputs/baseline_v1_2 \
  --pipeline v1_2 \
  --resume
```

**기준선 기록 항목 보완 — 플래그 기본값**

| 항목 | 코드 상 의미 |
| --- | --- |
| DP 후보 브리지 | 기본 **켜짐**. 끄려면 `--no-dp-candidate-bridge` |
| 열별 apex pull | 기본 **켜짐**. 끄려면 `--no-dp-column-apex-pull` |
| sharp peak preserve | 기본 **꺼짐**. 켜려면 `--use-sharp-peak-preserve` |
| contrast 보조 | 기본 **꺼짐**. 켜려면 `--use-contrast-aux` |
| ridge 후보 가산 | 기본 **꺼짐**. 켜려면 `--use-ridge-candidates` |
| 축 마스크 마진 | `--axis-mask-margin` (픽셀, 기본값 코드 참고) |

`--tune_json`, `--pipeline`, `--max_samples` 등 나머지는 `runner/batch_run.py`의 `argparse` 정의가 단일 출처다.

---

## 2. Main / Debug / Diagnosis 지표 (`eval/metrics.py`, `eval/gates.py`)

게이트·리포트에 쓰이는 **main** 키는 아래와 같다. 계획서의 예시 이름과 다른 부분만 교체하면 된다.

| 계획서 표기(예) | 실제 `main` 키 |
| --- | --- |
| `major_peak_error_px` | **`major_peak_x_error`** (픽셀 공간 주 피크 **x** 오차; y는 `debug.major_peak_y_error`) |
| `max_gap` | **`max_gap_px`** |
| `major_peak_2theta_error` | **`major_peak_x_error_2theta`** |

게이트(`eval/gates.py`)는 도메인별로 **`peak_recall`**을 **main**에 포함한다 (`precision`/`f1`은 debug).

**Diagnosis** (`metrics.compute_all_metrics`의 `diagnosis` 블록):

- `candidate_recall_per_column` — 계획서의 `candidate_recall`과 대응하는 열 단위 후보 커버리지 해석에 사용
- `empty_column_rate` — `empty_column_ratio` 명칭과 혼동하지 말 것
- `recovery_success_rate` — 계획서의 `recovery_rate` 해석에 대응
- `reentry_count`, `path_margin_instability` — DP 불안정 분석용

**Debug** 예시:

- `tail_mae_px`, `tail_collapse_rate` — 계획서의 “tail 오차” 분석에 사용 (`tail_error`라는 단일 키는 없음)

---

## 3. 데이터 스크립트 CLI (`scripts/`)

계획서의 예시와 실제 인자명이 다른 부분이다.

| 작업 | 계획서 예시 | 저장소 실제 |
| --- | --- | --- |
| 소스 스캔 | `--source_json_dir` | **`--source_root`** |
| 스캔 출력 | `data/all_samples.csv` | 기본값은 `data/metadata/all_samples.csv` (`--output_csv`로 지정 가능) |
| split 생성 | `--output_dir data/splits` | **`build_splits.py`**는 `--train_csv`, `--val_csv`, `--test_csv`, `--input_csv`, `--metadata_csv` 등 개별 경로 인자 사용 (기본 출력도 `data/metadata/` 쪽) |
| 무결성 검사 | `--output_json ...` | 현재 **`validate_dataset_integrity.py`는 JSON 출력 옵션이 없음** — 결과는 표준 출력이거나, 리다이렉트·후속 패치로 `research_plan_outputs/02_dataset/dataset_integrity_report.json`을 채운다 |
| 분포 요약 | `--manifest`, `--output_md` | **`--input_csv`**, **`--subset_csv`**, **`--output_txt`** |

---

## 4. 산출물 `split_leakage_report.json`

전용 스크립트는 없을 수 있다. `scripts/build_splits.py`는 **family 단위 누출 방지**를 전제로 한다. 누출 리포트는 train/val/test CSV 간 `family_id`(또는 동등 키) 교차 검사로 생성하면 계획서 의도와 맞다.

---

## 5. 평가 출력 표 (7장 테이블 정렬)

컬럼명은 리포트 생성 파이프라인에 맞추되, 피크 관련 main 지표는 **`major_peak_x_error`**, 간격은 **`max_gap_px`**, 수치 피크는 **`major_peak_x_error_2theta`**를 쓰는 것이 `eval/gates.py`와 일치한다.

---

이 문서와 함께 `00_baseline/baseline_config.md`에 실제로 돌린 명령·커밋 해시·플래그를 붙여 두면 “기준선 고정” 완료 조건을 충족하기 쉽다.

---

## 6. 모델 도입 baseline (B0 / B1 / M1)

- **B0 (공식 rule-only baseline)**: `dist/xrd_digitizer_model_v1_3` — 코드 상수는 `core/model_integration_baseline.BASELINE_B0_RULE_SNAPSHOT_DIR`.
- **B1 (진단 참조)**: 예: `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1` 산출물에 대한 eval 리포트 — **공식 baseline으로 사용하지 않는다.**
- **M1 (모델 보조 후보)**: B0와 동일 평가 조건에서 통합 비교 — **`ml.model_integration_compare`** 로 판정한다.

필수 문장은 `core.model_integration_baseline.required_final_decision_sentence_ko` 및 `research_plan_outputs/05_final_decision/final_decision.md`와 동일하다.
