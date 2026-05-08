# 연구 계획 진단 보고서 (실행 요약)

## Baseline 구분 (필수)

- **B0 (공식)**: 수정 실험 전 최고 성능 rule-only 스냅샷 **`dist/xrd_digitizer_model_v1_3`** — 모델 도입 여부는 **항상 B0 대비**로만 판단한다 (`core/model_integration_baseline.py` 참고).
- **B1 (진단 참조)**: 본 문서의 수치·`eval_reports/` JSON은 **`outputs/0504/runs/eval_ablation_08_ridge_m4_peak1`** 산출물을 재평가한 것으로, 실패 원인·ablation 참고용이다. **공식 baseline이 아니다.**
- **M1**: 동일 평가 조건에서 모델 보조를 붙인 후보 — **B0를 이겨야** 통합 검토가 가능하다.

모델 도입은 진단용 조합 대비 개선 여부가 아니라, 수정 전 최고 성능 rule-only 스냅샷인 **B0(`dist/xrd_digitizer_model_v1_3`)** 대비 개선 여부로 판단한다. 모델은 해당 baseline보다 clean/styled/real_like 통합 성능에서 명확히 개선될 때만 도입한다.

---

실행일 기준 스냅샷: **도메인당 평가 샘플 10장**, **B1 진단 배치** `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1` (v1_1 + ridge + axis_margin4 + peak_single_pass).  
별도로 **`pipeline v1_2` 기본 플래그 전체 클린 배치**는 `outputs/research_diag/baseline_v12_default/clean`에서 진행·재개 중일 수 있다(Synology 환경에서 장시간 소요).

---

## 1) 데이터 무결성 (`02_dataset`)

- **`dataset_integrity_report.json`**: clean/styled/real 해석용 매니페스트(각 styled_v5·real_v4 필터 후 **50행**) 모두 **오류 0건**.
- **`split_leakage_report.json`**: 동일 manifest 내 **source/family cross-split 이슈 미검출**(컬럼 부재 시에는 검사 생략됨).
- **`distribution_summary.txt`**: `data/metadata/all_samples.csv` + `dev_subset` 분포 요약 생성 완료.

경로 정규화 CSV: `manifest_*_resolved.csv`.

---

## 2) 스타일별 게이트 요약 (mvp → development → strict)

| 그룹 | 샘플 수 | mvp pass | dev pass | strict pass |
| --- | ---: | ---: | ---: | ---: |
| clean | 10 | 10% | 0% | 0% |
| styled | 10 | 60% | 30% | 0% |
| real_like | 10 | 60% | 30% | 0% |

development 평균 지표(메인) 요약:

| 그룹 | curve_y_mae_px | major_peak_x_error | numeric_y_mae_norm | major_peak_x_error_2theta |
| --- | --- | --- | --- | --- |
| clean | 13.62 | **39.15** | 0.048 | **2.72** |
| styled | 30.57* | 20.36 | 0.066 | 1.41 |
| real_like | 19.40* | 24.58 | 0.049 | 1.70 |

\* styled/real은 각각 **`pattern_72296`**, **`pattern_60890`** 등 소수 이상치로 평균이 과대될 수 있음.

상세 JSON: `eval_reports/report_*_*.json` (**B1 진단 전용**, README 참고)  
표 형태 CSV: `00_baseline/baseline_metrics.csv` (**동일 B1 스냅샷 요약**; B0 수치는 별도 확보)

---

## 3) 실패 taxonomy 상위

`failure_taxonomy_from_eval.md` 집계:

1. **`grid_confusion`** (압도적 다수)
2. **`peak_miss_after_smoothing`**

→ 계획서 §7.3 해석으로는 **clean도 피크 x 오차가 매우 커서 “후보/DP/피크 정렬” 복합 이슈**에 가깝고, styled/real은 **추가로 저대비·왜곡 이상치** 점검이 필요하다.

---

## 4) 단계별 결론 (파일별 상세는 `03_stage_diagnosis/`)

| 단계 | 결론 한 줄 |
| --- | --- |
| 전처리·축 | `grid_confusion` 중심 → **axis 마스크·격자 억제** 최우선 후보 |
| 후보 | 컬럼 recall은 높음 → **“후보 부족”보다 잘못된 경로 선택**이 더 큼 |
| DP | margin 불안정 라벨은 상대적으로 적음 → **비용/격자 간섭** 점검 |
| 복구 | 트리거가 자주 안 도는 샘플 많음 → **트리거·효과 검증** 필요 |
| 후처리 | `peak_miss_after_smoothing` 반복 → **스무딩·피크 분기** 튜닝 |
| 수치화 | `numeric_y_mae_norm`은 상대적으로 양호 → **앞 단계 안정화 우선** |

---

## 5) Ablation (`04_ablation/ablation_matrix.tsv`)

도메인 간 상충 가능성이 있어 **“단일 최적 조합” 미확정**. ridge·peak-single-pass·margin4 조합별 수치는 TSV 참조.

---

## 6) 모델 도입

**현 시점 보류** (`05_final_decision/model_integration_decision.md`). 통합 판정은 **`ml/model_integration_compare`** 로 B0 리포트와 대조한다.

---

## 재현 방법

```bash
python3 scripts/run_research_plan_diagnosis.py --skip-eval   # 데이터 산출물만
python3 scripts/run_research_plan_diagnosis.py --batch-output-base outputs/0504/runs/eval_ablation_08_ridge_m4_peak1
```

코드베이스와 계획 서술 차이는 `CODEBASE_ALIGNMENT.md` 참고.
