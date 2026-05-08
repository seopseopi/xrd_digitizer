# Baseline 고정 설정 (템플릿)

## 공식 B0 (모델 도입 판단용)

- **경로**: `dist/xrd_digitizer_model_v1_3` (코드: `core.model_integration_baseline.BASELINE_B0_RULE_SNAPSHOT_DIR`)
- 상세: `OFFICIAL_BASELINE_B0.md`

## 워킹 트리에서 배치·실험용 설정

아래 항목을 **실제 실행 후** 채운다. 빈 칸은 실행 시점에 기록한다.

## 실행 명령 (복사 후 수정)

```bash
python runner/batch_run.py \
  --manifest_csv <MANIFEST_CSV> \
  --output_dir outputs/baseline_v1_2 \
  --pipeline v1_2 \
  --resume
```

## 반드시 기록할 항목

| 항목 | 값 |
| --- | --- |
| pipeline_version | |
| CLI 전체 (위 블록에 추가된 모든 플래그 포함) | |
| tune_json 사용 여부·경로 | |
| `--use-sharp-peak-preserve` 사용 여부 | |
| `--use-contrast-aux` 사용 여부 | |
| `--axis-mask-margin` 값 | |
| DP 후보 브리지 (`--no-dp-candidate-bridge` 여부) | |
| apex pull (`--no-dp-column-apex-pull` 여부) | |
| `--use-ridge-candidates` 여부 | |
| 입력 manifest 경로 | |
| 출력 디렉터리 | |
| git commit hash | |

## 산출물 연결

- 요약: `baseline_run_summary.json` (배치/평가 스크립트 출력 경로에 맞게 저장)
- 지표 테이블: `baseline_metrics.csv`

참고: 저장소–계획서 정렬 사항은 상위 디렉터리의 `CODEBASE_ALIGNMENT.md`를 본다.
