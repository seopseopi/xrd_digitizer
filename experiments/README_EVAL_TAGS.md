# experiments/ 및 eval_ablation_* 태그에 대해

`scripts/run_eval_ablation_matrix.py` 가 생성하는 `eval_ablation_*` 출력과 `experiments/ablation_eval_summary.tsv`, `experiments/verdict_ablation.txt` 는 **플래그 스윕·진단 참조(B1 성격)** 용이다.

**모델 도입 공식 baseline(B0)** 은 `dist/xrd_digitizer_model_v1_3` 이며, 모델 통합 여부는 `ml.model_integration_compare` 로 B0 대비 판정한다.
