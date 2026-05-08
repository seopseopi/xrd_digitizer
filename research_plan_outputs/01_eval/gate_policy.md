# Gate 정책 요약

- 레벨: `mvp` → `development`(기본) → `strict`
- 도메인: `clean`, `styled`, `real_like` 각각 임계값 분리 (`eval/gates.GATES`)

보고 순서(계획서와 동일):

1. mvp pass rate  
2. development pass rate  
3. strict pass rate  
4. strict 실패 시 failure taxonomy 상위  
5. 스타일별 strict pass rate  

본 표는 **B1 진단 참조** 배치(10장/도메인, `outputs/0504/runs/eval_ablation_08_ridge_m4_peak1`)에 대한 재평가 스냅샷이다. **공식 B0 baseline(`dist/xrd_digitizer_model_v1_3`)이 아니다.**

| 도메인 | mvp | development | strict |
| --- | --- | --- | --- |
| clean | 10% | 0% | 0% |
| styled | 60% | 30% | 0% |
| real_like | 60% | 30% | 0% |
