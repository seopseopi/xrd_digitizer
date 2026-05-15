/**
 * 배향 지수 MVP: 무작위 시편 대비 상대 피크 강도 P = I_meas / I_ref
 * I_ref는 ICDD PDF 등에서 사용자가 입력합니다. 기하·흡수 보정 없음.
 */

/**
 * @param {Array<{ label?: string, intensityMeasured: number, intensityReference: number }>} rows
 * @returns {Object}
 */
export function computeTextureIndices(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return { success: false, error: '입력 행이 없습니다.' };
  }

  const valid = rows.filter(
    (r) =>
      r &&
      Number.isFinite(r.intensityMeasured) &&
      Number.isFinite(r.intensityReference) &&
      r.intensityReference > 0
  );

  if (valid.length === 0) {
    return { success: false, error: '유효한 (측정·참조) 강도 쌍이 없습니다.' };
  }

  const rawRatios = valid.map((r) => r.intensityMeasured / r.intensityReference);
  const logMean =
    rawRatios.reduce((s, v) => s + Math.log(Math.max(v, 1e-30)), 0) / rawRatios.length;
  const norm = Math.exp(logMean);

  const results = valid.map((r, i) => ({
    label: r.label || `피크 ${i + 1}`,
    intensityMeasured: r.intensityMeasured,
    intensityReference: r.intensityReference,
    rawRatio: rawRatios[i],
    orientationIndex: rawRatios[i] / norm,
  }));

  const warnings = [
    '선호 배향이 강하면 RIR·QPA 해석이 왜곡될 수 있습니다.',
    '참조 강도는 동일 파장·기하 조건의 표준 패턴 값을 사용하는 것이 좋습니다.',
  ];

  return {
    success: true,
    results,
    normalization: 'geometric_mean_of_raw_ratios',
    warnings,
  };
}
