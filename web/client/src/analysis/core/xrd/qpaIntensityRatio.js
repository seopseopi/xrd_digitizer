/**
 * 반정량 상 분율 (RIR·단일 대표 피크 근사)
 * V_i ∝ (I_i / RIR_i) / Σ_j (I_j / RIR_j)  (동일 μt·무텍스처 가정)
 */

/**
 * 가우시안 근사 적분 강도 ∝ peakHeight * FWHM (상수는 상쇄)
 */
export function approximateIntegratedIntensityGaussian(height, fwhmDeg) {
  if (!Number.isFinite(height) || !Number.isFinite(fwhmDeg) || fwhmDeg <= 0) return null;
  const sigmaDeg = fwhmDeg / (2 * Math.sqrt(2 * Math.LN2));
  return height * sigmaDeg * Math.sqrt(2 * Math.PI);
}

/**
 * @param {Array<{ phaseId: string, rir: number, intensity?: number, integratedIntensity?: number }>} phases
 */
export function estimateQPAPhaseFractions(phases) {
  if (!Array.isArray(phases) || phases.length < 2) {
    return {
      success: false,
      error: '반정량 상 분율 계산에는 상이 최소 2개 필요합니다.',
    };
  }

  const weights = [];
  const warnings = [
    '겹침 피크·텍스처·미세흡수에 민감합니다. 정밀 정량에는 리트벨트 등을 권장합니다.',
  ];

  for (const p of phases) {
    const rir = Number(p.rir);
    let I = p.integratedIntensity;
    if (I == null || !Number.isFinite(I)) {
      I = Number(p.intensity);
    }
    if (!Number.isFinite(I) || I <= 0 || !Number.isFinite(rir) || rir <= 0) {
      return {
        success: false,
        error: `상 "${p.phaseId || '?'}"의 강도·RIR이 유효하지 않습니다.`,
      };
    }
    weights.push({ phaseId: String(p.phaseId || 'unknown'), rir, intensityUsed: I, weight: I / rir });
  }

  const sumW = weights.reduce((s, w) => s + w.weight, 0);
  if (sumW <= 0) {
    return { success: false, error: '가중치 합이 0입니다.' };
  }

  const fractions = weights.map((w) => ({
    phaseId: w.phaseId,
    approximateVolumeFraction: w.weight / sumW,
    weightIOverRIR: w.weight,
    rir: w.rir,
    intensityUsed: w.intensityUsed,
  }));

  return {
    success: true,
    phases: fractions,
    warnings,
  };
}
