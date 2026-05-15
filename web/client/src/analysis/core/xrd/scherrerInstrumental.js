/**
 * Scherrer / Williamson–Hall용 FWHM(2θ, 도) 기기 넓이 보정
 * 관측 폭은 동일 스케일(2θ, 도)로 가정합니다.
 */

const MIN_BROADENING_DEG = 1e-4;

/**
 * @param {number} fwhmObsDeg - 관측 FWHM (2θ, °)
 * @param {number} fwhmInstDeg - 기기 기여 FWHM (2θ, °), 표준 시편 등
 * @param {'none'|'subtract'|'quadratic'} method
 * @returns {number} 보정된 FWHM (2θ, °)
 */
export function correctFwhmTwoThetaDeg(fwhmObsDeg, fwhmInstDeg, method = 'none') {
  const obs = Number(fwhmObsDeg);
  const inst = Math.max(0, Number(fwhmInstDeg) || 0);
  if (!Number.isFinite(obs) || obs <= 0) return obs;

  if (!inst || method === 'none') {
    return Math.max(MIN_BROADENING_DEG, obs);
  }

  if (method === 'subtract') {
    return Math.max(MIN_BROADENING_DEG, obs - inst);
  }

  // Gaussian 폭 근사: β_sample² = β_obs² - β_inst²
  const o2 = obs * obs;
  const i2 = inst * inst;
  const s2 = Math.max(0, o2 - i2);
  return Math.max(MIN_BROADENING_DEG, Math.sqrt(s2));
}
