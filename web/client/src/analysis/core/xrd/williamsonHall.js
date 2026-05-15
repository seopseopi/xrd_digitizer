/**
 * 표준 Williamson–Hall (UDM): β cos θ = Kλ/D + 4ε sin θ
 * β: FWHM을 2θ 스케일 라디안으로, θ: Bragg 각(2θ의 절반)
 */

import { correctFwhmTwoThetaDeg } from './scherrerInstrumental.js';

/**
 * 최소제곱 직선 y = a + b x
 * @param {number[]} xs
 * @param {number[]} ys
 */
function linearRegression(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i];
    sumY += ys[i];
    sumXX += xs[i] * xs[i];
    sumXY += xs[i] * ys[i];
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-24) return null;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;

  const meanY = sumY / n;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const yhat = intercept + slope * xs[i];
    ssTot += (ys[i] - meanY) ** 2;
    ssRes += (ys[i] - yhat) ** 2;
  }
  const rSquared = ssTot > 1e-30 ? 1 - ssRes / ssTot : 0;

  return { intercept, slope, rSquared, n };
}

/**
 * @param {Array<{ id?: string, angle: number, fwhm: number }>} peaks - angle: 2θ(°), fwhm: FWHM(2θ)(°)
 * @param {Object} options
 * @param {number} [options.wavelengthAngstrom]
 * @param {number} [options.kScherrer] - Scherrer K
 * @param {number} [options.instrumentalFwhmDeg]
 * @param {'none'|'subtract'|'quadratic'} [options.instrumentalCorrection]
 * @param {number} [options.minFwhmDeg]
 * @returns {Object}
 */
export function fitStandardWilliamsonHall(peaks, options = {}) {
  const {
    wavelengthAngstrom = 1.5406,
    kScherrer = 0.9,
    instrumentalFwhmDeg = 0,
    instrumentalCorrection = 'none',
    minFwhmDeg = 1e-4,
  } = options;

  if (!Array.isArray(peaks) || peaks.length < 2) {
    return {
      success: false,
      error: 'Williamson–Hall 분석에는 유효한 피크가 최소 2개 필요합니다.',
    };
  }

  const points = [];

  for (const p of peaks) {
    if (!p || p.angle == null || p.fwhm == null) continue;
    const twoTheta = Number(p.angle);
    const fwhmObs = Number(p.fwhm);
    if (!Number.isFinite(twoTheta) || twoTheta <= 0) continue;
    if (!Number.isFinite(fwhmObs) || fwhmObs <= 0) continue;

    const fwhmCorr = correctFwhmTwoThetaDeg(
      fwhmObs,
      instrumentalFwhmDeg,
      instrumentalCorrection
    );
    if (fwhmCorr < minFwhmDeg) continue;

    const thetaRad = (twoTheta / 2) * (Math.PI / 180);
    const betaRad = fwhmCorr * (Math.PI / 180);
    const sinTheta = Math.sin(thetaRad);
    const betaCosTheta = betaRad * Math.cos(thetaRad);

    points.push({
      peakId: p.id != null ? String(p.id) : null,
      twoThetaDeg: twoTheta,
      fwhmObsDeg: fwhmObs,
      fwhmCorrectedDeg: fwhmCorr,
      sinTheta,
      betaCosTheta,
    });
  }

  if (points.length < 2) {
    return {
      success: false,
      error: '보정 후 유효한 FWHM을 가진 피크가 2개 미만입니다. 기기 FWHM 또는 피크를 확인하세요.',
      points,
    };
  }

  const xs = points.map((q) => q.sinTheta);
  const ys = points.map((q) => q.betaCosTheta);
  const reg = linearRegression(xs, ys);

  if (!reg) {
    return { success: false, error: '선형 회귀에 실패했습니다.', points };
  }

  const { intercept, slope, rSquared, n } = reg;
  const microstrain = slope / 4;

  let crystalliteSizeNm = null;
  let crystalliteSizeAngstrom = null;
  if (intercept > 1e-12) {
    crystalliteSizeAngstrom = (kScherrer * wavelengthAngstrom) / intercept;
    crystalliteSizeNm = crystalliteSizeAngstrom / 10;
  }

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const steps = 25;
  const fittedLine = [];
  for (let i = 0; i <= steps; i++) {
    const x = xMin + (i / steps) * (xMax - xMin || 1e-6);
    fittedLine.push({ sinTheta: x, betaCosTheta: intercept + slope * x });
  }

  return {
    success: true,
    points,
    regression: {
      interceptBetaCosTheta: intercept,
      slopeBetaCosThetaPerSinTheta: slope,
      rSquared,
      pointCount: n,
    },
    microstrain,
    crystalliteSizeNm,
    crystalliteSizeAngstrom,
    kScherrer,
    wavelengthAngstrom,
    instrumentalFwhmDeg,
    instrumentalCorrection,
    plotData: {
      x: xs,
      y: ys,
      fittedSinTheta: fittedLine.map((f) => f.sinTheta),
      fittedBetaCosTheta: fittedLine.map((f) => f.betaCosTheta),
    },
  };
}
