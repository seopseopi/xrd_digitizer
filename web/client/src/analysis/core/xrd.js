/**
 * XRD 알고리즘 코어
 * 순수 함수만 포함 — DOM, Canvas, React 의존성 없음
 * 입출력 형식: async (input: Object) => { success, data, meta }
 */

import {
  applySavitzkyGolay,
  detectLocalMaxima,
  detectPeaksBySecondDerivative,
  detectPeaks as _detectPeaks,
} from './xrd/peakDetection.js';

import {
  fitPeaks as _fitPeaks,
} from './xrd/peakFitting.js';

import {
  calculateCrystallinity as _calculateCrystallinity,
} from './xrd/crystallinity.js';

import {
  indexPeaksWithStructureInfo as _indexMillerIndices,
  calculateDSpacing,
} from './xrd/millerIndex.js';

import {
  performMWHAnalysis,
  performMWAAnalysis,
  interpretDislocationCharacter,
  estimateLatticeConstantFromPeaks,
  estimateStructureFromPeaks,
} from './xrd/dislocationAnalysis.js';

import { correctFwhmTwoThetaDeg } from './xrd/scherrerInstrumental.js';
import { fitStandardWilliamsonHall } from './xrd/williamsonHall.js';
import {
  identifyPhaseCandidates as runIdentifyPhaseCandidatesAnalysis,
  matchWithCommonPhases as runMatchWithCommonPhases,
  getTheoreticalPeaksForPhase as runGetTheoreticalPeaksForPhase,
  estimateCrystalSystemFromDRatios as runEstimateCrystalSystem,
  COMMON_PHASES,
} from './xrd/phaseIdentification.js';
import { computeTextureIndices as computeTextureIndicesCore } from './xrd/textureIndices.js';
import { estimateQPAPhaseFractions } from './xrd/qpaIntensityRatio.js';
import { fitResidualStressFromPsiScan } from './xrd/residualStressSin2Psi.js';

// 공통 응답 생성 헬퍼
function makeResult(data, startTime) {
  return {
    success: true,
    data,
    meta: {
      processingTimeMs: Date.now() - startTime,
      processedAt: new Date().toISOString(),
      processingLocation: 'local',
    },
  };
}

function makeError(code, message, detail) {
  return {
    success: false,
    error: { code, message, detail: detail || null },
  };
}

const DEFAULT_WAVELENGTH = 1.5406; // Cu Kα (Å)

/**
 * XRD 피크 탐지
 * @param {Object} input
 * @param {Array<{angle: number, intensity: number}>} input.dataPoints
 * @param {Object} [input.options]
 */
export async function detectPeaks(input) {
  const t0 = Date.now();
  try {
    const { dataPoints, options = {} } = input;
    if (!dataPoints || dataPoints.length === 0) {
      return makeError('INVALID_INPUT', 'XRD 데이터가 비어 있습니다');
    }

    const {
      method = 'second_derivative',
      smoothingWindow = 5,
      minPeakHeightPercent = 5,
      minPeakDistanceDeg = 0.3,
    } = options;

    const stepSize = dataPoints.length > 1
      ? Math.abs(dataPoints[1].angle - dataPoints[0].angle)
      : 0.02;
    const minPeakDistancePts = Math.max(1, Math.round(minPeakDistanceDeg / stepSize));

    const peaks = _detectPeaks(dataPoints, {
      smoothingWindow: Math.max(3, smoothingWindow % 2 === 0 ? smoothingWindow + 1 : smoothingWindow),
      smoothingPolyOrder: 2,
      minPeakHeight: minPeakHeightPercent / 100,
      minPeakDistance: minPeakDistancePts,
      method: method === 'second_derivative' ? 'secondDerivative' : 'localMaxima',
    });

    const intensities = dataPoints.map(d => d.intensity);
    const smoothed = applySavitzkyGolay(intensities, smoothingWindow % 2 === 0 ? smoothingWindow + 1 : smoothingWindow, 2, 0);
    const smoothedData = dataPoints.map((d, i) => ({ angle: d.angle, intensity: smoothed[i] }));

    const formattedPeaks = peaks.map((p, i) => ({
      id: `peak_${Date.now()}_${i}`,
      index: p.index,
      angle: p.angle,
      intensity: p.intensity,
      dSpacing: calculateDSpacing(p.angle, DEFAULT_WAVELENGTH),
      isManual: false,
    }));

    return makeResult({ peaks: formattedPeaks, count: formattedPeaks.length, smoothedData }, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * XRD 피크 피팅 (Gaussian/Lorentzian/Voigt)
 * @param {Object} input
 * @param {Array<{angle, intensity}>} input.dataPoints
 * @param {Array<{id, angle, intensity}>} input.peaks
 * @param {Object} [input.options]
 */
export async function fitPeaks(input) {
  const t0 = Date.now();
  try {
    const { dataPoints, peaks, options = {} } = input;
    if (!peaks || peaks.length === 0) {
      return makeError('INVALID_INPUT', '피팅할 피크가 없습니다');
    }

    const { model = 'gaussian', backgroundType = 'linear', fitWindowDeg = 2.0 } = options;

    const result = _fitPeaks(dataPoints, peaks, { model, backgroundType, fitWindowDeg });

    if (!result.success) {
      return makeError('COMPUTATION_ERROR', result.error || '피크 피팅 실패');
    }

    // peakFitting은 fittedCurve, peaks, backgroundParams 반환
    const angles = dataPoints.map(d => d.angle);
    const bgParams = result.backgroundParams || [];
    const backgroundCurve = angles.map(angle => {
      if (backgroundType === 'linear' && bgParams.length >= 2) {
        return bgParams[0] * angle + bgParams[1];
      }
      if (backgroundType === 'polynomial' && bgParams.length > 0) {
        return bgParams.reduce((sum, c, i) => sum + c * Math.pow(angle, i), 0);
      }
      return 0;
    });

    const fittedPeaks = (result.peaks || []).map((p, i) => ({
      ...p,
      id: peaks[i]?.id || `peak_${i}`,
      intensity: peaks[i]?.intensity ?? p.height,
      index: p.originalIndex ?? (() => {
        let bestIdx = 0;
        let minDiff = Infinity;
        dataPoints.forEach((d, idx) => {
          const diff = Math.abs(d.angle - p.angle);
          if (diff < minDiff) { minDiff = diff; bestIdx = idx; }
        });
        return bestIdx;
      })(),
    }));

    return makeResult(
      {
        fittedPeaks,
        fittedCurve: result.fittedCurve || [],
        fittingCurve: result.fittedCurve || [],
        backgroundCurve: angles.map((angle, i) => ({ angle, intensity: backgroundCurve[i] })),
      },
      t0
    );
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 결정화도 계산
 * @param {Object} input
 * @param {Array<{angle, intensity}>} input.dataPoints
 * @param {Array<{id, area, angle}>} input.fittedPeaks
 * @param {Array<{angle, intensity}>} input.backgroundCurve
 */
export async function calculateCrystallinity(input) {
  const t0 = Date.now();
  try {
    const { dataPoints, fittedPeaks, backgroundCurve } = input;
    if (!fittedPeaks || fittedPeaks.length === 0) {
      return makeError('INVALID_INPUT', '피팅된 피크가 없습니다');
    }

    const result = _calculateCrystallinity(dataPoints, fittedPeaks, backgroundCurve);

    // crystallinity.js는 숫자(0-1)를 직접 반환함
    const crystallinityValue = typeof result === 'number' ? result : (result?.crystallinity ?? 0);

    return makeResult(
      {
        crystallinity: crystallinityValue,
        crystallineArea: result?.crystallineArea ?? 0,
        amorphousArea: result?.amorphousArea ?? 0,
        totalArea: result?.totalArea ?? 0,
      },
      t0
    );
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * Scherrer 식을 이용한 결정자(crystallite) 크기 계산
 * @param {Object} input
 * @param {Array<{id, angle, fwhm}>} input.fittedPeaks
 * @param {number} [input.wavelength] - Å, 기본 1.5406
 * @param {number} [input.shapeFactor] - K, 기본 0.9
 * @param {number} [input.instrumentalFwhmDeg] - 기기 FWHM (2θ, °)
 * @param {'none'|'subtract'|'quadratic'} [input.instrumentalCorrection]
 */
export async function calculateCrystalliteSizes(input) {
  const t0 = Date.now();
  try {
    const {
      fittedPeaks,
      wavelength = DEFAULT_WAVELENGTH,
      shapeFactor = 0.9,
      instrumentalFwhmDeg = 0,
      instrumentalCorrection = 'none',
    } = input;
    if (!fittedPeaks || fittedPeaks.length === 0) {
      return makeError('INVALID_INPUT', '피팅된 피크가 없습니다');
    }

    const crystalliteSizes = fittedPeaks
      .filter(p => p.fwhm > 0 && p.angle > 0)
      .map(p => {
        const fwhmCorr = correctFwhmTwoThetaDeg(
          p.fwhm,
          instrumentalFwhmDeg,
          instrumentalCorrection
        );
        const thetaRad = (p.angle / 2) * (Math.PI / 180);
        const fwhmRad = fwhmCorr * (Math.PI / 180);
        const sizeNm = (shapeFactor * wavelength) / (fwhmRad * Math.cos(thetaRad)) / 10;
        return {
          peakId: p.id,
          angle: p.angle,
          fwhmObsDeg: p.fwhm,
          fwhmCorrectedDeg: fwhmCorr,
          sizeNm,
        };
      });

    const sizes = crystalliteSizes.map(s => s.sizeNm);
    const avg = sizes.reduce((a, b) => a + b, 0) / sizes.length;
    const sorted = [...sizes].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const std = Math.sqrt(sizes.reduce((s, v) => s + (v - avg) ** 2, 0) / sizes.length);

    return makeResult(
      {
        crystalliteSizes,
        averageSizeNm: avg,
        medianSizeNm: median,
        stdSizeNm: std,
        instrumentalFwhmDeg,
        instrumentalCorrection,
      },
      t0
    );
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 표준 Williamson–Hall: β cos θ = Kλ/D + 4ε sin θ
 * @param {Object} input
 * @param {Array<{id?, angle, fwhm}>} input.fittedPeaks
 * @param {number} [input.wavelength]
 * @param {number} [input.shapeFactor] - Scherrer K
 * @param {number} [input.instrumentalFwhmDeg]
 * @param {'none'|'subtract'|'quadratic'} [input.instrumentalCorrection]
 */
export async function williamsonHallFit(input) {
  const t0 = Date.now();
  try {
    const {
      fittedPeaks,
      wavelength = DEFAULT_WAVELENGTH,
      shapeFactor = 0.9,
      instrumentalFwhmDeg = 0,
      instrumentalCorrection = 'none',
    } = input;

    if (!fittedPeaks || fittedPeaks.length < 2) {
      return makeError('INVALID_INPUT', 'Williamson–Hall 분석에는 피크가 최소 2개 필요합니다');
    }

    const result = fitStandardWilliamsonHall(fittedPeaks, {
      wavelengthAngstrom: wavelength,
      kScherrer: shapeFactor,
      instrumentalFwhmDeg,
      instrumentalCorrection,
    });

    if (!result.success) {
      return makeError(
        'INSUFFICIENT_DATA',
        result.error || 'Williamson–Hall 피팅에 실패했습니다',
        result.points ? { points: result.points } : null
      );
    }

    return makeResult(result, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 밀러지수 인덱싱
 * @param {Object} input
 * @param {Array<{id, angle, dSpacing}>} input.fittedPeaks
 * @param {Object} [input.structureInfo]
 * @param {number} [input.wavelength]
 * @param {Object} [input.options]
 */
export async function indexMillerIndices(input) {
  const t0 = Date.now();
  try {
    const { fittedPeaks, structureInfo, wavelength = DEFAULT_WAVELENGTH, options = {} } = input;
    if (!fittedPeaks || fittedPeaks.length === 0) {
      return makeError('INVALID_INPUT', '피팅된 피크가 없습니다');
    }

    const { maxHKL = 10, dSpacingTolerancePercent = 2.0 } = options;

    const result = _indexMillerIndices(fittedPeaks, structureInfo, wavelength, {
      maxHKL,
      tolerancePercent: dSpacingTolerancePercent,
    });

    // indexPeaksWithStructureInfo는 배열을 직접 반환함 (객체가 아님)
    const indexedPeaks = Array.isArray(result) ? result : (result?.indexedPeaks || []);

    return makeResult(
      {
        indexedPeaks,
        latticeParameterEstimate: result?.latticeParameterEstimate || { a: null, b: null, c: null },
      },
      t0
    );
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 전위 밀도 분석 (mWH/mWA)
 * @param {Object} input
 * @param {Array<{peakId, angle, fwhm, h, k, l}>} input.indexedPeaks
 * @param {Object} input.materialConstants
 * @param {number} [input.wavelength]
 */
export async function analyzeDislocation(input) {
  const t0 = Date.now();
  try {
    const { indexedPeaks, materialConstants, wavelength = DEFAULT_WAVELENGTH, method = 'mwh', xrdData = null, qFromMwh = null } = input;
    const wavelengthNm = Number(wavelength) > 1 ? Number(wavelength) / 10 : Number(wavelength);
    if (!indexedPeaks || indexedPeaks.length < 3) {
      return makeError('INSUFFICIENT_DATA', '전위 밀도 분석을 위해 최소 3개의 인덱싱된 피크가 필요합니다');
    }

    const constants = materialConstants || {
      structure: estimateStructureFromPeaks(indexedPeaks),
      latticeConstant: estimateLatticeConstantFromPeaks(indexedPeaks, wavelengthNm),
    };

    let result;
    if (method === 'mwa') {
      if (!xrdData || xrdData.length === 0) {
        return makeError('INVALID_INPUT', 'mWA 분석을 위해 XRD 원본 데이터(xrdData)가 필요합니다');
      }
      result = performMWAAnalysis(indexedPeaks, xrdData, constants, wavelengthNm, qFromMwh);
    } else {
      result = performMWHAnalysis(indexedPeaks, constants, wavelengthNm);
    }

    const character = interpretDislocationCharacter(
      result?.q ?? null,
      (constants?.structure ?? 'fcc').toString().toLowerCase()
    );

    return makeResult({ ...result, dislocationCharacter: character }, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 상 동정: Cubic(FCC/BCC/SC) · HCP(c/a 범위 탐색) · Tetragonal(P/I)
 * 소멸 조건 완전 적용, COMMON_PHASES 프리셋 매칭 포함
 *
 * @param {Object} input
 * @param {Array<{ angle: number, intensity?: number }>} input.peaks
 * @param {number} [input.wavelength]
 * @param {Object} [input.options]
 * @param {number}  [input.options.angleToleranceDeg=0.12]
 * @param {boolean} [input.options.includeTetragonal=true]
 * @param {boolean} [input.options.includeCommonPhases=true]
 */
export async function identifyPhaseCandidates(input) {
  const t0 = Date.now();
  try {
    const { peaks, wavelength = DEFAULT_WAVELENGTH, options = {} } = input;
    if (!peaks || peaks.length < 2) {
      return makeError('INVALID_INPUT', '상 동정을 위해 피크가 최소 2개 필요합니다');
    }
    const result = runIdentifyPhaseCandidatesAnalysis(peaks, wavelength, options);
    if (!result.success) {
      return makeError('INSUFFICIENT_DATA', result.error || '상 동정에 실패했습니다');
    }
    return makeResult(result, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * COMMON_PHASES 프리셋과 직접 매칭 (빠른 참조용)
 * @param {Object} input
 * @param {Array<{ angle: number, intensity?: number }>} input.peaks
 * @param {number} [input.wavelength]
 * @param {number} [input.options.tolDeg=0.15]
 */
export async function matchPhasesWithPresets(input) {
  const t0 = Date.now();
  try {
    const { peaks, wavelength = DEFAULT_WAVELENGTH, options = {} } = input;
    if (!peaks || peaks.length < 2) {
      return makeError('INVALID_INPUT', '피크가 최소 2개 필요합니다');
    }
    const results = runMatchWithCommonPhases(peaks, wavelength, options.tolDeg ?? 0.15);
    return makeResult({ presetMatches: results, topMatch: results[0] ?? null }, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 특정 재료(COMMON_PHASES 키)의 이론 피크 목록 반환
 * — XRD 패턴 오버레이 또는 참조 비교에 사용
 *
 * @param {Object} input
 * @param {string} input.phaseName  — COMMON_PHASES의 키 (예: 'Fe-BCC (α-Fe)')
 * @param {number} [input.wavelength]
 * @param {number} [input.twoThetaMin=5]
 * @param {number} [input.twoThetaMax=100]
 */
export async function getTheoreticalPeaksForPhase(input) {
  const t0 = Date.now();
  try {
    const { phaseName, wavelength = DEFAULT_WAVELENGTH, twoThetaMin = 5, twoThetaMax = 100 } = input;
    if (!phaseName) return makeError('INVALID_INPUT', 'phaseName이 필요합니다');
    const peaks = runGetTheoreticalPeaksForPhase(phaseName, wavelength, twoThetaMin, twoThetaMax);
    const phase = COMMON_PHASES[phaseName] ?? null;
    return makeResult({ phaseName, phase, theoreticalPeaks: peaks, count: peaks.length }, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * d-비율 지문으로 결정계 사전 추정
 * @param {Object} input
 * @param {Array<{ angle: number }>} input.peaks
 * @param {number} [input.wavelength]
 */
export async function estimateCrystalSystemHint(input) {
  const t0 = Date.now();
  try {
    const { peaks, wavelength = DEFAULT_WAVELENGTH } = input;
    const dValues = [...peaks]
      .map(p => {
        const sinT = Math.sin((p.angle / 2) * Math.PI / 180);
        return sinT > 0 ? wavelength / (2 * sinT) : null;
      })
      .filter(Boolean)
      .sort((a, b) => b - a);
    const hint = runEstimateCrystalSystem(dValues);
    return makeResult(hint, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 배향 지수 (I_meas / I_ref 기하평균 정규화)
 * @param {Object} input
 * @param {Array<{ label?: string, intensityMeasured: number, intensityReference: number }>} input.rows
 */
export async function computeTextureIndicesAnalysis(input) {
  const t0 = Date.now();
  try {
    const { rows } = input;
    const result = computeTextureIndicesCore(rows || []);
    if (!result.success) {
      return makeError('INVALID_INPUT', result.error || '배향 지수 계산 실패');
    }
    return makeResult(result, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 반정량 상 분율 (I/RIR 근사)
 * @param {Object} input
 * @param {Array<{ phaseId: string, rir: number, intensity?: number, integratedIntensity?: number }>} input.phases
 */
export async function estimateQPAPhaseFractionsAnalysis(input) {
  const t0 = Date.now();
  try {
    const { phases } = input;
    const result = estimateQPAPhaseFractions(phases || []);
    if (!result.success) {
      return makeError('INVALID_INPUT', result.error || 'QPA 계산 실패');
    }
    return makeResult(result, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * sin²ψ 잔류 응력 (d–sin²ψ 선형 모델)
 * @param {Object} input
 * @param {Array<{ psiDeg: number, twoThetaDeg: number }>} input.points
 * @param {number} [input.wavelength]
 * @param {{ youngModulusGPa: number, poissonRatio: number }} input.elastic
 */
export async function fitResidualStressSin2Psi(input) {
  const t0 = Date.now();
  try {
    const { points, wavelength = DEFAULT_WAVELENGTH, elastic } = input;
    const result = fitResidualStressFromPsiScan(points, wavelength, elastic);
    if (!result.success) {
      return makeError('INVALID_INPUT', result.error || '응력 피팅 실패');
    }
    return makeResult(result, t0);
  } catch (err) {
    return makeError('COMPUTATION_ERROR', err.message, err.stack);
  }
}

/**
 * 리트벨트·외부 정밀화 안내 (앱 내 미구현)
 */
export async function getRietveldGuidance(input) {
  const t0 = Date.now();
  void input;
  return makeResult(
    {
      implementedInApp: false,
      summary:
        '전체 패턴 리트벨트 정밀화는 본 앱 범위를 넘습니다. FullProf Suite, GSAS-II, TOPAS, MAUD 등과 ICDD PDF(라이선스 준수) 워크플로를 권장합니다.',
      notes: [
        '잔류 오스테나이트 분율 등은 리트벨트 또는 전체 패턴 피팅이 업계 표준에 가깝습니다.',
        '반정량 보조는 "반정량 상분율" 탭의 I/RIR 근사를 사용할 수 있습니다.',
      ],
    },
    t0
  );
}

// 저수준 유틸리티 재export
export {
  COMMON_PHASES,
  applySavitzkyGolay,
  detectLocalMaxima,
  detectPeaksBySecondDerivative,
  calculateDSpacing,
};
