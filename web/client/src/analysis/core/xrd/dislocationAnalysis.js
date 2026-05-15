// fft-js 라이브러리 대신 직접 구현한 푸리에 변환 사용

/**
 * Cubic orientation parameter Γ 계산
 * Γ = (h²k² + h²l² + k²l²) / (h² + k² + l²)²
 * 문헌의 mWH 식에서 Γ는 0~1/3 범위를 갖는다.
 * @param {number} h - 밀러지수 h
 * @param {number} k - 밀러지수 k
 * @param {number} l - 밀러지수 l
 * @returns {number} Γ 값
 */
export const calculateGamma = (h, k, l) => {
  const h2 = h * h;
  const k2 = k * k;
  const l2 = l * l;
  const denominator = h2 + k2 + l2;
  
  if (denominator === 0) return 0;
  
  const numerator = h2 * k2 + h2 * l2 + k2 * l2;
  return numerator / (denominator * denominator);
};

// 기존 호출부 호환을 위해 이름은 유지하되, 반환값은 문헌식 Γ로 정정한다.
export const calculateH2 = calculateGamma;

/**
 * K 파라미터 계산
 * K = 2sin(θ) / λ
 * @param {number} angle - 2θ 각도 (도)
 * @param {number} wavelength - X선 파장 (nm)
 * @returns {number} K 값 (nm⁻¹)
 */
export const calculateK = (angle, wavelength) => {
  const thetaRad = (angle / 2) * Math.PI / 180; // 2θ를 θ로 변환 후 라디안
  return (2 * Math.sin(thetaRad)) / wavelength;
};

/**
 * ΔK 파라미터 계산
 * ΔK = 2cos(θ)(Δθ) / λ
 * @param {number} angle - 2θ 각도 (도)
 * @param {number} fwhm - 반가폭 (도)
 * @param {number} wavelength - X선 파장 (nm)
 * @returns {number} ΔK 값 (nm⁻¹)
 */
export const calculateDeltaK = (angle, fwhm, wavelength) => {
  const thetaRad = (angle / 2) * Math.PI / 180;
  const deltaThetaRad = fwhm * Math.PI / 180; // FWHM을 라디안으로 변환 (FWHM은 이미 전체 폭)
  return (2 * Math.cos(thetaRad) * deltaThetaRad) / wavelength;
};

/**
 * 버거스 벡터 크기 계산
 * @param {number} latticeConstant - 격자 상수 (Å)
 * @param {string} structure - 결정 구조 ('fcc' 또는 'bcc')
 * @returns {number} 버거스 벡터 크기 (nm)
 */
export const calculateBurgersVector = (latticeConstant, structure) => {
  const a = latticeConstant * 0.1; // Å를 nm로 변환
  
  if (structure.toLowerCase() === 'fcc') {
    // fcc: b = a/√2
    return a / Math.sqrt(2);
  } else if (structure.toLowerCase() === 'bcc') {
    // bcc: b = a√3/2
    return a * Math.sqrt(3) / 2;
  } else {
    throw new Error(`지원되지 않는 결정 구조: ${structure}`);
  }
};

/**
 * 대비 인자 (Contrast Factor) 계산
 * C̄_hkl = C̄_h00 (1 - qΓ)
 * @param {number} h - 밀러지수 h
 * @param {number} k - 밀러지수 k
 * @param {number} l - 밀러지수 l
 * @param {number} q - 전위 특성 파라미터
 * @param {number} Ch00 - 평균 대비 인자 (h00)
 * @returns {number} 대비 인자
 */
export const calculateContrastFactor = (h, k, l, q, Ch00) => {
  const gamma = calculateGamma(h, k, l);
  return Ch00 * (1 - q * gamma);
};

/**
 * XRD 피크로부터 격자 상수 추정 (Cubic 구조)
 * @param {Array<{angle: number, millerIndices: Array, dSpacing?: number}>} peaks - 피크 데이터
 * @param {number} wavelength - X선 파장 (nm)
 * @returns {number|null} 추정된 격자 상수 (Å), 추정 실패 시 null
 */
export const estimateLatticeConstantFromPeaks = (peaks, wavelength) => {
  // 유효한 밀러지수가 있는 피크만 필터링
  const validPeaks = peaks.filter(peak => 
    peak.millerIndices && 
    peak.millerIndices.length > 0 &&
    peak.angle &&
    peak.angle > 0
  );

  if (validPeaks.length < 2) {
    return null;
  }

  // 각 피크에 대해 격자 상수 추정
  const estimates = [];
  
  for (const peak of validPeaks) {
    const millerIndex = peak.millerIndices[0];
    const { h, k, l } = millerIndex;
    
    if (h === undefined || k === undefined || l === undefined) continue;
    
    // d-spacing 계산 (이미 있으면 사용, 없으면 계산)
    let dSpacing = peak.dSpacing;
    if (!dSpacing && peak.angle) {
      const thetaRad = (peak.angle / 2) * Math.PI / 180;
      dSpacing = (wavelength * 10) / (2 * Math.sin(thetaRad)); // nm를 Å로 변환
    }
    
    if (!dSpacing || dSpacing <= 0) continue;
    
    // Cubic 구조: d = a / √(h² + k² + l²)
    // 따라서 a = d * √(h² + k² + l²)
    const hklSum = h * h + k * k + l * l;
    if (hklSum > 0) {
      const a = dSpacing * Math.sqrt(hklSum);
      estimates.push(a);
    }
  }
  
  if (estimates.length === 0) {
    return null;
  }
  
  // 평균값 반환 (이상치 제거를 위해 중앙값 사용)
  estimates.sort((a, b) => a - b);
  const median = estimates.length % 2 === 0
    ? (estimates[estimates.length / 2 - 1] + estimates[estimates.length / 2]) / 2
    : estimates[Math.floor(estimates.length / 2)];
  
  return median;
};

/**
 * XRD 피크 패턴으로부터 결정 구조 추정
 * @param {Array<{angle: number, millerIndices: Array}>} peaks - 피크 데이터
 * @returns {string|null} 추정된 결정 구조 ('fcc' 또는 'bcc'), 추정 실패 시 null
 */
export const estimateStructureFromPeaks = (peaks) => {
  // 유효한 밀러지수가 있는 피크만 필터링
  const validPeaks = peaks.filter(peak => 
    peak.millerIndices && 
    peak.millerIndices.length > 0
  );

  if (validPeaks.length < 2) {
    return null;
  }

  // 밀러지수 패턴 분석
  const hklValues = validPeaks.map(peak => {
    const { h, k, l } = peak.millerIndices[0];
    return { h: Math.abs(h || 0), k: Math.abs(k || 0), l: Math.abs(l || 0) };
  });

  // FCC 특징: (111), (200), (220), (311), (222), (400) 등
  // BCC 특징: (110), (200), (211), (220), (310), (222) 등
  
  // (111) 피크가 있으면 FCC 가능성 높음
  const has111 = hklValues.some(hkl => hkl.h === 1 && hkl.k === 1 && hkl.l === 1);
  
  // (110) 피크가 있고 (111)이 없으면 BCC 가능성 높음
  const has110 = hklValues.some(hkl => hkl.h === 1 && hkl.k === 1 && hkl.l === 0);
  
  if (has111 && !has110) {
    return 'fcc';
  } else if (has110 && !has111) {
    return 'bcc';
  }
  
  // 기본값으로 FCC 반환 (더 일반적)
  return 'fcc';
};

/**
 * 탄성 상수 기본값 반환
 * @param {string} structure - 결정 구조 ('fcc' 또는 'bcc')
 * @returns {Object} 탄성 상수 {C11, C12, C44} (GPa)
 */
export const getDefaultElasticConstants = (structure) => {
  if (structure.toLowerCase() === 'fcc') {
    // 오스테나이트강 (SUS304) 기본값
    return {
      C11: 204.6, // GPa
      C12: 137.7, // GPa
      C44: 126.2  // GPa
    };
  } else if (structure.toLowerCase() === 'bcc') {
    // 순수 철 (α-Fe) 기본값
    return {
      C11: 231.4, // GPa
      C12: 134.7, // GPa
      C44: 116.4  // GPa
    };
  } else {
    throw new Error(`지원되지 않는 결정 구조: ${structure}`);
  }
};

const DISLOCATION_CONTRAST_PRESETS = {
  bcc: {
    source: 'bcc Fe preset (Ungar/ISIJ values)',
    qEdge: 1.310,
    qScrew: 2.647,
    Ch00Edge: 0.256,
    Ch00Screw: 0.305,
  },
  fcc: {
    source: 'generic fcc preset; verify with material-specific contrast factors',
    qEdge: 1.71,
    qScrew: 2.46,
    Ch00Edge: 0.256,
    Ch00Screw: 0.305,
  },
};

const clamp01 = (value) => Math.max(0, Math.min(1, value));

export const getDislocationContrastPreset = (structure, contrastFactors = null) => {
  const key = (structure || 'fcc').toString().toLowerCase();
  const preset = DISLOCATION_CONTRAST_PRESETS[key] || DISLOCATION_CONTRAST_PRESETS.fcc;
  const merged = {
    ...preset,
    ...(contrastFactors || {}),
  };

  const qEdge = Number(merged.qEdge);
  const qScrew = Number(merged.qScrew);
  const Ch00Edge = Number(merged.Ch00Edge);
  const Ch00Screw = Number(merged.Ch00Screw);

  return {
    ...merged,
    qEdge,
    qScrew,
    Ch00Edge,
    Ch00Screw,
    min: Math.min(qEdge, qScrew),
    max: Math.max(qEdge, qScrew),
    edge: qEdge,
    screw: qScrew,
  };
};

export const calculateScrewFractionFromQ = (q, contrastPreset) => {
  const { qEdge, qScrew } = contrastPreset;
  const range = qScrew - qEdge;
  if (!isFinite(q) || !isFinite(range) || Math.abs(range) < 1e-12) {
    return 0.5;
  }
  return clamp01((q - qEdge) / range);
};

export const calculateEffectiveCh00 = (q, structure, contrastFactors = null) => {
  const preset = getDislocationContrastPreset(structure, contrastFactors);
  const screwFraction = calculateScrewFractionFromQ(q, preset);
  return preset.Ch00Edge * (1 - screwFraction) + preset.Ch00Screw * screwFraction;
};

/**
 * 평균 대비 인자 Ch00 계산
 * @param {Object} contrastFactors - 대비 인자 또는 과거 탄성 상수 객체
 * @param {string} structure - 결정 구조 ('fcc' 또는 'bcc')
 * @returns {number} Ch00 값
 */
export const calculateCh00 = (contrastFactors, structure, q = null) => {
  if (contrastFactors?.Ch00 != null) return Number(contrastFactors.Ch00);
  if (contrastFactors?.Ch00Edge != null || contrastFactors?.Ch00Screw != null) {
    return calculateEffectiveCh00(q, structure, contrastFactors);
  }
  return calculateEffectiveCh00(q, structure, null);
};

/**
 * 이론적 q 범위 반환
 * @param {string} structure - 결정 구조 ('fcc' 또는 'bcc')
 * @returns {Object} {min: number, max: number, screw: number, edge: number}
 */
export const getTheoreticalQRange = (structure) => {
  return getDislocationContrastPreset(structure);
};

/**
 * 피크 주변 프로파일 추출
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 데이터
 * @param {number} peakCenter - 피크 중심 각도 (도)
 * @param {number} peakWidth - 피크 너비 (도, 기본값: 5도)
 * @returns {Array<{angle: number, intensity: number}>} 추출된 프로파일
 */
export const extractPeakProfile = (xrdData, peakCenter, peakWidth = 5) => {
  const halfWidth = peakWidth / 2;
  const minAngle = peakCenter - halfWidth;
  const maxAngle = peakCenter + halfWidth;
  
  return xrdData.filter(point => 
    point.angle >= minAngle && point.angle <= maxAngle
  );
};

/**
 * 푸리에 변환 수행 (Warren-Averbach 분석용)
 * 각 L 값에 대한 푸리에 계수 A(L) 계산
 * Warren-Averbach 방법에서는 실수부 푸리에 계수만 필요
 * @param {Array<number>} profile - 강도 프로파일 배열
 * @param {number} L - 푸리에 변수 (L = na³, nm)
 * @param {number} dSpacing - d-spacing (Å)
 * @returns {Object} {real: number, imag: number, magnitude: number} 푸리에 계수
 */
export const fourierTransform = (profile, L, dSpacing) => {
  if (!profile || profile.length === 0 || !dSpacing || dSpacing <= 0 || !L || L <= 0) {
    return { real: 0, imag: 0, magnitude: 0 };
  }
  
  const N = profile.length;
  if (N < 2) {
    const val = profile[0] || 0;
    return { real: val, imag: 0, magnitude: Math.abs(val) };
  }
  
  // Warren-Averbach 분석에서 푸리에 계수 계산
  // A(L) = (1/N) * Σ I(s) * cos(2πsL)
  // 여기서 s = 2sin(θ)/λ = 1/d (역격자 공간)
  const s = 1 / dSpacing; // Å⁻¹
  
  let realSum = 0;
  let imagSum = 0;
  
  // 간단한 이산 푸리에 변환 (DFT)
  // 각 데이터 포인트에 대해 코사인/사인 성분 계산
  for (let i = 0; i < N; i++) {
    const intensity = profile[i] || 0;
    if (!isFinite(intensity)) continue;
    
    // 푸리에 변수에 해당하는 주파수 성분 계산
    // L에 해당하는 위상: 2π * s * L * (i/N)
    const normalizedIndex = i / N;
    const phase = 2 * Math.PI * s * L * normalizedIndex;
    
    realSum += intensity * Math.cos(phase);
    imagSum += intensity * Math.sin(phase);
  }
  
  const real = realSum / N;
  const imag = imagSum / N;
  const magnitude = Math.sqrt(real * real + imag * imag);
  
  // 유효하지 않은 값 체크
  if (!isFinite(real) || !isFinite(imag) || !isFinite(magnitude)) {
    return { real: 0, imag: 0, magnitude: 0 };
  }
  
  return { real, imag, magnitude };
};

/**
 * 최소 제곱법으로 선형 회귀 수행
 * @param {Array<number>} x - x 데이터
 * @param {Array<number>} y - y 데이터
 * @returns {Object} {slope: number, intercept: number, rSquared: number}
 */
const linearRegression = (x, y) => {
  const n = x.length;
  if (n === 0) {
    return { slope: 0, intercept: 0, rSquared: 0 };
  }
  
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  
  // R² 계산
  const yMean = sumY / n;
  const ssRes = y.reduce((sum, yi, i) => {
    const predicted = slope * x[i] + intercept;
    return sum + Math.pow(yi - predicted, 2);
  }, 0);
  const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
  const rSquared = ssTot === 0 ? 0 : 1 - (ssRes / ssTot);
  
  return { slope, intercept, rSquared };
};

/**
 * Modified Williamson-Hall (mWH) 분석 수행
 * @param {Array<{angle: number, fwhm: number, millerIndices: Array}>} peaks - 피크 데이터
 * @param {Object} materialConstants - 재료 상수
 * @param {string} materialConstants.structure - 결정 구조
 * @param {number} materialConstants.latticeConstant - 격자 상수 (Å)
 * @param {Object} materialConstants.elasticConstants - 탄성 상수
 * @param {number} wavelength - X선 파장 (nm)
 * @returns {Object} mWH 분석 결과
 */
export const performMWHAnalysis = (peaks, materialConstants, wavelength) => {
  const { structure, contrastFactors } = materialConstants;
  
  // 유효한 밀러지수가 있는 피크만 필터링
  const validPeaks = peaks.filter(peak => 
    peak.millerIndices && 
    peak.millerIndices.length > 0 && 
    peak.fwhm && 
    peak.fwhm > 0
  );
  
  if (validPeaks.length < 3) {
    throw new Error('mWH 분석을 위해서는 최소 3개의 유효한 피크가 필요합니다.');
  }
  
  const contrastPreset = getDislocationContrastPreset(structure, contrastFactors);
  
  // 각 피크에 대해 K, ΔK, Γ 계산
  const dataPoints = [];
  
  for (const peak of validPeaks) {
    const millerIndex = peak.millerIndices[0]; // 첫 번째 밀러지수 사용
    const { h, k, l } = millerIndex;
    
    if (h === undefined || k === undefined || l === undefined) continue;
    
    const K = calculateK(peak.angle, wavelength);
    const deltaK = calculateDeltaK(peak.angle, peak.fwhm, wavelength);
    const gamma = calculateGamma(h, k, l);
    
    if (K > 0 && deltaK > 0) {
      dataPoints.push({
        h, k, l,
        angle: peak.angle,
        fwhm: peak.fwhm,
        K,
        deltaK,
        gamma,
        H2: gamma
      });
    }
  }
  
  if (dataPoints.length < 3) {
    throw new Error('유효한 데이터 포인트가 부족합니다.');
  }
  
  // α 값을 조정하여 선형 관계 최적화
  let bestAlpha = 0;
  let bestRSquared = -Infinity;
  let bestSlope = 0;
  let bestIntercept = 0;
  let bestPlotData = { x: [], y: [] };
  
  // α 범위 탐색 (0부터 시작하여 증가)
  // deltaK²의 최대값을 기준으로 범위 설정
  const maxDeltaK2 = Math.max(...dataPoints.map(d => Math.pow(d.deltaK, 2)));
  const alphaMax = Math.min(maxDeltaK2 * 0.9, 0.01);
  const alphaRange = [];
  for (let alpha = 0; alpha <= alphaMax; alpha += alphaMax / 100) {
    alphaRange.push(alpha);
  }
  
  for (const alpha of alphaRange) {
    const x = dataPoints.map(d => d.gamma);
    const y = dataPoints.map(d => {
      const numerator = Math.pow(d.deltaK, 2) - alpha;
      return numerator > 0 ? numerator / Math.pow(d.K, 2) : 0;
    });
    
    // 유효한 데이터만 사용
    const validIndices = y.map((yi, i) => yi > 0 && isFinite(yi) ? i : -1).filter(i => i >= 0);
    if (validIndices.length < 3) continue;
    
    const validX = validIndices.map(i => x[i]);
    const validY = validIndices.map(i => y[i]);
    
    const regression = linearRegression(validX, validY);
    
    // R²가 유효하고 q 산출 가능한 기울기/절편인 경우만 고려
    if (
      isFinite(regression.rSquared) &&
      regression.rSquared > bestRSquared &&
      regression.slope < 0 &&
      regression.intercept > 0
    ) {
      bestRSquared = regression.rSquared;
      bestAlpha = alpha;
      bestSlope = regression.slope;
      bestIntercept = regression.intercept;
      const slope = regression.slope;
      const intercept = regression.intercept;
      bestPlotData = {
        x: validX,
        y: validY,
        fittedY: validX.map(xi => slope * xi + intercept)
      };
    }
  }
  
  // 최적화 실패 시 기본값 사용 (α = 0)
  if (bestRSquared === -Infinity || !isFinite(bestRSquared)) {
    // α = 0으로 직접 회귀 수행
    const x = dataPoints.map(d => d.gamma);
    const y = dataPoints.map(d => Math.pow(d.deltaK, 2) / Math.pow(d.K, 2));
    
    const regression = linearRegression(x, y);
    if (isFinite(regression.rSquared)) {
      bestRSquared = regression.rSquared;
      bestAlpha = 0;
      bestSlope = regression.slope;
      bestIntercept = regression.intercept;
      bestPlotData = {
        x: x,
        y: y,
        fittedY: x.map(xi => bestSlope * xi + bestIntercept)
      };
    }
  }
  
  // mWH 방정식: Y = [(ΔK)² - α]/K² = φ² C_h00 (1 - qΓ)
  // 선형화하면 intercept = φ² C_h00, slope = -φ² C_h00 q 이므로 q = -slope / intercept.
  const warnings = [];
  const qualityFlags = {
    nonPositiveIntercept: !(bestIntercept > 0),
    nonNegativeSlope: !(bestSlope < 0),
    qOutOfRange: false,
    lowRSquared: bestRSquared < 0.9,
  };

  let q = null;
  if (bestIntercept > 0 && bestSlope < 0 && isFinite(bestSlope) && isFinite(bestIntercept)) {
    q = -bestSlope / bestIntercept;
  } else {
    warnings.push('mWH 회귀의 절편/기울기 부호가 문헌식과 맞지 않아 q를 신뢰할 수 없습니다.');
  }

  if (q != null && isFinite(q) && (q < contrastPreset.min || q > contrastPreset.max)) {
    qualityFlags.qOutOfRange = true;
    warnings.push(`q=${q.toFixed(3)}가 기준 범위(${contrastPreset.min.toFixed(3)}~${contrastPreset.max.toFixed(3)}) 밖입니다.`);
  }
  if (qualityFlags.lowRSquared) {
    warnings.push(`mWH 회귀 R²=${bestRSquared.toFixed(4)}로 낮아 전위 특성 해석은 참고값입니다.`);
  }

  const screwFraction = q != null && isFinite(q)
    ? calculateScrewFractionFromQ(q, contrastPreset)
    : 0.5;
  const Ch00 = calculateEffectiveCh00(q, structure, contrastFactors);
  const phi = bestIntercept > 0 && Ch00 > 0 ? Math.sqrt(bestIntercept / Ch00) : null;
  
  // 결정 크기 D 계산: α = (0.9/D)²
  const D = bestAlpha > 0 ? 0.9 / Math.sqrt(bestAlpha) : Infinity;
  
  return {
    q,
    qRaw: q,
    alpha: bestAlpha,
    D: D === Infinity ? null : D * wavelength * 10, // nm로 변환
    Ch00,
    phi,
    screwFraction,
    edgeFraction: 1 - screwFraction,
    contrastPreset,
    warnings,
    qualityFlags,
    plotData: bestPlotData,
    rSquared: bestRSquared,
    dataPoints
  };
};

/**
 * Modified Warren-Averbach (mWA) 분석 수행
 * @param {Array<{angle: number, fwhm: number, millerIndices: Array}>} peaks - 피크 데이터
 * @param {Array<{angle: number, intensity: number}>} xrdData - 원본 XRD 데이터
 * @param {Object} materialConstants - 재료 상수
 * @param {number} wavelength - X선 파장 (nm)
 * @param {number} q - mWH에서 구한 q 파라미터
 * @returns {Object} mWA 분석 결과
 */
export const performMWAAnalysis = (peaks, xrdData, materialConstants, wavelength, q) => {
  const { structure, latticeConstant, contrastFactors } = materialConstants;
  
  // q가 없으면 이론적 중간값 사용 (대비 인자 계산용)
  const qRange = getTheoreticalQRange(structure);
  const qEffective = (q != null && isFinite(q)) ? q : (qRange.min + qRange.max) / 2;
  
  // 유효한 밀러지수가 있는 피크만 필터링
  const validPeaks = peaks.filter(peak => 
    peak.millerIndices && 
    peak.millerIndices.length > 0
  );
  
  if (validPeaks.length < 2) {
    throw new Error('mWA 분석을 위해서는 최소 2개의 유효한 피크가 필요합니다.');
  }
  
  // 대비 인자 및 Burgers vector 계산
  const Ch00 = calculateCh00(contrastFactors, structure, qEffective);
  const b = calculateBurgersVector(latticeConstant, structure); // nm
  
  // 각 피크에 대해 푸리에 변환 수행
  const fourierData = [];
  
  for (const peak of validPeaks) {
    const millerIndex = peak.millerIndices[0];
    const { h, k, l } = millerIndex;
    
    if (h === undefined || k === undefined || l === undefined) continue;
    
    // 피크 프로파일 추출
    const peakWidth = peak.fwhm * 3 || 5; // FWHM의 3배 범위
    const profile = extractPeakProfile(xrdData, peak.angle, peakWidth);
    
    if (profile.length < 10) continue; // 최소 데이터 포인트 필요
    
    // 강도 배열 추출
    const intensities = profile.map(p => p.intensity);
    
    // d-spacing 계산 (Å 단위)
    const thetaRad = (peak.angle / 2) * Math.PI / 180;
    const dSpacing = (wavelength * 10) / (2 * Math.sin(thetaRad)); // wavelength는 nm, dSpacing은 Å
    
    // 푸리에 변환 수행 (여러 L 값에 대해)
    const LValues = [];
    const lnAValues = [];
    const k2CValues = [];
    
    // L 값 범위 설정 (L = na³, n은 정수)
    const a = latticeConstant * 0.1; // Å를 nm로 변환
    const maxN = Math.min(50, Math.floor(profile.length / 2));
    
    for (let n = 1; n <= maxN; n++) {
      const L = n * a * a * a; // L = na³
      
      // 푸리에 변환 수행
      const ftResult = fourierTransform(intensities, L, dSpacing);
      const A = ftResult.magnitude;
      
      if (A > 0 && isFinite(A)) {
        const lnA = Math.log(A);
        if (isFinite(lnA)) {
          const C = calculateContrastFactor(h, k, l, qEffective, Ch00);
          const K = calculateK(peak.angle, wavelength);
          const k2C = K * K * C;
          
          LValues.push(L);
          lnAValues.push(lnA);
          k2CValues.push(k2C);
        }
      }
    }
    
    if (LValues.length > 0) {
      fourierData.push({
        peak,
        millerIndex: { h, k, l },
        LValues,
        lnAValues,
        k2CValues
      });
    }
  }
  
  if (fourierData.length === 0) {
    throw new Error('푸리에 변환 데이터를 생성할 수 없습니다.');
  }
  
  // lnA(L) vs K²C̄ 플롯 데이터 생성
  const plotData = {
    lnA: [],
    k2C: []
  };
  
  fourierData.forEach(fd => {
    plotData.lnA.push(...fd.lnAValues);
    plotData.k2C.push(...fd.k2CValues);
  });
  
  // Y(L) = -lnA(L) + lnAs(L) 계산
  // lnAs(L)는 결정 크기 관련 항으로, 간단히 평균값 사용
  const meanLnA = plotData.lnA.reduce((a, b) => a + b, 0) / plotData.lnA.length;
  
  // Y/L² vs lnL 플롯 데이터 생성
  const yOverL2Plot = {
    x: [], // lnL
    y: []  // Y/L²
  };
  
  fourierData.forEach(fd => {
    fd.LValues.forEach((L, i) => {
      if (L > 0 && fd.lnAValues[i] !== undefined) {
        const Y = -fd.lnAValues[i] + meanLnA;
        yOverL2Plot.x.push(Math.log(L));
        yOverL2Plot.y.push(Y / (L * L));
      }
    });
  });
  
  // 선형 회귀로 전위 밀도 계산
  // mWA 방정식: Y/L² = ρ * (π/2) * b² * (ln(Re) - ln(L))
  // 선형 회귀: Y/L² = intercept + slope * ln(L)
  // 여기서 slope = -ρ * (π/2) * b²
  // 따라서 ρ = -slope * 2 / (π * b²)
  if (yOverL2Plot.x.length < 3) {
    throw new Error('Y/L² vs lnL 플롯 데이터가 부족합니다.');
  }
  
  // 이상치 제거 (Y/L² 값이 너무 크거나 작은 경우)
  const filteredData = yOverL2Plot.x.map((x, i) => ({
    x,
    y: yOverL2Plot.y[i]
  })).filter(d => 
    isFinite(d.x) && isFinite(d.y) && 
    Math.abs(d.y) < 1e10 && // 너무 큰 값 제거
    d.y > -1e10 // 너무 작은 값 제거
  );
  
  if (filteredData.length < 3) {
    throw new Error('필터링 후 유효한 데이터 포인트가 부족합니다.');
  }
  
  const regression = linearRegression(
    filteredData.map(d => d.x),
    filteredData.map(d => d.y)
  );
  const slope = regression.slope;
  
  // 전위 밀도 계산 (m⁻²)
  // 단위 변환: b는 nm, 결과는 m⁻²
  const bInMeters = b * 1e-9; // nm를 m로 변환
  let dislocationDensity = 0;
  
  // slope가 음수여야 전위 밀도가 양수
  if (slope < 0 && isFinite(slope) && bInMeters > 0 && Math.abs(slope) > 1e-10) {
    dislocationDensity = -slope * 2 / (Math.PI * bInMeters * bInMeters);
    // 유효성 검사
    if (!isFinite(dislocationDensity) || dislocationDensity < 0) {
      dislocationDensity = 0;
    }
  } else if (slope > 0 && isFinite(slope) && bInMeters > 0 && Math.abs(slope) > 1e-10) {
    // slope가 양수인 경우 (데이터 특성상 가능)
    dislocationDensity = slope * 2 / (Math.PI * bInMeters * bInMeters);
    if (!isFinite(dislocationDensity) || dislocationDensity < 0) {
      dislocationDensity = 0;
    }
  }
  
  return {
    dislocationDensity,
    plotData,
    yOverL2Plot: {
      x: filteredData.map(d => d.x),
      y: filteredData.map(d => d.y),
      fittedY: filteredData.map(d => regression.slope * d.x + regression.intercept)
    },
    rSquared: regression.rSquared,
    fourierData
  };
};

/**
 * 전위 특성 해석
 * @param {number} q - q 파라미터
 * @param {string} structure - 결정 구조
 * @returns {Object} 해석 결과
 */
export const interpretDislocationCharacter = (q, structure) => {
  const qRange = getTheoreticalQRange(structure);
  
  let dislocationCharacter = 'mixed';
  let screwRatio = 0.5;
  let edgeRatio = 0.5;
  const warnings = [];
  const qualityFlags = { qOutOfRange: false };
  
  // q가 null, undefined, 0이거나 유효하지 않으면 혼합형 반환
  if (q == null || !isFinite(q) || q <= 0) {
    return {
      dislocationCharacter: 'mixed',
      screwRatio: 0.5,
      edgeRatio: 0.5,
      qRange,
      warnings: ['q를 산출할 수 없어 전위 특성은 혼합형 참고값으로 표시합니다.'],
      qualityFlags,
    };
  }
  
  screwRatio = calculateScrewFractionFromQ(q, qRange);
  edgeRatio = 1 - screwRatio;
  if (q < qRange.min || q > qRange.max) {
    qualityFlags.qOutOfRange = true;
    warnings.push('q가 기준 범위를 벗어나 screw/edge 비율을 경계값으로 제한했습니다.');
  }

  if (screwRatio >= 0.9) {
    dislocationCharacter = 'screw';
  } else if (screwRatio <= 0.1) {
    dislocationCharacter = 'edge';
  } else {
    dislocationCharacter = 'mixed';
  }
  
  return {
    dislocationCharacter,
    screwRatio,
    edgeRatio,
    qRange,
    warnings,
    qualityFlags,
  };
};

