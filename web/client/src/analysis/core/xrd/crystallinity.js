/**
 * 결정화도 계산
 * 결정성 피크의 면적을 전체 면적로 나눈 비율
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 데이터
 * @param {Array<{index: number, angle: number, intensity: number}>} peaks - 탐지된 피크
 * @param {Object} options - 계산 옵션
 * @param {number} options.backgroundLevel - 배경 레벨 (없으면 자동 계산)
 * @returns {number} 결정화도 (0-1)
 */
export const calculateCrystallinity = (xrdData, peaks, options = {}) => {
  if (xrdData.length === 0 || peaks.length === 0) {
    return 0;
  }

  const { backgroundLevel } = options;

  // 배경 레벨 계산 (없으면 최소값 사용)
  let bgLevel = backgroundLevel;
  if (bgLevel === undefined) {
    bgLevel = Math.min(...xrdData.map(d => d.intensity));
  }

  // 전체 면적 계산 (사다리꼴 공식)
  let totalArea = 0;
  for (let i = 0; i < xrdData.length - 1; i++) {
    const dx = xrdData[i + 1].angle - xrdData[i].angle;
    const y1 = Math.max(0, xrdData[i].intensity - bgLevel);
    const y2 = Math.max(0, xrdData[i + 1].intensity - bgLevel);
    totalArea += (y1 + y2) * dx / 2;
  }

  // 결정성 피크 면적 계산
  let crystallineArea = 0;
  
  for (const peak of peaks) {
    const peakIdx = peak.index;
    
    // 피크 주변 영역 찾기 (피크 높이의 10% 이상인 영역)
    const peakIntensity = peak.intensity;
    const threshold = bgLevel + (peakIntensity - bgLevel) * 0.1;
    
    let leftIdx = peakIdx;
    let rightIdx = peakIdx;
    
    // 왼쪽 경계 찾기
    while (leftIdx > 0 && xrdData[leftIdx].intensity > threshold) {
      leftIdx--;
    }
    
    // 오른쪽 경계 찾기
    while (rightIdx < xrdData.length - 1 && xrdData[rightIdx].intensity > threshold) {
      rightIdx++;
    }
    
    // 피크 면적 계산
    for (let i = leftIdx; i < rightIdx; i++) {
      const dx = xrdData[i + 1].angle - xrdData[i].angle;
      const y1 = Math.max(0, xrdData[i].intensity - bgLevel);
      const y2 = Math.max(0, xrdData[i + 1].intensity - bgLevel);
      crystallineArea += (y1 + y2) * dx / 2;
    }
  }

  // 결정화도 = 결정성 면적 / 전체 면적
  const crystallinity = totalArea > 0 ? crystallineArea / totalArea : 0;
  
  return Math.min(1, Math.max(0, crystallinity)); // 0-1 범위로 제한
};

/**
 * Scherrer 방정식을 사용한 결정립 크기 계산
 * D = Kλ / (βcosθ)
 * @param {number} fwhm - 반가폭 (Full Width at Half Maximum, 도 단위)
 * @param {number} angle - 브래그 각도 (2θ, 도 단위)
 * @param {Object} options - 계산 옵션
 * @param {number} options.wavelength - X선 파장 (Å, 기본값: Cu Kα = 1.5406)
 * @param {number} options.shapeFactor - 형태 인자 K (기본값: 0.9)
 * @returns {number} 결정립 크기 (nm)
 */
export const calculateCrystalliteSize = (fwhm, angle, options = {}) => {
  const {
    wavelength = 1.5406, // Cu Kα
    shapeFactor = 0.9
  } = options;

  // 각도를 라디안으로 변환
  const thetaRad = (angle / 2) * Math.PI / 180; // 2θ를 θ로 변환 후 라디안
  const fwhmRad = fwhm * Math.PI / 180; // FWHM을 라디안으로 변환

  // Scherrer 방정식: D = Kλ / (βcosθ)
  // β는 라디안 단위의 FWHM
  const denominator = fwhmRad * Math.cos(thetaRad);
  
  if (denominator === 0) {
    return 0;
  }

  const d = (shapeFactor * wavelength) / denominator;
  
  // Å를 nm로 변환
  return d / 10;
};

/**
 * 여러 피크에 대한 결정립 크기 계산
 * @param {Array<{angle: number, fwhm: number}>} peaks - 피크 정보 (angle: 2θ, fwhm: 반가폭)
 * @param {Object} options - 계산 옵션
 * @returns {Array<{angle: number, fwhm: number, crystalliteSize: number}>} 각 피크의 결정립 크기
 */
export const calculateCrystalliteSizes = (peaks, options = {}) => {
  return peaks.map(peak => ({
    ...peak,
    crystalliteSize: calculateCrystalliteSize(peak.fwhm, peak.angle, options)
  }));
};

/**
 * 평균 결정립 크기 계산
 * @param {Array<{crystalliteSize: number}>} crystalliteSizes - 결정립 크기 배열
 * @returns {Object} 통계 정보
 */
export const calculateAverageCrystalliteSize = (crystalliteSizes) => {
  if (crystalliteSizes.length === 0) {
    return {
      average: 0,
      median: 0,
      min: 0,
      max: 0,
      std: 0
    };
  }

  const sizes = crystalliteSizes
    .map(p => p.crystalliteSize)
    .filter(s => s > 0 && isFinite(s));

  if (sizes.length === 0) {
    return {
      average: 0,
      median: 0,
      min: 0,
      max: 0,
      std: 0
    };
  }

  const sorted = [...sizes].sort((a, b) => a - b);
  const sum = sizes.reduce((a, b) => a + b, 0);
  const average = sum / sizes.length;
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  
  // 표준편차
  const variance = sizes.reduce((acc, val) => acc + Math.pow(val - average, 2), 0) / sizes.length;
  const std = Math.sqrt(variance);

  return {
    average,
    median,
    min,
    max,
    std
  };
};

