/**
 * Savitzky-Golay 필터 계수 계산
 * @param {number} windowSize - 윈도우 크기 (홀수여야 함)
 * @param {number} polyOrder - 다항식 차수
 * @param {number} deriv - 미분 차수 (0: 스무딩, 1: 1차 미분, 2: 2차 미분)
 * @returns {Array<number>} 필터 계수 배열
 */
const calculateSavitzkyGolayCoefficients = (windowSize, polyOrder, deriv = 0) => {
  if (windowSize % 2 === 0) {
    throw new Error('윈도우 크기는 홀수여야 합니다.');
  }
  if (polyOrder >= windowSize) {
    throw new Error('다항식 차수는 윈도우 크기보다 작아야 합니다.');
  }

  const halfWindow = Math.floor(windowSize / 2);
  const coefficients = [];

  for (let i = -halfWindow; i <= halfWindow; i++) {
    let sum = 0;
    for (let j = 0; j <= polyOrder; j++) {
      const term = Math.pow(i, j);
      if (deriv === 0) {
        sum += term;
      } else if (deriv === 1) {
        sum += j * Math.pow(i, j - 1);
      } else if (deriv === 2) {
        sum += j * (j - 1) * Math.pow(i, j - 2);
      }
    }
    coefficients.push(sum);
  }

  // 정규화
  const sum = coefficients.reduce((a, b) => a + b, 0);
  return coefficients.map(c => c / sum);
};

/**
 * Savitzky-Golay 필터 적용
 * @param {Array<number>} data - 입력 데이터
 * @param {number} windowSize - 윈도우 크기
 * @param {number} polyOrder - 다항식 차수
 * @param {number} deriv - 미분 차수
 * @returns {Array<number>} 필터링된 데이터
 */
export const applySavitzkyGolay = (data, windowSize = 5, polyOrder = 2, deriv = 0) => {
  if (data.length < windowSize) {
    return [...data]; // 데이터가 너무 짧으면 원본 반환
  }

  const halfWindow = Math.floor(windowSize / 2);
  const coefficients = calculateSavitzkyGolayCoefficients(windowSize, polyOrder, deriv);
  const result = [];

  for (let i = 0; i < data.length; i++) {
    let value = 0;
    
    if (i < halfWindow) {
      // 시작 부분: 대칭 패딩
      for (let j = -halfWindow; j <= halfWindow; j++) {
        const idx = Math.max(0, Math.min(data.length - 1, i + j));
        value += data[idx] * coefficients[j + halfWindow];
      }
    } else if (i >= data.length - halfWindow) {
      // 끝 부분: 대칭 패딩
      for (let j = -halfWindow; j <= halfWindow; j++) {
        const idx = Math.max(0, Math.min(data.length - 1, i + j));
        value += data[idx] * coefficients[j + halfWindow];
      }
    } else {
      // 중간 부분: 정상 처리
      for (let j = -halfWindow; j <= halfWindow; j++) {
        value += data[i + j] * coefficients[j + halfWindow];
      }
    }
    
    result.push(value);
  }

  return result;
};

/**
 * 1차 미분 계산
 * @param {Array<number>} data - 입력 데이터
 * @returns {Array<number>} 1차 미분 결과
 */
export const calculateFirstDerivative = (data) => {
  const derivative = [];
  
  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      derivative.push(data[1] - data[0]);
    } else if (i === data.length - 1) {
      derivative.push(data[i] - data[i - 1]);
    } else {
      // 중앙 차분
      derivative.push((data[i + 1] - data[i - 1]) / 2);
    }
  }
  
  return derivative;
};

/**
 * 2차 미분 계산
 * @param {Array<number>} data - 입력 데이터
 * @returns {Array<number>} 2차 미분 결과
 */
export const calculateSecondDerivative = (data) => {
  const derivative = [];
  
  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      derivative.push(data[2] - 2 * data[1] + data[0]);
    } else if (i === data.length - 1) {
      derivative.push(data[i] - 2 * data[i - 1] + data[i - 2]);
    } else {
      // 중앙 차분
      derivative.push(data[i + 1] - 2 * data[i] + data[i - 1]);
    }
  }
  
  return derivative;
};

/**
 * 로컬 맥시마 탐지
 * @param {Array<number>} data - 입력 데이터
 * @param {number} minHeight - 최소 피크 높이 (상대값 또는 절대값)
 * @param {number} minDistance - 최소 피크 간격 (인덱스 단위)
 * @param {boolean} useRelativeThreshold - 상대 임계값 사용 여부
 * @returns {Array<number>} 피크 인덱스 배열
 */
export const detectLocalMaxima = (data, minHeight = 0, minDistance = 1, useRelativeThreshold = true) => {
  if (data.length === 0) return [];

  const maxValue = Math.max(...data);
  const minValue = Math.min(...data);
  const threshold = useRelativeThreshold 
    ? minValue + (maxValue - minValue) * minHeight
    : minHeight;

  const peaks = [];

  for (let i = 1; i < data.length - 1; i++) {
    // 로컬 맥시마 조건: 이전 값과 다음 값보다 큼
    if (data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] > threshold) {
      // 최소 거리 체크
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) {
        peaks.push(i);
      } else {
        // 더 높은 피크로 교체
        if (data[i] > data[peaks[peaks.length - 1]]) {
          peaks[peaks.length - 1] = i;
        }
      }
    }
  }

  return peaks;
};

/**
 * 2차 미분을 이용한 피크 탐색 (더 정확한 방법)
 * @param {Array<number>} data - 입력 데이터
 * @param {number} minHeight - 최소 피크 높이
 * @param {number} minDistance - 최소 피크 간격
 * @returns {Array<number>} 피크 인덱스 배열
 */
export const detectPeaksBySecondDerivative = (data, minHeight = 0, minDistance = 1) => {
  // 2차 미분 계산
  const secondDeriv = calculateSecondDerivative(data);
  
  // 2차 미분이 음수인 지점 찾기 (피크의 정상부)
  const peaks = [];
  const maxValue = Math.max(...data);
  const minValue = Math.min(...data);
  const threshold = minValue + (maxValue - minValue) * minHeight;

  for (let i = 1; i < secondDeriv.length - 1; i++) {
    // 2차 미분이 음수이고, 이전 값이 양수였던 지점 (피크의 시작)
    if (secondDeriv[i] < 0 && secondDeriv[i - 1] >= 0) {
      // 피크의 정확한 위치 찾기 (주변에서 최대값)
      let peakIdx = i;
      let maxIntensity = data[i];
      
      // 주변 영역에서 최대값 찾기
      for (let j = Math.max(0, i - 5); j < Math.min(data.length, i + 5); j++) {
        if (data[j] > maxIntensity) {
          maxIntensity = data[j];
          peakIdx = j;
        }
      }
      
      if (data[peakIdx] > threshold) {
        // 최소 거리 체크
        if (peaks.length === 0 || peakIdx - peaks[peaks.length - 1] >= minDistance) {
          peaks.push(peakIdx);
        } else if (data[peakIdx] > data[peaks[peaks.length - 1]]) {
          peaks[peaks.length - 1] = peakIdx;
        }
      }
    }
  }

  return peaks;
};

/**
 * 피크 탐색 메인 함수
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 데이터
 * @param {Object} options - 탐색 옵션
 * @param {number} options.smoothingWindow - 스무딩 윈도우 크기
 * @param {number} options.smoothingPolyOrder - 스무딩 다항식 차수
 * @param {number} options.minPeakHeight - 최소 피크 높이 (0-1 상대값)
 * @param {number} options.minPeakDistance - 최소 피크 간격 (데이터 포인트 수)
 * @param {string} options.method - 탐색 방법 ('localMaxima' | 'secondDerivative')
 * @returns {Array<{index: number, angle: number, intensity: number}>} 탐지된 피크 배열
 */
export const detectPeaks = (xrdData, options = {}) => {
  const {
    smoothingWindow = 5,
    smoothingPolyOrder = 2,
    minPeakHeight = 0.05,
    minPeakDistance = 5,
    method = 'secondDerivative'
  } = options;

  if (xrdData.length === 0) {
    return [];
  }

  // Intensity 배열 추출
  const intensities = xrdData.map(d => d.intensity);

  // 스무딩 적용
  const smoothed = applySavitzkyGolay(intensities, smoothingWindow, smoothingPolyOrder, 0);

  // 피크 탐색
  let peakIndices;
  if (method === 'secondDerivative') {
    peakIndices = detectPeaksBySecondDerivative(smoothed, minPeakHeight, minPeakDistance);
  } else {
    peakIndices = detectLocalMaxima(smoothed, minPeakHeight, minPeakDistance, true);
  }

  // 피크 정보 구성
  const peaks = peakIndices.map(idx => ({
    index: idx,
    angle: xrdData[idx].angle,
    intensity: xrdData[idx].intensity,
    smoothedIntensity: smoothed[idx]
  }));

  // 강도 순으로 정렬
  peaks.sort((a, b) => b.intensity - a.intensity);

  return peaks;
};

