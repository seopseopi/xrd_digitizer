import { levenbergMarquardt } from 'ml-levenberg-marquardt';

/**
 * 가우시안 함수
 * @param {number} x - 입력 값
 * @param {number} center - 피크 중심 위치
 * @param {number} height - 피크 높이
 * @param {number} width - 피크 너비 (표준편차)
 * @returns {number} 가우시안 값
 */
const gaussian = (x, center, height, width) => {
  const sigma = width / (2 * Math.sqrt(2 * Math.LN2)); // FWHM to sigma
  const exponent = -0.5 * Math.pow((x - center) / sigma, 2);
  return height * Math.exp(exponent);
};

/**
 * 로렌츠 함수
 * @param {number} x - 입력 값
 * @param {number} center - 피크 중심 위치
 * @param {number} height - 피크 높이
 * @param {number} width - 피크 너비 (FWHM)
 * @returns {number} 로렌츠 값
 */
const lorentzian = (x, center, height, width) => {
  const gamma = width / 2;
  const denominator = Math.pow(x - center, 2) + Math.pow(gamma, 2);
  return height * Math.pow(gamma, 2) / denominator;
};

/**
 * Voigt 함수 (가우시안과 로렌츠의 컨볼루션 근사)
 * @param {number} x - 입력 값
 * @param {number} center - 피크 중심 위치
 * @param {number} height - 피크 높이
 * @param {number} widthG - 가우시안 너비
 * @param {number} widthL - 로렌츠 너비
 * @returns {number} Voigt 값
 */
const voigt = (x, center, height, widthG, widthL) => {
  // 간단한 Voigt 근사 (Pseudo-Voigt)
  const g = gaussian(x, center, height, widthG);
  const l = lorentzian(x, center, height, widthL);
  // 가중 평균 (일반적으로 0.5:0.5 비율 사용)
  return 0.5 * g + 0.5 * l;
};

/**
 * 선형 배경
 * @param {number} x - 입력 값
 * @param {number} a - 기울기
 * @param {number} b - 절편
 * @returns {number} 배경 값
 */
const linearBackground = (x, a, b) => {
  return a * x + b;
};

/**
 * 다항식 배경
 * @param {number} x - 입력 값
 * @param {Array<number>} coeffs - 다항식 계수 [a0, a1, a2, ...]
 * @returns {number} 배경 값
 */
const polynomialBackground = (x, coeffs) => {
  let result = 0;
  for (let i = 0; i < coeffs.length; i++) {
    result += coeffs[i] * Math.pow(x, i);
  }
  return result;
};

/**
 * 단일 피크 피팅 함수 생성
 * @param {string} peakType - 피크 타입 ('gaussian', 'lorentzian', 'voigt')
 * @param {string} backgroundType - 배경 타입 ('linear', 'polynomial')
 * @param {number} backgroundOrder - 배경 다항식 차수 (backgroundType이 'polynomial'일 때)
 * @returns {Function} 피팅 함수
 */
const createFittingFunction = (peakType, backgroundType, backgroundOrder = 2) => {
  return (x, params) => {
    let result = 0;
    
    // 배경 추가
    let peakStartIdx;
    if (backgroundType === 'linear') {
      result += linearBackground(x, params[0], params[1]);
      peakStartIdx = 2;
    } else if (backgroundType === 'polynomial') {
      const bgCoeffs = params.slice(0, backgroundOrder + 1);
      result += polynomialBackground(x, bgCoeffs);
      peakStartIdx = backgroundOrder + 1;
    } else {
      peakStartIdx = 0;
    }
    
    // 피크 추가 (각 피크당 3개 파라미터: center, height, width)
    const numPeaks = (params.length - peakStartIdx) / 3;
    
    for (let i = 0; i < numPeaks; i++) {
      const center = params[peakStartIdx + i * 3];
      const height = params[peakStartIdx + i * 3 + 1];
      const width = params[peakStartIdx + i * 3 + 2];
      
      if (peakType === 'gaussian') {
        result += gaussian(x, center, height, width);
      } else if (peakType === 'lorentzian') {
        result += lorentzian(x, center, height, width);
      } else if (peakType === 'voigt') {
        // Voigt는 4개 파라미터 필요 (center, height, widthG, widthL)
        const widthG = width;
        const widthL = params[peakStartIdx + i * 4 + 3] || width;
        result += voigt(x, center, height, widthG, widthL);
      }
    }
    
    return result;
  };
};

/**
 * 초기 파라미터 추정
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 데이터
 * @param {Array<{index: number, angle: number, intensity: number}>} peaks - 탐지된 피크
 * @param {string} backgroundType - 배경 타입
 * @param {number} backgroundOrder - 배경 다항식 차수
 * @returns {Array<number>} 초기 파라미터 배열
 */
const estimateInitialParameters = (xrdData, peaks, backgroundType, backgroundOrder) => {
  const params = [];
  
  // 배경 파라미터 초기화
  if (backgroundType === 'linear') {
    // 선형 배경: 기울기와 절편
    const firstIntensity = xrdData[0].intensity;
    const lastIntensity = xrdData[xrdData.length - 1].intensity;
    const firstAngle = xrdData[0].angle;
    const lastAngle = xrdData[xrdData.length - 1].angle;
    
    const slope = (lastIntensity - firstIntensity) / (lastAngle - firstAngle);
    const intercept = firstIntensity - slope * firstAngle;
    
    params.push(slope, intercept);
  } else if (backgroundType === 'polynomial') {
    // 다항식 배경: 최소값 기준으로 초기화
    const minIntensity = Math.min(...xrdData.map(d => d.intensity));
    for (let i = 0; i <= backgroundOrder; i++) {
      params.push(i === 0 ? minIntensity : 0);
    }
  }
  
  // 피크 파라미터 초기화
  for (const peak of peaks) {
    params.push(peak.angle); // center
    params.push(peak.intensity * 0.8); // height (약간 낮게)
    
    // 피크 너비 추정 (주변 데이터로부터)
    const peakIdx = peak.index;
    const halfMax = peak.intensity / 2;
    let leftIdx = peakIdx;
    let rightIdx = peakIdx;
    
    // 왼쪽 절반 높이 지점 찾기
    while (leftIdx > 0 && xrdData[leftIdx].intensity > halfMax) {
      leftIdx--;
    }
    
    // 오른쪽 절반 높이 지점 찾기
    while (rightIdx < xrdData.length - 1 && xrdData[rightIdx].intensity > halfMax) {
      rightIdx++;
    }
    
    const fwhm = Math.abs(xrdData[rightIdx].angle - xrdData[leftIdx].angle);
    params.push(fwhm || 0.5); // 기본값 0.5도
  }
  
  return params;
};

/**
 * 피크 피팅 실행
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 데이터
 * @param {Array<{index: number, angle: number, intensity: number}>} peaks - 탐지된 피크
 * @param {Object} options - 피팅 옵션
 * @param {string} options.peakType - 피크 타입 ('gaussian', 'lorentzian', 'voigt')
 * @param {string} options.backgroundType - 배경 타입 ('linear', 'polynomial', 'none')
 * @param {number} options.backgroundOrder - 배경 다항식 차수
 * @returns {Object} 피팅 결과
 */
export const fitPeaks = (xrdData, peaks, options = {}) => {
  const {
    peakType = 'gaussian',
    backgroundType = 'linear',
    backgroundOrder = 2
  } = options;

  if (peaks.length === 0) {
    return {
      success: false,
      error: '피크가 없습니다.'
    };
  }

  // 데이터 준비
  const x = xrdData.map(d => d.angle);
  const y = xrdData.map(d => d.intensity);

  // 피팅 함수 생성
  const fittingFunction = createFittingFunction(peakType, backgroundType, backgroundOrder);

  // 초기 파라미터 추정
  const initialParams = estimateInitialParameters(xrdData, peaks, backgroundType, backgroundOrder);

  // 파라미터 범위 설정 (제약 조건)
  const minValues = initialParams.map((p, i) => {
    const peakIdx = Math.floor((i - (backgroundType === 'linear' ? 2 : backgroundOrder + 1)) / 3);
    if (i < (backgroundType === 'linear' ? 2 : backgroundOrder + 1)) {
      return -Infinity; // 배경 파라미터는 제약 없음
    } else if (i % 3 === 0) {
      // center: 피크 주변 ±5도
      return peaks[peakIdx].angle - 5;
    } else if (i % 3 === 1) {
      // height: 0 이상
      return 0;
    } else {
      // width: 0.1도 이상
      return 0.1;
    }
  });

  const maxValues = initialParams.map((p, i) => {
    const peakIdx = Math.floor((i - (backgroundType === 'linear' ? 2 : backgroundOrder + 1)) / 3);
    if (i < (backgroundType === 'linear' ? 2 : backgroundOrder + 1)) {
      return Infinity; // 배경 파라미터는 제약 없음
    } else if (i % 3 === 0) {
      // center: 피크 주변 ±5도
      return peaks[peakIdx].angle + 5;
    } else if (i % 3 === 1) {
      // height: 무제한
      return Infinity;
    } else {
      // width: 10도 이하
      return 10;
    }
  });

  try {
    // Levenberg-Marquardt 알고리즘 실행
    // parameterizedFunction: 파라미터 배열을 받아서 함수를 반환
    const parameterizedFunction = (params) => {
      return (x) => fittingFunction(x, params);
    };

    const result = levenbergMarquardt(
      { x, y },
      parameterizedFunction,
      {
        parameters: initialParams,
        minValues: minValues,
        maxValues: maxValues,
        maxIterations: 1000,
        errorTolerance: 1e-6
      }
    );

    // 피팅된 파라미터 추출
    const fittedParams = result.parameterValues;
    
    // 피팅된 곡선 생성
    const fittedCurve = x.map(xi => fittingFunction(xi, fittedParams));

    // 각 피크의 피팅 결과 추출
    const peakStartIdx = backgroundType === 'linear' ? 2 : backgroundOrder + 1;
    const fittedPeaks = [];
    
    for (let i = 0; i < peaks.length; i++) {
      const center = fittedParams[peakStartIdx + i * 3];
      const height = fittedParams[peakStartIdx + i * 3 + 1];
      const width = fittedParams[peakStartIdx + i * 3 + 2];
      
      fittedPeaks.push({
        originalIndex: peaks[i].index,
        center: center,
        height: height,
        fwhm: width,
        angle: center
      });
    }

    return {
      success: true,
      parameters: fittedParams,
      fittedCurve: fittedCurve,
      peaks: fittedPeaks,
      backgroundParams: fittedParams.slice(0, peakStartIdx),
      iterations: result.iterations || 0,
      error: result.parameterError
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
};

