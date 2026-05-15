import { fromBRML, fromPowDLLXY } from 'xrd-analysis';
import { parseCif } from 'crystcif-parse';

/**
 * XRD 파일 파싱 결과
 * @typedef {Object} XRDParseResult
 * @property {Object} metadata - 파일 메타데이터
 * @property {Array<{angle: number, intensity: number}>} data - 2θ-Intensity 데이터 배열
 * @property {Object} structureInfo - CIF 파일의 경우 구조 정보 (격자 상수 등)
 */

/**
 * 파일 확장자로 파일 형식 판별
 * @param {string} fileName - 파일명
 * @returns {string} 파일 형식 ('brml', 'powdll', 'csv', 'xy', 'txt', 'cif', 'unknown')
 */
export const detectFileFormat = (fileName) => {
  const ext = fileName.toLowerCase().split('.').pop();
  
  switch (ext) {
    case 'brml':
      return 'brml';
    case 'powdll':
    case 'xdd':
      return 'powdll';
    case 'csv':
      return 'csv';
    case 'xy':
      return 'xy';
    case 'txt':
      return 'txt';
    case 'cif':
      return 'cif';
    case 'json':
      return 'json';
    default:
      return 'unknown';
  }
};

/**
 * BRML 파일 파싱 (xrd-analysis 라이브러리 사용)
 * @param {string|Uint8Array} blob - BRML 파일 내용
 * @returns {Promise<XRDParseResult>}
 */
export const parseBRML = async (blob) => {
  try {
    const analysis = await fromBRML(blob);
    const spectrum = analysis.spectra[0];
    
    if (!spectrum || !spectrum.variables) {
      throw new Error('BRML 파일에서 스펙트럼 데이터를 찾을 수 없습니다.');
    }
    
    const xData = spectrum.variables.x.data;
    const yData = spectrum.variables.y.data;
    
    // xrd-analysis의 반환 형식에 따라 데이터 추출
    const data = xData.map((angle, i) => ({
      angle: angle,
      intensity: yData[i] || 0
    }));
    
    return {
      metadata: {
        format: 'BRML',
        title: spectrum.title || '',
        xUnits: spectrum.variables.x.label || '',
        yUnits: spectrum.variables.y.label || '',
        ...spectrum.meta
      },
      data: data,
      structureInfo: null
    };
  } catch (error) {
    throw new Error(`BRML 파싱 오류: ${error.message}`);
  }
};

/**
 * PowDLL 파일 파싱 (xrd-analysis 라이브러리 사용)
 * @param {string|Uint8Array} blob - PowDLL 파일 내용
 * @returns {XRDParseResult}
 */
export const parsePowDLL = (blob) => {
  try {
    const analysis = fromPowDLLXY(blob);
    const spectrum = analysis.spectra[0];
    
    if (!spectrum || !spectrum.variables) {
      throw new Error('PowDLL 파일에서 스펙트럼 데이터를 찾을 수 없습니다.');
    }
    
    const xData = spectrum.variables.x.data;
    const yData = spectrum.variables.y.data;
    
    const data = xData.map((angle, i) => ({
      angle: angle,
      intensity: yData[i] || 0
    }));
    
    return {
      metadata: {
        format: 'PowDLL',
        title: spectrum.title || '',
        xUnits: spectrum.variables.x.label || '',
        yUnits: spectrum.variables.y.label || '',
        ...spectrum.meta
      },
      data: data,
      structureInfo: null
    };
  } catch (error) {
    throw new Error(`PowDLL 파싱 오류: ${error.message}`);
  }
};

/**
 * CSV/XY/TXT 파일 파싱 (직접 구현)
 * 탭, 공백, 쉼표로 구분된 2θ-Intensity 쌍 파싱
 * @param {string} text - CSV/XY/TXT 파일 내용
 * @returns {XRDParseResult}
 */
export const parseCSVXY = (text) => {
  const lines = text.split('\n').filter(line => line.trim());
  const metadata = {};
  const data = [];
  let isDataSection = false;
  let headerFound = false;

  for (let i = 0; i < lines.length; i++) {
    const trimmedLine = lines[i].trim();
    if (!trimmedLine) continue;

    // BOM 제거
    const line = trimmedLine.replace(/^\uFEFF/, '');
    
    // 구분자 자동 감지 (탭, 쉼표, 공백)
    let parts;
    if (line.includes('\t')) {
      parts = line.split('\t').map(p => p.trim());
    } else if (line.includes(',')) {
      parts = line.split(',').map(p => p.trim());
    } else {
      // 공백으로 구분 (여러 공백도 처리)
      parts = line.split(/\s+/).filter(p => p.length > 0);
    }

    if (parts.length < 2) continue;

    const firstPart = parts[0].trim();
    const secondPart = parts[1].trim();

    // 숫자로 시작하면 데이터 섹션
    const firstNum = parseFloat(firstPart);
    const secondNum = parseFloat(secondPart);

    if (!isNaN(firstNum) && !isNaN(secondNum)) {
      isDataSection = true;
      data.push({
        angle: firstNum,
        intensity: secondNum
      });
    } else if (!isDataSection && !headerFound) {
      // 헤더 라인 처리 (예: "2Theta", "Intensity")
      if (firstPart.toLowerCase().includes('theta') || 
          firstPart.toLowerCase().includes('angle') ||
          firstPart.toLowerCase().includes('2theta') ||
          firstPart.toLowerCase().includes('2-theta')) {
        headerFound = true;
        continue; // 헤더 라인은 건너뛰기
      }
      
      // 메타데이터 섹션
      if (firstPart && secondPart) {
        metadata[firstPart] = secondPart;
      }
    }
  }

  if (data.length === 0) {
    throw new Error('XRD 데이터를 찾을 수 없습니다. 파일 형식을 확인해주세요.');
  }

  // 각도 순서 정렬
  data.sort((a, b) => a.angle - b.angle);

  return {
    metadata: {
      format: 'CSV/XY/TXT',
      dataPoints: data.length,
      ...metadata
    },
    data: data,
    structureInfo: null
  };
};

/**
 * CIF 파일 파싱 (crystcif-parse 라이브러리 사용)
 * @param {string} text - CIF 파일 내용
 * @returns {XRDParseResult}
 */
export const parseCIF = (text) => {
  try {
    const cifData = parseCif(text);
    
    // CIF 파일은 주로 구조 정보를 포함하므로, XRD 패턴 데이터가 직접 포함되지 않을 수 있음
    // 구조 정보 추출
    let structureInfo = null;
    
    if (cifData && cifData.length > 0) {
      const firstBlock = cifData[0];
      structureInfo = {
        cellParams: {
          system: firstBlock['_cell_space_group_name_H-M_alt'] || 'cubic',
          a: parseFloat(firstBlock['_cell_length_a']) || null,
          b: parseFloat(firstBlock['_cell_length_b']) || null,
          c: parseFloat(firstBlock['_cell_length_c']) || null,
          alpha: parseFloat(firstBlock['_cell_angle_alpha']) || null,
          beta: parseFloat(firstBlock['_cell_angle_beta']) || null,
          gamma: parseFloat(firstBlock['_cell_angle_gamma']) || null
        },
        spaceGroup: firstBlock['_space_group_name_H-M_alt'] || null,
        atoms: firstBlock['_atom_site_label'] || []
      };
    }

    // CIF에 XRD 데이터가 포함된 경우 (일반적으로는 없음)
    // 여기서는 구조 정보만 반환
    return {
      metadata: {
        format: 'CIF'
      },
      data: [], // CIF는 구조 정보만 포함, XRD 패턴은 별도 파일 필요
      structureInfo: structureInfo
    };
  } catch (error) {
    throw new Error(`CIF 파싱 오류: ${error.message}`);
  }
};

/**
 * opxrd 형식 JSON 파일 파싱 (two_theta_values, intensities 배열)
 * @param {string} text - JSON 문자열
 * @returns {XRDParseResult}
 */
export const parseXRDJSON = (text) => {
  let obj;
  try {
    obj = JSON.parse(text);
  } catch (e) {
    throw new Error(`JSON 파싱 오류: ${e.message}`);
  }

  const twoTheta = obj.two_theta_values || obj['two_theta_values'] || obj.twoTheta || obj.x;
  const intensities = obj.intensities || obj.intensity || obj.y;

  if (!Array.isArray(twoTheta) || !Array.isArray(intensities)) {
    throw new Error('JSON에 two_theta_values와 intensities 배열이 필요합니다.');
  }

  const data = twoTheta.map((angle, i) => ({
    angle: Number(angle),
    intensity: Number(intensities[i] ?? 0),
  }));

  if (data.length === 0) {
    throw new Error('XRD 데이터를 찾을 수 없습니다.');
  }

  const metadata = { format: 'JSON (opxrd)', dataPoints: data.length };
  if (obj.label && typeof obj.label === 'string') {
    try {
      const labelObj = JSON.parse(obj.label);
      if (labelObj.xray_info) {
        const xrayInfo = typeof labelObj.xray_info === 'string' ? JSON.parse(labelObj.xray_info) : labelObj.xray_info;
        if (xrayInfo.primary_wavelength) metadata.wavelength = parseFloat(xrayInfo.primary_wavelength);
      }
    } catch (_) { /* ignore */ }
  }
  if (obj.metadata && typeof obj.metadata === 'string') {
    try {
      metadata.rawMetadata = JSON.parse(obj.metadata);
    } catch (_) { /* ignore */ }
  }

  return { metadata, data, structureInfo: null };
};

/**
 * 이미 텍스트로 디코딩된 내용을 파일 형식에 따라 파싱 (FileReader 없이 직접 호출 가능)
 * callLocalDirectWithFile 경유로 전달된 텍스트 데이터 처리용
 * @param {string} text - 파일 텍스트 내용
 * @param {string} fileType - 'powdll' | 'csv' | 'xy' | 'txt' | 'cif' | 'json' | 'unknown'
 * @returns {Promise<XRDParseResult>}
 */
export const parseTextXRD = async (text, fileType) => {
  switch (fileType) {
    case 'powdll':
      return parsePowDLL(text);
    case 'cif':
      return parseCIF(text);
    case 'json':
      return parseXRDJSON(text);
    case 'csv':
    case 'xy':
    case 'txt':
      return parseCSVXY(text);
    default:
      return parseCSVXY(text);
  }
};

/**
 * 파일 형식에 따라 적절한 파서 선택하여 파싱
 * @param {File} file - 업로드된 파일 객체
 * @returns {Promise<XRDParseResult>}
 */
export const parseXRDFile = async (file) => {
  const format = detectFileFormat(file.name);
  
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = async (event) => {
      try {
        const text = event.target.result;
        let result;

        switch (format) {
          case 'brml':
            result = await parseBRML(text);
            break;
          case 'powdll':
            result = parsePowDLL(text);
            break;
          case 'json':
            result = parseXRDJSON(text);
            break;
          case 'csv':
          case 'xy':
          case 'txt':
            result = parseCSVXY(text);
            break;
          case 'cif':
            result = parseCIF(text);
            break;
          default:
            // 알 수 없는 형식이면 CSV/XY 파서로 시도
            result = parseCSVXY(text);
            result.metadata.format = 'Unknown (attempted CSV/XY parsing)';
        }

        resolve(result);
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => {
      reject(new Error('파일 읽기 오류가 발생했습니다.'));
    };

    reader.readAsText(file);
  });
};

