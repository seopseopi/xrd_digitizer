/**
 * 파서 알고리즘 코어
 * EBSD(h5oina, CTF), XRD 다양한 포맷 파싱
 * 순수 함수만 포함 — DOM, Canvas, React 의존성 없음
 */

import { parseH5OINA, calculateStatistics as calcEBSDStats } from './ebsd/h5oinaParser.js';
import { convertCTFToH5OINA } from './ebsd/ctfAdapter.js';
import {
  detectFileFormat,
  parseBRML,
} from './xrd/xrdParser.js';

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

/**
 * EBSD 파일 파싱 (h5oina 또는 ctf)
 * @param {Object} input
 * @param {ArrayBuffer} input.fileData - 파일 바이너리
 * @param {'h5oina'|'ctf'} input.fileType
 * @param {string} input.fileName
 */
export async function parseEBSD(input) {
  const t0 = Date.now();
  try {
    const { fileData, fileType, fileName } = input;
    if (!fileData) return makeError('INVALID_INPUT', '파일 데이터가 없습니다');

    let ebsdData;

    if (fileType === 'h5oina') {
      // File 객체로 래핑 (plain Array → Uint8Array 변환 필수)
      const uint8 = fileData instanceof Uint8Array ? fileData : new Uint8Array(fileData);
      const blob = new Blob([uint8]);
      const file = new File([blob], fileName || 'data.h5oina');
      ebsdData = await parseH5OINA(file);
    } else if (fileType === 'ctf') {
      const decoder = new TextDecoder('utf-8');
      const text = decoder.decode(fileData);
      ebsdData = convertCTFToH5OINA(text);
    } else {
      return makeError('INVALID_INPUT', `지원하지 않는 파일 형식: ${fileType}`);
    }

    const stats = calcEBSDStats(ebsdData);

    // 상 정보 정규화
    const phases = (ebsdData.phases || []).map((p, i) => ({
      id: i + 1,
      name: p.name || `Phase ${i + 1}`,
      color: p.color || `#${Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, '0')}`,
      latticeParameters: {
        a: p.latticeParams?.a || p.latticeParameters?.a || 0,
        b: p.latticeParams?.b || p.latticeParameters?.b || 0,
        c: p.latticeParams?.c || p.latticeParameters?.c || 0,
        alpha: p.latticeAngles?.alpha ?? p.latticeParameters?.alpha ?? 90,
        beta: p.latticeAngles?.beta ?? p.latticeParameters?.beta ?? 90,
        gamma: p.latticeAngles?.gamma ?? p.latticeParameters?.gamma ?? 90,
      },
      crystalSystem: p.crystalSystem || 'Unknown',
    }));

    return makeResult(
      {
        header: {
          xCells: ebsdData.header.xCells,
          yCells: ebsdData.header.yCells,
          xStep: ebsdData.header.xStep,
          yStep: ebsdData.header.yStep,
          magnification: ebsdData.header.magnification || null,
          acceleratingVoltage: ebsdData.header.acceleratingVoltage || null,
        },
        phases,
        grid: ebsdData.grid,
        metadata: ebsdData.metadata || {},
        totalPoints: stats.totalPoints,
        indexedPoints: stats.indexedPoints,
        indexingRate: stats.totalPoints > 0 ? stats.indexedPoints / stats.totalPoints : 0,
        averageBC: stats.averageBC || 0,
        averageBS: stats.averageBS || 0,
        phaseDistribution: stats.phaseDistribution || {},
      },
      t0
    );
  } catch (err) {
    return makeError('PARSE_ERROR', err.message, err.stack);
  }
}

/**
 * XRD 파일 파싱
 * @param {Object} input
 * @param {number[]|Uint8Array} [input.fileData] - callLocalDirectWithFile 경유 바이너리 데이터
 * @param {string} [input.fileContent] - 텍스트 파일은 string, 바이너리는 Base64
 * @param {string} input.fileType - 'brml' | 'powdll' | 'csv' | 'xy' | 'txt' | 'cif' | 'json'
 * @param {'text'|'base64'} [input.encoding]
 * @param {string} [input.fileName]
 */
export async function parseXRD(input) {
  const t0 = Date.now();
  try {
    const { fileData, fileContent: rawContent, fileType: explicitType, encoding = 'text', fileName } = input;

    const fileType = explicitType || (fileName ? detectFileFormat(fileName) : 'unknown');

    // fileData (Uint8Array/Array from callLocalDirectWithFile) → 적절한 형식으로 변환
    if (fileData && !rawContent) {
      const uint8 = fileData instanceof Uint8Array ? fileData : new Uint8Array(fileData);
      if (fileType === 'brml') {
        const blob = new Blob([uint8]);
        const result = await parseBRML(blob);
        return makeResult(
          {
            dataPoints: result.data || result.dataPoints || [],
            structureInfo: result.structureInfo || { latticeParameters: { a: null, b: null, c: null, alpha: 90, beta: 90, gamma: 90 }, spaceGroup: null, crystalSystem: null },
            metadata: { wavelength: result.metadata?.wavelength || 1.5406, scanRange: result.metadata?.scanRange || { min: null, max: null }, stepSize: result.metadata?.stepSize || null, sampleName: result.metadata?.sampleName || null, instrumentName: result.metadata?.instrumentName || null, measurementDate: result.metadata?.measurementDate || null, rawMetadata: result.metadata || {} },
          },
          t0
        );
      }
      // 텍스트 기반 포맷 → UTF-8 디코딩
      const decoder = new TextDecoder('utf-8');
      const text = decoder.decode(uint8);
      const { parseTextXRD } = await import('./xrd/xrdParser.js');
      const result = await parseTextXRD(text, fileType);
      return makeResult(
        {
          dataPoints: result.data || result.dataPoints || [],
          structureInfo: result.structureInfo || { latticeParameters: { a: null, b: null, c: null, alpha: 90, beta: 90, gamma: 90 }, spaceGroup: null, crystalSystem: null },
          metadata: { wavelength: result.metadata?.wavelength || 1.5406, scanRange: result.metadata?.scanRange || { min: null, max: null }, stepSize: result.metadata?.stepSize || null, sampleName: result.metadata?.sampleName || null, instrumentName: result.metadata?.instrumentName || null, measurementDate: result.metadata?.measurementDate || null, rawMetadata: result.metadata || {} },
        },
        t0
      );
    }

    const fileContent = rawContent;
    if (!fileContent) return makeError('INVALID_INPUT', '파일 내용이 없습니다');

    let result;

    if (fileType === 'brml') {
      let blob;
      if (encoding === 'base64') {
        const binary = atob(fileContent);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        blob = new Blob([bytes]);
      } else {
        blob = new Blob([fileContent]);
      }
      result = await parseBRML(blob);
    } else {
      // 텍스트 기반 포맷 (csv, xy, txt, powdll, cif)
      const { parseTextXRD } = await import('./xrd/xrdParser.js');
      result = await parseTextXRD(fileContent, fileType);
    }

    return makeResult(
      {
        dataPoints: result.data || result.dataPoints || [],
        structureInfo: result.structureInfo || {
          latticeParameters: { a: null, b: null, c: null, alpha: 90, beta: 90, gamma: 90 },
          spaceGroup: null,
          crystalSystem: null,
        },
        metadata: {
          wavelength: result.metadata?.wavelength || 1.5406,
          scanRange: result.metadata?.scanRange || { min: null, max: null },
          stepSize: result.metadata?.stepSize || null,
          sampleName: result.metadata?.sampleName || null,
          instrumentName: result.metadata?.instrumentName || null,
          measurementDate: result.metadata?.measurementDate || null,
          rawMetadata: result.metadata || {},
        },
      },
      t0
    );
  } catch (err) {
    return makeError('PARSE_ERROR', err.message, err.stack);
  }
}

// 저수준 파서 재export
export { detectFileFormat, convertCTFToH5OINA };
