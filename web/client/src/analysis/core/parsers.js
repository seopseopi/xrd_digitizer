/**
 * XRD-only parser — standalone version (EBSD/EDS removed)
 */
import {
  detectFileFormat,
  parseBRML,
} from './xrd/xrdParser.js';

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
  return { success: false, error: { code, message, detail: detail || null } };
}

export async function parseXRD(input) {
  const t0 = Date.now();
  try {
    const { fileData, fileContent: rawContent, fileType: explicitType, encoding = 'text', fileName } = input;
    const fileType = explicitType || (fileName ? detectFileFormat(fileName) : 'unknown');

    if (fileData && !rawContent) {
      const uint8 = fileData instanceof Uint8Array ? fileData : new Uint8Array(fileData);
      if (fileType === 'brml') {
        const blob = new Blob([uint8]);
        const result = await parseBRML(blob);
        return makeResult({ dataPoints: result.data || result.dataPoints || [], structureInfo: result.structureInfo || {}, metadata: { wavelength: result.metadata?.wavelength || 1.5406, ...result.metadata } }, t0);
      }
      const decoder = new TextDecoder('utf-8');
      const text = decoder.decode(uint8);
      const { parseTextXRD } = await import('./xrd/xrdParser.js');
      const result = await parseTextXRD(text, fileType);
      return makeResult({
        dataPoints: result.data || result.dataPoints || [],
        structureInfo: result.structureInfo || { latticeParameters: { a: null, b: null, c: null, alpha: 90, beta: 90, gamma: 90 }, spaceGroup: null, crystalSystem: null },
        metadata: { wavelength: result.metadata?.wavelength || 1.5406, scanRange: result.metadata?.scanRange || { min: null, max: null }, stepSize: result.metadata?.stepSize || null, sampleName: result.metadata?.sampleName || null, instrumentName: result.metadata?.instrumentName || null, measurementDate: result.metadata?.measurementDate || null, rawMetadata: result.metadata || {} },
      }, t0);
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
      const { parseTextXRD } = await import('./xrd/xrdParser.js');
      result = await parseTextXRD(fileContent, fileType);
    }

    return makeResult({
      dataPoints: result.data || result.dataPoints || [],
      structureInfo: result.structureInfo || { latticeParameters: { a: null, b: null, c: null, alpha: 90, beta: 90, gamma: 90 }, spaceGroup: null, crystalSystem: null },
      metadata: { wavelength: result.metadata?.wavelength || 1.5406, scanRange: result.metadata?.scanRange || { min: null, max: null }, stepSize: result.metadata?.stepSize || null, sampleName: result.metadata?.sampleName || null, instrumentName: result.metadata?.instrumentName || null, measurementDate: result.metadata?.measurementDate || null, rawMetadata: result.metadata || {} },
    }, t0);
  } catch (err) {
    return makeError('PARSE_ERROR', err.message, err.stack);
  }
}

export { detectFileFormat };
