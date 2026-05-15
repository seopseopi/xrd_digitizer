/**
 * XRD Digitizer — Python 파이프라인 subprocess 브릿지
 *
 * 환경변수:
 *   XRD_DIGITIZER_PATH      Python 프로젝트 루트 (절대 경로)
 *   XRD_DIGITIZER_PYTHON    Python 실행 파일 경로 (기본: 프로젝트 .venv/bin/python3 → python3)
 *   XRD_DIGITIZER_TIMEOUT_MS subprocess 타임아웃 ms (기본: 120000)
 *   XRD_USE_ML              'false' 로 설정하면 클래식 파이프라인으로 롤백 (기본: ML)
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// ── 경로 설정 ────────────────────────────────────────────────────────────────

const DIGITIZER_DIR = process.env.XRD_DIGITIZER_PATH
  || path.resolve(__dirname, '../python/xrd_digitizer');

const PYTHON = process.env.XRD_DIGITIZER_PYTHON
  || (() => {
    const isWin = process.platform === 'win32';
    const venvPython = isWin
      ? path.join(DIGITIZER_DIR, '.venv', 'Scripts', 'python.exe')
      : path.join(DIGITIZER_DIR, '.venv', 'bin', 'python3');
    if (fs.existsSync(venvPython)) return venvPython;
    return isWin ? 'python' : 'python3';
  })();

const TIMEOUT_MS = parseInt(process.env.XRD_DIGITIZER_TIMEOUT_MS || '120000', 10);

// 파이프라인 선택:
//   XRD_USE_ML=true     → ML CurveExtractorNet
//   XRD_USE_CLASSIC=true → classical (candidate + DP) 파이프라인
//   기본                → simple 픽셀 추출 (깨끗한 이미지에 강함; 양끝 fallback artifact 없음)
const USE_ML      = process.env.XRD_USE_ML === 'true';
const USE_CLASSIC = process.env.XRD_USE_CLASSIC === 'true';

// ── 임시 파일 헬퍼 ────────────────────────────────────────────────────────────

const TEMP_DIR = path.join(os.tmpdir(), 'materiai-xrd-digitizer');
if (!fs.existsSync(TEMP_DIR)) fs.mkdirSync(TEMP_DIR, { recursive: true });

function tempPath(suffix) {
  return path.join(TEMP_DIR, `${Date.now()}-${Math.random().toString(36).slice(2)}${suffix}`);
}

function safeUnlink(p) {
  if (p && fs.existsSync(p)) try { fs.unlinkSync(p); } catch { /* ignore */ }
}

// ── subprocess 실행 ───────────────────────────────────────────────────────────

function runPython(args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, args, {
      cwd: DIGITIZER_DIR,
      env: { ...process.env, PYTHONPATH: DIGITIZER_DIR },
    });

    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => { stdout += d.toString(); });
    proc.stderr.on('data', (d) => { stderr += d.toString(); });

    const timer = setTimeout(() => {
      proc.kill('SIGTERM');
      reject(new Error(`XRD digitizer timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    proc.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Python exited with code ${code}. stderr: ${stderr.slice(0, 500)}`));
      } else {
        resolve({ stdout, stderr });
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });
  });
}

// ── 메인 함수 ─────────────────────────────────────────────────────────────────

/**
 * XRD 패턴 이미지를 수치 데이터로 복원한다.
 *
 * @param {object} payload
 * @param {Buffer} payload.fileData        이미지 버퍼 (PNG/JPG/TIFF)
 * @param {string} payload.fileName        원본 파일명
 * @param {string} payload.manual_inputs   mi.json JSON 문자열
 * @returns {Promise<object>}              { success, data, meta }
 */
async function digitizeXRD(payload) {
  const t0 = Date.now();
  const { fileData, fileName, manual_inputs } = payload;

  // ── 입력 검증 ──
  if (!fileData) {
    return { success: false, error: { code: 'MISSING_FILE', message: '이미지 파일이 필요합니다' } };
  }
  if (!manual_inputs) {
    return { success: false, error: { code: 'MISSING_MANUAL_INPUTS', message: 'manual_inputs JSON이 필요합니다' } };
  }

  let miJson;
  try {
    miJson = typeof manual_inputs === 'string' ? JSON.parse(manual_inputs) : manual_inputs;
  } catch {
    return { success: false, error: { code: 'INVALID_MANUAL_INPUTS', message: 'manual_inputs가 유효한 JSON이 아닙니다' } };
  }

  // 필수 필드 확인
  const required = ['plot_box', 'x_axis_points', 'x_axis_values', 'y_axis_points', 'y_axis_values', 'color_sample_point'];
  const missing = required.filter((k) => !(k in miJson));
  if (missing.length > 0) {
    return { success: false, error: { code: 'INVALID_MANUAL_INPUTS', message: `manual_inputs 누락 필드: ${missing.join(', ')}` } };
  }

  // ── 임시 파일 저장 ──
  const ext = path.extname(fileName || '.png').toLowerCase() || '.png';
  const imgPath = tempPath(ext);
  const miPath  = tempPath('.json');

  try {
    fs.writeFileSync(imgPath, fileData);
    fs.writeFileSync(miPath, JSON.stringify({
      ...miJson,
      legend_ignore_boxes: miJson.legend_ignore_boxes ?? [],
      perspective_corners:  miJson.perspective_corners  ?? null,
      color_resample_points: miJson.color_resample_points ?? [],
    }, null, 2), 'utf-8');

    // ── Python 실행 ──
    let pythonArgs;
    if (USE_ML) {
      pythonArgs = ['-m', 'runner.run_ml_curve',
        '--image_path',         imgPath,
        '--manual_inputs_path', miPath,
        '--stdout',
        '--no-debug'];
    } else if (USE_CLASSIC) {
      pythonArgs = ['-m', 'runner.run_local',
        '--image_path',         imgPath,
        '--manual_inputs_path', miPath,
        '--stdout',
        '--no-debug',
        '--roi-upscale-factor', '2'];
    } else {
      pythonArgs = ['-m', 'runner.run_simple',
        '--image_path',         imgPath,
        '--manual_inputs_path', miPath,
        '--stdout',
        '--no-debug'];
    }
    const { stdout } = await runPython(pythonArgs, TIMEOUT_MS);

    // stdout에서 JSON 추출 (stderr 로그가 섞이지 않도록 --stdout은 stderr로 로그 분리됨)
    const jsonStart = stdout.indexOf('{');
    if (jsonStart === -1) {
      throw new Error('Python output에 JSON이 없습니다');
    }
    const result = JSON.parse(stdout.slice(jsonStart));

    return {
      success: true,
      data: {
        two_theta_values:    result.two_theta_values    ?? [],
        intensities:         result.intensities         ?? [],
        peaks_numeric_curve: result.peaks_numeric_curve ?? [],
        x_range:             result.x_range             ?? null,
        y_range:             result.y_range             ?? null,
        confidence:          result.confidence          ?? null,
        warnings:            result.warnings            ?? [],
      },
      meta: {
        processingTimeMs: Date.now() - t0,
        processedAt: new Date().toISOString(),
        processingLocation: 'backend',
        pointCount: (result.two_theta_values ?? []).length,
        peakCount:  (result.peaks_numeric_curve ?? []).length,
      },
    };
  } finally {
    safeUnlink(imgPath);
    safeUnlink(miPath);
  }
}

// ── 자동 감지 ─────────────────────────────────────────────────────────────────

/**
 * XRD 패턴 이미지에서 축선 픽셀 좌표를 자동 감지한다.
 *
 * @param {object} payload
 * @param {Buffer} payload.fileData   이미지 버퍼
 * @param {string} payload.fileName   원본 파일명
 * @returns {Promise<object>}         { success, data }
 */
async function detectRoiXRD(payload) {
  const t0 = Date.now();
  const { fileData, fileName } = payload;

  if (!fileData) {
    return { success: false, error: { code: 'MISSING_FILE', message: '이미지 파일이 필요합니다' } };
  }

  const ext = path.extname(fileName || '.png').toLowerCase() || '.png';
  const imgPath = tempPath(ext);

  try {
    fs.writeFileSync(imgPath, fileData);

    const { stdout } = await runPython([
      '-m', 'runner.run_detect',
      '--image_path', imgPath,
      '--stdout',
    ], 30000);

    const jsonStart = stdout.indexOf('{');
    if (jsonStart === -1) throw new Error('Python output에 JSON이 없습니다');
    const result = JSON.parse(stdout.slice(jsonStart));

    if (!result.success) {
      return { success: false, error: { code: 'DETECT_FAILED', message: result.error || '감지 실패' } };
    }

    return {
      success: true,
      data: {
        calib_points:       result.calib_points,
        curve_color:        result.curve_color        ?? null,
        color_sample_point: result.color_sample_point ?? null,
        axis_values:        result.axis_values        ?? null,
        confidence:         result.confidence         ?? null,
        ocr_available:      result.ocr_available      ?? false,
      },
      meta: { processingTimeMs: Date.now() - t0 },
    };
  } finally {
    safeUnlink(imgPath);
  }
}

module.exports = { digitizeXRD, detectRoiXRD };
