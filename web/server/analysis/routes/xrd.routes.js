const express = require('express');
const router = express.Router();
const { handleFileUpload } = require('../middleware/fileUpload');
const { createAnalysisHandler } = require('../controllers/analysisController');

const { parseXRD } = require('../core/parsers');
const { digitizeXRD, detectRoiXRD } = require('../core/xrdDigitizer');
const {
  detectPeaks,
  fitPeaks,
  calculateCrystallinity,
  calculateCrystalliteSizes,
  williamsonHallFit,
  identifyPhaseCandidates,
  computeTextureIndicesAnalysis,
  estimateQPAPhaseFractionsAnalysis,
  fitResidualStressSin2Psi,
  getRietveldGuidance,
  indexMillerIndices,
  analyzeDislocation,
} = require('../core/xrd');

/**
 * @openapi
 * /api/analysis/xrd/parse:
 *   post:
 *     tags: [분석-XRD]
 *     summary: XRD 파일 파싱
 *     description: XRD 데이터 파일(.xy, .dat, .csv 등)을 파싱하여 2θ-강도 데이터를 반환합니다. JWT 인증 필요.
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *     responses:
 *       200:
 *         description: 파싱된 XRD 데이터 (2θ 배열, 강도 배열)
 *       202:
 *         description: 비동기 처리 시작됨 (대용량 파일). jobId 반환.
 */
router.post('/parse', handleFileUpload, createAnalysisHandler('xrd', 'parse', parseXRD));
/**
 * @openapi
 * /api/analysis/xrd/detect-peaks:
 *   post:
 *     tags: [분석-XRD]
 *     summary: 피크 검출
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             description: 파싱된 XRD 데이터 + 피크 검출 파라미터
 *     responses:
 *       200:
 *         description: 검출된 피크 목록
 *
 * /api/analysis/xrd/fit-peaks:
 *   post:
 *     tags: [분석-XRD]
 *     summary: 피크 피팅 (Pseudo-Voigt)
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *     responses:
 *       200:
 *         description: 피팅 결과
 *
 * /api/analysis/xrd/calculate-crystallinity:
 *   post:
 *     tags: [분석-XRD]
 *     summary: 결정화도 계산
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *     responses:
 *       200:
 *         description: 결정화도 (%)
 *
 * /api/analysis/xrd/calculate-crystallite-sizes:
 *   post:
 *     tags: [분석-XRD]
 *     summary: 결정립 크기 계산 (Scherrer 방정식)
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *     responses:
 *       200:
 *         description: 결정립 크기 (nm)
 *
 * /api/analysis/xrd/index-miller:
 *   post:
 *     tags: [분석-XRD]
 *     summary: Miller 지수 인덱싱
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *     responses:
 *       200:
 *         description: Miller 지수 인덱싱 결과
 *
 * /api/analysis/xrd/analyze-dislocation:
 *   post:
 *     tags: [분석-XRD]
 *     summary: 전위 밀도 분석 (Williamson-Hall)
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *     responses:
 *       200:
 *         description: 전위 밀도 분석 결과
 */
/**
 * @openapi
 * /api/analysis/xrd/digitize:
 *   post:
 *     tags: [분석-XRD]
 *     summary: XRD 패턴 이미지 → 수치 데이터 복원 (Digitizer)
 *     description: |
 *       XRD 그래프 이미지와 축 보정 좌표(manual_inputs)를 받아
 *       (2θ, intensity) 수치 배열과 피크 목록을 반환합니다.
 *       내부적으로 Python deterministic pipeline을 subprocess로 실행합니다.
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             required: [file, manual_inputs]
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *                 description: XRD 패턴 이미지 (PNG / JPEG / TIFF)
 *               manual_inputs:
 *                 type: string
 *                 description: |
 *                   축 보정 JSON 문자열. 필수 필드:
 *                   plot_box, x_axis_points, x_axis_values,
 *                   y_axis_points, y_axis_values, color_sample_point
 *     responses:
 *       200:
 *         description: 복원된 수치 데이터
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success: { type: boolean }
 *                 data:
 *                   type: object
 *                   properties:
 *                     two_theta_values: { type: array, items: { type: number } }
 *                     intensities:      { type: array, items: { type: number } }
 *                     peaks_numeric_curve: { type: array }
 *                     x_range:   { type: array }
 *                     y_range:   { type: array }
 *                     confidence: { type: number }
 *                     warnings:  { type: array }
 */
router.post('/digitize', handleFileUpload, createAnalysisHandler('xrd', 'digitize', digitizeXRD));
router.post('/detect-roi', handleFileUpload, createAnalysisHandler('xrd', 'detect-roi', detectRoiXRD));

router.post('/detect-peaks', createAnalysisHandler('xrd', 'detect-peaks', detectPeaks));
router.post('/fit-peaks', createAnalysisHandler('xrd', 'fit-peaks', fitPeaks));
router.post('/calculate-crystallinity', createAnalysisHandler('xrd', 'calculate-crystallinity', calculateCrystallinity));
router.post('/calculate-crystallite-sizes', createAnalysisHandler('xrd', 'calculate-crystallite-sizes', calculateCrystalliteSizes));
router.post('/williamson-hall-fit', createAnalysisHandler('xrd', 'williamson-hall-fit', williamsonHallFit));
router.post('/identify-phase-candidates', createAnalysisHandler('xrd', 'identify-phase-candidates', identifyPhaseCandidates));
router.post('/compute-texture-indices', createAnalysisHandler('xrd', 'compute-texture-indices', computeTextureIndicesAnalysis));
router.post('/estimate-qpa-phase-fractions', createAnalysisHandler('xrd', 'estimate-qpa-phase-fractions', estimateQPAPhaseFractionsAnalysis));
router.post('/fit-residual-stress-sin2-psi', createAnalysisHandler('xrd', 'fit-residual-stress-sin2-psi', fitResidualStressSin2Psi));
router.post('/get-rietveld-guidance', createAnalysisHandler('xrd', 'get-rietveld-guidance', getRietveldGuidance));
router.post('/index-miller', createAnalysisHandler('xrd', 'index-miller', indexMillerIndices));
router.post('/analyze-dislocation', createAnalysisHandler('xrd', 'analyze-dislocation', analyzeDislocation));

module.exports = router;
