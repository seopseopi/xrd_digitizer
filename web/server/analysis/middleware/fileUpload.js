/**
 * 파일 업로드 미들웨어 (multer 기반)
 * 분석 API 엔드포인트에서 파일을 수신하고 임시 저장
 */

const multer = require('multer');
const path = require('path');
const os = require('os');
const fs = require('fs');

const TEMP_DIR = path.join(os.tmpdir(), 'xrd-analysis');

// 임시 디렉토리 생성
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}

// 허용 파일 확장자
const ALLOWED_EXTENSIONS = new Set([
  '.h5oina', '.ctf',           // EBSD
  '.brml', '.powdll', '.csv', '.xy', '.txt', '.cif', // XRD
  '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.json',            // FFT/SAED 이미지
  '.h5',                                               // EDS
]);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, TEMP_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const uniqueName = `${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`;
    cb(null, uniqueName);
  },
});

const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  if (ALLOWED_EXTENSIONS.has(ext)) {
    cb(null, true);
  } else {
    cb(new Error(`지원하지 않는 파일 형식: ${ext}`), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 2 * 1024 * 1024 * 1024, // 2GB 최대
  },
});

/**
 * 단일 파일 업로드 미들웨어
 */
const uploadSingle = upload.single('file');

/**
 * 오류 처리를 포함한 단일 파일 업로드 래퍼
 */
function handleFileUpload(req, res, next) {
  uploadSingle(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({
        success: false,
        error: { code: 'UPLOAD_ERROR', message: `파일 업로드 오류: ${err.message}` },
      });
    }
    if (err) {
      return res.status(400).json({
        success: false,
        error: { code: 'INVALID_FILE', message: err.message },
      });
    }
    next();
  });
}

/**
 * 임시 파일 삭제
 */
function cleanupTempFile(filePath) {
  if (filePath && fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
    } catch {
      // 무시
    }
  }
}

module.exports = { handleFileUpload, cleanupTempFile, TEMP_DIR };
