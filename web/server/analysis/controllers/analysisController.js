const fs = require('fs');
const path = require('path');
const { cleanupTempFile } = require('../middleware/fileUpload');

function createAnalysisHandler(toolType, action, syncHandler) {
  return async (req, res) => {
    const tempFilePath = req.file?.path;
    try {
      let payload;
      if (req.file) {
        const fileBuffer = fs.readFileSync(tempFilePath);
        payload = {
          ...(req.body || {}),
          fileData: fileBuffer,
          fileName: req.file.originalname,
          fileType: path.extname(req.file.originalname).slice(1).toLowerCase(),
        };
      } else {
        payload = req.body || {};
      }

      const result = await syncHandler(payload);
      res.json(result);
    } catch (err) {
      console.error(`[${toolType}/${action}] error:`, err.message);
      res.status(500).json({ success: false, error: { message: err.message } });
    } finally {
      cleanupTempFile(tempFilePath);
    }
  };
}

module.exports = { createAnalysisHandler };
