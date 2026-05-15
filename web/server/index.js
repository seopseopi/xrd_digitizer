require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: true, credentials: false }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// XRD analysis routes
const xrdRouter = require('./analysis/routes/xrd.routes');
app.use('/api/analysis/xrd', xrdRouter);

// Health check
app.get('/api/health', (req, res) => res.json({ ok: true }));

// Serve React build in production
const clientBuild = path.join(__dirname, '../client/build');
if (fs.existsSync(clientBuild)) {
  app.use(express.static(clientBuild));
  app.get('*', (req, res) => res.sendFile(path.join(clientBuild, 'index.html')));
}

// Cleanup temp files on shutdown
const { TEMP_DIR } = require('./analysis/middleware/fileUpload');
process.on('SIGINT', () => { process.exit(0); });

app.listen(PORT, () => console.log(`XRD server running on http://localhost:${PORT}`));
