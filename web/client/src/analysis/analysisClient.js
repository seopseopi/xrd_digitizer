import { getApiBase } from '../config/api';

const BACKEND_BASE = `${getApiBase()}/analysis`;

// ── settings (useProcessingLocation 호환) ────────────────────────────
const SETTINGS_KEY = 'xrd_analysis_settings';
const defaultSettings = {
  forceLocal: false,
  forceBackend: true,
  localThresholdMB: 50,
};

const settings = {
  get: () => {
    try {
      const saved = localStorage.getItem(SETTINGS_KEY);
      return saved ? { ...defaultSettings, ...JSON.parse(saved) } : { ...defaultSettings };
    } catch {
      return { ...defaultSettings };
    }
  },
  update: (partial) => {
    const current = settings.get();
    const updated = { ...current, ...partial };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(updated));
    return updated;
  },
  getEndpointTarget: () => 'backend',
};

// ── HTTP helpers ──────────────────────────────────────────────────────
async function callAPI(endpoint, body) {
  const url = `${BACKEND_BASE}/${endpoint}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
}

async function callWithFile(endpoint, file, extraFields = {}) {
  const url = `${BACKEND_BASE}/${endpoint}`;
  const form = new FormData();
  form.append('file', file);
  for (const [k, v] of Object.entries(extraFields)) form.append(k, v);
  const response = await fetch(url, { method: 'POST', body: form });
  return response.json();
}

// ── API ───────────────────────────────────────────────────────────────
export const analysisClient = {
  settings,
  xrd: {
    parse:   (input) => callWithFile('xrd/parse', input.file, { fileType: input.fileType ?? '' }),
    digitize: (imageFile, manualInputs) =>
      callWithFile('xrd/digitize', imageFile, { manual_inputs: JSON.stringify(manualInputs) }),
    detectRoi: (imageFile) => callWithFile('xrd/detect-roi', imageFile, {}),
    detectPeaks:                  (input) => callAPI('xrd/detect-peaks', input),
    fitPeaks:                     (input) => callAPI('xrd/fit-peaks', input),
    calculateCrystallinity:       (input) => callAPI('xrd/calculate-crystallinity', input),
    calculateCrystalliteSizes:    (input) => callAPI('xrd/calculate-crystallite-sizes', input),
    williamsonHallFit:            (input) => callAPI('xrd/williamson-hall-fit', input),
    identifyPhaseCandidates:      (input) => callAPI('xrd/identify-phase-candidates', input),
    computeTextureIndices:        (input) => callAPI('xrd/compute-texture-indices', input),
    estimateQPAPhaseFractions:    (input) => callAPI('xrd/estimate-qpa-phase-fractions', input),
    fitResidualStressSin2Psi:     (input) => callAPI('xrd/fit-residual-stress-sin2-psi', input),
    getRietveldGuidance:          (input) => callAPI('xrd/get-rietveld-guidance', input),
    indexMillerIndices:           (input) => callAPI('xrd/index-miller', input),
    analyzeDislocation:           (input) => callAPI('xrd/analyze-dislocation', input),
  },
};

export default analysisClient;
