const path = require('path');
const { pathToFileURL } = require('url');

const CORE_URL = pathToFileURL(
  path.resolve(__dirname, '../../client/src/analysis/core/xrd.js')
).href;

let _module = null;
async function loadModule() {
  if (!_module) _module = await import(CORE_URL);
  return _module;
}

const proxy = (fn) => async (input) => {
  const mod = await loadModule();
  return mod[fn](input);
};

module.exports = {
  detectPeaks:                       proxy('detectPeaks'),
  fitPeaks:                          proxy('fitPeaks'),
  calculateCrystallinity:            proxy('calculateCrystallinity'),
  calculateCrystalliteSizes:         proxy('calculateCrystalliteSizes'),
  williamsonHallFit:                 proxy('williamsonHallFit'),
  identifyPhaseCandidates:           proxy('identifyPhaseCandidates'),
  computeTextureIndicesAnalysis:     proxy('computeTextureIndicesAnalysis'),
  estimateQPAPhaseFractionsAnalysis: proxy('estimateQPAPhaseFractionsAnalysis'),
  fitResidualStressSin2Psi:          proxy('fitResidualStressSin2Psi'),
  getRietveldGuidance:               proxy('getRietveldGuidance'),
  indexMillerIndices:                proxy('indexMillerIndices'),
  analyzeDislocation:                proxy('analyzeDislocation'),
};
