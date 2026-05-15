const path = require('path');
const { pathToFileURL } = require('url');

const CORE_URL = pathToFileURL(
  path.resolve(__dirname, '../../client/src/analysis/core/parsers.js')
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
  parseXRD: proxy('parseXRD'),
};
