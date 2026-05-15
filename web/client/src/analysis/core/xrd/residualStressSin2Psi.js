/**
 * sin²ψ 법 (등방성·큐빅 근사): d = d₀ + (dd/dsin²ψ)·sin²ψ
 * σ ≈ E/(1+ν) · (1/d₀) · (dd/dsin²ψ)  — 응력장이 만드는 격자 변형의 1차 근사
 *
 * ψ: 시편 기울기(°), 2θ: 해당 ψ에서의 회절각(°)
 */

function linearRegression(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-30) return null;
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  const meanY = sy / n;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const yh = intercept + slope * xs[i];
    ssTot += (ys[i] - meanY) ** 2;
    ssRes += (ys[i] - yh) ** 2;
  }
  const rSquared = ssTot > 1e-30 ? 1 - ssRes / ssTot : 0;
  return { slope, intercept, rSquared, n };
}

/**
 * @param {Array<{ psiDeg: number, twoThetaDeg: number }>} points
 * @param {number} wavelengthAngstrom
 * @param {Object} elastic
 * @param {number} elastic.youngModulusGPa
 * @param {number} elastic.poissonRatio
 */
export function fitResidualStressFromPsiScan(points, wavelengthAngstrom, elastic) {
  if (!Array.isArray(points) || points.length < 2) {
    return { success: false, error: '잔류 응력 피팅에는 ψ–2θ 점이 최소 2개 필요합니다.' };
  }

  const E = Number(elastic?.youngModulusGPa);
  const nu = Number(elastic?.poissonRatio);
  if (!Number.isFinite(E) || E <= 0 || !Number.isFinite(nu) || nu <= -1 || nu >= 0.5) {
    return { success: false, error: '유효한 탄성계수(E, ν)가 필요합니다.' };
  }

  const xs = [];
  const ds = [];
  for (const p of points) {
    const psi = Number(p.psiDeg);
    const tt = Number(p.twoThetaDeg);
    if (!Number.isFinite(psi) || !Number.isFinite(tt) || tt <= 0) continue;
    const thetaRad = (tt / 2) * (Math.PI / 180);
    const s = Math.sin((psi * Math.PI) / 180);
    const sin2psi = s * s;
    const d = wavelengthAngstrom / (2 * Math.sin(thetaRad));
    if (!Number.isFinite(d) || d <= 0) continue;
    xs.push(sin2psi);
    ds.push(d);
  }

  if (xs.length < 2) {
    return { success: false, error: '유효한 (ψ, 2θ) 점이 2개 미만입니다.' };
  }

  const reg = linearRegression(xs, ds);
  if (!reg) {
    return { success: false, error: '선형 회귀 실패.' };
  }

  const d0 = reg.intercept;
  const slopeD = reg.slope;

  if (Math.abs(d0) < 1e-9) {
    return { success: false, error: 'd₀(절편)이 0에 가깝습니다. 데이터를 확인하세요.' };
  }

  const Epa = E * 1e9;
  const sigmaPa = (Epa / (1 + nu)) * (slopeD / d0);
  const sigmaMpa = sigmaPa / 1e6;

  const fitD = xs.map((x) => reg.intercept + reg.slope * x);

  const warnings = [
    '큐빅·등방성·{hkl} 고정 ψ 스캔 가정입니다. 질강·섬유질 재료는 해석이 달라질 수 있습니다.',
    '부호 규약(인장/압축)은 기기 좌표계와 ψ 정의에 따릅니다.',
  ];

  return {
    success: true,
    stressMPa: sigmaMpa,
    d0Angstrom: d0,
    slopeDAngstromPerSin2Psi: slopeD,
    rSquared: reg.rSquared,
    pointCount: reg.n,
    plotData: {
      sin2Psi: xs,
      dObservedAngstrom: ds,
      dFittedAngstrom: fitD,
    },
    warnings,
  };
}
