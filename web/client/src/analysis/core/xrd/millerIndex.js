/**
 * Bravais 격자 센터링 및 공간군 기저에 따른 반사 허용 여부 판단
 *
 * 지원 centering:
 *   'P'   : 조건 없음 (모든 hkl 허용)
 *   'I'   : h+k+l = 2n
 *   'F'   : h,k,l 모두 홀수 또는 모두 짝수
 *   'C'   : h+k = 2n
 *   'A'   : k+l = 2n
 *   'B'   : h+l = 2n
 *   'R'   : -h+k+l = 3n  (rhombohedral, obverse)
 *   'HCP' : P6₃/mmc 기저 소멸 포함
 *           → (h+2k) ≡ 0(mod 3) AND l 홀수 → 금지
 *
 * @param {number} h
 * @param {number} k
 * @param {number} l
 * @param {string} [centering='P']
 * @returns {boolean} 반사가 허용되면 true
 */
export function isSystematicallyAllowed(h, k, l, centering = 'P') {
  if (h === 0 && k === 0 && l === 0) return false;

  const c = (centering || 'P').toUpperCase();
  switch (c) {
    case 'P':
      return true;
    case 'I':
      return (h + k + l) % 2 === 0;
    case 'F': {
      const allOdd  = h % 2 !== 0 && k % 2 !== 0 && l % 2 !== 0;
      const allEven = h % 2 === 0 && k % 2 === 0 && l % 2 === 0;
      return allOdd || allEven;
    }
    case 'C':
      return (h + k) % 2 === 0;
    case 'A':
      return (k + l) % 2 === 0;
    case 'B':
      return (h + l) % 2 === 0;
    case 'R':
      return ((-h + k + l) % 3 + 3) % 3 === 0;
    case 'HCP': {
      // P6₃/mmc (#194): 기저 소멸
      // (h + 2k) ≡ 0 (mod 3) AND l 홀수 → |F| = 0
      const mod3 = ((h + 2 * k) % 3 + 3) % 3;
      if (mod3 === 0 && l % 2 !== 0) return false;
      return true;
    }
    default:
      return true;
  }
}

/**
 * Bragg 방정식을 사용한 d-spacing 계산
 * nλ = 2d sin(θ)
 * @param {number} angle - 브래그 각도 (2θ, 도 단위)
 * @param {number} wavelength - X선 파장 (Å, 기본값: Cu Kα = 1.5406)
 * @param {number} order - 회절 차수 (기본값: 1)
 * @returns {number} d-spacing (Å)
 */
export const calculateDSpacing = (angle, wavelength = 1.5406, order = 1) => {
  const thetaRad = (angle / 2) * Math.PI / 180; // 2θ를 θ로 변환 후 라디안
  const d = (order * wavelength) / (2 * Math.sin(thetaRad));
  return d;
};

// ── 큐빅·정방정 면족(family) — 등가 순열을 하나로 묶음 ─────────────────

/**
 * 큐빅에서 동일 d(동일 h²+k²+l²)인 순열의 정규지수: |h|,|k|,|l| 내림차순 (관례적으로 h≥k≥l)
 * 예: (112)(121)(211) → {211}
 */
export function cubicFamilyCanonicalIndices(h, k, l) {
  const tri = [Math.abs(h), Math.abs(k), Math.abs(l)].sort((a, b) => b - a);
  return { h: tri[0], k: tri[1], l: tri[2] };
}

export function cubicFamilyKey(h, k, l) {
  const { h: H, k: K, l: L } = cubicFamilyCanonicalIndices(h, k, l);
  return `${H},${K},${L}`;
}

/** 큐빅 면족 문자열 {hkl} */
export function formatCubicMillerFamily(h, k, l) {
  const { h: H, k: K, l: L } = cubicFamilyCanonicalIndices(h, k, l);
  return `{${H}${K}${L}}`;
}

/**
 * 정방정(a=b): (hkl)과 (khl) 동일 d → |h|,|k| 중 큰 값을 먼저 쓰는 정규형
 */
export function tetragonalFamilyCanonicalIndices(h, k, l) {
  const ah = Math.abs(h);
  const ak = Math.abs(k);
  const al = Math.abs(l);
  const hi = Math.max(ah, ak);
  const lo = Math.min(ah, ak);
  return { h: hi, k: lo, l: al };
}

export function tetragonalFamilyKey(h, k, l) {
  const { h: H, k: K, l: L } = tetragonalFamilyCanonicalIndices(h, k, l);
  return `${H},${K},${L}`;
}

export function formatTetragonalMillerFamily(h, k, l) {
  const { h: H, k: K, l: L } = tetragonalFamilyCanonicalIndices(h, k, l);
  return `{${H}${K}${L}}`;
}

/**
 * 결정계에 맞게 면족 단위로 후보 병합 (가장 좋은 매칭만 유지)
 * @param {Array<{h,k,l,matchScore?,diff?}>} candidates
 * @param {string} crystalSystem  — 'cubic' | 'tetragonal' | ...
 */
export function dedupeMillerCandidatesByCrystalSystem(candidates, crystalSystem = 'cubic') {
  if (!candidates?.length) return [];
  const sys = (crystalSystem || 'cubic').toLowerCase();

  const scoreOf = (c) => {
    if (c.matchScore != null) return c.matchScore;
    if (c.diff != null) return 1 / (1e-9 + c.diff);
    return 0;
  };

  if (sys === 'cubic') {
    const m = new Map();
    for (const c of candidates) {
      const key = cubicFamilyKey(c.h, c.k, c.l);
      const prev = m.get(key);
      if (!prev || scoreOf(c) > scoreOf(prev)) m.set(key, c);
    }
    return [...m.values()].sort((a, b) => scoreOf(b) - scoreOf(a));
  }

  if (sys === 'tetragonal') {
    const m = new Map();
    for (const c of candidates) {
      const key = tetragonalFamilyKey(c.h, c.k, c.l);
      const prev = m.get(key);
      if (!prev || scoreOf(c) > scoreOf(prev)) m.set(key, c);
    }
    return [...m.values()].sort((a, b) => scoreOf(b) - scoreOf(a));
  }

  return [...candidates];
}

/** UI·라벨: 결정계별 면족(또는 단순 (hkl)) */
export function formatMillerIndicesForDisplay(h, k, l, crystalSystem = 'cubic') {
  const sys = (crystalSystem || 'cubic').toLowerCase();
  if (sys === 'cubic') return formatCubicMillerFamily(h, k, l);
  if (sys === 'tetragonal') return formatTetragonalMillerFamily(h, k, l);
  const hs = h != null ? String(h) : '';
  const ks = k != null ? String(k) : '';
  const ls = l != null ? String(l) : '';
  return `(${hs}${ks}${ls})`;
}

/**
 * 관측 d-spacing과 후보 (hkl)가 맞다고 가정할 때 역산되는 격자상수(힌트).
 * d는 Bragg로 정해지나, (hkl)·면족 자체는 단위셀을 알아야 확정된다는 전제를 UI에 전달하기 위함.
 *
 * @param {number} dAngstrom
 * @param {number} h
 * @param {number} k
 * @param {number} l
 * @param {string} crystalSystem
 * @param {Object} [extra]
 * @param {number} [extra.c] — 정방정·육방정에서 문헌/CIF c (Å)
 * @returns {{ kind: string, a?: number, c?: number, text: string } | null}
 */
export function impliedLatticeFromPeak(dAngstrom, h, k, l, crystalSystem = 'cubic', extra = {}) {
  if (dAngstrom == null || dAngstrom <= 0 || !Number.isFinite(dAngstrom)) return null;
  if (h == null || k == null || l == null) return null;

  const sys = (crystalSystem || 'cubic').toLowerCase();

  if (sys === 'cubic') {
    const { h: H, k: K, l: L } = cubicFamilyCanonicalIndices(h, k, l);
    const N = H * H + K * K + L * L;
    if (N <= 0) return null;
    const a = dAngstrom * Math.sqrt(N);
    return {
      kind: 'cubic',
      a,
      text: `a ≈ ${a.toFixed(4)} Å`,
      detail: `큐빅, 면족 {${H}${K}${L}} 가정: a = d√(h²+k²+l²)`,
    };
  }

  if (sys === 'tetragonal') {
    const { h: H, k: K, l: L } = tetragonalFamilyCanonicalIndices(h, k, l);
    const invD2 = 1 / (dAngstrom * dAngstrom);
    const c = extra.c;
    if (L === 0 && (H > 0 || K > 0)) {
      const N = H * H + K * K;
      if (N <= 0) return null;
      const a = dAngstrom * Math.sqrt(N);
      return {
        kind: 'tetragonal',
        a,
        text: `a ≈ ${a.toFixed(4)} Å`,
        detail: '정방정, l=0: a = d√(h²+k²)',
      };
    }
    if (c && c > 0 && Number.isFinite(c)) {
      const termL = (L * L) / (c * c);
      const rest = invD2 - termL;
      const plane = H * H + K * K;
      if (rest <= 0 || plane <= 0) {
        return {
          kind: 'tetragonal',
          text: '역산 불가',
          detail: `주어진 c=${c}Å·지수와 d가 정합하지 않습니다.`,
        };
      }
      const a = Math.sqrt(plane / rest);
      return {
        kind: 'tetragonal',
        a,
        c,
        text: `a ≈ ${a.toFixed(4)} Å`,
        detail: `정방정, 문헌 c=${c}Å 가정`,
      };
    }
    return {
      kind: 'tetragonal',
      text: 'a,c 필요',
      detail: '정방정은 보통 여러 피크와 c값(CIF)·c/a로 연립 확인',
    };
  }

  if (sys === 'hexagonal') {
    const c = extra.c;
    const hexN = h * h + h * k + k * k;
    const LC = Math.abs(l);
    const invD2 = 1 / (dAngstrom * dAngstrom);
    if (!c || c <= 0) {
      return {
        kind: 'hexagonal',
        text: 'a 역산 조건 부족',
        detail: '육방정은 보통 CIF의 c(Å)가 있어야 a를 역산합니다.',
      };
    }
    const termL = (LC * LC) / (c * c);
    const rest = invD2 - termL;
    if (rest <= 0 || hexN <= 0) {
      return {
        kind: 'hexagonal',
        text: '역산 불가',
        detail: '지수·c·d 정합을 확인하세요.',
      };
    }
    const a = Math.sqrt((4 * hexN) / (3 * rest));
    return {
      kind: 'hexagonal',
      a,
      c,
      text: `a ≈ ${a.toFixed(4)} Å`,
      detail: `육방정, c=${c}Å 가정`,
    };
  }

  return {
    kind: sys || 'unknown',
    text: '—',
    detail: '이 결정계는 CIF·다중 피크로 격자를 맞추는 것이 안전합니다.',
  };
}

/**
 * 큐빅 가정일 때 피크들의 1순위 후보로부터 a 역산값 요약
 */
export function summarizeImpliedCubicAFromPeaks(peaks, crystalSystem = 'cubic') {
  const sys = (crystalSystem || 'cubic').toLowerCase();
  if (sys !== 'cubic' || !peaks?.length) return null;
  const values = [];
  for (const p of peaks) {
    const best = p?.millerIndices?.[0];
    if (!best || p.dSpacing == null) continue;
    const hint = impliedLatticeFromPeak(p.dSpacing, best.h, best.k, best.l, 'cubic', {});
    if (hint?.a != null && Number.isFinite(hint.a)) values.push(hint.a);
  }
  if (values.length === 0) return null;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const varSum = values.reduce((s, v) => s + (v - mean) ** 2, 0);
  const std = values.length > 1 ? Math.sqrt(varSum / (values.length - 1)) : 0;
  return { mean, std, n: values.length, values };
}

function dedupeCubicMillerByFamily(candidates) {
  const m = new Map();
  for (const c of candidates) {
    const key = cubicFamilyKey(c.h, c.k, c.l);
    const prev = m.get(key);
    if (!prev || c.diff < prev.diff) m.set(key, c);
  }
  return [...m.values()].sort((a, b) => a.diff - b.diff);
}

function dedupeTetragonalMillerByFamily(candidates) {
  const m = new Map();
  for (const c of candidates) {
    const key = tetragonalFamilyKey(c.h, c.k, c.l);
    const prev = m.get(key);
    if (!prev || c.diff < prev.diff) m.set(key, c);
  }
  return [...m.values()].sort((a, b) => a.diff - b.diff);
}

/**
 * 밀러지수 (hkl)로부터 d-spacing 계산
 * @param {number} h - 밀러지수 h
 * @param {number} k - 밀러지수 k
 * @param {number} l - 밀러지수 l
 * @param {Object} latticeParams - 격자 상수
 * @param {string} latticeParams.system - 결정계 ('cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'monoclinic', 'triclinic')
 * @param {number} latticeParams.a - 격자 상수 a (Å)
 * @param {number} latticeParams.b - 격자 상수 b (Å, tetragonal 이상)
 * @param {number} latticeParams.c - 격자 상수 c (Å, orthorhombic 이상)
 * @param {number} latticeParams.alpha - 각도 α (도, monoclinic 이상)
 * @param {number} latticeParams.beta - 각도 β (도, monoclinic 이상)
 * @param {number} latticeParams.gamma - 각도 γ (도, triclinic)
 * @returns {number} d-spacing (Å)
 */
export const calculateDSpacingFromMillerIndices = (h, k, l, latticeParams) => {
  const { system, a, b, c, alpha, beta, gamma } = latticeParams;

  if (!system || !a) {
    throw new Error('격자 상수가 필요합니다.');
  }

  let d;

  switch (system.toLowerCase()) {
    case 'cubic':
      // 1/d² = (h² + k² + l²) / a²
      d = a / Math.sqrt(h * h + k * k + l * l);
      break;

    case 'tetragonal':
      if (!c) throw new Error('격자 상수 c가 필요합니다.');
      // 1/d² = h²/a² + k²/a² + l²/c²
      d = 1 / Math.sqrt((h * h + k * k) / (a * a) + (l * l) / (c * c));
      break;

    case 'orthorhombic':
      if (!b || !c) throw new Error('격자 상수 b, c가 필요합니다.');
      // 1/d² = h²/a² + k²/b² + l²/c²
      d = 1 / Math.sqrt((h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c));
      break;

    case 'hexagonal':
      if (!c) throw new Error('격자 상수 c가 필요합니다.');
      // 1/d² = 4(h² + hk + k²)/(3a²) + l²/c²
      d = 1 / Math.sqrt(4 * (h * h + h * k + k * k) / (3 * a * a) + (l * l) / (c * c));
      break;

    case 'monoclinic':
      if (!b || !c || beta === undefined) throw new Error('격자 상수 b, c, beta가 필요합니다.');
      const betaRad = beta * Math.PI / 180;
      // 1/d² = h²/(a²sin²β) + k²/b² + l²/(c²sin²β) - 2hlcosβ/(acsin²β)
      const sin2Beta = Math.sin(betaRad) * Math.sin(betaRad);
      d = 1 / Math.sqrt(
        (h * h) / (a * a * sin2Beta) +
        (k * k) / (b * b) +
        (l * l) / (c * c * sin2Beta) -
        (2 * h * l * Math.cos(betaRad)) / (a * c * sin2Beta)
      );
      break;

    case 'triclinic':
      if (!b || !c || alpha === undefined || beta === undefined || gamma === undefined) {
        throw new Error('모든 격자 상수와 각도가 필요합니다.');
      }
      // 복잡한 공식 (일반적으로 라이브러리 사용 권장)
      // 여기서는 간단한 근사 사용
      throw new Error('Triclinic 시스템은 아직 지원되지 않습니다.');

    default:
      throw new Error(`지원되지 않는 결정계: ${system}`);
  }

  return d;
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Cubic 시스템)
 * @param {number} dSpacing - d-spacing (Å)
 * @param {number} a - 격자 상수 a (Å)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesCubic = (dSpacing, a, maxHKL = 10, wavelength = 1.5406, tolerance = 0.01) => {
  const candidates = [];

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        if (h === 0 && k === 0 && l === 0) continue; // (000) 제외

        const d = a / Math.sqrt(h * h + k * k + l * l);
        const diff = Math.abs(d - dSpacing);

        if (diff < tolerance) {
          // 2θ 계산
          const thetaRad = Math.asin(wavelength / (2 * d));
          const angle = 2 * thetaRad * 180 / Math.PI;

          candidates.push({
            h, k, l,
            d: d,
            angle: angle,
            diff: diff
          });
        }
      }
    }
  }

  // 차이값으로 정렬
  candidates.sort((a, b) => a.diff - b.diff);

  return dedupeCubicMillerByFamily(candidates);
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Tetragonal 시스템)
 * @param {number} dSpacing - d-spacing (Å)
 * @param {number} a - 격자 상수 a (Å)
 * @param {number} c - 격자 상수 c (Å)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesTetragonal = (dSpacing, a, c, maxHKL = 10, wavelength = 1.5406, tolerance = 0.01) => {
  const candidates = [];

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        if (h === 0 && k === 0 && l === 0) continue; // (000) 제외

        // 1/d² = (h²+k²)/a² + l²/c²
        const d = 1 / Math.sqrt((h * h + k * k) / (a * a) + (l * l) / (c * c));
        const diff = Math.abs(d - dSpacing);

        if (diff < tolerance) {
          const thetaRad = Math.asin(wavelength / (2 * d));
          const angle = 2 * thetaRad * 180 / Math.PI;

          candidates.push({
            h, k, l,
            d: d,
            angle: angle,
            diff: diff
          });
        }
      }
    }
  }

  candidates.sort((a, b) => a.diff - b.diff);
  return dedupeTetragonalMillerByFamily(candidates);
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Orthorhombic 시스템)
 * @param {number} dSpacing - d-spacing (Å)
 * @param {number} a - 격자 상수 a (Å)
 * @param {number} b - 격자 상수 b (Å)
 * @param {number} c - 격자 상수 c (Å)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesOrthorhombic = (dSpacing, a, b, c, maxHKL = 10, wavelength = 1.5406, tolerance = 0.01) => {
  const candidates = [];

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        if (h === 0 && k === 0 && l === 0) continue; // (000) 제외

        // 1/d² = h²/a² + k²/b² + l²/c²
        const d = 1 / Math.sqrt((h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c));
        const diff = Math.abs(d - dSpacing);

        if (diff < tolerance) {
          const thetaRad = Math.asin(wavelength / (2 * d));
          const angle = 2 * thetaRad * 180 / Math.PI;

          candidates.push({
            h, k, l,
            d: d,
            angle: angle,
            diff: diff
          });
        }
      }
    }
  }

  candidates.sort((a, b) => a.diff - b.diff);
  return candidates;
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Hexagonal 시스템)
 * @param {number} dSpacing - d-spacing (Å)
 * @param {number} a - 격자 상수 a (Å)
 * @param {number} c - 격자 상수 c (Å)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesHexagonal = (dSpacing, a, c, maxHKL = 10, wavelength = 1.5406, tolerance = 0.01) => {
  const candidates = [];

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        if (h === 0 && k === 0 && l === 0) continue; // (000) 제외

        // 1/d² = 4(h² + hk + k²)/(3a²) + l²/c²
        const d = 1 / Math.sqrt(4 * (h * h + h * k + k * k) / (3 * a * a) + (l * l) / (c * c));
        const diff = Math.abs(d - dSpacing);

        if (diff < tolerance) {
          const thetaRad = Math.asin(wavelength / (2 * d));
          const angle = 2 * thetaRad * 180 / Math.PI;

          candidates.push({
            h, k, l,
            d: d,
            angle: angle,
            diff: diff
          });
        }
      }
    }
  }

  candidates.sort((a, b) => a.diff - b.diff);
  return candidates;
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Monoclinic 시스템)
 * @param {number} dSpacing - d-spacing (Å)
 * @param {number} a - 격자 상수 a (Å)
 * @param {number} b - 격자 상수 b (Å)
 * @param {number} c - 격자 상수 c (Å)
 * @param {number} beta - 각도 β (도)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesMonoclinic = (dSpacing, a, b, c, beta, maxHKL = 10, wavelength = 1.5406, tolerance = 0.01) => {
  const candidates = [];
  const betaRad = beta * Math.PI / 180;
  const sin2Beta = Math.sin(betaRad) * Math.sin(betaRad);

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        if (h === 0 && k === 0 && l === 0) continue; // (000) 제외

        // 1/d² = h²/(a²sin²β) + k²/b² + l²/(c²sin²β) - 2hlcosβ/(acsin²β)
        const d = 1 / Math.sqrt(
          (h * h) / (a * a * sin2Beta) +
          (k * k) / (b * b) +
          (l * l) / (c * c * sin2Beta) -
          (2 * h * l * Math.cos(betaRad)) / (a * c * sin2Beta)
        );
        const diff = Math.abs(d - dSpacing);

        if (diff < tolerance) {
          const thetaRad = Math.asin(wavelength / (2 * d));
          const angle = 2 * thetaRad * 180 / Math.PI;

          candidates.push({
            h, k, l,
            d: d,
            angle: angle,
            diff: diff
          });
        }
      }
    }
  }

  candidates.sort((a, b) => a.diff - b.diff);
  return candidates;
};

/**
 * 역격자 행렬 계산 (Triclinic 시스템용)
 * @param {Object} latticeParams - 격자 상수
 * @returns {Array<Array<number>>} 역격자 행렬 (3x3)
 */
/* eslint-disable-next-line no-unused-vars */
const calculateReciprocalLatticeMatrix = (latticeParams) => {
  const { a, b, c, alpha, beta, gamma } = latticeParams;
  const alphaRad = alpha * Math.PI / 180;
  const betaRad = beta * Math.PI / 180;
  const gammaRad = gamma * Math.PI / 180;

  // 단위 셀 부피 계산
  const V = a * b * c * Math.sqrt(
    1 - Math.cos(alphaRad) * Math.cos(alphaRad) -
    Math.cos(betaRad) * Math.cos(betaRad) -
    Math.cos(gammaRad) * Math.cos(gammaRad) +
    2 * Math.cos(alphaRad) * Math.cos(betaRad) * Math.cos(gammaRad)
  );

  // 역격자 상수
  const aStar = (b * c * Math.sin(alphaRad)) / V;
  const bStar = (a * c * Math.sin(betaRad)) / V;
  const cStar = (a * b * Math.sin(gammaRad)) / V;

  // 역격자 각도 (alphaStar는 현재 행렬 구성에 미사용, 대칭성 유지용)
  const betaStar = Math.acos(
    (Math.cos(alphaRad) * Math.cos(gammaRad) - Math.cos(betaRad)) /
    (Math.sin(alphaRad) * Math.sin(gammaRad))
  );
  const gammaStar = Math.acos(
    (Math.cos(alphaRad) * Math.cos(betaRad) - Math.cos(gammaRad)) /
    (Math.sin(alphaRad) * Math.sin(betaRad))
  );

  // 역격자 행렬 구성
  const cosBetaStar = Math.cos(betaStar);
  const cosGammaStar = Math.cos(gammaStar);
  const sinGammaStar = Math.sin(gammaStar);

  return [
    [aStar, bStar * cosGammaStar, cStar * cosBetaStar],
    [0, bStar * sinGammaStar, cStar * (Math.cos(alphaRad) - Math.cos(betaRad) * cosGammaStar) / Math.sin(gammaRad)],
    [0, 0, cStar * Math.sin(betaRad)]
  ];
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Triclinic 시스템)
 * @param {number} dSpacing - d-spacing (Å)
 * @param {Object} latticeParams - 격자 상수 {a, b, c, alpha, beta, gamma}
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 8, triclinic은 계산량이 많아서 낮춤)
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesTriclinic = (dSpacing, latticeParams, maxHKL = 8, wavelength = 1.5406, tolerance = 0.01) => {
  const candidates = [];
  const { a, b, c, alpha, beta, gamma } = latticeParams;

  // 역격자 벡터 계산
  const alphaRad = alpha * Math.PI / 180;
  const betaRad = beta * Math.PI / 180;
  const gammaRad = gamma * Math.PI / 180;

  // 단위 셀 부피
  const V = a * b * c * Math.sqrt(
    1 - Math.cos(alphaRad) * Math.cos(alphaRad) -
    Math.cos(betaRad) * Math.cos(betaRad) -
    Math.cos(gammaRad) * Math.cos(gammaRad) +
    2 * Math.cos(alphaRad) * Math.cos(betaRad) * Math.cos(gammaRad)
  );

  // 역격자 상수
  const aStar = (b * c * Math.sin(alphaRad)) / V;
  const bStar = (a * c * Math.sin(betaRad)) / V;
  const cStar = (a * b * Math.sin(gammaRad)) / V;

  // 역격자 각도
  const cosAlphaStar = (Math.cos(betaRad) * Math.cos(gammaRad) - Math.cos(alphaRad)) /
    (Math.sin(betaRad) * Math.sin(gammaRad));
  const cosBetaStar = (Math.cos(alphaRad) * Math.cos(gammaRad) - Math.cos(betaRad)) /
    (Math.sin(alphaRad) * Math.sin(gammaRad));
  const cosGammaStar = (Math.cos(alphaRad) * Math.cos(betaRad) - Math.cos(gammaRad)) /
    (Math.sin(alphaRad) * Math.sin(betaRad));

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        if (h === 0 && k === 0 && l === 0) continue; // (000) 제외

        // 역격자 벡터의 크기 계산
        const hStar = h * aStar;
        const kStar = k * bStar;
        const lStar = l * cStar;

        // 역격자 벡터의 내적
        const dStarSq = hStar * hStar + kStar * kStar + lStar * lStar +
          2 * hStar * kStar * cosGammaStar +
          2 * hStar * lStar * cosBetaStar +
          2 * kStar * lStar * cosAlphaStar;

        const d = 1 / Math.sqrt(dStarSq);
        const diff = Math.abs(d - dSpacing);

        if (diff < tolerance) {
          const thetaRad = Math.asin(wavelength / (2 * d));
          const angle = 2 * thetaRad * 180 / Math.PI;

          candidates.push({
            h, k, l,
            d: d,
            angle: angle,
            diff: diff
          });
        }
      }
    }
  }

  candidates.sort((a, b) => a.diff - b.diff);
  return candidates;
};

/**
 * 결정계에 따라 적절한 역인덱싱 함수 호출
 * @param {number} dSpacing - d-spacing (Å)
 * @param {Object} latticeParams - 격자 상수
 * @param {string} latticeParams.system - 결정계
 * @param {number} latticeParams.a - 격자 상수 a (Å)
 * @param {number} latticeParams.b - 격자 상수 b (Å, tetragonal 이상)
 * @param {number} latticeParams.c - 격자 상수 c (Å, orthorhombic 이상)
 * @param {number} latticeParams.alpha - 각도 α (도, monoclinic 이상)
 * @param {number} latticeParams.beta - 각도 β (도, monoclinic 이상)
 * @param {number} latticeParams.gamma - 각도 γ (도, triclinic)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @param {number} wavelength - X선 파장 (Å, 기본값: 1.5406)
 * @param {number} tolerance - 허용 오차 (Å, 기본값: 0.01)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number, diff: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndicesForSystem = (dSpacing, latticeParams, maxHKL = 10, wavelength = 1.5406, tolerance = 0.01) => {
  const { system, a, b, c, alpha, beta, gamma } = latticeParams;

  if (!system || !a) {
    throw new Error('격자 상수가 필요합니다.');
  }

  const systemLower = system.toLowerCase();

  switch (systemLower) {
    case 'cubic':
      return calculatePossibleMillerIndicesCubic(dSpacing, a, maxHKL, wavelength, tolerance);

    case 'tetragonal':
      if (!c) throw new Error('격자 상수 c가 필요합니다.');
      return calculatePossibleMillerIndicesTetragonal(dSpacing, a, c, maxHKL, wavelength, tolerance);

    case 'orthorhombic':
      if (!b || !c) throw new Error('격자 상수 b, c가 필요합니다.');
      return calculatePossibleMillerIndicesOrthorhombic(dSpacing, a, b, c, maxHKL, wavelength, tolerance);

    case 'hexagonal':
      if (!c) throw new Error('격자 상수 c가 필요합니다.');
      return calculatePossibleMillerIndicesHexagonal(dSpacing, a, c, maxHKL, wavelength, tolerance);

    case 'monoclinic':
      if (!b || !c || beta === undefined) throw new Error('격자 상수 b, c, beta가 필요합니다.');
      return calculatePossibleMillerIndicesMonoclinic(dSpacing, a, b, c, beta, maxHKL, wavelength, tolerance);

    case 'triclinic':
      if (!b || !c || alpha === undefined || beta === undefined || gamma === undefined) {
        throw new Error('모든 격자 상수와 각도가 필요합니다.');
      }
      return calculatePossibleMillerIndicesTriclinic(
        dSpacing,
        { a, b, c, alpha, beta, gamma },
        Math.min(maxHKL, 8), // triclinic은 계산량이 많아서 제한
        wavelength,
        tolerance
      );

    default:
      throw new Error(`지원되지 않는 결정계: ${system}`);
  }
};

/**
 * d-spacing으로부터 가능한 밀러지수 계산 (Cubic 시스템) - 하위 호환성
 * @param {number} dSpacing - d-spacing (Å)
 * @param {number} a - 격자 상수 a (Å)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @returns {Array<{h: number, k: number, l: number, d: number, angle: number}>} 가능한 밀러지수 배열
 */
export const calculatePossibleMillerIndices = (dSpacing, a, maxHKL = 10, wavelength = 1.5406) => {
  return calculatePossibleMillerIndicesCubic(dSpacing, a, maxHKL, wavelength);
};

/**
 * 피팅된 피크에서 FWHM 추출
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 데이터
 * @param {number} peakCenter - 피크 중심 각도
 * @param {number} peakHeight - 피크 높이
 * @returns {number} FWHM (도)
 */
export const extractFWHMFromPeak = (xrdData, peakCenter, peakHeight) => {
  const halfMax = peakHeight / 2;
  
  // 피크 중심 인덱스 찾기
  let centerIdx = 0;
  let minDiff = Infinity;
  for (let i = 0; i < xrdData.length; i++) {
    const diff = Math.abs(xrdData[i].angle - peakCenter);
    if (diff < minDiff) {
      minDiff = diff;
      centerIdx = i;
    }
  }

  // 왼쪽 절반 높이 지점 찾기
  let leftIdx = centerIdx;
  while (leftIdx > 0 && xrdData[leftIdx].intensity > halfMax) {
    leftIdx--;
  }
  
  // 오른쪽 절반 높이 지점 찾기
  let rightIdx = centerIdx;
  while (rightIdx < xrdData.length - 1 && xrdData[rightIdx].intensity > halfMax) {
    rightIdx++;
  }

  // 선형 보간으로 정확한 FWHM 계산
  let leftAngle = xrdData[leftIdx].angle;
  if (leftIdx > 0 && xrdData[leftIdx].intensity < halfMax) {
    // 선형 보간
    const y1 = xrdData[leftIdx].intensity;
    const y2 = xrdData[leftIdx + 1].intensity;
    const x1 = xrdData[leftIdx].angle;
    const x2 = xrdData[leftIdx + 1].angle;
    const slope = (y2 - y1) / (x2 - x1);
    leftAngle = x1 + (halfMax - y1) / slope;
  }

  let rightAngle = xrdData[rightIdx].angle;
  if (rightIdx < xrdData.length - 1 && xrdData[rightIdx].intensity < halfMax) {
    // 선형 보간
    const y1 = xrdData[rightIdx - 1].intensity;
    const y2 = xrdData[rightIdx].intensity;
    const x1 = xrdData[rightIdx - 1].angle;
    const x2 = xrdData[rightIdx].angle;
    const slope = (y2 - y1) / (x2 - x1);
    rightAngle = x1 + (halfMax - y1) / slope;
  }

  return Math.abs(rightAngle - leftAngle);
};

/**
 * 이론적 XRD 피크 생성 (격자 상수와 결정계로부터)
 *
 * @param {Object} latticeParams - 격자 상수
 * @param {string} latticeParams.system    - 결정계
 * @param {number} latticeParams.a         - 격자 상수 a (Å)
 * @param {number} [latticeParams.b]       - 격자 상수 b (Å)
 * @param {number} [latticeParams.c]       - 격자 상수 c (Å)
 * @param {number} [latticeParams.alpha]   - 각도 α (도)
 * @param {number} [latticeParams.beta]    - 각도 β (도)
 * @param {number} [latticeParams.gamma]   - 각도 γ (도)
 * @param {number} [maxHKL=10]             - 최대 밀러지수
 * @param {number} [wavelength=1.5406]     - X선 파장 (Å)
 * @param {number} [maxAngle=90]           - 최대 2θ (도)
 * @param {string} [centering='P']         - Bravais 센터링 ('P','I','F','C','A','B','R','HCP')
 * @returns {Array<{h,k,l,d,angle,relativeIntensity,millerLabel}>}
 */
export const generateTheoreticalPeaks = (
  latticeParams,
  maxHKL = 10,
  wavelength = 1.5406,
  maxAngle = 90,
  centering = 'P'
) => {
  const theoreticalPeaks = [];
  const seen = new Map();

  for (let h = 0; h <= maxHKL; h++) {
    for (let k = 0; k <= maxHKL; k++) {
      for (let l = 0; l <= maxHKL; l++) {
        // 소멸 조건 적용 (000 포함)
        if (!isSystematicallyAllowed(h, k, l, centering)) continue;

        let d;
        try {
          d = calculateDSpacingFromMillerIndices(h, k, l, latticeParams);
        } catch {
          continue;
        }

        // d가 너무 작으면 asin 불가
        if (!d || d <= wavelength / 2) continue;

        const sinTheta = wavelength / (2 * d);
        if (sinTheta > 1) continue;
        const angle = 2 * Math.asin(sinTheta) * 180 / Math.PI;
        if (angle > maxAngle) continue;

        // 중복 피크 제거 (등가 hkl은 동일 각도)
        const key = angle.toFixed(3);
        if (seen.has(key)) continue;
        seen.set(key, true);

        // 구조인자 기반 상대 강도 근사
        // Lorenz-polarization 보정 제외(간이), 구조인자 크기만 반영
        const relativeIntensity = estimateStructureFactorSq(h, k, l, centering, latticeParams.system);

        theoreticalPeaks.push({
          h, k, l,
          millerLabel: `(${h}${k}${l})`,
          d,
          angle,
          relativeIntensity,
        });
      }
    }
  }

  theoreticalPeaks.sort((a, b) => a.angle - b.angle);
  return theoreticalPeaks;
};

/**
 * 구조인자 제곱 근사값 (단위셀 원자 모두 동종 가정 — 단순화)
 * 정확한 계산은 원자 위치와 원자 산란 인자가 필요하나, 위치 선별용으로 충분
 * @param {number} h
 * @param {number} k
 * @param {number} l
 * @param {string} centering
 * @param {string} crystalSystem
 * @returns {number} 0~100 사이 상대 강도
 */
function estimateStructureFactorSq(h, k, l, centering, crystalSystem) {
  const c = (centering || 'P').toUpperCase();
  switch (c) {
    case 'F': return 16;   // |F|² = (4f)² 정규화 → 16
    case 'I': return 4;    // |F|² = (2f)² → 4
    case 'HCP': {
      // P6₃/mmc 2-원자 기저
      // |F|² ∝ 2(1 + cos(2π(h+2k)/3 + πl))
      const phi = 2 * Math.PI * (h + 2 * k) / 3 + Math.PI * l;
      const fSq = 2 * (1 + Math.cos(phi));
      return Math.max(0, fSq / 4 * 100); // 0~100으로 정규화
    }
    case 'R': return 9;
    default:
      // P, C, A, B: 다양 — 간이 추정 (낮은 지수일수록 강도 대체로 큼)
      return Math.max(5, 100 / (1 + 0.3 * (h * h + k * k + l * l)));
  }
}

/**
 * 관측 피크와 이론적 피크를 매칭하여 밀러지수 할당
 * @param {Array<{angle: number, intensity?: number}>} observedPeaks - 관측된 피크
 * @param {Array<{h: number, k: number, l: number, d: number, angle: number, relativeIntensity?: number}>} theoreticalPeaks - 이론적 피크
 * @param {number} angleTolerance - 각도 허용 오차 (도, 기본값: 0.5)
 * @param {boolean} useIntensity - 강도 정보 활용 여부 (기본값: false)
 * @returns {Array<{peak: Object, millerIndices: Array<{h: number, k: number, l: number, d: number, angle: number, matchScore: number}>}>} 매칭 결과
 */
export const matchPeaksToMillerIndices = (observedPeaks, theoreticalPeaks, angleTolerance = 0.5, useIntensity = false) => {
  const matches = [];

  for (const observedPeak of observedPeaks) {
    const candidates = [];
    const observedAngle = observedPeak.angle;
    const observedIntensity = observedPeak.intensity || 0;

    // 각도 차이가 허용 오차 내인 이론적 피크 찾기
    for (const theoreticalPeak of theoreticalPeaks) {
      const angleDiff = Math.abs(theoreticalPeak.angle - observedAngle);

      if (angleDiff <= angleTolerance) {
        // 매칭 점수 계산
        let matchScore = 1.0 - (angleDiff / angleTolerance); // 각도 차이 기반 점수 (0-1)

        // 강도 정보 활용 (선택적)
        if (useIntensity && theoreticalPeak.relativeIntensity && observedIntensity > 0) {
          // 상대 강도 차이를 고려 (단순화된 모델)
          const intensityRatio = Math.min(
            observedIntensity / (theoreticalPeak.relativeIntensity * 10),
            theoreticalPeak.relativeIntensity * 10 / observedIntensity
          );
          matchScore *= (0.5 + 0.5 * intensityRatio); // 강도 일치도 반영
        }

        candidates.push({
          h: theoreticalPeak.h,
          k: theoreticalPeak.k,
          l: theoreticalPeak.l,
          d: theoreticalPeak.d,
          angle: theoreticalPeak.angle,
          matchScore: matchScore,
          angleDiff: angleDiff
        });
      }
    }

    // 매칭 점수로 정렬
    candidates.sort((a, b) => b.matchScore - a.matchScore);

    matches.push({
      peak: observedPeak,
      millerIndices: candidates
    });
  }

  return matches;
};

/**
 * 자동 인덱싱 최적화 함수
 * @param {Array<{angle: number, intensity?: number}>} observedPeaks - 관측된 피크
 * @param {Object} latticeParams - 격자 상수
 * @param {number} wavelength - X선 파장 (Å, 기본값: 1.5406)
 * @param {number} angleTolerance - 각도 허용 오차 (도, 기본값: 0.5)
 * @param {number} maxHKL - 최대 밀러지수 (기본값: 10)
 * @returns {Array<{angle: number, intensity?: number, dSpacing: number, millerIndices: Array, confidence: number}>} 인덱싱된 피크
 */
export const autoIndexPeaks = (observedPeaks, latticeParams, wavelength = 1.5406, angleTolerance = 0.5, maxHKL = 10) => {
  // 이론적 피크 생성
  let theoreticalPeaks = [];
  try {
    theoreticalPeaks = generateTheoreticalPeaks(latticeParams, maxHKL, wavelength);
  } catch (error) {
    console.warn('이론적 피크 생성 실패:', error);
    // 이론적 피크 생성 실패 시 빈 배열 반환
    return observedPeaks.map(peak => ({
      ...peak,
      dSpacing: calculateDSpacing(peak.angle, wavelength),
      millerIndices: [],
      confidence: 0,
      bestMatch: null
    }));
  }

  // 이론적 피크가 없으면 빈 배열 반환
  if (theoreticalPeaks.length === 0) {
    console.warn('이론적 피크가 생성되지 않았습니다. 격자 상수를 확인하세요.');
    return observedPeaks.map(peak => ({
      ...peak,
      dSpacing: calculateDSpacing(peak.angle, wavelength),
      millerIndices: [],
      confidence: 0,
      bestMatch: null
    }));
  }

  // 피크 매칭
  const matches = matchPeaksToMillerIndices(observedPeaks, theoreticalPeaks, angleTolerance, false);

  // 결과 구성
  return matches.map(match => {
    const sys = (latticeParams.system || 'cubic').toLowerCase();
    const rawCand = match.millerIndices.slice(0, 40);
    let millerIndices = dedupeMillerCandidatesByCrystalSystem(rawCand, sys).slice(0, 5);

    if (millerIndices.length === 0) {
      // 매칭 실패 시 d-spacing 기반 역인덱싱 시도
      const dSpacing = calculateDSpacing(match.peak.angle, wavelength);
      try {
        const candidates = calculatePossibleMillerIndicesForSystem(
          dSpacing,
          latticeParams,
          maxHKL,
          wavelength,
          0.05 // 더 관대한 tolerance
        );
        millerIndices = candidates.slice(0, 5);
      } catch (error) {
        console.warn('역인덱싱 실패:', error);
      }
    }

    const bestMatch = millerIndices[0] || null;
    const confidence = bestMatch ? (bestMatch.matchScore != null ? bestMatch.matchScore : 0.5) : 0;

    return {
      ...match.peak,
      dSpacing: calculateDSpacing(match.peak.angle, wavelength),
      millerIndices: millerIndices,
      confidence: confidence,
      bestMatch: bestMatch ? { h: bestMatch.h, k: bestMatch.k, l: bestMatch.l } : null
    };
  });
};

/**
 * CIF 구조 정보를 사용한 밀러지수 인덱싱 (개선된 버전)
 * @param {Array<{angle: number, fwhm?: number, intensity?: number}>} peaks - 피크 정보
 * @param {Object} structureInfo - CIF 구조 정보
 * @param {number} wavelength - X선 파장 (Å)
 * @param {number} angleTolerance - 각도 허용 오차 (도, 기본값: 0.5)
 * @param {boolean} useAutoIndexing - 자동 인덱싱 사용 여부 (기본값: true)
 * @returns {Array<{angle: number, dSpacing: number, millerIndices: Array, fwhm?: number, intensity?: number, confidence?: number}>} 인덱싱된 피크
 */
export const indexPeaksWithStructureInfo = (peaks, structureInfo, wavelength = 1.5406, angleTolerance = 0.5, useAutoIndexing = true) => {
  if (!structureInfo || !structureInfo.cellParams) {
    // 구조 정보가 없어도 d-spacing 기반 역인덱싱 시도
    return peaks.map(peak => {
      const dSpacing = calculateDSpacing(peak.angle, wavelength);
      let millerIndices = [];
      
      // 기본 Cubic 가정으로 인덱싱 시도 (첫 번째 피크로부터 격자 상수 추정)
      if (peaks.length > 0) {
        const firstDSpacing = calculateDSpacing(peaks[0].angle, wavelength);
        const estimatedA = firstDSpacing * Math.sqrt(3); // (111) 가정
        
        try {
          const candidates = calculatePossibleMillerIndicesCubic(
            dSpacing,
            estimatedA,
            15,
            wavelength,
            0.1
          );
          millerIndices = candidates.slice(0, 5);
        } catch (error) {
          console.warn('기본 인덱싱 실패:', error);
        }
      }
      
      return {
        ...peak,
        dSpacing: dSpacing,
        millerIndices: millerIndices
      };
    });
  }

  const { cellParams } = structureInfo;
  const system = cellParams.system || 'cubic';
  const a = cellParams.a;
  const b = cellParams.b || a;
  const c = cellParams.c || a;
  const alpha = cellParams.alpha;
  const beta = cellParams.beta;
  const gamma = cellParams.gamma;

  // 격자 상수 a가 없거나 유효하지 않으면 기본값 사용하지 않고 역인덱싱으로 폴백
  if (!a || a <= 0 || a > 100) {
    console.warn('격자 상수 a가 유효하지 않습니다. 기본 인덱싱으로 폴백합니다.');
    return peaks.map(peak => {
      const dSpacing = calculateDSpacing(peak.angle, wavelength);
      let millerIndices = [];
      
      if (peaks.length > 0) {
        const firstDSpacing = calculateDSpacing(peaks[0].angle, wavelength);
        const estimatedA = firstDSpacing * Math.sqrt(3);
        
        try {
          const candidates = calculatePossibleMillerIndicesCubic(
            dSpacing,
            estimatedA,
            15,
            wavelength,
            0.1
          );
          millerIndices = candidates.slice(0, 5);
        } catch (error) {
          console.warn('기본 인덱싱 실패:', error);
        }
      }
      
      return {
        ...peak,
        dSpacing: dSpacing,
        millerIndices: millerIndices
      };
    });
  }

  const latticeParams = {
    system: system,
    a: a,
    b: b,
    c: c,
    alpha: alpha,
    beta: beta,
    gamma: gamma
  };

  // 자동 인덱싱 사용
  if (useAutoIndexing) {
    const result = autoIndexPeaks(peaks, latticeParams, wavelength, angleTolerance);
    
    // 결과 검증: 모든 피크에 밀러지수가 없으면 역인덱싱으로 폴백
    const hasAnyMillerIndices = result.some(peak => peak.millerIndices && peak.millerIndices.length > 0);
    
    if (!hasAnyMillerIndices) {
      console.warn('자동 인덱싱이 실패했습니다. 역인덱싱으로 폴백합니다.');
      return peaks.map(peak => {
        const dSpacing = calculateDSpacing(peak.angle, wavelength);
        let millerIndices = [];
        
        try {
          const candidates = calculatePossibleMillerIndicesForSystem(
            dSpacing,
            latticeParams,
            15,
            wavelength,
            0.1 // 더 관대한 tolerance
          );
          millerIndices = candidates.slice(0, 5);
        } catch (error) {
          console.warn('역인덱싱 실패:', error);
        }

        return {
          ...peak,
          dSpacing: dSpacing,
          millerIndices: millerIndices
        };
      });
    }
    
    return result;
  }

  // 기존 방식 (개선된 버전)
  return peaks.map(peak => {
    const dSpacing = calculateDSpacing(peak.angle, wavelength);
    
    // 가능한 밀러지수 계산 (모든 결정계 지원)
    let millerIndices = [];
    try {
      const candidates = calculatePossibleMillerIndicesForSystem(
        dSpacing,
        latticeParams,
        15, // maxHKL 증가
        wavelength,
        0.1 // 더 관대한 tolerance
      );
      millerIndices = candidates.slice(0, 5); // 상위 5개만
    } catch (error) {
      console.warn('밀러지수 계산 오류:', error);
      millerIndices = [];
    }

    return {
      ...peak,
      dSpacing: dSpacing,
      millerIndices: millerIndices
    };
  });
};

