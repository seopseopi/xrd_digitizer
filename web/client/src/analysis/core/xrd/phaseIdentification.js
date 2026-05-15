/**
 * XRD 상 동정 (Phase Identification)
 *
 * 지원 결정 구조:
 *   Cubic   — FCC (F), BCC (I), SC (P)
 *   Hexagonal — HCP P6₃/mmc (소멸 조건 적용), 일반 Hexagonal P
 *   Tetragonal — P, I
 *
 * 알고리즘:
 *   1. 조대(coarse) 격자 상수 그리드 탐색
 *   2. 세밀(fine) 정제
 *   3. 매칭 cost = Σ(Δ2θ)² + 미매칭 패널티
 *   4. 강도 가중치 반영 (구조인자 제곱 근사)
 *
 * 주의: PDF 카드 자동 매칭은 라이선스 이슈로 미포함.
 *       COMMON_PHASES 프리셋 또는 오픈 CIF 연동 권장.
 */

import { calculateDSpacingFromMillerIndices, isSystematicallyAllowed } from './millerIndex.js';

/**
 * 큐빅 분말 회절 등가면 개수 (정육면체 Laue군 근사, 교육용)
 * @param {number} h
 * @param {number} k
 * @param {number} l
 */
export function cubicPowderMultiplicity(h, k, l) {
  const [a, b, c] = [Math.abs(h), Math.abs(k), Math.abs(l)].sort((x, y) => x - y);
  if (a + b + c === 0) return 0;
  if (!a && !b) return 6;
  if (!a && b === c) return 12;
  if (!a) return 24;
  if (a === b && b === c) return 8;
  if (a === b || b === c || a === c) return 24;
  return 24;
}

function powderMultiplicityForStructure(h, k, l, structureKey) {
  if (h == null || k == null || l == null) return null;
  const s = (structureKey || '').toLowerCase();
  if (s === 'fcc' || s === 'bcc' || s === 'sc') return cubicPowderMultiplicity(h, k, l);
  return null;
}

// ─────────────────────────────────────────────────────────
// 공통 금속·산화물 프리셋 (라이선스 free, 문헌 격자 상수)
// ─────────────────────────────────────────────────────────
export const COMMON_PHASES = {
  'Fe-BCC (α-Fe)':    { system: 'cubic',      centering: 'I',   a: 2.866 },
  'Fe-FCC (γ-Fe)':    { system: 'cubic',      centering: 'F',   a: 3.591 },
  'Al':               { system: 'cubic',      centering: 'F',   a: 4.046 },
  'Cu':               { system: 'cubic',      centering: 'F',   a: 3.615 },
  'Ni':               { system: 'cubic',      centering: 'F',   a: 3.524 },
  'Cr':               { system: 'cubic',      centering: 'I',   a: 2.884 },
  'W':                { system: 'cubic',      centering: 'I',   a: 3.165 },
  'Mo':               { system: 'cubic',      centering: 'I',   a: 3.147 },
  'V':                { system: 'cubic',      centering: 'I',   a: 3.024 },
  'Ti-α (HCP)':       { system: 'hexagonal',  centering: 'HCP', a: 2.951, c: 4.684 },
  'Ti-β (BCC)':       { system: 'cubic',      centering: 'I',   a: 3.306 },
  'Mg':               { system: 'hexagonal',  centering: 'HCP', a: 3.209, c: 5.211 },
  'Zn':               { system: 'hexagonal',  centering: 'HCP', a: 2.665, c: 4.947 },
  'Zr-α':             { system: 'hexagonal',  centering: 'HCP', a: 3.231, c: 5.147 },
  'Co-HCP':           { system: 'hexagonal',  centering: 'HCP', a: 2.507, c: 4.069 },
  'Martensite (C~0.2%)': { system: 'tetragonal', centering: 'I', a: 2.845, c: 2.980 },
  'FeO (wüstite)':    { system: 'cubic',      centering: 'F',   a: 4.307 },
  'Fe₃O₄ (magnetite)':{ system: 'cubic',     centering: 'F',   a: 8.396 },
  'α-Fe₂O₃ (hematite)':{ system: 'hexagonal', centering: 'R', a: 5.035, c: 13.747 },
  'α-Al₂O₃ (corundum)':{ system: 'hexagonal', centering: 'R', a: 4.758, c: 12.991 },
  'TiN':              { system: 'cubic',      centering: 'F',   a: 4.242 },
  'TiC':              { system: 'cubic',      centering: 'F',   a: 4.328 },
  'CrN':              { system: 'cubic',      centering: 'F',   a: 4.149 },
  'ZrO₂ (cubic)':     { system: 'cubic',      centering: 'F',   a: 5.090 },
  'Si':               { system: 'cubic',      centering: 'F',   a: 5.431 },
  'Ge':               { system: 'cubic',      centering: 'F',   a: 5.658 },
};

// ─────────────────────────────────────────────────────────
// 내부 유틸리티
// ─────────────────────────────────────────────────────────

/** 2θ → d-spacing (Bragg) */
function dFromTwoTheta(twoThetaDeg, wavelengthAngstrom) {
  const sinT = Math.sin((twoThetaDeg / 2) * Math.PI / 180);
  if (sinT <= 0) return null;
  return wavelengthAngstrom / (2 * sinT);
}

/** d-spacing → 2θ (Bragg). d ≤ λ/2 이면 null */
function twoThetaFromD(dAngstrom, wavelengthAngstrom) {
  if (!dAngstrom || dAngstrom <= wavelengthAngstrom / 2) return null;
  const sinT = wavelengthAngstrom / (2 * dAngstrom);
  if (sinT > 1) return null;
  return (2 * Math.asin(sinT) * 180) / Math.PI;
}

// ─────────────────────────────────────────────────────────
// 이론 피크 생성 — 결정계별
// ─────────────────────────────────────────────────────────

/**
 * Cubic 이론 피크 목록
 * @param {'fcc'|'bcc'|'sc'} structure
 * @param {number} a  — 격자 상수 (Å)
 */
export function listCubicTheoreticalPeaks(structure, a, wavelengthAngstrom, twoThetaMin, twoThetaMax) {
  const centering = structure === 'fcc' ? 'F' : structure === 'bcc' ? 'I' : 'P';
  const out = [];
  const seen = new Map();
  const maxN = 14;
  for (let h = 0; h <= maxN; h++) {
    for (let k = 0; k <= h; k++) {          // h >= k 조건으로 중복 축소
      for (let l = 0; l <= k; l++) {        // k >= l
        if (h === 0 && k === 0 && l === 0) continue;
        if (!isSystematicallyAllowed(h, k, l, centering)) continue;
        const d = a / Math.sqrt(h * h + k * k + l * l);
        const tt = twoThetaFromD(d, wavelengthAngstrom);
        if (tt == null || tt < twoThetaMin - 0.5 || tt > twoThetaMax + 0.5) continue;
        const key = tt.toFixed(3);
        if (!seen.has(key)) {
          seen.set(key, true);
          const fSq = centering === 'F' ? 16 : centering === 'I' ? 4 : 1;
          out.push({ twoTheta: tt, h, k, l, d, relativeIntensity: fSq });
        }
      }
    }
  }
  out.sort((x, y) => x.twoTheta - y.twoTheta);
  return out;
}

/**
 * HCP 이론 피크 목록 — P6₃/mmc 소멸 조건 적용
 * @param {number} a
 * @param {number} c
 */
export function listHcpTheoreticalPeaks(a, c, wavelengthAngstrom, twoThetaMin, twoThetaMax) {
  const latticeParams = { system: 'hexagonal', a, b: a, c, alpha: 90, beta: 90, gamma: 120 };
  const out = [];
  const seen = new Map();
  for (let h = 0; h <= 12; h++) {
    for (let k = 0; k <= 12; k++) {
      for (let l = 0; l <= 20; l++) {
        // P6₃/mmc 소멸 조건 적용
        if (!isSystematicallyAllowed(h, k, l, 'HCP')) continue;

        let d;
        try {
          d = calculateDSpacingFromMillerIndices(h, k, l, latticeParams);
        } catch {
          continue;
        }
        const tt = twoThetaFromD(d, wavelengthAngstrom);
        if (tt == null || tt < twoThetaMin - 0.5 || tt > twoThetaMax + 0.5) continue;
        const key = tt.toFixed(3);
        if (!seen.has(key)) {
          seen.set(key, true);
          // 구조인자 근사 (P6₃/mmc 2-원자 기저)
          const phi = 2 * Math.PI * (h + 2 * k) / 3 + Math.PI * l;
          const fSq = Math.max(0, 2 * (1 + Math.cos(phi)));
          out.push({ twoTheta: tt, h, k, l, d, relativeIntensity: fSq * 25 }); // 0~100 범위
        }
      }
    }
  }
  out.sort((x, y) => x.twoTheta - y.twoTheta);
  return out;
}

/**
 * Tetragonal 이론 피크 목록
 * @param {number} a
 * @param {number} c
 * @param {'P'|'I'} centering
 */
export function listTetragonalTheoreticalPeaks(a, c, centering = 'P', wavelengthAngstrom, twoThetaMin, twoThetaMax) {
  const out = [];
  const seen = new Map();
  for (let h = 0; h <= 14; h++) {
    for (let k = 0; k <= h; k++) {
      for (let l = 0; l <= 20; l++) {
        if (!isSystematicallyAllowed(h, k, l, centering)) continue;
        const invD2 = (h * h + k * k) / (a * a) + (l * l) / (c * c);
        if (invD2 === 0) continue;
        const d = 1 / Math.sqrt(invD2);
        const tt = twoThetaFromD(d, wavelengthAngstrom);
        if (tt == null || tt < twoThetaMin - 0.5 || tt > twoThetaMax + 0.5) continue;
        const key = tt.toFixed(3);
        if (!seen.has(key)) {
          seen.set(key, true);
          const fSq = centering === 'I' ? 4 : 1;
          out.push({ twoTheta: tt, h, k, l, d, relativeIntensity: fSq * 25 });
        }
      }
    }
  }
  out.sort((x, y) => x.twoTheta - y.twoTheta);
  return out;
}

// ─────────────────────────────────────────────────────────
// 피크 매칭
// ─────────────────────────────────────────────────────────

/**
 * 측정 피크와 이론 피크를 일대일 최근접 매칭
 *
 * @param {Array<{angle: number, intensity?: number}>} measuredPeaks
 * @param {Array<{twoTheta, h, k, l, d, relativeIntensity}>} theoretical
 * @param {number} tolDeg  — 허용 각도 오차 (°)
 * @param {number} wavelengthAngstrom
 * @returns {{ cost, matched, assignments, meanResidualDeg }}
 */
function matchPeaksToTheoretical(measuredPeaks, theoretical, tolDeg, wavelengthAngstrom, structureKey = null) {
  const theo = [...theoretical].sort((a, b) => a.twoTheta - b.twoTheta);
  const usedTheo = new Set();
  const assignments = [];
  let cost = 0;
  let matched = 0;

  const normalizedMeas = measuredPeaks.map((mp) => {
    const ang = typeof mp === 'number' ? mp : mp.angle;
    const measuredIntensity = (typeof mp === 'object' && mp != null && mp.intensity != null) ? mp.intensity : null;
    return { ang, measuredIntensity };
  });
  const maxMeasI = Math.max(...normalizedMeas.map((m) => m.measuredIntensity || 0), 1e-12);
  const wIntensity = 0.45;

  const byIntensity = [...normalizedMeas.entries()]
    .sort((a, b) => (b[1].measuredIntensity || 0) - (a[1].measuredIntensity || 0));

  for (const [, { ang, measuredIntensity }] of byIntensity) {
    const inTol = [];
    for (let i = 0; i < theo.length; i++) {
      if (usedTheo.has(i)) continue;
      const delta = Math.abs(theo[i].twoTheta - ang);
      if (delta <= tolDeg) {
        inTol.push({
          i,
          delta,
          theoRel: theo[i].relativeIntensity != null ? Math.max(theo[i].relativeIntensity, 1e-9) : 1,
        });
      }
    }

    let pick = null;
    if (inTol.length > 0) {
      const maxRel = Math.max(...inTol.map((c) => c.theoRel));
      let bestScore = Infinity;
      for (const c of inTol) {
        let intensityTerm = 0;
        if (measuredIntensity != null) {
          const mn = measuredIntensity / maxMeasI;
          const tn = c.theoRel / maxRel;
          intensityTerm = wIntensity * (tolDeg * tolDeg) * (mn - tn) * (mn - tn);
        }
        const score = c.delta * c.delta + intensityTerm;
        if (score < bestScore) {
          bestScore = score;
          pick = { ...c, score };
        }
      }
    }

    let bestIdx = -1;
    let bestDelta = Infinity;
    if (!pick) {
      for (let i = 0; i < theo.length; i++) {
        if (usedTheo.has(i)) continue;
        const delta = Math.abs(theo[i].twoTheta - ang);
        if (delta < bestDelta) {
          bestDelta = delta;
          bestIdx = i;
        }
      }
    }

    if (pick) {
      const bestIdxP = pick.i;
      bestDelta = pick.delta;
      usedTheo.add(bestIdxP);
      cost += pick.score;
      matched++;
      const t = theo[bestIdxP];
      const dMeasured = dFromTwoTheta(ang, wavelengthAngstrom);
      assignments.push({
        measuredTwoThetaDeg: ang,
        measuredIntensity,
        theoreticalTwoThetaDeg: t.twoTheta,
        deltaDeg: bestDelta,
        h: t.h, k: t.k, l: t.l,
        millerLabel: `(${t.h}${t.k}${t.l})`,
        dTheoreticalAngstrom: t.d,
        dMeasuredAngstrom: dMeasured,
        relativeIntensityEstimate: t.relativeIntensity ?? null,
        multiplicity: powderMultiplicityForStructure(t.h, t.k, t.l, structureKey),
      });
    } else if (bestIdx >= 0 && bestDelta <= tolDeg) {
      usedTheo.add(bestIdx);
      cost += bestDelta * bestDelta;
      matched++;
      const t = theo[bestIdx];
      const dMeasured = dFromTwoTheta(ang, wavelengthAngstrom);
      assignments.push({
        measuredTwoThetaDeg: ang,
        measuredIntensity,
        theoreticalTwoThetaDeg: t.twoTheta,
        deltaDeg: bestDelta,
        h: t.h, k: t.k, l: t.l,
        millerLabel: `(${t.h}${t.k}${t.l})`,
        dTheoreticalAngstrom: t.d,
        dMeasuredAngstrom: dMeasured,
        relativeIntensityEstimate: t.relativeIntensity ?? null,
        multiplicity: powderMultiplicityForStructure(t.h, t.k, t.l, structureKey),
      });
    } else {
      cost += (tolDeg * tolDeg) * 8;
      const dMeasured = dFromTwoTheta(ang, wavelengthAngstrom);
      assignments.push({
        measuredTwoThetaDeg: ang,
        measuredIntensity,
        theoreticalTwoThetaDeg: null,
        deltaDeg: null,
        h: null, k: null, l: null,
        millerLabel: null,
        dTheoreticalAngstrom: null,
        dMeasuredAngstrom: dMeasured,
        relativeIntensityEstimate: null,
        multiplicity: null,
      });
    }
  }

  assignments.sort((a, b) => (a.measuredTwoThetaDeg ?? 0) - (b.measuredTwoThetaDeg ?? 0));

  const matched_assignments = assignments.filter(a => a.deltaDeg != null);
  const meanResidualDeg =
    matched_assignments.length > 0
      ? Math.sqrt(matched_assignments.reduce((s, a) => s + a.deltaDeg * a.deltaDeg, 0) / matched_assignments.length)
      : null;

  return { cost, matched, assignments, meanResidualDeg };
}

// ─────────────────────────────────────────────────────────
// Cubic 격자 탐색 + 정제
// ─────────────────────────────────────────────────────────

function refineCubicLatticeA(structure, wavelengthAngstrom, measuredPeakObjects, tolDeg, aGuess) {
  const angles = measuredPeakObjects.map((p) => p.angle);
  const span = 0.08;
  const step = 0.0005;
  const minA = Math.max(2.0, aGuess - span);
  const maxA = Math.min(24, aGuess + span);
  const tmin = Math.min(...angles) - 1;
  const tmax = Math.max(...angles) + 1;
  let bestA = aGuess, bestScore = Infinity, bestMatch = null;
  for (let a = minA; a <= maxA; a += step) {
    const theo = listCubicTheoreticalPeaks(structure, a, wavelengthAngstrom, tmin, tmax);
    const m = matchPeaksToTheoretical(measuredPeakObjects, theo, tolDeg, wavelengthAngstrom, structure);
    if (m.cost < bestScore) { bestScore = m.cost; bestA = a; bestMatch = m; }
  }
  return { a: bestA, match: bestMatch, score: bestScore };
}

// ─────────────────────────────────────────────────────────
// HCP 격자 탐색 + 정제  (a AND c/a 동시 탐색)
// ─────────────────────────────────────────────────────────

// 대표 HCP c/a 값 (실제 원소 기반)
const HCP_CA_PRESETS = [1.568, 1.585, 1.588, 1.593, 1.623, 1.624, 1.633, 1.856, 1.886];

function searchHcpLattice(measuredPeakObjects, wavelengthAngstrom, tolDeg, aMin, aMax, aStep) {
  const angles = measuredPeakObjects.map((p) => p.angle);
  const tmin = Math.min(...angles) - 1;
  const tmax = Math.max(...angles) + 1;

  // c/a 탐색 목록: 프리셋 + 범위 탐색(1.45~1.95, step 0.025)
  const caSet = new Set(HCP_CA_PRESETS.map(v => +v.toFixed(3)));
  for (let ca = 1.45; ca <= 1.95; ca += 0.025) caSet.add(+(ca.toFixed(3)));
  const caList = [...caSet].sort((a, b) => a - b);

  // HCP c-축 물리적 상한: 실제 HCP 금속 c ≤ 5.21Å (Mg), Zr 5.15Å, Zn 4.95Å
  const HCP_C_MAX = 5.5;

  let best = null;
  const coarseTolHCP = tolDeg * 3;
  for (const cOverA of caList) {
    for (let a = aMin; a <= aMax; a += aStep) {
      const c = a * cOverA;
      if (c > HCP_C_MAX) continue;   // c-축 상한 초과 → 비물리적 HCP 격자 제외
      const theo = listHcpTheoreticalPeaks(a, c, wavelengthAngstrom, tmin, tmax);
      if (theo.length === 0) continue;
      const m = matchPeaksToTheoretical(measuredPeakObjects, theo, coarseTolHCP, wavelengthAngstrom, 'hcp');
      if (!best || m.cost < best.cost) {
        best = { latticeA: a, latticeC: c, cOverA, ...m };
      }
    }
  }
  if (!best) return null;

  // 세밀 정제: best 주변 ±0.05Å (a), ±0.03 c/a
  return refineHcpAandCA(measuredPeakObjects, wavelengthAngstrom, tolDeg, best.latticeA, best.cOverA, tmin, tmax);
}

function refineHcpAandCA(measuredPeakObjects, wavelengthAngstrom, tolDeg, aGuess, caGuess, tmin, tmax) {
  let bestA = aGuess, bestCA = caGuess, bestScore = Infinity, bestMatch = null;
  for (let da = -0.06; da <= 0.06; da += 0.001) {
    const a = +(aGuess + da).toFixed(4);
    if (a < 2.0) continue;
    for (let dca = -0.04; dca <= 0.04; dca += 0.005) {
      const ca = +(caGuess + dca).toFixed(4);
      if (ca < 1.40 || ca > 2.05) continue;
      const c = a * ca;
      if (c > 5.5) continue;   // c-축 상한 초과 → 비물리적 HCP 격자 제외
      const theo = listHcpTheoreticalPeaks(a, c, wavelengthAngstrom, tmin, tmax);
      if (theo.length === 0) continue;
      const m = matchPeaksToTheoretical(measuredPeakObjects, theo, tolDeg, wavelengthAngstrom, 'hcp');
      if (m.cost < bestScore) { bestScore = m.cost; bestA = a; bestCA = ca; bestMatch = m; }
    }
  }
  return {
    structure: 'hcp',
    latticeA: bestA,
    latticeC: +(bestA * bestCA).toFixed(4),
    cOverA: bestCA,
    score: bestScore,
    matched: bestMatch ? bestMatch.matched : 0,
    assignments: bestMatch ? bestMatch.assignments : [],
    meanResidualDeg: bestMatch ? bestMatch.meanResidualDeg : null,
  };
}

// ─────────────────────────────────────────────────────────
// Tetragonal 격자 탐색 + 정제
// ─────────────────────────────────────────────────────────

function searchTetragonalLattice(measuredPeakObjects, wavelengthAngstrom, tolDeg, centering, aMin, aMax, aStep) {
  const angles = measuredPeakObjects.map((p) => p.angle);
  const tmin = Math.min(...angles) - 1;
  const tmax = Math.max(...angles) + 1;
  // c/a 탐색: 0.4 ~ 3.5
  const caValues = [];
  for (let ca = 0.4; ca <= 3.5; ca += 0.1) caValues.push(+(ca.toFixed(2)));

  const coarseTolTet = tolDeg * 3;
  let best = null;
  for (const cOverA of caValues) {
    for (let a = aMin; a <= Math.min(aMax, 8); a += aStep) {
      const c = a * cOverA;
      const theo = listTetragonalTheoreticalPeaks(a, c, centering, wavelengthAngstrom, tmin, tmax);
      if (theo.length === 0) continue;
      const m = matchPeaksToTheoretical(measuredPeakObjects, theo, coarseTolTet, wavelengthAngstrom, null);
      if (!best || m.cost < best.cost) {
        best = { latticeA: a, latticeC: c, cOverA, centering, ...m };
      }
    }
  }
  if (!best) return null;

  // 세밀 정제
  let bA = best.latticeA, bC = best.latticeC, bScore = Infinity, bMatch = null;
  for (let da = -0.08; da <= 0.08; da += 0.002) {
    for (let dc = -0.15; dc <= 0.15; dc += 0.005) {
      const a = +(bA + da).toFixed(4);
      const c = +(bC + dc).toFixed(4);
      if (a < 2.0 || c < 1.5) continue;
      const theo = listTetragonalTheoreticalPeaks(a, c, centering, wavelengthAngstrom, tmin, tmax);
      if (theo.length === 0) continue;
      const m = matchPeaksToTheoretical(measuredPeakObjects, theo, tolDeg, wavelengthAngstrom, null);
      if (m.cost < bScore) { bScore = m.cost; bA = a; bC = c; bMatch = m; }
    }
  }

  return {
    structure: `tetragonal-${centering}`,
    centering,
    latticeA: bA,
    latticeC: bC,
    cOverA: +(bC / bA).toFixed(4),
    score: bScore,
    matched: bMatch ? bMatch.matched : 0,
    assignments: bMatch ? bMatch.assignments : [],
    meanResidualDeg: bMatch ? bMatch.meanResidualDeg : null,
  };
}

// ─────────────────────────────────────────────────────────
// d-비율 사전 분류기
// ─────────────────────────────────────────────────────────

/**
 * d-spacing 비율로 결정계를 빠르게 추정 (격자 상수와 무관한 지문)
 * @param {number[]} sortedDValues  — 큰 값(작은 2θ) 순 정렬
 * @returns {{ likelySystem: string, scores: Object }}
 */
export function estimateCrystalSystemFromDRatios(sortedDValues) {
  if (sortedDValues.length < 2) return { likelySystem: 'unknown', scores: {} };

  const d0 = sortedDValues[0];
  const ratios = sortedDValues.slice(0, 6).map(d => d / d0);

  // 이론 d-비율 지문 (처음 6 피크 기준)
  const fingerprints = {
    'FCC': [1.000, 0.816, 0.707, 0.577, 0.548, 0.500],
    'BCC': [1.000, 0.707, 0.577, 0.500, 0.447, 0.408],
    'HCP(ideal)': [1.000, 0.946, 0.832, 0.823, 0.756, 0.673],
    'SC':  [1.000, 0.707, 0.577, 0.500, 0.447, 0.408],
    // 정방정 P, c/a≈1 근사 (101),(110),(002) 패밀리 혼합 지문)
    'TET(c/a~1)': [1.000, 0.894, 0.707, 0.640, 0.577, 0.500],
  };

  const scores = {};
  let bestKey = 'unknown', bestScore = Infinity;
  for (const [label, fp] of Object.entries(fingerprints)) {
    const n = Math.min(ratios.length, fp.length);
    let rmse = 0;
    for (let i = 0; i < n; i++) rmse += (ratios[i] - fp[i]) ** 2;
    rmse = Math.sqrt(rmse / n);
    scores[label] = rmse;
    if (rmse < bestScore) { bestScore = rmse; bestKey = label; }
  }
  return { likelySystem: bestKey, scores };
}

// ─────────────────────────────────────────────────────────
// COMMON_PHASES 프리셋 기반 매칭
// ─────────────────────────────────────────────────────────

/**
 * 내장 프리셋 재료와 측정 피크를 직접 비교
 * @param {Array<{angle, intensity?}>} peaks
 * @param {number} wavelengthAngstrom
 * @param {number} [tolDeg=0.15]
 * @returns {Array<{name, score, matched, assignments, ...}>}
 */
export function matchWithCommonPhases(peaks, wavelengthAngstrom, tolDeg = 0.15) {
  const sortedObjs = preparePeakObjects(peaks);
  if (sortedObjs.length === 0) return [];
  const tmin = sortedObjs[0].angle - 1;
  const tmax = sortedObjs[sortedObjs.length - 1].angle + 1;
  const results = [];

  for (const [name, phase] of Object.entries(COMMON_PHASES)) {
    let theo = [];
    let structureKey = null;
    if (phase.system === 'cubic') {
      structureKey = phase.centering === 'F' ? 'fcc' : phase.centering === 'I' ? 'bcc' : 'sc';
      theo = listCubicTheoreticalPeaks(structureKey, phase.a, wavelengthAngstrom, tmin, tmax);
    } else if (phase.system === 'hexagonal') {
      if (phase.centering === 'HCP') {
        theo = listHcpTheoreticalPeaks(phase.a, phase.c, wavelengthAngstrom, tmin, tmax);
      } else {
        // R centering hexagonal — 간이 처리 (Bragg법으로 임시 생성)
        theo = listHcpTheoreticalPeaks(phase.a, phase.c, wavelengthAngstrom, tmin, tmax);
      }
    } else if (phase.system === 'tetragonal') {
      theo = listTetragonalTheoreticalPeaks(phase.a, phase.c, phase.centering, wavelengthAngstrom, tmin, tmax);
    }

    if (theo.length === 0) continue;
    const m = matchPeaksToTheoretical(sortedObjs, theo, tolDeg, wavelengthAngstrom, structureKey);
    const matchRate = sortedObjs.length > 0 ? m.matched / sortedObjs.length : 0;

    results.push({
      name,
      phase,
      score: m.cost,
      matched: m.matched,
      totalPeaks: sortedObjs.length,
      matchRate,
      meanResidualDeg: m.meanResidualDeg,
      assignments: m.assignments,
    });
  }

  results.sort((a, b) => {
    const dr = b.matchRate - a.matchRate;
    if (Math.abs(dr) > 0.05) return dr;
    return a.score - b.score;
  });

  return results;
}

// ─────────────────────────────────────────────────────────
// 메인 공개 API
// ─────────────────────────────────────────────────────────

function preparePeakObjects(peaks, maxPeaksToUse = 14) {
  return [...peaks]
    .filter(p => p && Number(p.angle ?? p) > 0)
    .sort((a, b) => ((b.intensity ?? 0) - (a.intensity ?? 0)))
    .slice(0, maxPeaksToUse)
    .map((p) => ({
      angle: Number(p.angle ?? p),
      intensity: p.intensity != null ? Number(p.intensity) : null,
    }))
    .sort((a, b) => a.angle - b.angle);
}

/**
 * 주 상 동정 함수
 *
 * @param {Array<{angle: number, intensity?: number}>} peaks  — 탐지된 피크
 * @param {number} wavelengthAngstrom  — X선 파장 (Å, 기본 Cu Kα = 1.5406)
 * @param {Object} [options]
 * @param {number}  [options.angleToleranceDeg=0.12]
 * @param {number}  [options.aMin=2.5]
 * @param {number}  [options.aMax=16]
 * @param {number}  [options.aStepCoarse=0.04]
 * @param {number}  [options.maxPeaksToUse=14]
 * @param {boolean} [options.includeTetragonal=true]
 * @param {boolean} [options.includeCommonPhases=true]
 * @returns {{ success, candidates, presetMatches?, peaksUsed, warnings, latticeReportTop }}
 */
export function identifyPhaseCandidates(peaks, wavelengthAngstrom, options = {}) {
  const {
    angleToleranceDeg = 0.12,
    aMin = 2.5,
    aMax = 16,
    aStepCoarse = 0.04,
    maxPeaksToUse = 14,
    includeTetragonal = true,
    includeCommonPhases = true,
  } = options;

  const sortedObjs = preparePeakObjects(peaks, maxPeaksToUse);

  if (sortedObjs.length < 2) {
    return { success: false, error: '상 동정을 위해 피크가 최소 2개 필요합니다.' };
  }

  const tmin = sortedObjs[0].angle - 1;
  const tmax = sortedObjs[sortedObjs.length - 1].angle + 1;
  const candidates = [];

  /**
   * AIC 스타일 모델 복잡도 보정 스코어 계산
   *
   * scoreDOF = cost
   *          + numFreeParams × matched × tolDeg² × 0.5   ← AIC 패널티
   *          + physPenalty                                ← 결정계별 물리 패널티
   *
   * 원리: 자유 매개변수가 많을수록 "허용 오차 단위의 기회비용"을 부과.
   *       cost ≈ 0 인 경우에도 파라미터 수 차이를 올바르게 구분 (나눗셈 방식의 한계 극복).
   * physPenalty (결정계별):
   *   hexagonal  — HCP 금속 a 통상 < 3.5Å (Zr 3.23 최대). a > 4.0Å부터 급증.
   *   tetragonal — a > 8Å 비현실.
   *   cubic      — a > 10Å 비현실.
   */
  function calcScoreDOF(score, matched, numFreeParams, latticeA, tolDeg, crystalSystem) {
    // AIC 패널티: 각 자유 매개변수 × 매칭된 피크 수 × 허용 오차²의 절반
    const paramPenalty = numFreeParams * matched * (tolDeg ** 2) * 0.5;
    let physPenalty = 0;
    if (crystalSystem === 'hexagonal') {
      // HCP: a > 4.0Å는 극히 드묾 (Mg 3.21, Ti 2.95, Zn 2.67, Zr 3.23)
      if (latticeA != null && latticeA > 4.0) {
        physPenalty = Math.pow(latticeA - 4.0, 2) * 0.5;
      }
    } else if (crystalSystem === 'tetragonal') {
      if (latticeA != null && latticeA > 8) {
        physPenalty = Math.pow(latticeA - 8, 2) * 0.05;
      }
    } else { // cubic
      // a > 6Å: 슈퍼셀 억제 (순수금속 cubic 통상 ≤5Å, 일부 복잡 산화물 예외)
      if (latticeA != null && latticeA > 6) {
        physPenalty = Math.pow(latticeA - 6, 2) * 0.5;
      }
    }
    return score + paramPenalty + physPenalty;
  }

  // ── 1. Cubic (FCC / BCC / SC) ────────────────────────────
  // 조대 탐색: 허용치 3배 (격자간격 0.04Å → 피크 이동 ~0.4° 가능)
  // 물리 패널티를 코아스 탐색에서도 적용: a > 6Å 슈퍼셀 억제
  const coarseTol = angleToleranceDeg * 3;
  for (const structure of ['fcc', 'bcc', 'sc']) {
    let globalBest = null;
    for (let a = aMin; a <= aMax; a += aStepCoarse) {
      const theo = listCubicTheoreticalPeaks(structure, a, wavelengthAngstrom, tmin, tmax);
      const m = matchPeaksToTheoretical(sortedObjs, theo, coarseTol, wavelengthAngstrom, structure);
      // 슈퍼셀 억제: a > 6Å부터 감점 (순수 금속 cubic 최대 ~5Å, 일부 산화물 제외)
      const coarsePhysP = a > 6 ? Math.pow(a - 6, 2) * 0.5 : 0;
      const adjustedCost = m.cost + coarsePhysP;
      if (!globalBest || adjustedCost < globalBest.adjustedCost) globalBest = { a, m, adjustedCost };
    }
    if (globalBest) {
      const refined = refineCubicLatticeA(structure, wavelengthAngstrom, sortedObjs, angleToleranceDeg, globalBest.a);
      const cellVolume = refined.a ** 3;
      const matchedCount = refined.match.matched;
      candidates.push({
        structure,
        centering: structure === 'fcc' ? 'F' : structure === 'bcc' ? 'I' : 'P',
        crystalSystem: 'cubic',
        latticeA: refined.a,
        latticeB: refined.a,
        latticeC: refined.a,
        cOverA: 1,
        score: refined.score,
        scoreDOF: calcScoreDOF(refined.score, matchedCount, 1, refined.a, angleToleranceDeg, 'cubic'),
        numFreeParams: 1,
        matchedCount,
        totalPeaks: sortedObjs.length,
        matchRate: matchedCount / sortedObjs.length,
        meanResidualDeg: refined.match.meanResidualDeg,
        assignments: refined.match.assignments,
        cellVolumeAngstrom3: cellVolume,
      });
    }
  }

  // ── 2. HCP ────────────────────────────────────────────────
  // HCP 금속 격자 상수 현실 범위: a ≤ 4.5Å 제한
  // (Mg 3.21, Ti 2.95, Zn 2.67, Zr 3.23 — 모두 3.5Å 이하)
  const hcpAMax = Math.min(aMax, 4.5);
  const hcp = searchHcpLattice(sortedObjs, wavelengthAngstrom, angleToleranceDeg, aMin, hcpAMax, aStepCoarse);
  if (hcp) {
    const vol = (Math.sqrt(3) / 2) * hcp.latticeA * hcp.latticeA * hcp.latticeC;
    candidates.push({
      structure: 'hcp',
      centering: 'HCP',
      crystalSystem: 'hexagonal',
      latticeA: hcp.latticeA,
      latticeB: hcp.latticeA,
      latticeC: hcp.latticeC,
      cOverA: hcp.cOverA,
      score: hcp.score,
      scoreDOF: calcScoreDOF(hcp.score, hcp.matched, 2, hcp.latticeA, angleToleranceDeg, 'hexagonal'),
      numFreeParams: 2,
      matchedCount: hcp.matched,
      totalPeaks: sortedObjs.length,
      matchRate: hcp.matched / sortedObjs.length,
      meanResidualDeg: hcp.meanResidualDeg,
      assignments: hcp.assignments,
      cellVolumeAngstrom3: vol,
    });
  }

  // ── 3. Tetragonal (P, I) ──────────────────────────────────
  if (includeTetragonal) {
    for (const ct of ['P', 'I']) {
      const tet = searchTetragonalLattice(sortedObjs, wavelengthAngstrom, angleToleranceDeg, ct, aMin, Math.min(aMax, 8), aStepCoarse);
      if (tet) {
        const vol = tet.latticeA * tet.latticeA * tet.latticeC;
        candidates.push({
          structure: `tetragonal-${ct}`,
          centering: ct,
          crystalSystem: 'tetragonal',
          latticeA: tet.latticeA,
          latticeB: tet.latticeA,
          latticeC: tet.latticeC,
          cOverA: tet.cOverA,
          score: tet.score,
          scoreDOF: calcScoreDOF(tet.score, tet.matched, 2, tet.latticeA, angleToleranceDeg, 'tetragonal'),
          numFreeParams: 2,
          matchedCount: tet.matched,
          totalPeaks: sortedObjs.length,
          matchRate: tet.matched / sortedObjs.length,
          meanResidualDeg: tet.meanResidualDeg,
          assignments: tet.assignments,
          cellVolumeAngstrom3: vol,
        });
      }
    }
  }

  // 정렬: 매칭률 내림차순 → scoreDOF(자유도 보정) 오름차순
  // scoreDOF = cost / (matched - numFreeParams) : 매개변수 많은 모델의 과적합 억제
  candidates.sort((x, y) => {
    const dr = y.matchRate - x.matchRate;
    if (Math.abs(dr) > 0.04) return dr;
    return (x.scoreDOF ?? x.score) - (y.scoreDOF ?? y.score);
  });

  // ── 4. d-비율 사전 분류기 ─────────────────────────────────
  const dValues = sortedObjs.map((o) => {
    const d = dFromTwoTheta(o.angle, wavelengthAngstrom);
    return d;
  }).filter(Boolean).sort((a, b) => b - a);
  const dRatioHint = estimateCrystalSystemFromDRatios(dValues);

  // ── 5. 프리셋 매칭 ────────────────────────────────────────
  let presetMatches = null;
  if (includeCommonPhases) {
    presetMatches = matchWithCommonPhases(peaks, wavelengthAngstrom, angleToleranceDeg);
  }

  // 경고 메시지
  const warnings = [];
  if (candidates[0] && candidates[0].matchedCount < sortedObjs.length) {
    warnings.push('일부 피크가 이론 반사와 매칭되지 않습니다. 다상·비정질·측정 오차를 확인하세요.');
  }
  warnings.push('자동 상 동정은 참고용입니다. PDF/CIF 데이터베이스와의 대조를 권장합니다.');

  const latticeReportTop = candidates.length > 0 ? summarizeLatticeReport(candidates[0]) : null;

  return {
    success: true,
    candidates,
    presetMatches,
    peaksUsed: sortedObjs.map((o) => o.angle),
    wavelengthAngstrom,
    angleToleranceDeg,
    dRatioHint,
    warnings,
    latticeReportTop,
  };
}

/**
 * 선택된 후보의 격자 리포트 요약
 * @param {Object} candidate
 * @returns {Object|null}
 */
export function summarizeLatticeReport(candidate) {
  if (!candidate || !candidate.assignments) return null;
  const withDelta = candidate.assignments.filter(a => a.deltaDeg != null);
  return {
    structure: candidate.structure,
    centering: candidate.centering,
    crystalSystem: candidate.crystalSystem,
    a: candidate.latticeA,
    b: candidate.latticeB,
    c: candidate.latticeC,
    cOverA: candidate.cOverA,
    cellVolumeAngstrom3: candidate.cellVolumeAngstrom3,
    matchedReflections: withDelta.length,
    totalPeaks: candidate.totalPeaks,
    matchRate: candidate.matchRate,
    rmsDeltaDeg:
      withDelta.length > 0
        ? Math.sqrt(withDelta.reduce((s, a) => s + a.deltaDeg * a.deltaDeg, 0) / withDelta.length)
        : null,
    peakAssignments: withDelta.map(a => ({
      twoTheta: a.measuredTwoThetaDeg,
      millerLabel: a.millerLabel,
      dMeasured: a.dMeasuredAngstrom,
      dTheoretical: a.dTheoreticalAngstrom,
      deltaDeg: a.deltaDeg,
      deltaDAngstrom:
        a.dMeasuredAngstrom != null && a.dTheoreticalAngstrom != null
          ? Math.abs(a.dMeasuredAngstrom - a.dTheoreticalAngstrom)
          : null,
      multiplicity: a.multiplicity,
      relativeIntensityEstimate: a.relativeIntensityEstimate,
    })),
  };
}

/**
 * 자동 상동정 후보의 격자로부터 이론 피크 목록 (차트 오버레이 등)
 * @param {Object} candidate — identifyPhaseCandidates의 candidates[] 요소
 * @param {number} wavelengthAngstrom
 * @param {number} [twoThetaMin=5]
 * @param {number} [twoThetaMax=100]
 */
export function listTheoreticalPeaksForCandidate(candidate, wavelengthAngstrom, twoThetaMin = 5, twoThetaMax = 100) {
  if (!candidate || candidate.latticeA == null) return [];
  const s = (candidate.structure || '').toLowerCase();
  if (s === 'fcc' || s === 'bcc' || s === 'sc') {
    return listCubicTheoreticalPeaks(s, candidate.latticeA, wavelengthAngstrom, twoThetaMin, twoThetaMax)
      .map((p) => ({ ...p, millerLabel: `(${p.h}${p.k}${p.l})` }));
  }
  if (s === 'hcp' && candidate.latticeC != null) {
    return listHcpTheoreticalPeaks(candidate.latticeA, candidate.latticeC, wavelengthAngstrom, twoThetaMin, twoThetaMax)
      .map((p) => ({ ...p, millerLabel: `(${p.h}${p.k}${p.l})` }));
  }
  if (s.startsWith('tetragonal-') && candidate.latticeC != null) {
    const ct = (candidate.centering || s.replace('tetragonal-', '') || 'P').toUpperCase();
    return listTetragonalTheoreticalPeaks(
      candidate.latticeA,
      candidate.latticeC,
      ct === 'I' ? 'I' : 'P',
      wavelengthAngstrom,
      twoThetaMin,
      twoThetaMax
    ).map((p) => ({ ...p, millerLabel: `(${p.h}${p.k}${p.l})` }));
  }
  return [];
}

/**
 * COMMON_PHASES에서 이름으로 이론 피크 목록 생성
 * @param {string} phaseName  — COMMON_PHASES key
 * @param {number} wavelengthAngstrom
 * @param {number} [twoThetaMin=5]
 * @param {number} [twoThetaMax=100]
 * @returns {Array<{twoTheta, h, k, l, d, relativeIntensity, millerLabel}>}
 */
export function getTheoreticalPeaksForPhase(phaseName, wavelengthAngstrom, twoThetaMin = 5, twoThetaMax = 100) {
  const phase = COMMON_PHASES[phaseName];
  if (!phase) return [];
  if (phase.system === 'cubic') {
    const structure = phase.centering === 'F' ? 'fcc' : phase.centering === 'I' ? 'bcc' : 'sc';
    return listCubicTheoreticalPeaks(structure, phase.a, wavelengthAngstrom, twoThetaMin, twoThetaMax)
      .map(p => ({ ...p, millerLabel: `(${p.h}${p.k}${p.l})` }));
  } else if (phase.system === 'hexagonal') {
    return listHcpTheoreticalPeaks(phase.a, phase.c, wavelengthAngstrom, twoThetaMin, twoThetaMax)
      .map(p => ({ ...p, millerLabel: `(${p.h}${p.k}${p.l})` }));
  } else if (phase.system === 'tetragonal') {
    return listTetragonalTheoreticalPeaks(phase.a, phase.c, phase.centering, wavelengthAngstrom, twoThetaMin, twoThetaMax)
      .map(p => ({ ...p, millerLabel: `(${p.h}${p.k}${p.l})` }));
  }
  return [];
}
