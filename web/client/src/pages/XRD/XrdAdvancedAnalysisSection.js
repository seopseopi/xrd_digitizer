import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  createStressSin2PsiPlotData,
  createStressSin2PsiPlotOptions,
} from './xrdVisualization';

/**
 * 계획서 2~4단계 UI: 상 동정·격자, 배향, QPA, sin²ψ 응력, 리트벨트 안내
 */
export default function XrdAdvancedAnalysisSection({
  indexedPeaks,
  wavelength,
  fileName,
  disabled,
  phaseIdentificationResult,
  theoryOverlayCount = 0,
  onShowTheoryOverlay,
  onClearTheoryOverlay,
  textureResult,
  qpaResult,
  stressResult,
  rietveldInfo,
  isPhaseLoading,
  isTextureLoading,
  isQpaLoading,
  isStressLoading,
  isRietveldLoading,
  onIdentifyPhase,
  onComputeTexture,
  onComputeQpa,
  onComputeStress,
  onLoadRietveldInfo,
}) {
  const [activeTab, setActiveTab] = useState('structure');
  const [textureRefs, setTextureRefs] = useState([]);
  const [qpaRows, setQpaRows] = useState([
    { phaseId: '상 A', rir: 1, intensity: '' },
    { phaseId: '상 B', rir: 1, intensity: '' },
  ]);
  const [stressLines, setStressLines] = useState('0,30.0\n15,30.05\n30,30.1');
  const [youngGpa, setYoungGpa] = useState(200);
  const [poisson, setPoisson] = useState(0.3);

  const peakOptions = useMemo(
    () =>
      (indexedPeaks || []).map((p, i) => ({
        i,
        label: `${(p.angle ?? 0).toFixed(2)}° I≈${(p.intensity ?? 0).toFixed(0)}`,
        intensity: p.intensity,
      })),
    [indexedPeaks]
  );

  const textureInitPeakCount = useRef(-1);
  useEffect(() => {
    textureInitPeakCount.current = -1;
  }, [fileName]);

  useEffect(() => {
    const n = indexedPeaks?.length || 0;
    if (n === 0) {
      textureInitPeakCount.current = -1;
      return;
    }
    if (textureInitPeakCount.current === n) return;
    textureInitPeakCount.current = n;
    setTextureRefs(
      peakOptions.slice(0, Math.min(5, peakOptions.length)).map((o) => ({
        peakIndex: o.i,
        label: o.label,
        intensityReference: 100,
      }))
    );
  }, [indexedPeaks, peakOptions, fileName]);

  const runTexture = () => {
    const rows = textureRefs
      .map((r) => {
        const peak = indexedPeaks[r.peakIndex];
        if (!peak) return null;
        return {
          label: r.label,
          intensityMeasured: peak.intensity,
          intensityReference: Number(r.intensityReference),
        };
      })
      .filter(Boolean);
    if (rows.length === 0) {
      window.alert('유효한 피크–참조 강도 행이 없습니다.');
      return;
    }
    onComputeTexture(rows);
  };

  const runQpa = () => {
    const phases = qpaRows
      .filter((r) => r.phaseId && r.intensity !== '')
      .map((r) => ({
        phaseId: r.phaseId,
        rir: Number(r.rir) || 1,
        intensity: Number(r.intensity),
      }));
    if (phases.length < 2) {
      window.alert('상 분율 계산에는 상이 최소 2개(이름·RIR·강도) 필요합니다.');
      return;
    }
    onComputeQpa(phases);
  };

  const runStress = () => {
    const lines = stressLines.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const points = [];
    for (const line of lines) {
      const parts = line.split(/[,;\t]/).map((s) => s.trim());
      if (parts.length < 2) continue;
      const psiDeg = Number(parts[0]);
      const twoThetaDeg = Number(parts[1]);
      if (Number.isFinite(psiDeg) && Number.isFinite(twoThetaDeg)) {
        points.push({ psiDeg, twoThetaDeg });
      }
    }
    if (points.length < 2) {
      window.alert('ψ–2θ 점이 최소 2개 필요합니다.');
      return;
    }
    onComputeStress(points, { youngModulusGPa: youngGpa, poissonRatio: poisson });
  };

  const tabBtn = (id, label) => (
    <button
      type="button"
      key={id}
      className="btn-rtg"
      onClick={() => setActiveTab(id)}
      style={{
        padding: '8px 14px',
        fontSize: '13px',
        background: activeTab === id ? 'var(--color-primary)' : 'var(--color-sub-2)',
        color: activeTab === id ? 'white' : 'var(--color-text-1)',
        border: 'none',
        borderRadius: '6px',
        cursor: 'pointer',
      }}
    >
      {label}
    </button>
  );

  return (
    <div className="card-col gap10" style={{ marginTop: '24px' }}>
      <h3 style={{ margin: 0 }}>고급 분석 (상·배향·QPA·응력)</h3>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {tabBtn('structure', '구조·상 동정')}
        {tabBtn('texture', '배향 지수')}
        {tabBtn('qpa', '반정량 상분율')}
        {tabBtn('stress', 'sin²ψ 잔류응력')}
        {tabBtn('rietveld', '리트벨트 안내')}
      </div>

      {activeTab === 'structure' && (
        <div className="card-col gap10">
          <p style={{ fontSize: '12px', color: 'var(--color-text-2)', margin: 0 }}>
            FCC/BCC/단순 큐빅·HCP(c/a 그리드 탐색)·정방정(P/I)에 대해 격자를 탐색하고 피크를 매칭합니다. 강도 순·구조인자 근사로 동률 매칭을 보조합니다.
            Match!·PDF-4 수준의 ICDD 카드 서치는 라이선스가 필요하며, 추후 COD·오픈 CIF/PDF 조각과의 자동 대조를 염두에 두고 있습니다.
          </p>
          <button
            type="button"
            className="btn-rtg btn-primary"
            disabled={disabled || isPhaseLoading || !indexedPeaks?.length}
            onClick={onIdentifyPhase}
          >
            {isPhaseLoading ? '분석 중...' : '상 후보 분석 실행'}
          </button>
          {phaseIdentificationResult?.dRatioHint && (
            <div
              style={{
                fontSize: '12px',
                padding: '8px 10px',
                background: 'var(--color-sub-2)',
                borderRadius: '8px',
              }}
            >
              <strong>d-비율 지문 추정:</strong> {phaseIdentificationResult.dRatioHint.likelySystem}
              {phaseIdentificationResult.dRatioHint.scores &&
                Object.keys(phaseIdentificationResult.dRatioHint.scores).length > 0 && (
                  <span style={{ marginLeft: '8px', color: 'var(--color-text-2)' }}>
                    (RMSE 점수:{' '}
                    {Object.entries(phaseIdentificationResult.dRatioHint.scores)
                      .map(([k, v]) => `${k} ${typeof v === 'number' ? v.toFixed(3) : v}`)
                      .join(', ')}
                    )
                  </span>
                )}
            </div>
          )}
          {phaseIdentificationResult?.candidates?.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
              <button
                type="button"
                className="btn-rtg"
                disabled={disabled || !onShowTheoryOverlay}
                onClick={onShowTheoryOverlay}
              >
                1순위 후보 이론 피크 → 차트 오버레이
              </button>
              <button
                type="button"
                className="btn-rtg"
                disabled={!theoryOverlayCount || !onClearTheoryOverlay}
                onClick={onClearTheoryOverlay}
              >
                오버레이 지우기 {theoryOverlayCount ? `(${theoryOverlayCount})` : ''}
              </button>
              <span style={{ fontSize: '11px', color: 'var(--color-text-2)' }}>
                「피크 탐색」단계 메인 차트에 주황 점선으로 표시됩니다.
              </span>
            </div>
          )}
          {phaseIdentificationResult?.candidates?.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--color-sub-2)' }}>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>순위</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>구조</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>a (Å)</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>c (Å)</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>c/a</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>매칭</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>잔차 RMS (°)</th>
                    <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>V (Å³)</th>
                  </tr>
                </thead>
                <tbody>
                  {phaseIdentificationResult.candidates.slice(0, 6).map((c, idx) => (
                    <tr key={`${c.structure}-${idx}`}>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>{idx + 1}</td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>{c.structure}</td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>{c.latticeA?.toFixed(4)}</td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                        {c.crystalSystem === 'cubic' ? '—' : c.latticeC?.toFixed(4) ?? '—'}
                      </td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                        {c.crystalSystem === 'cubic' ? '—' : c.cOverA != null ? Number(c.cOverA).toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                        {c.matchedCount}/{c.totalPeaks}
                      </td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                        {c.meanResidualDeg != null ? c.meanResidualDeg.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                        {c.cellVolumeAngstrom3 != null ? c.cellVolumeAngstrom3.toFixed(2) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {phaseIdentificationResult?.latticeReportTop?.peakAssignments?.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <strong style={{ fontSize: '13px' }}>1순위 후보 — 피크별 (hkl) · d</strong>
              <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse', marginTop: '8px' }}>
                <thead>
                  <tr style={{ background: 'var(--color-sub-2)' }}>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>2θ_meas</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>(hkl)</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>d_meas</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>d_theory</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>Δd</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>Δ2θ°</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>M</th>
                    <th style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>I_est</th>
                  </tr>
                </thead>
                <tbody>
                  {phaseIdentificationResult.latticeReportTop.peakAssignments.map((row, i) => (
                    <tr key={i}>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>
                        {row.twoTheta != null ? row.twoTheta.toFixed(3) : '—'}
                      </td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>{row.millerLabel ?? '—'}</td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>
                        {row.dMeasured != null ? row.dMeasured.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>
                        {row.dTheoretical != null ? row.dTheoretical.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>
                        {row.deltaDAngstrom != null ? row.deltaDAngstrom.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>
                        {row.deltaDeg != null ? row.deltaDeg.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>{row.multiplicity ?? '—'}</td>
                      <td style={{ padding: '6px', border: '1px solid var(--color-monotone-2)' }}>
                        {row.relativeIntensityEstimate != null ? row.relativeIntensityEstimate.toFixed(1) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {phaseIdentificationResult?.latticeReportTop && (
            <div
              style={{
                fontSize: '12px',
                padding: '10px',
                background: 'var(--color-sub-2)',
                borderRadius: '8px',
              }}
            >
              <strong>1순위 후보 격자 요약</strong>
              <div>
                RMS Δ2θ:{' '}
                {phaseIdentificationResult.latticeReportTop.rmsDeltaDeg != null
                  ? `${phaseIdentificationResult.latticeReportTop.rmsDeltaDeg.toFixed(4)}°`
                  : '—'}
                , 매칭 반사 수: {phaseIdentificationResult.latticeReportTop.matchedReflections}
              </div>
            </div>
          )}
          {phaseIdentificationResult?.warnings?.map((w, i) => (
            <p key={i} style={{ fontSize: '11px', color: 'var(--color-text-2)', margin: 0 }}>
              ※ {w}
            </p>
          ))}
        </div>
      )}

      {activeTab === 'texture' && (
        <div className="card-col gap10">
          <p style={{ fontSize: '12px', color: 'var(--color-text-2)', margin: 0 }}>
            각 피크에 대해 PDF 등에서 얻은 <strong>참조 상대 강도</strong>를 입력하세요. 배향 지수는 기하평균으로 1 근처로 정규화합니다.
          </p>
          {textureRefs.map((row, idx) => (
            <div key={idx} style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
              <select
                value={row.peakIndex}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  const next = [...textureRefs];
                  next[idx] = { ...next[idx], peakIndex: v, label: peakOptions[v]?.label || '' };
                  setTextureRefs(next);
                }}
                style={{ minWidth: '200px', padding: '6px' }}
              >
                {peakOptions.map((o) => (
                  <option key={o.i} value={o.i}>
                    {o.label}
                  </option>
                ))}
              </select>
              <label style={{ fontSize: '12px' }}>
                I_ref
                <input
                  type="number"
                  min="0.001"
                  step="0.1"
                  value={row.intensityReference}
                  onChange={(e) => {
                    const next = [...textureRefs];
                    next[idx] = { ...next[idx], intensityReference: parseFloat(e.target.value) || 0 };
                    setTextureRefs(next);
                  }}
                  style={{ marginLeft: '6px', width: '90px', padding: '4px' }}
                />
              </label>
            </div>
          ))}
          <button
            type="button"
            className="btn-rtg btn-primary"
            disabled={disabled || isTextureLoading || textureRefs.length === 0}
            onClick={runTexture}
          >
            {isTextureLoading ? '계산 중...' : '배향 지수 계산'}
          </button>
          {textureResult?.warnings?.map((w, i) => (
            <p key={i} style={{ fontSize: '11px', color: 'var(--color-text-2)', margin: 0 }}>
              ※ {w}
            </p>
          ))}
          {textureResult?.results?.length > 0 && (
            <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'var(--color-sub-2)' }}>
                  <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>피크</th>
                  <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>P (정규화)</th>
                  <th style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>I_meas/I_ref</th>
                </tr>
              </thead>
              <tbody>
                {textureResult.results.map((r, i) => (
                  <tr key={i}>
                    <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>{r.label}</td>
                    <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                      {r.orientationIndex.toFixed(3)}
                    </td>
                    <td style={{ padding: '8px', border: '1px solid var(--color-monotone-2)' }}>
                      {r.rawRatio.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {activeTab === 'qpa' && (
        <div className="card-col gap10">
          <p style={{ fontSize: '12px', color: 'var(--color-text-2)', margin: 0 }}>
            각 상의 <strong>대표 피크 강도(높이)</strong>와 PDF의 <strong>RIR(I/Ic)</strong>를 입력합니다. 부피분율은 I/RIR 비로 근사합니다.
          </p>
          {qpaRows.map((row, idx) => (
            <div key={idx} style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
              <input
                placeholder="상 이름"
                value={row.phaseId}
                onChange={(e) => {
                  const next = [...qpaRows];
                  next[idx] = { ...next[idx], phaseId: e.target.value };
                  setQpaRows(next);
                }}
                style={{ width: '120px', padding: '6px' }}
              />
              <label style={{ fontSize: '12px' }}>
                RIR
                <input
                  type="number"
                  min="0.001"
                  step="0.1"
                  value={row.rir}
                  onChange={(e) => {
                    const next = [...qpaRows];
                    next[idx] = { ...next[idx], rir: parseFloat(e.target.value) || 1 };
                    setQpaRows(next);
                  }}
                  style={{ marginLeft: '6px', width: '70px', padding: '4px' }}
                />
              </label>
              <label style={{ fontSize: '12px' }}>
                I(강도)
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={row.intensity}
                  onChange={(e) => {
                    const next = [...qpaRows];
                    next[idx] = { ...next[idx], intensity: e.target.value };
                    setQpaRows(next);
                  }}
                  style={{ marginLeft: '6px', width: '100px', padding: '4px' }}
                />
              </label>
            </div>
          ))}
          <button
            type="button"
            className="btn-rtg"
            onClick={() => setQpaRows([...qpaRows, { phaseId: `상 ${qpaRows.length + 1}`, rir: 1, intensity: '' }])}
          >
            행 추가
          </button>
          <button
            type="button"
            className="btn-rtg btn-primary"
            disabled={disabled || isQpaLoading}
            onClick={runQpa}
          >
            {isQpaLoading ? '계산 중...' : '상 분율 계산'}
          </button>
          {qpaResult?.phases?.length > 0 && (
            <ul style={{ fontSize: '13px' }}>
              {qpaResult.phases.map((p) => (
                <li key={p.phaseId}>
                  <strong>{p.phaseId}</strong>: 부피분율 ≈ {(p.approximateVolumeFraction * 100).toFixed(2)}%
                </li>
              ))}
            </ul>
          )}
          {qpaResult?.warnings?.map((w, i) => (
            <p key={i} style={{ fontSize: '11px', color: 'var(--color-text-2)', margin: 0 }}>
              ※ {w}
            </p>
          ))}
        </div>
      )}

      {activeTab === 'stress' && (
        <div className="card-col gap10">
          <p style={{ fontSize: '12px', color: 'var(--color-text-2)', margin: 0 }}>
            한 줄에 <code>ψ(°),2θ(°)</code> 형식으로 입력합니다. 동일 (hkl)에 대해 ψ만 변화한 데이터여야 합니다.
          </p>
          <textarea
            value={stressLines}
            onChange={(e) => setStressLines(e.target.value)}
            rows={5}
            style={{ width: '100%', fontFamily: 'monospace', fontSize: '12px', padding: '8px' }}
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
            <label style={{ fontSize: '12px' }}>
              E (GPa)
              <input
                type="number"
                value={youngGpa}
                onChange={(e) => setYoungGpa(parseFloat(e.target.value) || 0)}
                style={{ marginLeft: '6px', width: '80px', padding: '4px' }}
              />
            </label>
            <label style={{ fontSize: '12px' }}>
              ν
              <input
                type="number"
                step="0.01"
                value={poisson}
                onChange={(e) => setPoisson(parseFloat(e.target.value) || 0)}
                style={{ marginLeft: '6px', width: '60px', padding: '4px' }}
              />
            </label>
          </div>
          <button
            type="button"
            className="btn-rtg btn-primary"
            disabled={disabled || isStressLoading}
            onClick={runStress}
          >
            {isStressLoading ? '피팅 중...' : '잔류 응력 계산'}
          </button>
          {Number.isFinite(stressResult?.stressMPa) && (
            <>
              <div className="info-item">
                <span className="info-label">추정 응력 σ (MPa):</span>
                <span className="info-value">{stressResult.stressMPa.toFixed(2)}</span>
              </div>
              <div className="info-item">
                <span className="info-label">d₀ (Å):</span>
                <span className="info-value">{stressResult.d0Angstrom?.toFixed(5)}</span>
              </div>
              <div className="info-item">
                <span className="info-label">R²:</span>
                <span className="info-value">{stressResult.rSquared?.toFixed(4)}</span>
              </div>
              {createStressSin2PsiPlotData(stressResult) && (
                <div className="chart-container" style={{ height: '320px', minHeight: '320px' }}>
                  <Line
                    data={createStressSin2PsiPlotData(stressResult)}
                    options={createStressSin2PsiPlotOptions()}
                  />
                </div>
              )}
            </>
          )}
          {stressResult?.warnings?.map((w, i) => (
            <p key={i} style={{ fontSize: '11px', color: 'var(--color-text-2)', margin: 0 }}>
              ※ {w}
            </p>
          ))}
        </div>
      )}

      {activeTab === 'rietveld' && (
        <div className="card-col gap10">
          <p style={{ fontSize: '13px', color: 'var(--color-text-2)', lineHeight: 1.5 }}>
            앱 내 전체 패턴 리트벨트 엔진은 포함하지 않습니다. 정밀 정량·구조 정밀화는 전용 프로그램과 PDF 라이선스를 갖춘 워크플로를 권장합니다.
          </p>
          <button
            type="button"
            className="btn-rtg btn-primary"
            disabled={isRietveldLoading}
            onClick={onLoadRietveldInfo}
          >
            {isRietveldLoading ? '불러오는 중...' : '권장 도구·유의사항 보기'}
          </button>
          {rietveldInfo && (
            <pre
              style={{
                whiteSpace: 'pre-wrap',
                fontSize: '12px',
                padding: '12px',
                background: 'var(--color-sub-2)',
                borderRadius: '8px',
              }}
            >
              {JSON.stringify(rietveldInfo, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
