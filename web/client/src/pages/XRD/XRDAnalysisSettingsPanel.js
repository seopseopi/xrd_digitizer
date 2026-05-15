import React, { useCallback, useRef, useState } from 'react';

const IconFileUp = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinejoin="round" />
    <path d="M14 2v6h6 M12 18v-6 M9 15l3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const IconFileUpLarge = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinejoin="round" />
    <path d="M14 2v6h6 M12 18v-6 M9 15l3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const IconPeakSearch = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M4 19V5 M8 19v-8 M12 19V9 M16 19v-5 M20 19v-12" strokeLinecap="round" />
  </svg>
);

const IconPeakFit = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M3 18c2-6 4-9 7-9s4 3 7 9" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M3 18h18" strokeLinecap="round" />
  </svg>
);

const IconCrystal = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M12 2l8 5v10l-8 5-8-5V7z" strokeLinejoin="round" />
    <path d="M12 2v20 M4 7l8 5 8-5 M4 17l8 5 8-5" strokeLinejoin="round" />
  </svg>
);

const Chevron = ({ expanded }) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    {expanded ? <path d="M18 15l-6-6-6 6" strokeLinecap="round" strokeLinejoin="round" /> : <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />}
  </svg>
);

function AccordionSection({ id, title, icon, expanded, onToggle, children }) {
  return (
    <div className="xrd-settings-accordion-section" data-section={id}>
      <button
        type="button"
        className="xrd-settings-accordion-header"
        onClick={() => onToggle(id)}
        aria-expanded={expanded}
      >
        <span className="xrd-settings-accordion-title">
          <span className="xrd-settings-accordion-icon">{icon}</span>
          {title}
        </span>
        <span className="xrd-settings-accordion-chevron">
          <Chevron expanded={expanded} />
        </span>
      </button>
      {expanded && <div className="xrd-settings-accordion-body">{children}</div>}
    </div>
  );
}

const defaultOpen = {
  upload: true,
  peakSearch: true,
  peakFit: false,
  crystallography: false,
};

/**
 * XRD 분석 설정 UI (앱 우측 Toolbar 패널에 마운트).
 */
const XRDAnalysisSettingsPanel = ({
  settings,
  setSettings,
  fileInputId,
  fileName,
  isProcessing,
  onFileChange,
  onLoadSampleData,
  onRunAnalysis,
  isAnalyzing,
  hasData,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const [dropActive, setDropActive] = useState(false);
  const inputRef = useRef(null);

  const toggle = useCallback((id) => {
    setOpen((prev) => ({ ...prev, [id]: !prev[id] }));
  }, []);

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDropActive(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDropActive(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDropActive(false);
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    const synthetic = { target: { files: [f], value: '' } };
    onFileChange(synthetic);
  };

  const openFilePicker = () => inputRef.current?.click();

  return (
    <div className="xrd-analysis-settings-panel">
      <div className="xrd-settings-accordion">
        <AccordionSection
          id="upload"
          title="파일 업로드"
          icon={<IconFileUp />}
          expanded={open.upload}
          onToggle={toggle}
        >
          <input
            ref={inputRef}
            type="file"
            id={fileInputId}
            className="xrd-settings-file-input"
            accept=".brml,.powdll,.csv,.xy,.xye,.dat,.raw,.txt,.cif,.json"
            onChange={onFileChange}
          />
          <div
            role="button"
            tabIndex={0}
            className={`xrd-settings-dropzone ${dropActive ? 'xrd-settings-dropzone--active' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={openFilePicker}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openFilePicker();
              }
            }}
          >
            <div className="xrd-settings-dropzone-icon">
              <IconFileUpLarge />
            </div>
            <p className="xrd-settings-dropzone-main">파일을 드래그하거나 클릭</p>
            <p className="xrd-settings-dropzone-hint">.xy .xye .csv .dat .raw 등</p>
            {fileName && (
              <p className="xrd-settings-dropzone-file" title={fileName}>
                {fileName}
              </p>
            )}
          </div>
          <button
            type="button"
            className="xrd-settings-sample-btn"
            onClick={(e) => {
              e.stopPropagation();
              onLoadSampleData();
            }}
            disabled={isProcessing}
          >
            샘플 데이터 로드
          </button>
        </AccordionSection>

        <AccordionSection
          id="peakSearch"
          title="피크 탐색"
          icon={<IconPeakSearch />}
          expanded={open.peakSearch}
          onToggle={toggle}
        >
          <label className="xrd-settings-field">
            <span className="xrd-settings-label">탐색 방법</span>
            <select
              className="xrd-settings-select"
              value={settings.peakDetectionMethod}
              onChange={(e) => setSettings((prev) => ({ ...prev, peakDetectionMethod: e.target.value }))}
            >
              <option value="localMaxima">로컬 맥시마</option>
              <option value="secondDerivative">2차 미분</option>
            </select>
          </label>

          <div className="xrd-settings-slider-row">
            <span className="xrd-settings-label">평활화 창</span>
            <span className="xrd-settings-slider-value">{settings.smoothingWindow}</span>
          </div>
          <input
            type="range"
            className="xrd-settings-range"
            min="3"
            max="15"
            step="2"
            value={settings.smoothingWindow}
            onChange={(e) => setSettings((prev) => ({ ...prev, smoothingWindow: parseInt(e.target.value, 10) }))}
          />

          <div className="xrd-settings-slider-row">
            <span className="xrd-settings-label">최소 피크 높이</span>
            <span className="xrd-settings-slider-value">{(settings.minPeakHeight * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            className="xrd-settings-range"
            min="0.01"
            max="0.2"
            step="0.01"
            value={settings.minPeakHeight}
            onChange={(e) => setSettings((prev) => ({ ...prev, minPeakHeight: parseFloat(e.target.value) }))}
          />

          <div className="xrd-settings-slider-row">
            <span className="xrd-settings-label">최소 피크 간격</span>
            <span className="xrd-settings-slider-value">{settings.minPeakDistance.toFixed(1)}°</span>
          </div>
          <input
            type="range"
            className="xrd-settings-range"
            min="0.05"
            max="2"
            step="0.05"
            value={settings.minPeakDistance}
            onChange={(e) => setSettings((prev) => ({ ...prev, minPeakDistance: parseFloat(e.target.value) }))}
          />
        </AccordionSection>

        <AccordionSection
          id="peakFit"
          title="피크 피팅"
          icon={<IconPeakFit />}
          expanded={open.peakFit}
          onToggle={toggle}
        >
          <label className="xrd-settings-field xrd-settings-check">
            <input
              type="checkbox"
              checked={settings.enableFitting}
              onChange={(e) => setSettings((prev) => ({ ...prev, enableFitting: e.target.checked }))}
            />
            <span>피크 피팅 수행</span>
          </label>
          <label className="xrd-settings-field">
            <span className="xrd-settings-label">피크 모델</span>
            <select
              className="xrd-settings-select"
              value={settings.peakType}
              onChange={(e) => setSettings((prev) => ({ ...prev, peakType: e.target.value }))}
            >
              <option value="gaussian">가우시안</option>
              <option value="lorentzian">로렌츠</option>
              <option value="voigt">Voigt</option>
            </select>
          </label>
          <label className="xrd-settings-field">
            <span className="xrd-settings-label">배경 타입</span>
            <select
              className="xrd-settings-select"
              value={settings.backgroundType}
              onChange={(e) => setSettings((prev) => ({ ...prev, backgroundType: e.target.value }))}
            >
              <option value="linear">선형</option>
              <option value="polynomial">다항식</option>
              <option value="none">없음</option>
            </select>
          </label>
          <div className="xrd-settings-divider" />
          <span className="xrd-settings-subheading">표시</span>
          <label className="xrd-settings-field xrd-settings-check">
            <input
              type="checkbox"
              checked={settings.showFittedCurve}
              onChange={(e) => setSettings((prev) => ({ ...prev, showFittedCurve: e.target.checked }))}
            />
            <span>피팅 곡선</span>
          </label>
          <label className="xrd-settings-field xrd-settings-check">
            <input
              type="checkbox"
              checked={settings.showPeakMarkers}
              onChange={(e) => setSettings((prev) => ({ ...prev, showPeakMarkers: e.target.checked }))}
            />
            <span>피크 마커</span>
          </label>
          <label className="xrd-settings-field xrd-settings-check">
            <input
              type="checkbox"
              checked={settings.showIndexedPeaks}
              onChange={(e) => setSettings((prev) => ({ ...prev, showIndexedPeaks: e.target.checked }))}
            />
            <span>인덱싱된 피크</span>
          </label>
        </AccordionSection>

        <AccordionSection
          id="crystallography"
          title="결정학 설정"
          icon={<IconCrystal />}
          expanded={open.crystallography}
          onToggle={toggle}
        >
          <label className="xrd-settings-field">
            <span className="xrd-settings-label">X선 파장 (Å)</span>
            <input
              type="number"
              className="xrd-settings-input"
              min="0.5"
              max="3"
              step="0.0001"
              value={settings.wavelength}
              onChange={(e) => setSettings((prev) => ({ ...prev, wavelength: parseFloat(e.target.value) }))}
            />
          </label>
          <label className="xrd-settings-field">
            <span className="xrd-settings-label" title="Scherrer 형상 인자 K">
              Scherrer K
            </span>
            <input
              type="number"
              className="xrd-settings-input"
              min="0.5"
              max="1.2"
              step="0.01"
              value={settings.shapeFactor}
              onChange={(e) => setSettings((prev) => ({ ...prev, shapeFactor: parseFloat(e.target.value) }))}
            />
          </label>
          <label className="xrd-settings-field">
            <span className="xrd-settings-label" title="기기 FWHM (2θ, °)">
              기기 FWHM (2θ, °)
            </span>
            <input
              type="number"
              className="xrd-settings-input"
              min="0"
              max="2"
              step="0.001"
              value={settings.instrumentalFwhmDeg}
              onChange={(e) =>
                setSettings((prev) => ({ ...prev, instrumentalFwhmDeg: parseFloat(e.target.value) || 0 }))
              }
            />
          </label>
          <label className="xrd-settings-field">
            <span className="xrd-settings-label">기기 넓이 보정</span>
            <select
              className="xrd-settings-select"
              value={settings.instrumentalCorrection}
              onChange={(e) => setSettings((prev) => ({ ...prev, instrumentalCorrection: e.target.value }))}
            >
              <option value="none">없음</option>
              <option value="subtract">차감 (β − β_inst)</option>
              <option value="quadratic">가우시안 (β² − β_inst²)^(1/2)</option>
            </select>
          </label>
        </AccordionSection>
      </div>

      <div className="xrd-settings-actions">
        <button
          type="button"
          className="xrd-settings-run-btn"
          onClick={onRunAnalysis}
          disabled={!hasData || isAnalyzing}
        >
          {isAnalyzing ? '분석 중…' : '분석 실행'}
        </button>
      </div>
    </div>
  );
};

export default XRDAnalysisSettingsPanel;
