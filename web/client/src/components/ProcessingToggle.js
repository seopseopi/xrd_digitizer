/**
 * ProcessingToggle / ProcessingLocationBar
 *
 * 모든 분석기에서 공통으로 사용하는 로컬/서버 처리 위치 선택 바.
 *
 * 기능:
 *  1. 로컬 / 서버 강제 전환 토글
 *  2. 대용량 파일 자동 전환 알림 배너
 *  3. 백엔드 비동기 작업 진행률 바
 *
 * 사용법 (권장 - 분석기에서):
 *   import ProcessingLocationBar from '../../../components/ProcessingToggle';
 *   <ProcessingLocationBar />
 *
 * 사용법 (hook 직접 사용이 필요한 경우):
 *   const hook = useProcessingLocation();
 *   <ProcessingToggle hook={hook} />
 */
import React, { useState } from 'react';
import { useProcessingLocation } from '../hooks/useProcessingLocation';

/* ────────────────────────────────── 아이콘 헬퍼 ── */
const Icon = ({ name, size = 18, className = '', style = {} }) => {
  const sizeClass = [13, 14, 15, 16].includes(size) ? ` pl-icon-${size}` : '';
  const cn = `material-symbols-rounded pl-icon${sizeClass} ${className}`.trim();
  return (
    <span
      className={cn}
      style={[13, 14, 15, 16].includes(size) ? { lineHeight: 1, ...style } : { fontSize: size, lineHeight: 1, ...style }}
    >
      {name}
    </span>
  );
};

/* ────────────────────────────────── 뱃지 ── */
const LocationBadge = ({ location }) => {
  const map = {
    local: { label: '로컬', icon: 'computer' },
    'auto-local': { label: '자동 (로컬)', icon: 'autorenew' },
    backend: { label: '서버', icon: 'cloud' },
    'auto-backend': { label: '자동 (서버)', icon: 'cloud_sync' },
  };
  const { label, icon } = map[location] || map['auto-local'];
  const dataLocation = map[location] ? location : 'auto-local';
  return (
    <span className="pl-badge" data-location={dataLocation}>
      <Icon name={icon} size={13} />
      {label}
    </span>
  );
};

/* ────────────────────────────────── 진행률 바 ── */
const JobProgressBar = ({ jobProgress, onCancel }) => {
  if (!jobProgress) return null;
  const { status, progress, message } = jobProgress;
  const isFailed = status === 'failed';
  const isDone = status === 'completed';
  const barPct = isDone ? 100 : isFailed ? 100 : progress;
  const statusKey = isFailed ? 'failed' : isDone ? 'completed' : 'running';

  return (
    <div className="pl-progress-bar">
      <div className="pl-progress-header">
        <span className="pl-progress-message" data-status={statusKey}>
          <Icon
            name={isFailed ? 'error' : isDone ? 'check_circle' : 'hourglass_top'}
            size={14}
          />
          <span>{message}</span>
        </span>
        <div className="pl-progress-actions">
          <span className="pl-progress-pct">{barPct}%</span>
          {!isDone && !isFailed && (
            <button
              type="button"
              className="pl-progress-cancel-btn"
              onClick={onCancel}
              title="취소"
            >
              <Icon name="close" size={14} />
            </button>
          )}
        </div>
      </div>
      <div className="pl-progress-track">
        <div
          className="pl-progress-fill"
          data-status={statusKey}
          style={{ width: `${barPct}%` }}
        />
      </div>
    </div>
  );
};

/* ────────────────────────────────── 대용량 파일 경고 배너 ── */
const LargeFileWarning = ({ warning, onDismiss }) => {
  if (!warning) return null;
  return (
    <div className="pl-large-file-warning">
      <Icon name="warning" size={15} />
      <span className="pl-message">
        <strong>{warning.fileName}</strong>({warning.sizeMB} MB) — 파일이 크기 때문에{' '}
        <strong>{warning.targetLocation === 'backend' ? '서버' : '로컬'}</strong>에서 처리됩니다.
      </span>
      <button
        type="button"
        className="pl-dismiss-btn"
        onClick={onDismiss}
        title="닫기"
      >
        <Icon name="close" size={14} />
      </button>
    </div>
  );
};

/* ────────────────────────────────── 메인 컴포넌트 ── */
export function ProcessingToggle({ hook, className = '', style: outerStyle = {} }) {
  const {
    settings,
    swAvailable,
    effectiveLocation,
    largeFileWarning,
    jobProgress,
    setForceLocal,
    setForceBackend,
    setThresholdMB,
    dismissLargeFileWarning,
    stopJobPolling,
  } = hook;

  const [expanded, setExpanded] = useState(false);

  /* 현재 모드 레이블 */
  const modeLabel =
    settings.forceLocal
      ? '로컬 고정'
      : settings.forceBackend
      ? '서버 고정'
      : '자동';

  return (
    <div
      className={`processing-location-bar ${className}`.trim()}
      style={outerStyle && Object.keys(outerStyle).length ? { ...outerStyle, marginBottom: 0 } : undefined}
    >
      {/* ── 컨트롤 바 ── */}
      <div className="pl-control-bar">
        {/* 처리 위치 뱃지 */}
        <Icon name="memory" size={16} />
        <span className="pl-label">처리 위치:</span>
        <LocationBadge location={effectiveLocation} />

        {/* SW 가용 여부 표시 (정보용) */}
        {!swAvailable && !settings.forceBackend && (
          <span
            className="pl-direct-hint"
            title="Service Worker 없이 Web Worker로 직접 로컬 처리합니다."
          >
            <Icon name="bolt" size={13} />
            직접 처리
          </span>
        )}

        <div className="pl-spacer" />

        {/* 모드 레이블 */}
        <span className="pl-mode-label">모드: {modeLabel}</span>

        {/* 펼치기/접기 */}
        <button
          type="button"
          className="pl-expand-btn"
          onClick={() => setExpanded((v) => !v)}
          title="처리 위치 설정"
        >
          <Icon name={expanded ? 'expand_less' : 'tune'} size={16} />
        </button>
      </div>

      {/* ── 설정 패널 (펼쳤을 때) ── */}
      {expanded && (
        <div className="pl-settings-panel">
          {/* 로컬 강제 */}
          <label className="pl-settings-label">
            <input
              type="checkbox"
              className="toggle"
              checked={!!settings.forceLocal}
              onChange={(e) => setForceLocal(e.target.checked)}
            />
            <span className="pl-label-content">
              <Icon name="computer" size={14} className="pl-icon-local" />
              항상 로컬 처리
            </span>
          </label>

          {/* 서버 강제 */}
          <label className="pl-settings-label">
            <input
              type="checkbox"
              className="toggle"
              checked={!!settings.forceBackend}
              onChange={(e) => setForceBackend(e.target.checked)}
            />
            <span className="pl-label-content">
              <Icon name="cloud" size={14} className="pl-icon-backend" />
              항상 서버 처리
            </span>
          </label>

          {/* 크기 임계값 */}
          {!settings.forceLocal && !settings.forceBackend && (
            <div className="pl-threshold-row">
              <Icon name="data_usage" size={14} className="pl-icon-data" />
              <span className="pl-label">자동 전환 기준:</span>
              <input
                type="number"
                className="pl-threshold-input"
                min={1}
                max={2000}
                value={settings.localThresholdMB ?? 50}
                onChange={(e) => setThresholdMB(e.target.value)}
              />
              <span className="pl-threshold-hint">MB 이상 시 서버 처리</span>
            </div>
          )}

          {/* SW 상태 */}
          <div
            className="pl-sw-status"
            data-available={swAvailable ? 'true' : 'false'}
          >
            <Icon
              name={swAvailable ? 'check_circle' : 'bolt'}
              size={13}
            />
            {swAvailable
              ? 'Service Worker 활성'
              : 'Web Worker 직접 처리 (Service Worker 불필요)'}
          </div>
        </div>
      )}

      {/* ── 대용량 파일 경고 배너 ── */}
      <LargeFileWarning warning={largeFileWarning} onDismiss={dismissLargeFileWarning} />

      {/* ── 백엔드 작업 진행률 바 ── */}
      <JobProgressBar jobProgress={jobProgress} onCancel={stopJobPolling} />
    </div>
  );
}

/**
 * ProcessingLocationBar - 훅을 내부에 포함한 통합 컴포넌트.
 * 모든 분석기에서 동일하게 사용할 수 있도록 단일 import로 사용.
 */
export function ProcessingLocationBar(props) {
  const hook = useProcessingLocation();
  return <ProcessingToggle hook={hook} {...props} />;
}

export default ProcessingLocationBar;
