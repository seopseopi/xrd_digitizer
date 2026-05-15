import { useState, useCallback } from 'react';
import { analysisClient } from '../analysis/analysisClient';

export function useProcessingLocation() {
  const [settings, setSettings] = useState(() => analysisClient.settings.get());
  const [jobProgress] = useState(null);
  const [largeFileWarning] = useState(null);

  const updateSettings = useCallback((partial) => {
    const updated = analysisClient.settings.update(partial);
    setSettings({ ...updated });
  }, []);

  const setForceLocal = useCallback(
    (value) => updateSettings({ forceLocal: value, forceBackend: value ? false : settings.forceBackend }),
    [settings, updateSettings]
  );
  const setForceBackend = useCallback(
    (value) => updateSettings({ forceBackend: value, forceLocal: value ? false : settings.forceLocal }),
    [settings, updateSettings]
  );
  const setThresholdMB = useCallback(
    (mb) => updateSettings({ localThresholdMB: Number(mb) }),
    [updateSettings]
  );

  return {
    settings,
    swAvailable: false,
    effectiveLocation: 'backend',
    largeFileWarning,
    jobProgress,
    setForceLocal,
    setForceBackend,
    setThresholdMB,
    checkFileAndRoute: () => 'backend',
    dismissLargeFileWarning: () => {},
    startJobPolling: () => {},
    stopJobPolling: () => {},
  };
}

export default useProcessingLocation;
