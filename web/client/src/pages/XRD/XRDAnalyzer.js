import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Chart } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { analysisClient } from '../../analysis/analysisClient';
import ProcessingLocationBar from '../../components/ProcessingToggle';
import { useToolbarContent } from '../../context/ToolbarContentContext';
import { 
  createXRDChartData, 
  createXRDChartOptions,
  createPeakTableData,
  createMWHPlotData,
  createMWHPlotOptions,
  createMWAPlotData,
  createMWAPlotOptions,
  createMWAYOverL2PlotData,
  createMWAYOverL2PlotOptions,
  createStandardWilliamsonHallPlotData,
  createStandardWilliamsonHallPlotOptions,
  ensureXrdTheoryOverlayRegistered,
} from './xrdVisualization';
import MaterialConstantsModal from './MaterialConstantsModal';
import XRDDigitizer from './XRDDigitizer';
import XrdAdvancedAnalysisSection from './XrdAdvancedAnalysisSection';
import XRDAnalysisSettingsPanel from './XRDAnalysisSettingsPanel';
import './XRDAnalyzer.css';
import {
  extractFWHMFromPeak,
  calculateDSpacing,
  calculateDSpacingFromMillerIndices,
  calculatePossibleMillerIndicesCubic,
  formatMillerIndicesForDisplay,
  impliedLatticeFromPeak,
  summarizeImpliedCubicAFromPeaks,
} from '../../analysis/core/xrd/millerIndex';
import { listTheoreticalPeaksForCandidate } from '../../analysis/core/xrd/phaseIdentification';

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin
);
ensureXrdTheoryOverlayRegistered();

/** 데모용 합성 XRD 패턴 (샘플 데이터 로드) */
function generateDemoXrdPattern() {
  const gauss = (t, c, a, w) => a * Math.exp(-((t - c) ** 2) / (2 * w * w));
  const points = [];
  for (let t = 10; t <= 90; t += 0.02) {
    let y = 120 + 0.35 * t;
    y += gauss(t, 31.76, 4200, 0.11);
    y += gauss(t, 45.52, 2800, 0.13);
    y += gauss(t, 56.6, 1900, 0.14);
    y += gauss(t, 67.0, 800, 0.16);
    y += 8 * Math.sin(t * 3.17 + 0.4);
    points.push({ angle: Math.round(t * 100) / 100, intensity: Math.max(0, y) });
  }
  return points;
}

/** Tabbar 직하단 스테퍼 — DataInput과 동일 마크업/클래스 */
const XRD_ANALYSIS_STEPS = [
  { id: 'peak-search', label: '피크 탐색' },
  { id: 'peak-info', label: '피크 분석' },
  { id: 'dislocation', label: '전위밀도 분석' },
  { id: 'advanced', label: '고급분석' },
];

const XRDAnalyzer = () => {
  const { setToolbarContent, setToolbarFooterContent } = useToolbarContent();

  // 모드: 'analyze' | 'digitize'
  const [mode, setMode] = useState('analyze');

  // State
  const [xrdData, setXrdData] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [structureInfo, setStructureInfo] = useState(null);
  const [metadata, setMetadata] = useState(null);
  
  // 분석 결과
  const [detectedPeaks, setDetectedPeaks] = useState([]);
  const [manualPeaks, setManualPeaks] = useState([]); // 수동 추가 피크
  const [, setFittedResult] = useState(null);
  const [indexedPeaks, setIndexedPeaks] = useState([]);
  const [crystallinity, setCrystallinity] = useState(null);
  const [crystalliteStats, setCrystalliteStats] = useState(null);
  
  // 전위 밀도 분석 관련 상태
  const [dislocationResults, setDislocationResults] = useState(null);
  const [materialConstants, setMaterialConstants] = useState(null);
  const [showDislocationAnalysis, setShowDislocationAnalysis] = useState(false);
  const [showMaterialConstantsModal, setShowMaterialConstantsModal] = useState(false);
  const [isAnalyzingDislocation, setIsAnalyzingDislocation] = useState(false);
  const [whStandardResult, setWhStandardResult] = useState(null);
  const [isAnalyzingWh, setIsAnalyzingWh] = useState(false);

  const [phaseIdentificationResult, setPhaseIdentificationResult] = useState(null);
  const [phaseTheoryOverlayLines, setPhaseTheoryOverlayLines] = useState([]);
  /** 피크 표·모달의 면족/{hkl} 표기 기준 (CIF 없을 때는 큐빅 가정과 동일) */
  const millerCrystalSystem = structureInfo?.cellParams?.system || 'cubic';
  const [textureResult, setTextureResult] = useState(null);
  const [qpaResult, setQpaResult] = useState(null);
  const [stressResult, setStressResult] = useState(null);
  const [rietveldInfo, setRietveldInfo] = useState(null);
  const [isPhaseLoading, setIsPhaseLoading] = useState(false);
  const [isTextureLoading, setIsTextureLoading] = useState(false);
  const [isQpaLoading, setIsQpaLoading] = useState(false);
  const [isStressLoading, setIsStressLoading] = useState(false);
  const [isRietveldLoading, setIsRietveldLoading] = useState(false);
  
  // 로딩 상태
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // RAW 데이터 표시 상태
  const [showRawData, setShowRawData] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);
  
  // 피크 추가 모달 상태
  const [showAddPeakModal, setShowAddPeakModal] = useState(false);
  const [newPeak, setNewPeak] = useState({ angle: '', intensity: '' });
  
  // 밀러지수 상세 정보 모달 상태
  const [showMillerIndicesModal, setShowMillerIndicesModal] = useState(false);
  const [selectedPeakIndex, setSelectedPeakIndex] = useState(null);
  
  // 밀러지수 편집 상태
  const [editingMillerIndices, setEditingMillerIndices] = useState({ h: '', k: '', l: '' });
  
  // 차트 클릭 메뉴 상태
  const [clickMenu, setClickMenu] = useState(null); // {x, y, angle, intensity, isExistingPeak}
  const chartRef = useRef(null);
  const dragStartRef = useRef(null); // 드래그 시작 위치 추적

  const [analysisProgressStep, setAnalysisProgressStep] = useState(0);
  
  // 설정
  const [settings, setSettings] = useState({
    // 피크 탐색 설정
    smoothingWindow: 3,
    smoothingPolyOrder: 2,
    minPeakHeight: 0.03,
    minPeakDistance: 0.3, // 도 단위 (0.3° = 일반 XRD 스텝 15포인트)
    peakDetectionMethod: 'localMaxima', // 기본값: 로컬 맥시마
    
    // 피크 피팅 설정
    peakType: 'gaussian', // 기본값: 가우시안
    backgroundType: 'polynomial', // 기본값: 다항식
    backgroundOrder: 2,
    enableFitting: true,
    
    // Scherrer / Williamson–Hall
    wavelength: 1.5406, // Cu Kα
    shapeFactor: 0.9,
    instrumentalFwhmDeg: 0,
    instrumentalCorrection: 'quadratic',
    
    // 시각화 설정
    showFittedCurve: true,
    showPeakMarkers: true,
    showIndexedPeaks: true
  });

  // 파일 업로드 핸들러
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setIsProcessing(true);
    setFileName(file.name);
    
    try {
      const response = await analysisClient.xrd.parse({ file });
      if (!response.success) throw new Error(response.error?.message || 'XRD 파일 파싱 실패');

      setXrdData(response.data.dataPoints);
      setStructureInfo(response.data.structureInfo);
      setMetadata(response.data.metadata || {});
      
      // 초기화
      setDetectedPeaks([]);
      setManualPeaks([]);
      setFittedResult(null);
      setIndexedPeaks([]);
      setCrystallinity(null);
      setCrystalliteStats(null);
      setWhStandardResult(null);
      setPhaseIdentificationResult(null);
      setPhaseTheoryOverlayLines([]);
      setTextureResult(null);
      setQpaResult(null);
      setStressResult(null);
      setRietveldInfo(null);
      
      setIsProcessing(false);
    } catch (error) {
      console.error('XRD 파일 파싱 오류:', error);
      alert('XRD 파일 파싱 중 오류가 발생했습니다: ' + error.message);
      setIsProcessing(false);
    }
    
    event.target.value = '';
  };

  const handleLoadSampleData = () => {
    setFileName('demo_powder.xy');
    setXrdData(generateDemoXrdPattern());
    setStructureInfo(null);
    setMetadata({});
    setDetectedPeaks([]);
    setManualPeaks([]);
    setFittedResult(null);
    setIndexedPeaks([]);
    setCrystallinity(null);
    setCrystalliteStats(null);
    setWhStandardResult(null);
    setPhaseIdentificationResult(null);
    setPhaseTheoryOverlayLines([]);
    setTextureResult(null);
    setQpaResult(null);
    setStressResult(null);
    setRietveldInfo(null);
  };

  const handleDigitizeComplete = (jsonData) => {
    if (!jsonData?.two_theta_values?.length) return;
    const dataPoints = jsonData.two_theta_values.map((angle, i) => ({
      angle,
      intensity: jsonData.intensities?.[i] ?? 0,
    }));
    setXrdData(dataPoints);
    setFileName('digitized_image.xy');
    setStructureInfo(null);
    setMetadata({});
    setDetectedPeaks([]);
    setManualPeaks([]);
    setFittedResult(null);
    setIndexedPeaks([]);
    setCrystallinity(null);
    setCrystalliteStats(null);
    setWhStandardResult(null);
    setPhaseIdentificationResult(null);
    setPhaseTheoryOverlayLines([]);
    setTextureResult(null);
    setQpaResult(null);
    setStressResult(null);
    setRietveldInfo(null);
    setMode('analyze');
  };

  // 분석 실행
  const runAnalysis = async () => {
    if (!xrdData || xrdData.length === 0) {
      alert('XRD 데이터가 없습니다.');
      return;
    }

    setIsAnalyzing(true);
    setWhStandardResult(null);
    setPhaseIdentificationResult(null);
    setPhaseTheoryOverlayLines([]);
    setTextureResult(null);
    setQpaResult(null);
    setStressResult(null);

    try {
      // 1. 피크 탐색 (analysisClient 경유)
      const peakResponse = await analysisClient.xrd.detectPeaks({
        dataPoints: xrdData,
        options: {
          method: settings.peakDetectionMethod,
          smoothingWindow: settings.smoothingWindow,
          minPeakHeightPercent: settings.minPeakHeight * 100,
          minPeakDistanceDeg: settings.minPeakDistance,
        },
      });

      const autoPeaks = peakResponse.success ? peakResponse.data.peaks : [];
      setDetectedPeaks(autoPeaks);
      
      // 탐지된 피크와 수동 추가 피크 합치기
      const allPeaks = [...autoPeaks, ...manualPeaks];
      allPeaks.sort((a, b) => a.angle - b.angle);

      // 2. 피크 피팅 (analysisClient 경유)
      let fittingResult = null;
      if (settings.enableFitting && allPeaks.length > 0) {
        const fitResponse = await analysisClient.xrd.fitPeaks({
          dataPoints: xrdData,
          peaks: allPeaks,
          options: {
            model: settings.peakType,
            backgroundType: settings.backgroundType,
            fitWindowDeg: 2.0,
          },
        });

        if (fitResponse.success) {
          fittingResult = { success: true, peaks: fitResponse.data.fittedPeaks, ...fitResponse.data };
          setFittedResult(fittingResult);
        }
      }

      // 3. FWHM 추출 (피팅 결과 또는 직접 추출)
      const peaksWithFWHM = allPeaks.map(peak => {
        let fwhm = peak.fwhm;
        
        if (fittingResult && fittingResult.success) {
          const fittedPeak = fittingResult.peaks.find(p => Math.abs(p.angle - peak.angle) < 0.1);
          if (fittedPeak) fwhm = fittedPeak.fwhm;
        }
        
        if (!fwhm) {
          fwhm = extractFWHMFromPeak(xrdData, peak.angle, peak.intensity);
        }

        return { ...peak, fwhm };
      });

      // 4. 결정립 크기 계산 (analysisClient 경유)
      const crystalliteSizeResponse = await analysisClient.xrd.calculateCrystalliteSizes({
        fittedPeaks: peaksWithFWHM.map(p => ({ id: p.id || `peak_${p.angle}`, angle: p.angle || 0, fwhm: p.fwhm || 0 })),
        wavelength: settings.wavelength,
        shapeFactor: settings.shapeFactor,
        instrumentalFwhmDeg: settings.instrumentalFwhmDeg,
        instrumentalCorrection: settings.instrumentalCorrection,
      });

      const crystalliteSizeMap = {};
      if (crystalliteSizeResponse.success) {
        crystalliteSizeResponse.data.crystalliteSizes.forEach(s => {
          crystalliteSizeMap[s.peakId] = s.sizeNm;
        });
        setCrystalliteStats({
          average: crystalliteSizeResponse.data.averageSizeNm,
          median: crystalliteSizeResponse.data.medianSizeNm,
          std: crystalliteSizeResponse.data.stdSizeNm,
        });
      }

      const peaksForIndexing = peaksWithFWHM.map(p => ({
        ...p,
        crystalliteSize: crystalliteSizeMap[p.id || `peak_${p.angle}`] || null,
        dSpacing: calculateDSpacing(p.angle, settings.wavelength),
      }));

      // 5. 밀러지수 인덱싱 (analysisClient 경유)
      let indexed = peaksForIndexing;
      const millerResponse = await analysisClient.xrd.indexMillerIndices({
        fittedPeaks: peaksForIndexing,
        structureInfo: structureInfo || null,
        wavelength: settings.wavelength,
        options: { maxHKL: 15, dSpacingTolerancePercent: 2.0 },
      });

      if (millerResponse.success && millerResponse.data.indexedPeaks) {
        indexed = millerResponse.data.indexedPeaks;
      } else {
        // 폴백: d-spacing 및 기본 후보 계산 (경량 유틸)
        const estimatedA = peaksForIndexing[0]?.dSpacing ? peaksForIndexing[0].dSpacing * Math.sqrt(3) : null;
        indexed = peaksForIndexing.map(peak => {
          let millerIndices = [];
          if (estimatedA && estimatedA > 0 && peak.dSpacing) {
            try {
              millerIndices = calculatePossibleMillerIndicesCubic(peak.dSpacing, estimatedA, 15, settings.wavelength, 0.1).slice(0, 5);
            } catch (e) { /* ignore */ }
          }
          return { ...peak, millerIndices };
        });
      }

      setIndexedPeaks(indexed);

      // 6. 결정화도 계산 (analysisClient 경유)
      const crystallinityResponse = await analysisClient.xrd.calculateCrystallinity({
        dataPoints: xrdData,
        fittedPeaks: fittingResult?.peaks || allPeaks,
        backgroundCurve: fittingResult?.backgroundCurve || [],
      });
      if (crystallinityResponse.success) {
        setCrystallinity(crystallinityResponse.data.crystallinity);
      }

    } catch (error) {
      console.error('분석 오류:', error);
      alert('분석 중 오류가 발생했습니다: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 차트 데이터: 탭0 피크 탐색용 (수동 피크 포함, 피팅·인덱싱 없음)
  const allPeaksForChart = [...detectedPeaks, ...manualPeaks];

  /** 탭0(피크 탐색): 탐지된 피크만 — 피팅·인덱싱 오버레이 없음 */
  const chartDataPeakSearchOnly = xrdData ? createXRDChartData(xrdData, {
    fittedCurve: null,
    peaks: allPeaksForChart.length > 0 ? allPeaksForChart : [],
    indexedPeaks: [],
  }) : null;

  // 차트 클릭 처리 함수 (공통 로직)
  const processChartClick = (clientX, clientY, target) => {
    if (!chartRef.current || !xrdData || xrdData.length === 0) return;
    
    const chartInstance = chartRef.current.getChart ? chartRef.current.getChart() : null;
    if (!chartInstance) return;
    
    const canvas = target || (chartInstance.canvas);
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;
    
    // Chart.js의 getRelativePosition 사용
    const nativeEvent = {
      clientX: clientX,
      clientY: clientY,
      target: canvas
    };
    const canvasPosition = Chart.helpers.getRelativePosition(nativeEvent, chartInstance);
    const xScale = chartInstance.scales.x;
    const clickedX = xScale.getValueForPixel(canvasPosition.x);
    
    // 클릭한 X 좌표에 가장 가까운 데이터 포인트 찾기
    let closestPoint = null;
    let minDistance = Infinity;
    
    for (let i = 0; i < xrdData.length; i++) {
      const point = xrdData[i];
      const distance = Math.abs(point.angle - clickedX);
      if (distance < minDistance) {
        minDistance = distance;
        closestPoint = {
          index: i,
          angle: point.angle,
          intensity: point.intensity
        };
      }
    }
    
    if (!closestPoint) return;
    
    // 해당 위치에 피크가 있는지 확인
    const allPeaks = [...detectedPeaks, ...manualPeaks, ...indexedPeaks];
    const existingPeak = allPeaks.find(p => Math.abs(p.angle - closestPoint.angle) < 0.5);
    
    // 메뉴 표시 위치 계산 (차트 컨테이너 기준)
    setClickMenu({
      x: canvasX,
      y: canvasY,
      angle: closestPoint.angle,
      intensity: closestPoint.intensity,
      isExistingPeak: !!existingPeak,
      existingPeak: existingPeak
    });
  };
  
  // 차트 클릭 핸들러 (Chart.js의 onClick 옵션 사용)
  const handleChartClick = (event, elements) => {
    if (!event || !event.native) return;
    
    // 드래그가 아닌 실제 클릭인지 확인
    if (dragStartRef.current) {
      const dragDistance = Math.sqrt(
        Math.pow(event.native.clientX - dragStartRef.current.x, 2) +
        Math.pow(event.native.clientY - dragStartRef.current.y, 2)
      );
      
      // 드래그 거리가 5px 이상이면 드래그로 간주하고 메뉴 표시하지 않음
      if (dragDistance > 5) {
        dragStartRef.current = null;
        return;
      }
      dragStartRef.current = null;
    }
    
    processChartClick(event.native.clientX, event.native.clientY, event.native.target);
  };
  
  // 메뉴 외부 클릭 시 닫기
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (clickMenu && !event.target.closest('.chart-click-menu')) {
        setClickMenu(null);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [clickMenu]);
  
  // 피크 선택 핸들러 (차트에서 클릭한 위치에 피크 추가)
  const handleSelectPeak = () => {
    if (!clickMenu) return;
    
    // 해당 각도에서 가장 가까운 데이터 포인트 찾기
    let closestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < xrdData.length; i++) {
      const diff = Math.abs(xrdData[i].angle - clickMenu.angle);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    
    const newPeakObj = {
      index: closestIdx,
      angle: clickMenu.angle,
      intensity: clickMenu.intensity,
      isManual: true
    };
    
    setManualPeaks([...manualPeaks, newPeakObj]);
    setClickMenu(null);
  };
  
  // 피크 삭제 핸들러 (차트에서 클릭한 위치의 피크 삭제)
  const handleDeletePeakFromChart = () => {
    if (!clickMenu || !clickMenu.existingPeak) return;
    
    const peak = clickMenu.existingPeak;
    
    // 수동 추가 피크인지 확인
    if (peak.isManual) {
      setManualPeaks(manualPeaks.filter((p, i) => {
        return Math.abs(p.angle - peak.angle) > 0.01;
      }));
    } else {
      // 자동 탐지된 피크는 detectedPeaks에서 제거
      setDetectedPeaks(detectedPeaks.filter((p, i) => {
        return Math.abs(p.angle - peak.angle) > 0.01;
      }));
    }
    
    // 분석 결과 초기화
    setFittedResult(null);
    setIndexedPeaks([]);
    setCrystallinity(null);
    setCrystalliteStats(null);
    
    setClickMenu(null);
  };

  const chartOptions = createXRDChartOptions({
    onClick: handleChartClick,
    showLegend: true,
    interactive: true,
    theoreticalOverlayLines: phaseTheoryOverlayLines,
  });

  // 피크 테이블 데이터
  const peakTableData = indexedPeaks && indexedPeaks.length > 0 
    ? createPeakTableData(indexedPeaks, millerCrystalSystem)
    : [];

  const latticeRefineExtra = useMemo(
    () => ({ c: structureInfo?.cellParams?.c }),
    [structureInfo?.cellParams?.c]
  );

  const impliedCubicSummary = useMemo(
    () => summarizeImpliedCubicAFromPeaks(indexedPeaks, millerCrystalSystem),
    [indexedPeaks, millerCrystalSystem]
  );
  
  // 피크 추가 핸들러
  const handleAddPeak = () => {
    const angle = parseFloat(newPeak.angle);
    const intensity = parseFloat(newPeak.intensity);
    
    if (isNaN(angle) || isNaN(intensity)) {
      alert('각도와 강도를 올바르게 입력해주세요.');
      return;
    }
    
    if (!xrdData || xrdData.length === 0) {
      alert('XRD 데이터가 없습니다.');
      return;
    }
    
    // 해당 각도에서 가장 가까운 데이터 포인트 찾기
    let closestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < xrdData.length; i++) {
      const diff = Math.abs(xrdData[i].angle - angle);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    
    const newPeakObj = {
      index: closestIdx,
      angle: angle,
      intensity: intensity,
      isManual: true // 수동 추가 표시
    };
    
    setManualPeaks([...manualPeaks, newPeakObj]);
    setNewPeak({ angle: '', intensity: '' });
    setShowAddPeakModal(false);
  };
  
  // 피크 삭제 핸들러
  const handleDeletePeak = (peakIndex) => {
    // indexedPeaks에서 해당 피크 찾기
    const peak = indexedPeaks[peakIndex];
    if (!peak) return;
    
    // 수동 추가 피크인지 확인
    if (peak.isManual) {
      setManualPeaks(manualPeaks.filter((p, i) => {
        // 각도로 매칭
        return Math.abs(p.angle - peak.angle) > 0.01;
      }));
    } else {
      // 자동 탐지된 피크는 detectedPeaks에서 제거
      setDetectedPeaks(detectedPeaks.filter((p, i) => {
        return Math.abs(p.angle - peak.angle) > 0.01;
      }));
    }
    
    // 분석 결과 초기화 (재분석 필요)
    setFittedResult(null);
    setIndexedPeaks([]);
    setCrystallinity(null);
    setCrystalliteStats(null);
  };

  // 밀러지수 수동 편집 핸들러
  const handleEditMillerIndices = (peakIndex, h, k, l) => {
    if (peakIndex === null || peakIndex === undefined) return;
    
    const updatedPeaks = [...indexedPeaks];
    const peak = updatedPeaks[peakIndex];
    
    if (!peak) return;
    
    // 새로운 밀러지수 생성
    const newMillerIndex = { h: parseInt(h) || 0, k: parseInt(k) || 0, l: parseInt(l) || 0 };
    
    // d-spacing 검증 (구조 정보가 있는 경우)
    if (structureInfo && structureInfo.cellParams) {
      try {
        const calculatedD = calculateDSpacingFromMillerIndices(
          newMillerIndex.h,
          newMillerIndex.k,
          newMillerIndex.l,
          structureInfo.cellParams
        );
        const observedD = peak.dSpacing;
        const diff = Math.abs(calculatedD - observedD);
        
        if (diff > 0.1) {
          alert(`경고: 계산된 d-spacing (${calculatedD.toFixed(3)} Å)이 관측값 (${observedD.toFixed(3)} Å)과 크게 다릅니다. 차이: ${diff.toFixed(3)} Å`);
        }
      } catch (error) {
        console.warn('d-spacing 검증 오류:', error);
      }
    }
    
    // 밀러지수 업데이트 (수동 편집 표시)
    updatedPeaks[peakIndex] = {
      ...peak,
      millerIndices: [newMillerIndex],
      isManualIndex: true, // 수동 편집 표시
      bestMatch: newMillerIndex
    };
    
    setIndexedPeaks(updatedPeaks);
    setShowMillerIndicesModal(false);
    setSelectedPeakIndex(null);
  };

  /** 표준 Williamson–Hall (β cos θ = Kλ/D + 4ε sin θ) — 밀러 지수 불필요 */
  const runStandardWilliamsonHall = async () => {
    const peaks = (indexedPeaks || []).filter(p => p && p.angle > 0 && p.fwhm > 0);
    if (peaks.length < 2) {
      alert('FWHM이 있는 피크가 2개 이상 필요합니다. 먼저 분석 실행으로 피크 피팅을 수행하세요.');
      return;
    }
    setIsAnalyzingWh(true);
    setWhStandardResult(null);
    try {
      const res = await analysisClient.xrd.williamsonHallFit({
        fittedPeaks: peaks.map(p => ({
          id: p.id || `peak_${p.angle}`,
          angle: p.angle,
          fwhm: p.fwhm,
        })),
        wavelength: settings.wavelength,
        shapeFactor: settings.shapeFactor,
        instrumentalFwhmDeg: settings.instrumentalFwhmDeg,
        instrumentalCorrection: settings.instrumentalCorrection,
      });
      if (!res.success) {
        alert(res.error?.message || 'Williamson–Hall 분석 실패');
        return;
      }
      setWhStandardResult(res.data);
    } catch (error) {
      console.error('Williamson–Hall 분석 오류:', error);
      alert('Williamson–Hall 분석 중 오류가 발생했습니다: ' + error.message);
    } finally {
      setIsAnalyzingWh(false);
    }
  };

  const handleIdentifyPhase = async () => {
    if (!indexedPeaks || indexedPeaks.length < 2) {
      alert('상 동정을 위해 피크가 최소 2개 필요합니다. 먼저 분석 실행을 수행하세요.');
      return;
    }
    setIsPhaseLoading(true);
    try {
      const res = await analysisClient.xrd.identifyPhaseCandidates({
        peaks: indexedPeaks,
        wavelength: settings.wavelength,
        options: {},
      });
      if (!res.success) {
        alert(res.error?.message || '상 동정 실패');
        return;
      }
      setPhaseIdentificationResult(res.data);
      setPhaseTheoryOverlayLines([]);
    } catch (err) {
      console.error(err);
      alert(err.message || '상 동정 오류');
    } finally {
      setIsPhaseLoading(false);
    }
  };

  const handleShowTopCandidateTheoryOverlay = () => {
    const c = phaseIdentificationResult?.candidates?.[0];
    if (!c || !xrdData?.length) {
      alert('상 동정 결과와 XRD 패턴이 필요합니다.');
      return;
    }
    const wl = settings.wavelength;
    const angles = xrdData.map((d) => d.angle);
    const tmin = Math.min(...angles) - 0.5;
    const tmax = Math.max(...angles) + 0.5;
    const peaks = listTheoreticalPeaksForCandidate(c, wl, tmin, tmax);
    setPhaseTheoryOverlayLines(peaks);
  };

  const handleClearPhaseTheoryOverlay = () => setPhaseTheoryOverlayLines([]);

  const handleComputeTexture = async (rows) => {
    setIsTextureLoading(true);
    try {
      const res = await analysisClient.xrd.computeTextureIndices({ rows });
      if (!res.success) {
        alert(res.error?.message || '배향 지수 계산 실패');
        setTextureResult(null);
        return;
      }
      setTextureResult(res.data);
    } catch (err) {
      alert(err.message || '오류');
    } finally {
      setIsTextureLoading(false);
    }
  };

  const handleComputeQpa = async (phases) => {
    setIsQpaLoading(true);
    try {
      const res = await analysisClient.xrd.estimateQPAPhaseFractions({ phases });
      if (!res.success) {
        alert(res.error?.message || 'QPA 계산 실패');
        setQpaResult(null);
        return;
      }
      setQpaResult(res.data);
    } catch (err) {
      alert(err.message || '오류');
    } finally {
      setIsQpaLoading(false);
    }
  };

  const handleComputeStress = async (points, elastic) => {
    setIsStressLoading(true);
    try {
      const res = await analysisClient.xrd.fitResidualStressSin2Psi({
        points,
        wavelength: settings.wavelength,
        elastic,
      });
      if (!res.success) {
        alert(res.error?.message || '잔류 응력 피팅 실패');
        setStressResult(null);
        return;
      }
      setStressResult(res.data);
    } catch (err) {
      alert(err.message || '오류');
    } finally {
      setIsStressLoading(false);
    }
  };

  const handleLoadRietveldGuidance = async () => {
    setIsRietveldLoading(true);
    try {
      const res = await analysisClient.xrd.getRietveldGuidance({});
      if (res.success) setRietveldInfo(res.data);
    } catch (err) {
      alert(err.message || '오류');
    } finally {
      setIsRietveldLoading(false);
    }
  };

  // 전위 밀도 분석 실행
  const runDislocationAnalysis = async () => {
    // 필수 데이터 검증
    if (!indexedPeaks || indexedPeaks.length < 3) {
      alert('전위 밀도 분석을 위해서는 최소 3개의 인덱싱된 피크가 필요합니다.');
      return;
    }

    // 밀러지수와 FWHM이 있는 피크만 필터링
    const validPeaks = indexedPeaks.filter(peak => 
      peak.millerIndices && 
      peak.millerIndices.length > 0 && 
      peak.fwhm && 
      peak.fwhm > 0
    );

    if (validPeaks.length < 3) {
      alert('전위 밀도 분석을 위해서는 최소 3개의 유효한 피크(밀러지수 및 FWHM 포함)가 필요합니다.');
      return;
    }

    // 재료 상수 확인
    if (!materialConstants) {
      alert('재료 물성 상수를 먼저 설정해주세요.');
      setShowMaterialConstantsModal(true);
      return;
    }

    setIsAnalyzingDislocation(true);

    try {
      const wavelength = materialConstants.wavelength || 0.15405;

      // mWH/mWA 통합 분석 (analysisClient 경유)
      const dislocationResponse = await analysisClient.xrd.analyzeDislocation({
        indexedPeaks: validPeaks,
        materialConstants,
        wavelength,
        method: 'mwh',
      });

      if (!dislocationResponse.success) {
        throw new Error(dislocationResponse.error?.message || '전위 밀도 분석 실패');
      }

      const { dislocationCharacter, ...mwhResult } = dislocationResponse.data;

      // mWA 분석 별도 실행 (xrdData, mWH의 q 전달 필요)
      const mwaResponse = await analysisClient.xrd.analyzeDislocation({
        indexedPeaks: validPeaks,
        materialConstants,
        wavelength,
        method: 'mwa',
        xrdData,
        qFromMwh: dislocationResponse.data?.q ?? null,
      });

      const interpretation = dislocationCharacter && typeof dislocationCharacter === 'object'
        ? dislocationCharacter
        : { dislocationCharacter: 'mixed', screwRatio: 0.5, edgeRatio: 0.5 };
      setDislocationResults({
        mwh: mwhResult,
        mwa: mwaResponse.success ? mwaResponse.data : { dislocationDensity: 0, rSquared: 0, plotData: { lnA: [] }, yOverL2Plot: { x: [] } },
        interpretation,
      });

      setShowDislocationAnalysis(true);
    } catch (error) {
      console.error('전위 밀도 분석 오류:', error);
      alert('전위 밀도 분석 중 오류가 발생했습니다: ' + error.message);
    } finally {
      setIsAnalyzingDislocation(false);
    }
  };

  // 재료 상수 저장 핸들러
  const handleSaveMaterialConstants = (constants) => {
    setMaterialConstants(constants);
  };

  const toolbarCallbacksRef = useRef({
    handleFileUpload: () => {},
    handleLoadSampleData: () => {},
    runAnalysis: () => {},
  });
  toolbarCallbacksRef.current = {
    handleFileUpload,
    handleLoadSampleData,
    runAnalysis,
  };

  useEffect(() => {
    // digitize 모드일 때는 XRDDigitizer가 툴바를 직접 제어
    if (mode !== 'analyze') return;

    setToolbarContent(
      <div className="xrd-analysis-settings-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* 입력 모드 토글 */}
        <div style={{ padding: '12px 16px 10px', borderBottom: '1px solid #eeeeee', flexShrink: 0 }}>
          <div style={{ display: 'flex', borderRadius: 8, overflow: 'hidden', border: '1px solid #e2e8f0' }}>
            {[{ key: 'analyze', label: '📊 파일 분석' }, { key: 'digitize', label: '🔬 이미지 디지타이저' }].map(({ key, label }) => (
              <button
                key={key}
                type="button"
                onClick={() => setMode(key)}
                style={{
                  flex: 1, padding: '7px 4px', fontSize: 11, fontWeight: mode === key ? 700 : 400,
                  background: mode === key ? '#2563eb' : '#fff',
                  color: mode === key ? '#fff' : '#475569',
                  border: 'none', cursor: 'pointer', transition: 'all 0.15s', lineHeight: 1.3,
                }}
              >{label}</button>
            ))}
          </div>
        </div>
        {/* 기존 분석 설정 패널 */}
        <XRDAnalysisSettingsPanel
          settings={settings}
          setSettings={setSettings}
          fileInputId="xrd-upload"
          fileName={fileName}
          isProcessing={isProcessing}
          onFileChange={(e) => toolbarCallbacksRef.current.handleFileUpload(e)}
          onLoadSampleData={() => toolbarCallbacksRef.current.handleLoadSampleData()}
          onRunAnalysis={() => toolbarCallbacksRef.current.runAnalysis()}
          isAnalyzing={isAnalyzing}
          hasData={!!xrdData && xrdData.length > 0}
        />
      </div>
    );
    setToolbarFooterContent(null);
    return () => {
      setToolbarContent(null);
      setToolbarFooterContent(null);
    };
  }, [
    mode,
    setMode,
    settings,
    fileName,
    isProcessing,
    isAnalyzing,
    xrdData,
    setToolbarContent,
    setToolbarFooterContent,
  ]);

  useEffect(() => {
    if (!xrdData) setAnalysisProgressStep(0);
  }, [xrdData]);

  const hasPeaks = detectedPeaks.length + manualPeaks.length > 0;
  const hasIndexed = indexedPeaks.length > 0;

  useEffect(() => {
    setAnalysisProgressStep((s) => {
      if (!hasPeaks && s > 0) return 0;
      if (hasPeaks && !hasIndexed && s > 1) return 1;
      return s;
    });
  }, [hasPeaks, hasIndexed]);

  const hasDislocation = Boolean(dislocationResults);
  const hasAdvanced = Boolean(
    phaseIdentificationResult || textureResult || qpaResult || stressResult || rietveldInfo
  );

  const xrdStepperDone = [hasPeaks && !!xrdData, hasIndexed, hasDislocation, hasAdvanced];

  const canJumpToXrdStepperIndex = (index) => {
    if (index === 0) return true;
    if (index === 1) return hasPeaks;
    if (index === 2) return hasIndexed;
    if (index === 3) return hasIndexed;
    return false;
  };

  const xrdStepperLastIndex = XRD_ANALYSIS_STEPS.length - 1;
  const xrdStepperNextDisabled =
    analysisProgressStep >= xrdStepperLastIndex ||
    !canJumpToXrdStepperIndex(analysisProgressStep + 1);

  const goToXrdAnalysisStep = (index) => {
    if (!canJumpToXrdStepperIndex(index)) return;
    setAnalysisProgressStep(index);
  };

  const xrdAnalysisStepperNav = (
    <nav className="data-input-stepper xrd-analyzer-stepper" aria-label="XRD 분석 단계">
      <div className="data-input-stepper-bar">
        <button
          type="button"
          className="btn-sqr btn-trans"
          aria-label="이전 단계"
          title="이전 단계"
          disabled={analysisProgressStep === 0}
          onClick={() => goToXrdAnalysisStep(Math.max(0, analysisProgressStep - 1))}
        >
          <span className="material-symbols-rounded data-input-stepper-arrow-icon">chevron_left</span>
        </button>
        <div className="data-input-stepper-track">
          <div className="data-input-stepper-steps-inner">
            {XRD_ANALYSIS_STEPS.map((s, index) => {
              const isActive = index === analysisProgressStep;
              const isDone = Boolean(xrdStepperDone[index]);
              const jumpOk = canJumpToXrdStepperIndex(index);
              return (
                <button
                  key={s.id}
                  type="button"
                  className={`data-input-stepper-item${isActive ? ' is-active' : ''}${isDone ? ' is-done' : ''}${!jumpOk ? ' is-disabled' : ''}`}
                  disabled={!jumpOk}
                  onClick={() => jumpOk && goToXrdAnalysisStep(index)}
                  aria-current={isActive ? 'step' : undefined}
                  title={s.id === 'peak-info' ? '피크 분석 (피크 정보)' : s.label}
                >
                  <span className="data-input-stepper-circle">{index + 1}</span>
                  <span className="data-input-stepper-label">{s.label}</span>
                </button>
              );
            })}
          </div>
        </div>
        <button
          type="button"
          className="btn-sqr btn-trans"
          aria-label="다음 단계"
          title="다음 단계"
          disabled={xrdStepperNextDisabled}
          onClick={() => goToXrdAnalysisStep(Math.min(xrdStepperLastIndex, analysisProgressStep + 1))}
        >
          <span className="material-symbols-rounded data-input-stepper-arrow-icon">chevron_right</span>
        </button>
      </div>
    </nav>
  );

  return (
    <div className="data-management">
      {mode === 'digitize' && (
        <XRDDigitizer
          onDigitizeComplete={handleDigitizeComplete}
          mode={mode}
          setMode={setMode}
          setToolbarContent={setToolbarContent}
        />
      )}

      {mode === 'analyze' && (
        <div className="xrd-analyzer-root">
          <ProcessingLocationBar style={{ marginBottom: 8 }} />
          {xrdAnalysisStepperNav}

          <div className="xrd-analyzer-main">
          {isProcessing && (
            <div className="processing-indicator">
              <div className="spinner"></div>
              <p>XRD 파일 처리 중...</p>
            </div>
          )}

          {!xrdData && !isProcessing && (
            <div className="xrd-empty-state" role="status" aria-live="polite">
              <div className="xrd-empty-state-hero" aria-hidden="true">
                <span className="material-symbols-rounded xrd-empty-state-hero-icon">show_chart</span>
              </div>
              <h2 className="xrd-empty-state-title">XRD 패턴을 준비 중입니다</h2>
              <p className="xrd-empty-state-lead">
                차트와 분석 결과는 데이터를 불러온 뒤 이곳에 표시됩니다.
              </p>
              <ul className="xrd-empty-state-hints">
                <li className="xrd-empty-state-hint">
                  <span className="material-symbols-rounded xrd-empty-state-hint-icon" aria-hidden="true">
                    dock_to_right
                  </span>
                  <span className="xrd-empty-state-hint-text">
                    <strong>오른쪽 도구 패널</strong>을 열고 &lsquo;파일 업로드&rsquo;에서 데이터를 선택하세요.
                  </span>
                </li>
                <li className="xrd-empty-state-hint">
                  <span className="material-symbols-rounded xrd-empty-state-hint-icon" aria-hidden="true">
                    upload_file
                  </span>
                  <span className="xrd-empty-state-hint-text">
                    파일을 <strong>드래그 앤 드롭</strong>하거나 영역을 눌러 탐색기에서 고를 수 있습니다.
                  </span>
                </li>
                <li className="xrd-empty-state-hint">
                  <span className="material-symbols-rounded xrd-empty-state-hint-icon" aria-hidden="true">
                    biotech
                  </span>
                  <span className="xrd-empty-state-hint-text">
                    바로 확인하려면 같은 패널의 <strong>샘플 데이터 로드</strong>를 눌러 보세요.
                  </span>
                </li>
              </ul>
            </div>
          )}

          {xrdData && (
            <>
          {analysisProgressStep === 0 && (
          <div className="card-col gap10">
            <h3>XRD 패턴</h3>
            {chartDataPeakSearchOnly && (
              <div
                className="chart-container"
                style={{
                  flex: '0 0 auto',
                  height: '360px',
                  minHeight: '360px',
                  maxHeight: '360px',
                }}
                onMouseDown={(e) => {
                  if (e.target.tagName === 'CANVAS' || e.target.closest('canvas')) {
                    dragStartRef.current = {
                      x: e.clientX,
                      y: e.clientY,
                      time: Date.now()
                    };
                  }
                }}
                onMouseUp={(e) => {
                  // 마우스 업 시 드래그가 아니었는지 확인
                  if (dragStartRef.current) {
                    const dragDistance = Math.sqrt(
                      Math.pow(e.clientX - dragStartRef.current.x, 2) +
                      Math.pow(e.clientY - dragStartRef.current.y, 2)
                    );
                    const dragTime = Date.now() - dragStartRef.current.time;
                    
                    // 드래그가 아니면 (5px 미만이고 300ms 이내) 클릭으로 간주하고 클릭 처리
                    if (dragDistance <= 5 && dragTime < 300) {
                      // 약간의 지연을 두어 zoom 플러그인의 드래그 처리와 충돌 방지
                      setTimeout(() => {
                        const canvas = e.target;
                        if (canvas && (canvas.tagName === 'CANVAS' || canvas.closest('canvas'))) {
                          processChartClick(e.clientX, e.clientY, canvas);
                        }
                      }, 50);
                    }
                    dragStartRef.current = null;
                  }
                }}
              >
                <Line 
                  ref={chartRef}
                  data={chartDataPeakSearchOnly} 
                  options={chartOptions}
                />
                
                {/* 클릭 메뉴 */}
                {clickMenu && (
                  <div
                    className="chart-click-menu"
                    style={{
                      position: 'absolute',
                      left: `${clickMenu.x}px`,
                      top: `${clickMenu.y}px`,
                      backgroundColor: 'white',
                      border: '1px solid var(--color-monotone-2)',
                      borderRadius: '8px',
                      padding: '10px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                      zIndex: 1000,
                      minWidth: '150px'
                    }}
                  >
                    <div style={{ marginBottom: '8px', fontSize: '12px', color: 'var(--color-text-2)' }}>
                      2θ: {clickMenu.angle.toFixed(2)}°<br />
                      Intensity: {clickMenu.intensity.toFixed(1)}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                      {!clickMenu.isExistingPeak ? (
                        <button
                          onClick={handleSelectPeak}
                          style={{
                            padding: '6px 12px',
                            background: 'var(--color-primary)',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '13px'
                          }}
                        >
                          피크 선택
                        </button>
                      ) : (
                        <button
                          onClick={handleDeletePeakFromChart}
                          style={{
                            padding: '6px 12px',
                            background: 'var(--color-failure)',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '13px'
                          }}
                        >
                          피크 삭제
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* RAW 데이터 토글 */}
            {xrdData && xrdData.length > 0 && (
              <div className="xrd-raw-data" style={{ marginTop: '20px' }}>
                <button
                  type="button"
                  className="raw-data-toggle"
                  onClick={() => setShowRawData(!showRawData)}
                >
                  {showRawData ? '▼ RAW 데이터 숨기기' : '▶ RAW 데이터 보기'} 
                  ({xrdData.length}개 데이터 포인트)
                </button>
                
                {showRawData && (
                  <div className="raw-data-table-container">
                    <table className="raw-data-table">
                      <thead>
                        <tr>
                          <th>2θ (°)</th>
                          <th>Intensity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {xrdData.map((point, index) => (
                          <tr key={index}>
                            <td>{point.angle !== undefined && point.angle !== null ? point.angle.toFixed(2) : 'N/A'}</td>
                            <td>{point.intensity !== undefined && point.intensity !== null ? point.intensity.toFixed(1) : 'N/A'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
            
            {/* 메타데이터 토글 */}
            {metadata && Object.keys(metadata).length > 0 && (
              <div className="xrd-raw-data" style={{ marginTop: '20px' }}>
                <button
                  type="button"
                  className="raw-data-toggle"
                  onClick={() => setShowMetadata(!showMetadata)}
                >
                  {showMetadata ? '▼ 메타데이터 숨기기' : '▶ 메타데이터 보기'} 
                  ({Object.keys(metadata).length}개 항목)
                </button>
                
                {showMetadata && (
                  <div className="raw-data-table-container">
                    <div className="xrd-metadata" style={{ padding: '15px' }}>
                      <div className="metadata-grid">
                        {Object.entries(metadata).map(([key, value]) => {
                          // 값이 객체나 배열인 경우 문자열로 변환
                          let displayValue = value;
                          if (value === null || value === undefined) {
                            displayValue = 'N/A';
                          } else if (typeof value === 'object') {
                            displayValue = JSON.stringify(value);
                          } else {
                            displayValue = String(value);
                          }
                          
                          // 키 이름을 읽기 쉽게 변환
                          const displayKey = key
                            .replace(/_/g, ' ')
                            .replace(/([A-Z])/g, ' $1')
                            .replace(/^./, str => str.toUpperCase())
                            .trim();
                          
                          return (
                            <div key={key} className="metadata-item">
                              <span className="metadata-key">{displayKey}</span>
                              <span className="metadata-value" title={displayValue}>
                                {displayValue.length > 50 ? displayValue.substring(0, 50) + '...' : displayValue}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          )}
          
          {analysisProgressStep === 1 && (detectedPeaks.length > 0 || manualPeaks.length > 0) && (
            <>
              {/* 피크 정보 */}
              <div className="card-col gap10" style={{ marginTop: '20px' }}>
                <div className="xrd-peak-info-header">
                  <h3 style={{ margin: 0 }}>피크 정보</h3>
                  <button
                    type="button"
                    className="btn-rtg btn-primary"
                    onClick={() => setShowAddPeakModal(true)}
                    style={{ padding: '8px 16px', fontSize: '14px', flexShrink: 0 }}
                  >
                    + 피크 추가
                  </button>
                </div>
                <p
                  style={{
                    margin: '0 0 10px 0',
                    fontSize: '12px',
                    lineHeight: 1.55,
                    color: 'var(--color-text-2)',
                  }}
                >
                  <strong>d-spacing</strong>는 Bragg 식으로 측정 각·파장에서 고정되지만,{' '}
                  <strong>어느 면족의 회절인지</strong>는 단위셀(격자상수·대칭)을 알아야 확정할 수 있습니다.
                  아래 <strong>격자 역산</strong> 열은 «1순위 (hkl)·면족 후보가 맞다면» 역으로 구한 격자상수입니다.
                  상용 <strong>PDF-4·ICDD</strong> 카드 매칭(Match! 등)은 라이선스가 필요하며, 장기적으로는 같은 패턴을{' '}
                  <strong>COD·오픈 CIF</strong> 등과 대조해 상·성분을 검증하는 흐름을 목표로 합니다.
                </p>
                {impliedCubicSummary && impliedCubicSummary.n >= 1 && (
                  <div
                    style={{
                      fontSize: '12px',
                      marginBottom: '10px',
                      padding: '10px 12px',
                      background: 'var(--color-sub-2)',
                      borderRadius: '8px',
                      color: 'var(--color-text-1)',
                    }}
                  >
                    <strong>큐빅 가정 검증용:</strong> 1순위 면족이 모두 맞다면 a는 피크마다 같은 값이어야 합니다.
                    {impliedCubicSummary.n === 1 ? (
                      <>
                        {' '}
                        현재 피크 1개: 역산 a ≈ <strong>{impliedCubicSummary.mean.toFixed(4)} Å</strong>.
                      </>
                    ) : (
                      <>
                        {' '}
                        피크 {impliedCubicSummary.n}개 역산 평균 <strong>{impliedCubicSummary.mean.toFixed(4)} Å</strong>
                        {impliedCubicSummary.std > 1e-6 && (
                          <> (표준편차 ±{impliedCubicSummary.std.toFixed(4)} Å)</>
                        )}
                        . 편차가 크면 다상·비큐빅·오인덱싱을 의심하세요.
                      </>
                    )}
                  </div>
                )}
                <div className="box-col pd0">
                  <table style={{ width: '100%' }}>
                    <thead>
                      <tr style={{ backgroundColor: 'var(--color-sub-2)' }}>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>#</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>2θ (°)</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>Intensity</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>FWHM (°)</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>d-spacing (Å)</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>면족 · (hkl)</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>격자 역산 (가정)</th>
                        <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>결정자 Scherrer (nm)</th>
                        <th style={{ padding: '10px', textAlign: 'center', border: '1px solid var(--color-monotone-2)' }}>작업</th>
                      </tr>
                    </thead>
                    <tbody>
                      {peakTableData.map((row, index) => {
                        const peak = indexedPeaks[index];
                        const isManual = peak && peak.isManual;
                        return (
                          <tr key={row.id} style={{ borderBottom: '1px solid var(--color-monotone-2)' }}>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                              {row.id}
                              {isManual && <span style={{ marginLeft: '5px', fontSize: '10px', color: 'var(--color-primary)' }}>(수동)</span>}
                            </td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>{row.angle}</td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>{row.intensity}</td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>{row.fwhm}</td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>{row.dSpacing}</td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                              {peak && peak.millerIndices && peak.millerIndices.length > 0 ? (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap' }}>
                                  {peak.millerIndices.slice(0, 2).map((hkl, idx) => {
                                    const label = formatMillerIndicesForDisplay(hkl.h, hkl.k, hkl.l, millerCrystalSystem);
                                    return (
                                      <span
                                        key={`${label}-${idx}`}
                                        style={{
                                          padding: '2px 6px',
                                          backgroundColor: idx === 0 ? 'var(--color-primary)' : 'var(--color-sub-2)',
                                          color: idx === 0 ? 'white' : 'var(--color-text-1)',
                                          borderRadius: '4px',
                                          fontSize: '11px',
                                          fontWeight: idx === 0 ? 'bold' : 'normal'
                                        }}
                                      >
                                        {label}
                                      </span>
                                    );
                                  })}
                                  {peak.millerIndices.length > 2 && (
                                    <button
                                      onClick={() => {
                                        setSelectedPeakIndex(index);
                                        setShowMillerIndicesModal(true);
                                      }}
                                      style={{
                                        padding: '2px 6px',
                                        fontSize: '10px',
                                        background: 'transparent',
                                        color: 'var(--color-primary)',
                                        border: '1px solid var(--color-primary)',
                                        borderRadius: '4px',
                                        cursor: 'pointer'
                                      }}
                                    >
                                      +{peak.millerIndices.length - 2}
                                    </button>
                                  )}
                                  {peak.millerIndices.length <= 2 && (
                                    <button
                                      onClick={() => {
                                        setSelectedPeakIndex(index);
                                        setShowMillerIndicesModal(true);
                                      }}
                                      style={{
                                        padding: '2px 6px',
                                        fontSize: '10px',
                                        background: 'transparent',
                                        color: 'var(--color-primary)',
                                        border: '1px solid var(--color-primary)',
                                        borderRadius: '4px',
                                        cursor: 'pointer',
                                        marginLeft: '5px'
                                      }}
                                    >
                                      상세
                                    </button>
                                  )}
                                </div>
                              ) : (
                                <span style={{ color: 'var(--color-text-2)', fontSize: '12px' }}>N/A</span>
                              )}
                            </td>
                            <td
                              style={{ padding: '10px', border: '1px solid var(--color-monotone-2)', fontSize: '12px' }}
                              title={
                                peak && peak.millerIndices?.[0] && peak.dSpacing != null
                                  ? impliedLatticeFromPeak(
                                      Number(peak.dSpacing),
                                      peak.millerIndices[0].h,
                                      peak.millerIndices[0].k,
                                      peak.millerIndices[0].l,
                                      millerCrystalSystem,
                                      latticeRefineExtra
                                    )?.detail
                                  : undefined
                              }
                            >
                              {(() => {
                                const best = peak?.millerIndices?.[0];
                                if (!best || peak?.dSpacing == null) return '—';
                                const hint = impliedLatticeFromPeak(
                                  Number(peak.dSpacing),
                                  best.h,
                                  best.k,
                                  best.l,
                                  millerCrystalSystem,
                                  latticeRefineExtra
                                );
                                return hint?.text ?? '—';
                              })()}
                            </td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>{row.crystalliteSize}</td>
                            <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)', textAlign: 'center' }}>
                              <button
                                onClick={() => handleDeletePeak(index)}
                                style={{
                                  padding: '4px 8px',
                                  fontSize: '12px',
                                  background: 'var(--color-failure)',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: '4px',
                                  cursor: 'pointer'
                                }}
                              >
                                삭제
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 통계 · Scherrer 요약 (표준 W–H는 전위밀도 탭) */}
              <div className="card-col gap10" style={{ marginTop: '20px' }}>
                <h3>분석 결과</h3>
                <div className="grid">
                  <div className="info-item">
                    <span className="info-label">탐지된 피크 수:</span>
                    <span className="info-value">{detectedPeaks.length}</span>
                  </div>
                  {manualPeaks.length > 0 && (
                    <div className="info-item">
                      <span className="info-label">수동 추가 피크 수:</span>
                      <span className="info-value">{manualPeaks.length}</span>
                    </div>
                  )}
                  <div className="info-item">
                    <span className="info-label">총 피크 수:</span>
                    <span className="info-value">{detectedPeaks.length + manualPeaks.length}</span>
                  </div>
                  {crystallinity !== null && (
                    <div className="info-item">
                      <span className="info-label">결정화도:</span>
                      <span className="info-value">{(crystallinity * 100).toFixed(2)}%</span>
                    </div>
                  )}
                  {crystalliteStats && (
                    <>
                      <div className="info-item">
                        <span className="info-label">평균 결정자 크기 (Scherrer):</span>
                        <span className="info-value">{crystalliteStats.average.toFixed(2)} nm</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">중간값 결정자 크기 (Scherrer):</span>
                        <span className="info-value">{crystalliteStats.median.toFixed(2)} nm</span>
                      </div>
                    </>
                  )}
                </div>

                <p style={{ fontSize: '12px', color: 'var(--color-text-2)', marginTop: '8px', maxWidth: '720px' }}>
                  Scherrer 등으로 추정한 결정자(crystallite)는 회절에 코히런트한 영역이며, 광학현미경·SEM에서 보이는 결정립(grain)과 반드시 같지 않습니다. 표준 Williamson–Hall(UDM) 분석은 <strong>전위밀도 분석</strong> 탭에서 실행할 수 있습니다.
                </p>
              </div>
            </>
          )}

            {analysisProgressStep === 2 && hasIndexed && (
            <>
              <div className="card-col gap10" style={{ marginTop: '20px' }}>
                <h3 style={{ margin: '0 0 8px' }}>표준 Williamson–Hall (UDM)</h3>
                <p style={{ fontSize: '13px', color: 'var(--color-text-2)', margin: '0 0 12px', maxWidth: '720px' }}>
                  균일 변형 모델: β cos θ 대 sin θ 선형 회귀로 미세 변형 ε와 결정자 크기 D를 추정합니다. FWHM이 유효한 인덱싱 피크가 2개 이상이어야 합니다.
                </p>
                <div className="btn-group" style={{ flexWrap: 'wrap', gap: '8px' }}>
                  <button
                    type="button"
                    className="btn-rtg btn-primary"
                    onClick={runStandardWilliamsonHall}
                    disabled={isAnalyzingWh || indexedPeaks.filter(p => p && p.fwhm > 0).length < 2}
                    style={{ padding: '8px 16px', fontSize: '14px' }}
                  >
                    {isAnalyzingWh ? '분석 중...' : '표준 Williamson–Hall 분석'}
                  </button>
                </div>

                {whStandardResult && whStandardResult.regression && (
                  <div className="grid" style={{ marginTop: '16px' }}>
                    <div className="info-item">
                      <span className="info-label">미세 변형률 ε (UDM):</span>
                      <span className="info-value">{whStandardResult.microstrain.toExponential(4)}</span>
                    </div>
                    {whStandardResult.crystalliteSizeNm != null ? (
                      <div className="info-item">
                        <span className="info-label">결정자 크기 D (W–H 절편):</span>
                        <span className="info-value">{whStandardResult.crystalliteSizeNm.toFixed(2)} nm</span>
                      </div>
                    ) : (
                      <div className="info-item" style={{ gridColumn: '1 / -1' }}>
                        <span className="info-label">결정자 크기 D (W–H 절편):</span>
                        <span className="info-value" style={{ color: 'var(--color-text-2)' }}>
                          산출 불가 (절편 ≤ 0 — 피크 폭·기기 보정을 확인하세요)
                        </span>
                      </div>
                    )}
                    <div className="info-item">
                      <span className="info-label">W–H R²:</span>
                      <span className="info-value">{whStandardResult.regression.rSquared.toFixed(4)}</span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">피팅 점 수:</span>
                      <span className="info-value">{whStandardResult.regression.pointCount}</span>
                    </div>
                  </div>
                )}

                {whStandardResult && createStandardWilliamsonHallPlotData(whStandardResult) && (
                  <div className="card-col gap10" style={{ marginTop: '16px' }}>
                    <h4 style={{ margin: 0 }}>표준 Williamson–Hall 플롯 (β cos θ vs sin θ)</h4>
                    <div className="chart-container" style={{ height: '380px', minHeight: '380px' }}>
                      <Line
                        data={createStandardWilliamsonHallPlotData(whStandardResult)}
                        options={createStandardWilliamsonHallPlotOptions()}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="card-col gap10 xrd-dislocation-control-card" style={{ marginTop: '20px' }}>
                <h3 style={{ margin: '0 0 8px' }}>전위 밀도 분석</h3>
                <p style={{ fontSize: '13px', color: 'var(--color-text-2)', margin: '0 0 12px', maxWidth: '720px' }}>
                  인덱싱된 피크(밀러지수·FWHM)를 사용해 mWH/mWA 전위 밀도를 계산합니다. 최소 3개의 유효 피크가 필요합니다.
                </p>
                <div className="btn-group xrd-dislocation-actions">
                  <button
                    className="btn-rtg btn-primary"
                    onClick={() => setShowMaterialConstantsModal(true)}
                    style={{ padding: '8px 16px', fontSize: '14px' }}
                  >
                    재료 상수 설정
                  </button>
                  <button
                    className="btn-rtg btn-primary"
                    onClick={runDislocationAnalysis}
                    disabled={isAnalyzingDislocation || indexedPeaks.length < 3}
                    style={{ padding: '8px 16px', fontSize: '14px' }}
                  >
                    {isAnalyzingDislocation ? '분석 중...' : '전위 밀도 분석'}
                  </button>
                  {materialConstants && (
                    <div style={{
                      padding: '8px 12px',
                      backgroundColor: 'var(--color-sub-2)',
                      borderRadius: '4px',
                      fontSize: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '5px'
                    }}>
                      <span>재료:</span>
                      <span style={{ fontWeight: 'bold' }}>
                        {materialConstants.structure.toUpperCase()}{' '}
                        (a={materialConstants.latticeConstant} Å)
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {showDislocationAnalysis && dislocationResults && (
                <div className="card-col gap10 xrd-dislocation-results-card" style={{ marginTop: '12px' }}>
                  <div className="xrd-dislocation-result-header">
                    <h3 style={{ margin: 0 }}>전위 밀도 분석 결과</h3>
                    <button
                      type="button"
                      onClick={() => setShowDislocationAnalysis(false)}
                      style={{
                        padding: '6px 12px',
                        background: 'var(--color-monotone-2)',
                        color: 'var(--color-text-1)',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      닫기
                    </button>
                  </div>

                  <div className="grid xrd-dislocation-metrics">
                    <div className="info-item">
                      <span className="info-label">전위 밀도 (ρ):</span>
                      <span className="info-value">
                        {dislocationResults.mwa.dislocationDensity.toExponential(2)} m⁻²
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">q 파라미터:</span>
                      <span className="info-value">
                        {Number.isFinite(dislocationResults.mwh.q) ? dislocationResults.mwh.q.toFixed(3) : '산출 불가'}
                      </span>
                    </div>
                    {dislocationResults.mwh.contrastPreset && (
                      <div className="info-item">
                        <span className="info-label">q 기준 범위:</span>
                        <span className="info-value">
                          {dislocationResults.mwh.contrastPreset.min.toFixed(3)} ~ {dislocationResults.mwh.contrastPreset.max.toFixed(3)}
                        </span>
                      </div>
                    )}
                    {dislocationResults.mwh.phi != null && (
                      <div className="info-item">
                        <span className="info-label">φ 파라미터:</span>
                        <span className="info-value">{dislocationResults.mwh.phi.toExponential(3)}</span>
                      </div>
                    )}
                    <div className="info-item">
                      <span className="info-label">해석 상태:</span>
                      <span className="info-value">
                        {dislocationResults.mwh.qualityFlags?.qOutOfRange || dislocationResults.mwh.qualityFlags?.lowRSquared
                          ? '참고값'
                          : '정상'}
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">전위 특성:</span>
                      <span className="info-value">
                        {dislocationResults.interpretation.dislocationCharacter === 'screw' ? '나선형' :
                         dislocationResults.interpretation.dislocationCharacter === 'edge' ? '칼날형' : '혼합형'}
                      </span>
                    </div>
                    {dislocationResults.mwh.D && (
                      <div className="info-item">
                        <span className="info-label">결정 크기 (D):</span>
                        <span className="info-value">{dislocationResults.mwh.D.toFixed(2)} nm</span>
                      </div>
                    )}
                    <div className="info-item">
                      <span className="info-label">나선형 비율:</span>
                      <span className="info-value">
                        {(dislocationResults.interpretation.screwRatio * 100).toFixed(1)}%
                        {(dislocationResults.mwh.qualityFlags?.qOutOfRange || dislocationResults.mwh.qualityFlags?.lowRSquared) ? ' (참고)' : ''}
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">칼날형 비율:</span>
                      <span className="info-value">
                        {(dislocationResults.interpretation.edgeRatio * 100).toFixed(1)}%
                        {(dislocationResults.mwh.qualityFlags?.qOutOfRange || dislocationResults.mwh.qualityFlags?.lowRSquared) ? ' (참고)' : ''}
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">mWH R²:</span>
                      <span className="info-value">{dislocationResults.mwh.rSquared.toFixed(4)}</span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">mWA R²:</span>
                      <span className="info-value">{dislocationResults.mwa.rSquared.toFixed(4)}</span>
                    </div>
                  </div>

                  {[
                    ...(dislocationResults.mwh.warnings || []),
                    ...(dislocationResults.interpretation.warnings || []),
                  ].length > 0 && (
                    <div className="xrd-dislocation-warning">
                      <strong>mWH 해석 주의:</strong>
                      <ul style={{ margin: '8px 0 0', paddingLeft: '18px' }}>
                        {[...new Set([
                          ...(dislocationResults.mwh.warnings || []),
                          ...(dislocationResults.interpretation.warnings || []),
                        ])].map((warning) => (
                          <li key={warning}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="xrd-dislocation-chart-grid">
                  {dislocationResults.mwh.plotData && dislocationResults.mwh.plotData.x.length > 0 && (
                    <div className="card-col gap10 xrd-dislocation-chart-card">
                      <h4>Modified Williamson-Hall (mWH)</h4>
                      <div className="chart-container xrd-dislocation-chart">
                        <Line
                          data={createMWHPlotData(dislocationResults.mwh)}
                          options={createMWHPlotOptions()}
                        />
                      </div>
                    </div>
                  )}

                  {dislocationResults.mwa.plotData && dislocationResults.mwa.plotData.lnA.length > 0 && (
                    <div className="card-col gap10 xrd-dislocation-chart-card">
                      <h4>mWA - lnA(L) vs K²C̄</h4>
                      <div className="chart-container xrd-dislocation-chart">
                        <Line
                          data={createMWAPlotData(dislocationResults.mwa)}
                          options={createMWAPlotOptions()}
                        />
                      </div>
                    </div>
                  )}

                  {dislocationResults.mwa.yOverL2Plot && dislocationResults.mwa.yOverL2Plot.x.length > 0 && (
                    <div className="card-col gap10 xrd-dislocation-chart-card">
                      <h4>mWA - Y/L² vs lnL</h4>
                      <div className="chart-container xrd-dislocation-chart">
                        <Line
                          data={createMWAYOverL2PlotData(dislocationResults.mwa)}
                          options={createMWAYOverL2PlotOptions()}
                        />
                      </div>
                    </div>
                  )}
                  </div>
                </div>
              )}
            </>
            )}

            {analysisProgressStep === 3 && hasIndexed && (
              <XrdAdvancedAnalysisSection
                indexedPeaks={indexedPeaks}
                wavelength={settings.wavelength}
                fileName={fileName}
                disabled={!xrdData}
                phaseIdentificationResult={phaseIdentificationResult}
                theoryOverlayCount={phaseTheoryOverlayLines.length}
                onShowTheoryOverlay={handleShowTopCandidateTheoryOverlay}
                onClearTheoryOverlay={handleClearPhaseTheoryOverlay}
                textureResult={textureResult}
                qpaResult={qpaResult}
                stressResult={stressResult}
                rietveldInfo={rietveldInfo}
                isPhaseLoading={isPhaseLoading}
                isTextureLoading={isTextureLoading}
                isQpaLoading={isQpaLoading}
                isStressLoading={isStressLoading}
                isRietveldLoading={isRietveldLoading}
                onIdentifyPhase={handleIdentifyPhase}
                onComputeTexture={handleComputeTexture}
                onComputeQpa={handleComputeQpa}
                onComputeStress={handleComputeStress}
                onLoadRietveldInfo={handleLoadRietveldGuidance}
              />
            )}

              {/* 밀러지수 상세 정보 모달 */}
              {showMillerIndicesModal && selectedPeakIndex !== null && (
                <div style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: 'rgba(0, 0, 0, 0.5)',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  zIndex: 1000
                }}>
                  <div style={{
                    backgroundColor: 'white',
                    padding: '30px',
                    borderRadius: '8px',
                    minWidth: '500px',
                    maxWidth: '700px',
                    maxHeight: '80vh',
                    overflowY: 'auto'
                  }}>
                    <h3 style={{ marginTop: 0, marginBottom: '20px' }}>
                      밀러지수 후보 목록 - 피크 #{selectedPeakIndex + 1}
                    </h3>
                    {indexedPeaks[selectedPeakIndex] && (
                      <>
                        <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: 'var(--color-sub-2)', borderRadius: '4px' }}>
                          <div><strong>2θ:</strong> {indexedPeaks[selectedPeakIndex].angle?.toFixed(3)}°</div>
                          <div><strong>Intensity:</strong> {indexedPeaks[selectedPeakIndex].intensity?.toFixed(2)}</div>
                          <div><strong>d-spacing:</strong> {indexedPeaks[selectedPeakIndex].dSpacing?.toFixed(3)} Å</div>
                          {indexedPeaks[selectedPeakIndex].confidence !== undefined && (
                            <div><strong>신뢰도:</strong> {(indexedPeaks[selectedPeakIndex].confidence * 100).toFixed(1)}%</div>
                          )}
                        </div>
                        {indexedPeaks[selectedPeakIndex].millerIndices && indexedPeaks[selectedPeakIndex].millerIndices.length > 0 ? (
                          <div>
                            <h4 style={{ marginBottom: '15px' }}>면족 후보 (동일 d의 순열은 하나로 묶음):</h4>
                            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '20px' }}>
                              <thead>
                                <tr style={{ backgroundColor: 'var(--color-sub-2)' }}>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>순위</th>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>면족 / (hkl)</th>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>격자 역산 (가정)</th>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>d-spacing (Å)</th>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>2θ (°)</th>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>매칭 점수</th>
                                  <th style={{ padding: '10px', textAlign: 'left', border: '1px solid var(--color-monotone-2)' }}>각도 차이</th>
                                  <th style={{ padding: '10px', textAlign: 'center', border: '1px solid var(--color-monotone-2)' }}>선택</th>
                                </tr>
                              </thead>
                              <tbody>
                                {indexedPeaks[selectedPeakIndex].millerIndices.map((hkl, idx) => (
                                  <tr key={idx} style={{ borderBottom: '1px solid var(--color-monotone-2)' }}>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                                      {idx === 0 ? (
                                        <span style={{ color: 'var(--color-primary)', fontWeight: 'bold' }}>★ {idx + 1}</span>
                                      ) : (
                                        idx + 1
                                      )}
                                    </td>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                                      <span style={{
                                        padding: '4px 8px',
                                        backgroundColor: idx === 0 ? 'var(--color-primary)' : 'var(--color-sub-2)',
                                        color: idx === 0 ? 'white' : 'var(--color-text-1)',
                                        borderRadius: '4px',
                                        fontWeight: idx === 0 ? 'bold' : 'normal'
                                      }}>
                                        {formatMillerIndicesForDisplay(hkl.h, hkl.k, hkl.l, millerCrystalSystem)}
                                      </span>
                                    </td>
                                    <td
                                      style={{ padding: '10px', border: '1px solid var(--color-monotone-2)', fontSize: '12px' }}
                                      title={
                                        indexedPeaks[selectedPeakIndex].dSpacing != null
                                          ? impliedLatticeFromPeak(
                                              Number(indexedPeaks[selectedPeakIndex].dSpacing),
                                              hkl.h,
                                              hkl.k,
                                              hkl.l,
                                              millerCrystalSystem,
                                              latticeRefineExtra
                                            )?.detail
                                          : undefined
                                      }
                                    >
                                      {(() => {
                                        const pd = indexedPeaks[selectedPeakIndex].dSpacing;
                                        if (pd == null) return '—';
                                        return impliedLatticeFromPeak(
                                          Number(pd),
                                          hkl.h,
                                          hkl.k,
                                          hkl.l,
                                          millerCrystalSystem,
                                          latticeRefineExtra
                                        )?.text ?? '—';
                                      })()}
                                    </td>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                                      {hkl.d?.toFixed(3) || 'N/A'}
                                    </td>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                                      {hkl.angle?.toFixed(3) || 'N/A'}
                                    </td>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                                      {hkl.matchScore !== undefined ? (hkl.matchScore * 100).toFixed(1) + '%' : 'N/A'}
                                    </td>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)' }}>
                                      {hkl.angleDiff !== undefined ? hkl.angleDiff.toFixed(3) + '°' : 'N/A'}
                                    </td>
                                    <td style={{ padding: '10px', border: '1px solid var(--color-monotone-2)', textAlign: 'center' }}>
                                      <button
                                        onClick={() => handleEditMillerIndices(selectedPeakIndex, hkl.h, hkl.k, hkl.l)}
                                        style={{
                                          padding: '4px 8px',
                                          fontSize: '12px',
                                          background: 'var(--color-primary)',
                                          color: 'white',
                                          border: 'none',
                                          borderRadius: '4px',
                                          cursor: 'pointer'
                                        }}
                                      >
                                        선택
                                      </button>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            
                            {/* 수동 입력 섹션 */}
                            <div style={{ marginTop: '20px', padding: '15px', backgroundColor: 'var(--color-sub-2)', borderRadius: '4px' }}>
                              <h4 style={{ marginTop: 0, marginBottom: '15px' }}>수동 입력:</h4>
                              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                  h: <input
                                    type="number"
                                    value={editingMillerIndices.h}
                                    onChange={(e) => setEditingMillerIndices({ ...editingMillerIndices, h: e.target.value })}
                                    style={{
                                      width: '60px',
                                      padding: '5px',
                                      borderRadius: '4px',
                                      border: '1px solid var(--color-monotone-2)'
                                    }}
                                  />
                                </label>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                  k: <input
                                    type="number"
                                    value={editingMillerIndices.k}
                                    onChange={(e) => setEditingMillerIndices({ ...editingMillerIndices, k: e.target.value })}
                                    style={{
                                      width: '60px',
                                      padding: '5px',
                                      borderRadius: '4px',
                                      border: '1px solid var(--color-monotone-2)'
                                    }}
                                  />
                                </label>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                  l: <input
                                    type="number"
                                    value={editingMillerIndices.l}
                                    onChange={(e) => setEditingMillerIndices({ ...editingMillerIndices, l: e.target.value })}
                                    style={{
                                      width: '60px',
                                      padding: '5px',
                                      borderRadius: '4px',
                                      border: '1px solid var(--color-monotone-2)'
                                    }}
                                  />
                                </label>
                                <button
                                  onClick={() => {
                                    if (editingMillerIndices.h !== '' || editingMillerIndices.k !== '' || editingMillerIndices.l !== '') {
                                      handleEditMillerIndices(
                                        selectedPeakIndex,
                                        editingMillerIndices.h,
                                        editingMillerIndices.k,
                                        editingMillerIndices.l
                                      );
                                      setEditingMillerIndices({ h: '', k: '', l: '' });
                                    }
                                  }}
                                  className="btn-primary"
                                  style={{ padding: '6px 12px' }}
                                >
                                  적용
                                </button>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--color-text-2)' }}>
                            <div style={{ marginBottom: '15px' }}>밀러지수 후보가 없습니다.</div>
                            {/* 수동 입력 섹션 */}
                            <div style={{ padding: '15px', backgroundColor: 'var(--color-sub-2)', borderRadius: '4px' }}>
                              <h4 style={{ marginTop: 0, marginBottom: '15px' }}>수동 입력:</h4>
                              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                  h: <input
                                    type="number"
                                    value={editingMillerIndices.h}
                                    onChange={(e) => setEditingMillerIndices({ ...editingMillerIndices, h: e.target.value })}
                                    style={{
                                      width: '60px',
                                      padding: '5px',
                                      borderRadius: '4px',
                                      border: '1px solid var(--color-monotone-2)'
                                    }}
                                  />
                                </label>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                  k: <input
                                    type="number"
                                    value={editingMillerIndices.k}
                                    onChange={(e) => setEditingMillerIndices({ ...editingMillerIndices, k: e.target.value })}
                                    style={{
                                      width: '60px',
                                      padding: '5px',
                                      borderRadius: '4px',
                                      border: '1px solid var(--color-monotone-2)'
                                    }}
                                  />
                                </label>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                  l: <input
                                    type="number"
                                    value={editingMillerIndices.l}
                                    onChange={(e) => setEditingMillerIndices({ ...editingMillerIndices, l: e.target.value })}
                                    style={{
                                      width: '60px',
                                      padding: '5px',
                                      borderRadius: '4px',
                                      border: '1px solid var(--color-monotone-2)'
                                    }}
                                  />
                                </label>
                                <button
                                  onClick={() => {
                                    if (editingMillerIndices.h !== '' || editingMillerIndices.k !== '' || editingMillerIndices.l !== '') {
                                      handleEditMillerIndices(
                                        selectedPeakIndex,
                                        editingMillerIndices.h,
                                        editingMillerIndices.k,
                                        editingMillerIndices.l
                                      );
                                      setEditingMillerIndices({ h: '', k: '', l: '' });
                                    }
                                  }}
                                  className="btn-primary"
                                  style={{ padding: '6px 12px' }}
                                >
                                  적용
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                    <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '20px' }}>
                      <button
                        onClick={() => {
                          setShowMillerIndicesModal(false);
                          setSelectedPeakIndex(null);
                          setEditingMillerIndices({ h: '', k: '', l: '' });
                        }}
                        className="btn-primary"
                        style={{ padding: '8px 16px' }}
                      >
                        닫기
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* 피크 추가 모달 */}
              {showAddPeakModal && (
                <div style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: 'rgba(0, 0, 0, 0.5)',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  zIndex: 1000
                }}>
                  <div style={{
                    backgroundColor: 'white',
                    padding: '30px',
                    borderRadius: '8px',
                    minWidth: '400px',
                    maxWidth: '500px'
                  }}>
                    <h3 style={{ marginTop: 0, marginBottom: '20px' }}>피크 추가</h3>
                    <div style={{ marginBottom: '15px' }}>
                      <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        2θ 각도 (°)
                      </label>
                      <input
                        type="number"
                        step="0.01"
                        value={newPeak.angle}
                        onChange={(e) => setNewPeak({ ...newPeak, angle: e.target.value })}
                        style={{
                          width: '100%',
                          padding: '8px',
                          borderRadius: '4px',
                          border: '1px solid var(--color-monotone-2)'
                        }}
                        placeholder="예: 43.5"
                      />
                    </div>
                    <div style={{ marginBottom: '20px' }}>
                      <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Intensity
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={newPeak.intensity}
                        onChange={(e) => setNewPeak({ ...newPeak, intensity: e.target.value })}
                        style={{
                          width: '100%',
                          padding: '8px',
                          borderRadius: '4px',
                          border: '1px solid var(--color-monotone-2)'
                        }}
                        placeholder="예: 1000"
                      />
                    </div>
                    <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => {
                          setShowAddPeakModal(false);
                          setNewPeak({ angle: '', intensity: '' });
                        }}
                        style={{
                          padding: '8px 16px',
                          background: 'var(--color-monotone-2)',
                          color: 'var(--color-text-1)',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer'
                        }}
                      >
                        취소
                      </button>
                      <button
                        onClick={handleAddPeak}
                        className="btn-primary"
                        style={{ padding: '8px 16px' }}
                      >
                        추가
                      </button>
                    </div>
                  </div>
                </div>
              )}
        </>
      )}
          </div>

          {/* 재료 물성 상수 모달 */}
          <MaterialConstantsModal
            isOpen={showMaterialConstantsModal}
            onClose={() => setShowMaterialConstantsModal(false)}
            onSave={handleSaveMaterialConstants}
            initialValues={materialConstants}
            indexedPeaks={indexedPeaks}
            wavelength={settings.wavelength}
          />
        </div>
      )}
    </div>
  );
};

export default XRDAnalyzer;

