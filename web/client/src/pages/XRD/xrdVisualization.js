import { Chart } from 'chart.js';
import { formatMillerIndicesForDisplay } from '../../analysis/core/xrd/millerIndex.js';

/**
 * 상 동정 1순위 후보의 이론 Bragg 각도를 점선으로 표시 (Chart.js 플러그인)
 */
export const xrdTheoryOverlayPlugin = {
  id: 'xrdTheoryOverlay',
  afterDatasetsDraw(chart) {
    const lines = chart.options.plugins?.xrdTheoryOverlay?.lines;
    if (!lines?.length) return;
    const { ctx, chartArea } = chart;
    const meta0 = chart.getDatasetMeta(0);
    if (!chartArea || !meta0?.data?.length) return;

    const labels = chart.data.labels || [];
    ctx.save();
    ctx.strokeStyle = 'rgba(217, 119, 6, 0.88)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    const yTop = chartArea.top;
    const yBot = chartArea.bottom;

    for (const line of lines) {
      const tt = line.twoTheta != null ? line.twoTheta : line.angle;
      if (tt == null || !Number.isFinite(tt)) continue;
      let bestI = -1;
      let bestD = Infinity;
      for (let i = 0; i < labels.length; i++) {
        const d = Math.abs(Number(labels[i]) - tt);
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI < 0 || bestD > 0.2) continue;
      const px = meta0.data[bestI]?.x;
      if (px == null || px < chartArea.left || px > chartArea.right) continue;
      ctx.beginPath();
      ctx.moveTo(px, yTop);
      ctx.lineTo(px, yBot);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();
  },
};

let xrdTheoryOverlayRegistered = false;
export function ensureXrdTheoryOverlayRegistered() {
  if (!xrdTheoryOverlayRegistered) {
    Chart.register(xrdTheoryOverlayPlugin);
    xrdTheoryOverlayRegistered = true;
  }
}

/**
 * Chart.js를 사용한 XRD 차트 데이터 생성
 * @param {Array<{angle: number, intensity: number}>} xrdData - XRD 원본 데이터
 * @param {Object} options - 시각화 옵션
 * @param {Array<number>} options.fittedCurve - 피팅된 곡선 데이터 (선택)
 * @param {Array<{angle: number, intensity: number}>} options.peaks - 탐지된 피크 (선택)
 * @param {Array<{angle: number, intensity: number, millerIndices?: Array}>} options.indexedPeaks - 인덱싱된 피크 (선택)
 * @returns {Object} Chart.js 데이터 객체
 */
export const createXRDChartData = (xrdData, options = {}) => {
  const {
    fittedCurve = null,
    peaks = [],
    indexedPeaks = []
  } = options;

  if (xrdData.length === 0) {
    return null;
  }

  const angles = xrdData.map(d => d.angle);
  const intensities = xrdData.map(d => d.intensity);

  const datasets = [
    {
      label: 'XRD Pattern',
      data: intensities,
      borderColor: 'rgb(59, 130, 246)',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      tension: 0.1,
      order: 2
    }
  ];

  // 피팅된 곡선 추가
  if (fittedCurve && fittedCurve.length === xrdData.length) {
    datasets.push({
      label: 'Fitted Curve',
      data: fittedCurve,
      borderColor: 'rgb(239, 68, 68)',
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
      borderDash: [5, 5],
      order: 1
    });
  }

  // 피크 마커 추가
  if (peaks.length > 0) {
    const peakAngles = peaks.map(p => p.angle);
    const peakIntensities = peaks.map(p => p.intensity);

    datasets.push({
      label: 'Detected Peaks',
      data: peakIntensities.map((intensity, i) => ({
        x: peakAngles[i],
        y: intensity,
        peakIndex: i,
        peakData: peaks[i]
      })),
      borderColor: 'rgb(34, 197, 94)',
      backgroundColor: 'rgba(34, 197, 94, 0.5)',
      pointRadius: 6,
      pointHoverRadius: 8,
      pointStyle: 'circle',
      showLine: false,
      order: 0
    });
  }

  // 인덱싱된 피크 추가 (밀러지수 표시)
  if (indexedPeaks.length > 0) {
    const indexedAngles = indexedPeaks.map(p => p.angle);
    const indexedIntensities = indexedPeaks.map(p => p.intensity);

    datasets.push({
      label: 'Indexed Peaks',
      data: indexedIntensities.map((intensity, i) => ({
        x: indexedAngles[i],
        y: intensity,
        peakIndex: i,
        peakData: indexedPeaks[i]
      })),
      borderColor: 'rgb(168, 85, 247)',
      backgroundColor: 'rgba(168, 85, 247, 0.5)',
      pointRadius: 5,
      pointHoverRadius: 7,
      pointStyle: 'triangle',
      showLine: false,
      order: 0
    });
  }

  return {
    labels: angles,
    datasets: datasets
  };
};

/**
 * Chart.js 옵션 생성
 * @param {Object} options - 차트 옵션
 * @param {boolean} options.showLegend - 범례 표시 여부
 * @param {boolean} options.interactive - 인터랙티브 모드
 * @param {Function} options.onClick - 클릭 핸들러
 * @param {Array<{twoTheta?: number, angle?: number, millerLabel?: string}>} [options.theoreticalOverlayLines] — 이론 피크 세로선
 * @returns {Object} Chart.js 옵션 객체
 */
export const createXRDChartOptions = (options = {}) => {
  const {
    showLegend = true,
    interactive = true,
    onClick = null,
    theoreticalOverlayLines = null,
  } = options;

  return {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: undefined,
    interaction: {
      mode: interactive ? 'index' : 'nearest',
      intersect: false
    },
    plugins: {
      legend: {
        display: showLegend,
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        enabled: interactive,
        callbacks: {
          title: (context) => {
            return `2θ = ${context[0].label}°`;
          },
          label: (context) => {
            const datasetLabel = context.dataset.label || '';
            const value = context.parsed.y !== null ? context.parsed.y.toFixed(2) : 'N/A';
            return `${datasetLabel}: ${value} counts`;
          },
          afterLabel: (context) => {
            // 인덱싱된 피크의 경우 밀러지수 표시
            if (context.dataset.label === 'Indexed Peaks' && context.raw && context.raw.peakData) {
              const peakData = context.raw.peakData;
              const labels = [];
              
              if (peakData.dSpacing !== undefined && peakData.dSpacing !== null) {
                labels.push(`d-spacing: ${peakData.dSpacing.toFixed(3)} Å`);
              }
              
              if (peakData.millerIndices && peakData.millerIndices.length > 0) {
                const bestMatch = peakData.millerIndices[0];
                if (bestMatch.h !== undefined && bestMatch.k !== undefined && bestMatch.l !== undefined) {
                  labels.push(`면족: ${formatMillerIndicesForDisplay(bestMatch.h, bestMatch.k, bestMatch.l, 'cubic')}`);
                  if (peakData.millerIndices.length > 1) {
                    labels.push(`+ ${peakData.millerIndices.length - 1} more candidates`);
                  }
                }
              }
              
              if (peakData.confidence !== undefined && peakData.confidence !== null) {
                labels.push(`Confidence: ${(peakData.confidence * 100).toFixed(1)}%`);
              }
              
              return labels;
            }
            return '';
          }
        }
      },
      zoom: interactive ? {
        zoom: {
          wheel: {
            enabled: false // 스크롤 확대 비활성화
          },
          pinch: {
            enabled: false // 핀치 확대 비활성화
          },
          drag: {
            enabled: true, // 드래그로 영역 선택 확대
            modifierKey: null, // 수정 키 없이 드래그만으로 확대
            threshold: 5 // 최소 드래그 거리 (5px)
          },
          mode: 'x', // X축만 확대
          limits: {
            x: { min: 'original', max: 'original' },
            y: { min: 'original', max: 'original' }
          }
        },
        pan: {
          enabled: true,
          mode: 'x', // X축만 패닝
          modifierKey: 'shift' // Shift 키를 누른 상태에서 드래그로 패닝
        }
      } : undefined,
      xrdTheoryOverlay: {
        lines: Array.isArray(theoreticalOverlayLines) ? theoreticalOverlayLines : [],
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: '2θ (degree)',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          maxTicksLimit: 20,
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Intensity (counts)',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        beginAtZero: true,
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    onClick: onClick || undefined
  };
};

/**
 * 피크 정보를 차트에 표시하기 위한 어노테이션 데이터 생성
 * @param {Array<{angle: number, intensity: number, millerIndices?: Array}>} peaks - 피크 정보
 * @returns {Array<Object>} 어노테이션 배열
 */
export const createPeakAnnotations = (peaks) => {
  return peaks.map((peak, index) => {
    const annotation = {
      type: 'point',
      xValue: peak.angle,
      yValue: peak.intensity,
      backgroundColor: 'rgba(34, 197, 94, 0.5)',
      borderColor: 'rgb(34, 197, 94)',
      borderWidth: 2,
      radius: 6
    };

    // 밀러지수가 있으면 라벨 추가
    if (peak.millerIndices && peak.millerIndices.length > 0) {
      const hkl = peak.millerIndices[0];
      annotation.label = {
        content: formatMillerIndicesForDisplay(hkl.h, hkl.k, hkl.l, 'cubic'),
        enabled: true,
        position: 'top',
        backgroundColor: 'rgba(168, 85, 247, 0.8)',
        color: 'white',
        font: {
          size: 10,
          weight: 'bold'
        },
        padding: 4
      };
    }

    return annotation;
  });
};

/**
 * 피크 테이블 데이터 생성
 * @param {Array<{angle: number, intensity: number, fwhm?: number, dSpacing?: number, millerIndices?: Array, crystalliteSize?: number}>} peaks - 피크 정보
 * @returns {Array<Object>} 테이블 행 데이터
 */
export const createPeakTableData = (peaks, crystalSystem = 'cubic') => {
  if (!peaks || !Array.isArray(peaks) || peaks.length === 0) {
    return [];
  }

  return peaks.map((peak, index) => {
    if (!peak) {
      return {
        id: index + 1,
        angle: 'N/A',
        intensity: 'N/A',
        fwhm: 'N/A',
        dSpacing: 'N/A',
        millerIndices: 'N/A',
        crystalliteSize: 'N/A'
      };
    }

    const row = {
      id: index + 1,
      angle: (peak.angle !== undefined && peak.angle !== null) ? peak.angle.toFixed(3) : 'N/A',
      intensity: (peak.intensity !== undefined && peak.intensity !== null) ? peak.intensity.toFixed(2) : 'N/A',
      fwhm: (peak.fwhm !== undefined && peak.fwhm !== null) ? peak.fwhm.toFixed(3) : 'N/A',
      dSpacing: (peak.dSpacing !== undefined && peak.dSpacing !== null) ? peak.dSpacing.toFixed(3) : 'N/A',
      millerIndices: peak.millerIndices && Array.isArray(peak.millerIndices) && peak.millerIndices.length > 0
        ? peak.millerIndices.map(hkl => {
            if (hkl && typeof hkl === 'object') {
              return formatMillerIndicesForDisplay(hkl.h, hkl.k, hkl.l, crystalSystem);
            }
            return '';
          }).filter(s => s).join(', ')
        : 'N/A',
      crystalliteSize: (peak.crystalliteSize !== undefined && peak.crystalliteSize !== null) 
        ? peak.crystalliteSize.toFixed(2) + ' nm' 
        : 'N/A'
    };
    return row;
  });
};

/**
 * Modified Williamson-Hall (mWH) 플롯 데이터 생성
 * @param {Object} mwhResult - mWH 분석 결과
 * @returns {Object} Chart.js 데이터 객체
 */
export const createMWHPlotData = (mwhResult) => {
  if (!mwhResult || !mwhResult.plotData) {
    return null;
  }

  const { plotData } = mwhResult;

  return {
    labels: plotData.x,
    datasets: [
      {
        label: 'Data Points',
        data: plotData.y.map((y, i) => ({
          x: plotData.x[i],
          y: y
        })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false,
        order: 1
      },
      {
        label: 'Fitted Line',
        data: plotData.fittedY ? plotData.fittedY.map((y, i) => ({
          x: plotData.x[i],
          y: y
        })) : [],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        order: 0
      }
    ]
  };
};

/**
 * Modified Williamson-Hall (mWH) 플롯 옵션 생성
 * @returns {Object} Chart.js 옵션 객체
 */
export const createMWHPlotOptions = () => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: undefined,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      tooltip: {
        callbacks: {
          title: (context) => {
            return `Γ = ${context[0].label}`;
          },
          label: (context) => {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(6)}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Γ',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: '[(ΔK)² - α] / K²',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };
};

/**
 * Modified Warren-Averbach (mWA) 플롯 데이터 생성 (lnA vs K²C̄)
 * @param {Object} mwaResult - mWA 분석 결과
 * @returns {Object} Chart.js 데이터 객체
 */
export const createMWAPlotData = (mwaResult) => {
  if (!mwaResult || !mwaResult.plotData) {
    return null;
  }

  const { plotData } = mwaResult;

  return {
    labels: plotData.k2C,
    datasets: [
      {
        label: 'lnA(L) vs K²C̄',
        data: plotData.lnA.map((lnA, i) => ({
          x: plotData.k2C[i],
          y: lnA
        })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        pointRadius: 3,
        pointHoverRadius: 5,
        showLine: false,
        order: 1
      }
    ]
  };
};

/**
 * Modified Warren-Averbach (mWA) 플롯 옵션 생성 (lnA vs K²C̄)
 * @returns {Object} Chart.js 옵션 객체
 */
export const createMWAPlotOptions = () => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: undefined,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      tooltip: {
        callbacks: {
          title: (context) => {
            return `K²C̄ = ${context[0].label}`;
          },
          label: (context) => {
            return `lnA(L) = ${context.parsed.y.toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'K²C̄',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'lnA(L)',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };
};

/**
 * Modified Warren-Averbach (mWA) Y/L² vs lnL 플롯 데이터 생성
 * @param {Object} mwaResult - mWA 분석 결과
 * @returns {Object} Chart.js 데이터 객체
 */
export const createMWAYOverL2PlotData = (mwaResult) => {
  if (!mwaResult || !mwaResult.yOverL2Plot) {
    return null;
  }

  const { yOverL2Plot } = mwaResult;

  return {
    labels: yOverL2Plot.x,
    datasets: [
      {
        label: 'Data Points',
        data: yOverL2Plot.y.map((y, i) => ({
          x: yOverL2Plot.x[i],
          y: y
        })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        pointRadius: 4,
        pointHoverRadius: 6,
        showLine: false,
        order: 1
      },
      {
        label: 'Fitted Line',
        data: yOverL2Plot.fittedY ? yOverL2Plot.fittedY.map((y, i) => ({
          x: yOverL2Plot.x[i],
          y: y
        })) : [],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        order: 0
      }
    ]
  };
};

/**
 * Modified Warren-Averbach (mWA) Y/L² vs lnL 플롯 옵션 생성
 * @returns {Object} Chart.js 옵션 객체
 */
export const createMWAYOverL2PlotOptions = () => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: undefined,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      tooltip: {
        callbacks: {
          title: (context) => {
            return `lnL = ${context[0].label}`;
          },
          label: (context) => {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(6)}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'lnL',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Y / L²',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };
};

/**
 * 표준 Williamson–Hall (UDM): β cos θ vs sin θ
 * @param {Object} whData - williamsonHallFit 성공 시 data
 */
export const createStandardWilliamsonHallPlotData = (whData) => {
  if (!whData || !whData.plotData) return null;
  const { plotData } = whData;
  const scatter = (plotData.x || []).map((x, i) => ({
    x,
    y: plotData.y[i],
  }));
  const line = (plotData.fittedSinTheta || []).map((x, i) => ({
    x,
    y: plotData.fittedBetaCosTheta[i],
  }));

  return {
    datasets: [
      {
        label: '실측 (β cos θ)',
        data: scatter,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        pointRadius: 6,
        showLine: false,
        order: 1,
      },
      {
        label: '선형 피팅',
        data: line,
        borderColor: 'rgb(220, 38, 38)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0,
        order: 0,
      },
    ],
  };
};

export const createStandardWilliamsonHallPlotOptions = () => ({
  responsive: true,
  maintainAspectRatio: false,
  aspectRatio: undefined,
  plugins: {
    legend: { display: true, position: 'top' },
    tooltip: {
      callbacks: {
        title: (ctx) => `sin θ = ${ctx[0].parsed.x?.toFixed(5) ?? ''}`,
        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(7) ?? ''}`,
      },
    },
  },
  scales: {
    x: {
      type: 'linear',
      title: {
        display: true,
        text: 'sin θ',
        font: { size: 14, weight: 'bold' },
      },
      grid: { color: 'rgba(0, 0, 0, 0.1)' },
    },
    y: {
      type: 'linear',
      title: {
        display: true,
        text: 'β cos θ (rad)',
        font: { size: 14, weight: 'bold' },
      },
      grid: { color: 'rgba(0, 0, 0, 0.1)' },
    },
  },
});

/**
 * sin²ψ 잔류 응력 피팅: d vs sin²ψ
 * @param {Object} stressData - fitResidualStressFromPsiScan 성공 data
 */
export const createStressSin2PsiPlotData = (stressData) => {
  if (!stressData?.plotData?.sin2Psi?.length) return null;
  const { sin2Psi, dObservedAngstrom, dFittedAngstrom } = stressData.plotData;
  return {
    datasets: [
      {
        label: 'd (측정, Å)',
        data: sin2Psi.map((x, i) => ({ x, y: dObservedAngstrom[i] })),
        borderColor: 'rgb(37, 99, 235)',
        backgroundColor: 'rgba(37, 99, 235, 0.5)',
        pointRadius: 6,
        showLine: false,
        order: 1,
      },
      {
        label: 'd (선형 피팅)',
        data: sin2Psi.map((x, i) => ({ x, y: dFittedAngstrom[i] })),
        borderColor: 'rgb(220, 38, 38)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0,
        order: 0,
      },
    ],
  };
};

export const createStressSin2PsiPlotOptions = () => ({
  responsive: true,
  maintainAspectRatio: false,
  aspectRatio: undefined,
  plugins: {
    legend: { display: true, position: 'top' },
    tooltip: {
      callbacks: {
        title: (ctx) => `sin²ψ = ${ctx[0].parsed.x?.toFixed(5) ?? ''}`,
        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(5) ?? ''} Å`,
      },
    },
  },
  scales: {
    x: {
      type: 'linear',
      title: {
        display: true,
        text: 'sin²ψ',
        font: { size: 14, weight: 'bold' },
      },
      grid: { color: 'rgba(0, 0, 0, 0.1)' },
    },
    y: {
      type: 'linear',
      title: {
        display: true,
        text: 'd (Å)',
        font: { size: 14, weight: 'bold' },
      },
      grid: { color: 'rgba(0, 0, 0, 0.1)' },
    },
  },
});

