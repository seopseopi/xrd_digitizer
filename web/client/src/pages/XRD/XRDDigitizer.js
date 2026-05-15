import React, { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import { analysisClient } from '../../analysis/analysisClient';

// ── 상수 ─────────────────────────────────────────────────────────────────────
// 영역 정의 3점: 원점(origin), X축 끝, Y축 끝 → plot_box 결정
const BOX_COLORS  = ['#64748b', '#ef4444', '#8b5cf6'];
const BOX_LABELS  = ['원점', 'X끝', 'Y끝'];

// 캘리브레이션 4점: x축 2점 + y축 2점 (임의 tick mark, 반드시 끝점 아니어도 됨)
const CALIB_COLORS = ['#f59e0b', '#f97316', '#3b82f6', '#22c55e'];
const CALIB_LABELS = ['X1', 'X2', 'Y1', 'Y2'];

// ── 색상 유틸 ─────────────────────────────────────────────────────────────────
function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0').toUpperCase()).join('');
}

function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  };
}

function findClosestPixelInRoi(img, targetR, targetG, targetB) {
  if (!img) return null;
  const offscreen = document.createElement('canvas');
  offscreen.width = img.naturalWidth;
  offscreen.height = img.naturalHeight;
  const ctx = offscreen.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const w = img.naturalWidth, h = img.naturalHeight;
  const STEP = Math.max(1, Math.round(Math.sqrt(w * h) / 100));
  const data = ctx.getImageData(0, 0, w, h).data;
  let bestDist = Infinity, bestX = 0, bestY = 0;
  for (let py = 0; py < h; py += STEP) {
    for (let px = 0; px < w; px += STEP) {
      const i = (py * w + px) * 4;
      const dr = data[i] - targetR, dg = data[i + 1] - targetG, db = data[i + 2] - targetB;
      const dist = dr * dr + dg * dg + db * db;
      if (dist < bestDist) { bestDist = dist; bestX = px; bestY = py; }
    }
  }
  return { x: bestX, y: bestY };
}

// ── 돋보기 ────────────────────────────────────────────────────────────────────
function drawMagnifier(ctx, cx, cy, dr, imgEl, canvasW, canvasH, boxCorners, calibPts) {
  if (!imgEl) return;
  const MR = 36;
  const SRC_SIZE = 20;
  const DISPLAY_SIZE = SRC_SIZE * 6;
  const MAG_SCALE = DISPLAY_SIZE / SRC_SIZE;

  const imgX = (cx - dr.x) / dr.scale;
  const imgY = (cy - dr.y) / dr.scale;

  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.35)';
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(cx, cy, MR + 1, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.01)';
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, MR, 0, Math.PI * 2);
  ctx.clip();

  const iW = imgEl.naturalWidth, iH = imgEl.naturalHeight;
  const srcX = Math.max(0, Math.min(imgX - SRC_SIZE / 2, iW - SRC_SIZE));
  const srcY = Math.max(0, Math.min(imgY - SRC_SIZE / 2, iH - SRC_SIZE));
  const offsetX = (imgX - SRC_SIZE / 2 - srcX) * MAG_SCALE;
  const offsetY = (imgY - SRC_SIZE / 2 - srcY) * MAG_SCALE;
  ctx.drawImage(
    imgEl, srcX, srcY, SRC_SIZE, SRC_SIZE,
    cx - DISPLAY_SIZE / 2 - offsetX, cy - DISPLAY_SIZE / 2 - offsetY, DISPLAY_SIZE, DISPLAY_SIZE
  );

  // box corner points in magnifier
  if (boxCorners) {
    boxCorners.forEach((pt, i) => {
      if (!pt.px) return;
      const dxi = pt.px.x - imgX, dyi = pt.px.y - imgY;
      if (Math.abs(dxi) > SRC_SIZE / 2 + 2 || Math.abs(dyi) > SRC_SIZE / 2 + 2) return;
      const dotX = cx + dxi * MAG_SCALE, dotY = cy + dyi * MAG_SCALE;
      ctx.beginPath(); ctx.arc(dotX, dotY, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();
      ctx.beginPath(); ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
      ctx.fillStyle = BOX_COLORS[i]; ctx.fill();
      ctx.font = 'bold 8px sans-serif'; ctx.fillStyle = '#fff';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(BOX_LABELS[i], dotX, dotY);
    });
  }

  // calib points in magnifier
  if (calibPts) {
    calibPts.forEach((pt, i) => {
      if (!pt.px) return;
      const dxi = pt.px.x - imgX, dyi = pt.px.y - imgY;
      if (Math.abs(dxi) > SRC_SIZE / 2 + 2 || Math.abs(dyi) > SRC_SIZE / 2 + 2) return;
      const dotX = cx + dxi * MAG_SCALE, dotY = cy + dyi * MAG_SCALE;
      ctx.beginPath(); ctx.arc(dotX, dotY, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();
      ctx.beginPath(); ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
      ctx.fillStyle = CALIB_COLORS[i]; ctx.fill();
      ctx.font = 'bold 9px sans-serif'; ctx.fillStyle = '#fff';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(CALIB_LABELS[i], dotX, dotY);
    });
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  }

  // crosshair
  ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)';
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  const CL = 9;
  ctx.beginPath();
  ctx.moveTo(cx - CL, cy); ctx.lineTo(cx + CL, cy);
  ctx.moveTo(cx, cy - CL); ctx.lineTo(cx, cy + CL);
  ctx.stroke();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(cx, cy, MR, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ── 하위 컴포넌트 ─────────────────────────────────────────────────────────────
function ChevronIcon({ open }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      {open
        ? <path d="M18 15l-6-6-6 6" strokeLinecap="round" strokeLinejoin="round" />
        : <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />}
    </svg>
  );
}

function AccordionSection({ title, open, onToggle, children }) {
  return (
    <div className="xrd-settings-accordion-section">
      <button type="button" className="xrd-settings-accordion-header" onClick={onToggle} aria-expanded={open}>
        <span className="xrd-settings-accordion-title">{title}</span>
        <span className="xrd-settings-accordion-chevron"><ChevronIcon open={open} /></span>
      </button>
      {open && <div className="xrd-settings-accordion-body">{children}</div>}
    </div>
  );
}

function initBoxCorners() {
  // 3점으로 그래프 영역(plot_box) 결정. 값 입력 없이 위치만.
  return [
    { px: null, autoDetected: false }, // p1: 원점 (origin, bottom-left)
    { px: null, autoDetected: false }, // p2: X축 끝 (x-end, bottom-right)
    { px: null, autoDetected: false }, // p3: Y축 끝 (y-end, top-left)
  ];
}

function initCalibPts() {
  // 4점으로 pixel↔물리값 매핑. 반드시 축 끝점이 아닌 임의 tick이어도 됨.
  return [
    { px: null, val: '', autoDetected: false }, // X1: x축 첫 번째 tick
    { px: null, val: '', autoDetected: false }, // X2: x축 두 번째 tick
    { px: null, val: '', autoDetected: false }, // Y1: y축 첫 번째 tick
    { px: null, val: '', autoDetected: false }, // Y2: y축 두 번째 tick
  ];
}

// ── 메인 컴포넌트 ─────────────────────────────────────────────────────────────
export default function XRDDigitizer({ onDigitizeComplete, mode, setMode, setToolbarContent }) {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [naturalSize, setNaturalSize] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // 영역 정의 3점 (원점, X끝, Y끝) + 활성 입력 모드
  const [boxCorners, setBoxCorners] = useState(initBoxCorners);
  const [boxMode, setBoxMode]       = useState(null); // null | 0 | 1 | 2

  // 캘리브레이션 4점 (X1, X2, Y1, Y2) + 활성 입력 모드
  const [calibPts, setCalibPts]   = useState(initCalibPts);
  const [calibMode, setCalibMode] = useState(null); // null | 0 | 1 | 2 | 3

  const [colorRgb, setColorRgb] = useState(null);
  const [colorPt, setColorPt] = useState(null);
  const [isEyedropperMode, setIsEyedropperMode] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [runError, setRunError] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isDetectingColor, setIsDetectingColor] = useState(false);
  const [detectResult, setDetectResult] = useState(null);

  // plot_box: boxCorners 3점에서 파생 (자동 감지 또는 수동)
  const detectedBox = React.useMemo(() => {
    const [origin, xEnd, yEnd] = boxCorners;
    if (origin?.px && xEnd?.px && yEnd?.px) {
      return [origin.px.x, yEnd.px.y, xEnd.px.x, origin.px.y];
    }
    return null;
  }, [boxCorners]);

  const [openSections, setOpenSections] = useState({ file: true, box: true, calib: true, color: true });

  const zoomRef = useRef({ level: 1, panX: 0, panY: 0 });
  const [zoomDisplay, setZoomDisplay] = useState(1);
  const [overlayData, setOverlayData] = useState(null);
  const [showOverlay, setShowOverlay] = useState(true);

  const fileInputRef    = useRef(null);
  const replaceInputRef = useRef(null);
  const colorInputRef   = useRef(null);
  const imgContainerRef = useRef(null);
  const imgRef          = useRef(null);
  const canvasRef       = useRef(null);
  const magnifierPosRef = useRef(null);
  const isPanningRef    = useRef(false);
  const panStartRef     = useRef(null);

  const isSelectingMode = calibMode !== null || boxMode !== null || isEyedropperMode;

  // ── 캔버스 헬퍼 ────────────────────────────────────────────────────────────
  const getDisplayRect = useCallback(() => {
    const img = imgRef.current;
    const container = imgContainerRef.current;
    if (!img || !img.naturalWidth || !container || !container.clientWidth) return null;
    const cw = container.clientWidth, ch = container.clientHeight;
    const nw = img.naturalWidth, nh = img.naturalHeight;
    const z = zoomRef.current;
    const baseScale = Math.min(cw / nw, ch / nh);
    const scale = baseScale * z.level;
    const dw = nw * scale, dh = nh * scale;
    return { x: (cw - dw) / 2 + z.panX, y: (ch - dh) / 2 + z.panY, w: dw, h: dh, scale };
  }, []);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const dr = getDisplayRect();
    if (!dr) return;

    // 결과 오버레이
    if (showOverlay && overlayData) {
      const [cx1, cx2, cy1, cy2] = calibPts;
      const xv1 = parseFloat(cx1?.val), xv2 = parseFloat(cx2?.val);
      const yv1 = parseFloat(cy1?.val), yv2 = parseFloat(cy2?.val);
      if (cx1?.px && cx2?.px && cy1?.px && cy2?.px
          && Number.isFinite(xv1) && Number.isFinite(xv2)
          && Number.isFinite(yv1) && Number.isFinite(yv2)) {
        const xScale = (cx2.px.x - cx1.px.x) / (xv2 - xv1);
        const yScale = (cy2.px.y - cy1.px.y) / (yv2 - yv1);
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
        ctx.lineWidth = 1.5 / zoomRef.current.level;
        overlayData.two_theta_values.forEach((theta, idx) => {
          const intensity = overlayData.intensities[idx];
          const imgX = cx1.px.x + (theta - xv1) * xScale;
          const imgY = cy1.px.y + (intensity - yv1) * yScale;
          const ox = imgX * dr.scale + dr.x;
          const oy = imgY * dr.scale + dr.y;
          if (idx === 0) ctx.moveTo(ox, oy); else ctx.lineTo(ox, oy);
        });
        ctx.stroke();
        ctx.restore();
      }
    }

    // 영역 plot_box
    if (detectedBox) {
      const [bx0, by0, bx1, by1] = detectedBox;
      const sx0 = bx0 * dr.scale + dr.x, sy0 = by0 * dr.scale + dr.y;
      const sx1 = bx1 * dr.scale + dr.x, sy1 = by1 * dr.scale + dr.y;
      ctx.save();
      ctx.strokeStyle = 'rgba(99, 179, 237, 0.9)';
      ctx.lineWidth = 2 / zoomRef.current.level;
      ctx.setLineDash([6 / zoomRef.current.level, 3 / zoomRef.current.level]);
      ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      ctx.fillStyle = 'rgba(99, 179, 237, 0.08)';
      ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      ctx.restore();
    }

    // 영역 3점 (다이아몬드 모양)
    boxCorners.forEach((pt, i) => {
      if (!pt.px) return;
      const px = pt.px.x * dr.scale + dr.x;
      const py = pt.px.y * dr.scale + dr.y;
      const R = 7;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(px, py - R); ctx.lineTo(px + R, py);
      ctx.lineTo(px, py + R); ctx.lineTo(px - R, py);
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();
      ctx.fillStyle = BOX_COLORS[i]; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.restore();
      ctx.fillStyle = BOX_COLORS[i];
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(BOX_LABELS[i], px + 9, py - 4);
    });

    // 캘리브레이션 4점 (원형)
    calibPts.forEach((pt, i) => {
      if (!pt.px) return;
      const px = pt.px.x * dr.scale + dr.x;
      const py = pt.px.y * dr.scale + dr.y;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = CALIB_COLORS[i]; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = CALIB_COLORS[i];
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText(CALIB_LABELS[i], px + 10, py - 6);
    });

    // 돋보기
    const magPos = magnifierPosRef.current;
    if (isSelectingMode && magPos) {
      drawMagnifier(ctx, magPos.cx, magPos.cy, dr, imgRef.current, canvas.width, canvas.height, boxCorners, calibPts);
    }
  }, [boxCorners, calibPts, isSelectingMode, getDisplayRect, overlayData, showOverlay, detectedBox]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  const positionImg = useCallback(() => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth) return;
    const dr = getDisplayRect();
    if (!dr) return;
    img.style.left = `${dr.x}px`;
    img.style.top = `${dr.y}px`;
    img.style.width = `${dr.w}px`;
    img.style.height = `${dr.h}px`;
  }, [getDisplayRect]);

  const redraw = useCallback(() => {
    positionImg();
    drawCanvas();
  }, [positionImg, drawCanvas]);

  useLayoutEffect(() => {
    if (naturalSize) positionImg();
  }, [positionImg, naturalSize]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = imgContainerRef.current;
    if (!canvas || !container) return;
    const ro = new ResizeObserver(() => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      redraw();
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, [imageFile, redraw]);

  // ── 줌 / 패닝 ─────────────────────────────────────────────────────────────
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    const container = imgContainerRef.current;
    const img = imgRef.current;
    if (!canvas || !container || !img || !img.naturalWidth) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const dr = getDisplayRect();
    if (!dr) return;
    const cw = container.clientWidth, ch = container.clientHeight;
    const nw = img.naturalWidth, nh = img.naturalHeight;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const z = zoomRef.current;
    const newLevel = Math.min(8, Math.max(0.5, z.level * factor));
    const baseScale = Math.min(cw / nw, ch / nh);
    const newScale = baseScale * newLevel;
    const imgX = (cx - dr.x) / dr.scale, imgY = (cy - dr.y) / dr.scale;
    const newX = cx - imgX * newScale, newY = cy - imgY * newScale;
    zoomRef.current = { level: newLevel, panX: newX - (cw - nw * newScale) / 2, panY: newY - (ch - nh * newScale) / 2 };
    setZoomDisplay(newLevel);
    redraw();
  }, [getDisplayRect, redraw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  const adjustZoom = useCallback((dir) => {
    const z = zoomRef.current;
    zoomRef.current = { ...z, level: Math.min(8, Math.max(0.5, z.level * (dir > 0 ? 1.3 : 1 / 1.3))) };
    setZoomDisplay(zoomRef.current.level);
    redraw();
  }, [redraw]);

  const resetZoom = useCallback(() => {
    zoomRef.current = { level: 1, panX: 0, panY: 0 };
    setZoomDisplay(1);
    redraw();
  }, [redraw]);

  const handleMouseDown = useCallback((e) => {
    if (isSelectingMode) return;
    isPanningRef.current = true;
    const z = zoomRef.current;
    panStartRef.current = { x: e.clientX, y: e.clientY, panX: z.panX, panY: z.panY };
  }, [isSelectingMode]);

  const handleMouseUp = useCallback(() => { isPanningRef.current = false; }, []);

  // ── 파일 처리 ──────────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    setImageUrl(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(file); });
    setImageFile(file);
    setNaturalSize(null);
    setColorRgb(null);
    setColorPt(null);
    setIsEyedropperMode(false);
    setBoxCorners(initBoxCorners());
    setBoxMode(null);
    setCalibPts(initCalibPts());
    setCalibMode(null);
    setRunError(null);
    setDetectResult(null);
    setOverlayData(null);
    zoomRef.current = { level: 1, panX: 0, panY: 0 };
    setZoomDisplay(1);
    magnifierPosRef.current = null;
  }, []);

  // ── 자동 감지 ─────────────────────────────────────────────────────────────
  const handleAutoDetect = useCallback(async () => {
    if (!imageFile || isDetecting) return;
    setIsDetecting(true);
    setDetectResult(null);
    try {
      const res = await analysisClient.xrd.detectRoi(imageFile);
      if (!res?.success) throw new Error(res?.error?.message || res?.message || '감지 실패');
      const { calib_points: cp, curve_color: cc, color_sample_point: csp, axis_values: av, confidence, ocr_available } = res.data;
      const p1 = cp?.p1, p2 = cp?.p2, p3 = cp?.p3;

      // 영역 정의 3점 채우기
      setBoxCorners([
        { px: p1 ? { x: p1.x, y: p1.y } : null, autoDetected: !!p1 },
        { px: p2 ? { x: p2.x, y: p2.y } : null, autoDetected: !!p2 },
        { px: p3 ? { x: p3.x, y: p3.y } : null, autoDetected: !!p3 },
      ]);

      // 캘리브레이션 4점 채우기 (초기값은 축 끝점, 사용자가 interior tick으로 이동 가능)
      setCalibPts([
        { px: p1 ? { x: p1.x, y: p1.y } : null, val: av?.x_min != null ? String(av.x_min) : '', autoDetected: !!p1 },
        { px: p2 ? { x: p2.x, y: p2.y } : null, val: av?.x_max != null ? String(av.x_max) : '', autoDetected: !!p2 },
        { px: p1 ? { x: p1.x, y: p1.y } : null, val: av?.y_min != null ? String(av.y_min) : '', autoDetected: !!p1 },
        { px: p3 ? { x: p3.x, y: p3.y } : null, val: av?.y_max != null ? String(av.y_max) : '', autoDetected: !!p3 },
      ]);

      if (cc) setColorRgb({ r: cc[0], g: cc[1], b: cc[2] });
      if (csp) setColorPt({ x: csp[0], y: csp[1] });
      setDetectResult({ confidence, ocr_available, hasColor: !!cc });
    } catch (err) {
      setDetectResult({ error: err.message || '자동 감지 실패' });
    } finally {
      setIsDetecting(false);
    }
  }, [imageFile, isDetecting]);

  const handleAutoDetectColor = useCallback(async () => {
    if (!imageFile || isDetectingColor) return;
    setIsDetectingColor(true);
    try {
      const res = await analysisClient.xrd.detectRoi(imageFile);
      if (!res?.success) throw new Error(res?.error?.message || res?.message || '색상 감지 실패');
      const { curve_color: cc, color_sample_point: csp } = res.data;
      if (!cc) throw new Error('색상을 감지할 수 없습니다');
      setColorRgb({ r: cc[0], g: cc[1], b: cc[2] });
      if (csp) setColorPt({ x: csp[0], y: csp[1] });
    } catch (err) {
      alert(err.message || '색상 자동 감지 실패');
    } finally {
      setIsDetectingColor(false);
    }
  }, [imageFile, isDetectingColor]);

  const handleFileInputChange = useCallback((e) => {
    handleFile(e.target.files?.[0]);
    e.target.value = '';
  }, [handleFile]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  }, [handleFile]);

  const handleImageLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
  }, []);

  // ── 캔버스 이벤트 ─────────────────────────────────────────────────────────
  const getCanvasPos = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (isPanningRef.current && panStartRef.current && !isSelectingMode) {
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      zoomRef.current = { ...zoomRef.current, panX: panStartRef.current.panX + dx, panY: panStartRef.current.panY + dy };
      canvasRef.current.style.cursor = 'grabbing';
      redraw();
      return;
    }
    const { x: cx, y: cy } = getCanvasPos(e);
    const canvas = canvasRef.current;
    if (isSelectingMode) {
      canvas.style.cursor = 'none';
      magnifierPosRef.current = { cx, cy };
      drawCanvas();
    } else {
      canvas.style.cursor = zoomRef.current.level > 1 ? 'grab' : 'default';
    }
  }, [getCanvasPos, isSelectingMode, drawCanvas, redraw]);

  const handleMouseLeave = useCallback(() => {
    magnifierPosRef.current = null;
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = useCallback((e) => {
    const { x: cx, y: cy } = getCanvasPos(e);
    const dr = getDisplayRect();
    if (!dr) return;
    const imgX = Math.max(0, Math.min(Math.round((cx - dr.x) / dr.scale), (naturalSize?.w ?? 1) - 1));
    const imgY = Math.max(0, Math.min(Math.round((cy - dr.y) / dr.scale), (naturalSize?.h ?? 1) - 1));

    if (boxMode !== null) {
      setBoxCorners(prev => prev.map((p, i) => i === boxMode ? { ...p, px: { x: imgX, y: imgY }, autoDetected: false } : p));
      setBoxMode(null);
      magnifierPosRef.current = null;
      return;
    }

    if (calibMode !== null) {
      setCalibPts(prev => prev.map((p, i) => i === calibMode ? { ...p, px: { x: imgX, y: imgY }, autoDetected: false } : p));
      setCalibMode(null);
      magnifierPosRef.current = null;
      return;
    }

    if (!isEyedropperMode) return;
    const img = imgRef.current;
    const offscreen = document.createElement('canvas');
    offscreen.width = img.naturalWidth;
    offscreen.height = img.naturalHeight;
    const octx = offscreen.getContext('2d');
    octx.drawImage(img, 0, 0);
    const px = octx.getImageData(imgX, imgY, 1, 1).data;
    setColorRgb({ r: px[0], g: px[1], b: px[2] });
    setColorPt({ x: imgX, y: imgY });
    setIsEyedropperMode(false);
    magnifierPosRef.current = null;
  }, [boxMode, calibMode, isEyedropperMode, getCanvasPos, getDisplayRect, naturalSize]);

  const handleColorPickerChange = useCallback((e) => {
    const { r, g, b } = hexToRgb(e.target.value);
    setColorRgb({ r, g, b });
    const pt = findClosestPixelInRoi(imgRef.current, r, g, b);
    if (pt) setColorPt(pt);
  }, []);

  const toggleSection = useCallback((key) => {
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // ── 유효성 ────────────────────────────────────────────────────────────────
  const boxValid   = boxCorners.every(p => p.px !== null) && detectedBox !== null;
  const calibValid = calibPts.every(p => p.px !== null && p.val !== '' && !isNaN(parseFloat(p.val)));
  const canRun     = !!(imageFile && boxValid && calibValid && colorRgb && colorPt);

  // ── 실행 ──────────────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    if (!canRun || !colorPt) return;
    setIsLoading(true);
    setRunError(null);
    try {
      const [cx1, cx2, cy1, cy2] = calibPts;
      const [bx0, by0, bx1, by1] = detectedBox;
      const manualInputs = {
        plot_box: [bx0 + 2, by0 + 2, bx1 - 2, by1],
        x_axis_points: [[cx1.px.x, cx1.px.y], [cx2.px.x, cx2.px.y]],
        x_axis_values: [parseFloat(cx1.val), parseFloat(cx2.val)],
        y_axis_points: [[cy1.px.x, cy1.px.y], [cy2.px.x, cy2.px.y]],
        y_axis_values: [parseFloat(cy1.val), parseFloat(cy2.val)],
        color_sample_point: [colorPt.x, colorPt.y],
        curve_color_rgb: [colorRgb.r, colorRgb.g, colorRgb.b],
      };
      const res = await analysisClient.xrd.digitize(imageFile, manualInputs);
      if (!res?.success) throw new Error(res?.error?.message || '디지타이즈 실패');
      onDigitizeComplete?.(res.data);
      setOverlayData({ two_theta_values: res.data.two_theta_values, intensities: res.data.intensities });
      setShowOverlay(true);
    } catch (err) {
      setRunError(err.message || '알 수 없는 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  }, [canRun, colorPt, colorRgb, calibPts, detectedBox, imageFile, onDigitizeComplete]);

  // ── 툴바 주입 ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!setToolbarContent) return;

    const colorHex = colorRgb ? rgbToHex(colorRgb.r, colorRgb.g, colorRgb.b) : '#000000';

    const boxMeta = [
      { label: '원점 (origin)', hint: '차트 좌하단 코너', guide: '그래프 영역의 원점(좌하단)을 클릭하세요.' },
      { label: 'X축 끝 (x-end)', hint: '차트 우하단 코너', guide: '그래프 영역의 x축 끝(우하단)을 클릭하세요.' },
      { label: 'Y축 끝 (y-end)', hint: '차트 좌상단 코너', guide: '그래프 영역의 y축 끝(좌상단)을 클릭하세요.' },
    ];

    const calibMeta = [
      { label: 'X1 — x축 첫 번째 tick', hint: 'x축 위의 알려진 tick 위치', guide: 'x축 위 숫자가 표시된 tick mark를 클릭하세요.', valueLabel: '2θ' },
      { label: 'X2 — x축 두 번째 tick', hint: 'x축 위의 또 다른 알려진 tick', guide: 'X1과 다른 tick mark를 클릭하세요 (멀수록 정밀).', valueLabel: '2θ' },
      { label: 'Y1 — y축 첫 번째 tick', hint: 'y축 위의 알려진 tick 위치', guide: 'y축 위 숫자가 표시된 tick mark를 클릭하세요.', valueLabel: 'I' },
      { label: 'Y2 — y축 두 번째 tick', hint: 'y축 위의 또 다른 알려진 tick', guide: 'Y1과 다른 tick mark를 클릭하세요 (멀수록 정밀).', valueLabel: 'I' },
    ];

    const autoDetectBadge = (isAuto) => isAuto ? (
      <span style={{ fontSize: 9, fontWeight: 600, padding: '1px 5px', borderRadius: 3, background: '#dbeafe', color: '#2563eb', marginLeft: 4 }}>
        자동감지
      </span>
    ) : null;

    setToolbarContent(
      <div className="xrd-analysis-settings-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

        {/* 모드 토글 */}
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

        <div className="xrd-settings-accordion" style={{ flex: 1, overflowY: 'auto' }}>

          {/* 파일 */}
          <AccordionSection title="이미지 파일" open={openSections.file} onToggle={() => toggleSection('file')}>
            {imageFile ? (
              <>
                <div style={{ fontSize: 12, color: '#475569', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: 8 }}>
                  {imageFile.name}
                </div>
                <button type="button" className="xrd-settings-sample-btn" style={{ background: '#475569' }}
                  onClick={() => replaceInputRef.current?.click()}>
                  이미지 교체
                </button>
              </>
            ) : (
              <div style={{ fontSize: 12, color: '#94a3b8' }}>이미지를 업로드해 주세요.</div>
            )}
          </AccordionSection>

          {/* 자동 감지 버튼 (두 섹션에 공통) */}
          <div style={{ padding: '8px 12px 0' }}>
            <button
              type="button"
              onClick={handleAutoDetect}
              disabled={!imageFile || isDetecting}
              style={{
                width: '100%', padding: '9px 0', marginBottom: 6,
                borderRadius: 7, border: 'none',
                background: isDetecting ? '#e0e7ff' : '#2563eb',
                color: isDetecting ? '#3730a3' : '#fff',
                fontSize: 12, fontWeight: 600, cursor: imageFile && !isDetecting ? 'pointer' : 'not-allowed',
                opacity: imageFile ? 1 : 0.5,
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              }}
            >
              {isDetecting
                ? <>처리 중<span style={{ animation: 'dotBlink 1.2s 0.0s ease-in-out infinite' }}>.</span><span style={{ animation: 'dotBlink 1.2s 0.4s ease-in-out infinite' }}>.</span><span style={{ animation: 'dotBlink 1.2s 0.8s ease-in-out infinite' }}>.</span></>
                : '🔍 영역 + 축 자동 감지'}
            </button>
            {detectResult && !detectResult.error && (
              <div style={{ marginBottom: 6, padding: '5px 8px', borderRadius: 5, fontSize: 11, lineHeight: 1.5, background: '#f0fdf4', border: '1px solid #86efac', color: '#166534' }}>
                ✓ 자동 감지 완료 — 신뢰도 {Math.round((detectResult.confidence ?? 0) * 100)}%
                {detectResult.hasColor ? ' · 색상 감지됨' : ''}
                {detectResult.ocr_available ? ' · 축값 채워짐' : ' · 축값은 직접 입력해주세요'}
              </div>
            )}
            {detectResult?.error && (
              <div style={{ marginBottom: 6, padding: '5px 8px', borderRadius: 5, fontSize: 11, background: '#fef2f2', border: '1px solid #fca5a5', color: '#dc2626' }}>
                ✗ {detectResult.error}
              </div>
            )}
          </div>

          {/* ── 영역 설정 (3점) ── */}
          <AccordionSection title="① 영역 설정 (3점)" open={openSections.box} onToggle={() => toggleSection('box')}>
            <div style={{ padding: '4px 8px 8px', borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0', fontSize: 11, color: '#64748b', lineHeight: 1.5, marginBottom: 8 }}>
              그래프 영역을 정의하는 3개의 코너 점을 지정하세요.<br />
              <b>원점</b> → <b>X축 끝</b> → <b>Y축 끝</b> 순서로 클릭.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {boxCorners.map((pt, i) => {
                const isActive = boxMode === i;
                const isDone   = pt.px !== null;
                const isAuto   = pt.autoDetected && isDone;
                const color    = BOX_COLORS[i];
                const meta     = boxMeta[i];

                return (
                  <div
                    key={i}
                    style={{
                      borderRadius: 8,
                      border: `1px solid ${isActive ? color : isDone ? (isAuto ? '#bfdbfe' : '#d1fae5') : '#e2e8f0'}`,
                      background: isActive ? `${color}15` : isDone ? (isAuto ? '#eff6ff' : '#f0fdf4') : '#f8fafc',
                      overflow: 'hidden',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 10px' }}>
                      {/* 다이아몬드 아이콘 */}
                      <svg width="18" height="18" viewBox="0 0 18 18" style={{ flexShrink: 0 }}>
                        <polygon points="9,2 16,9 9,16 2,9" fill={isDone ? color : '#e2e8f0'} stroke="#fff" strokeWidth="1" />
                        {isDone && <text x="9" y="13" textAnchor="middle" fontSize="8" fontWeight="bold" fill="#fff">✓</text>}
                      </svg>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: '#334155', display: 'flex', alignItems: 'center' }}>
                          {meta.label}{autoDetectBadge(isAuto)}
                        </div>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 1 }}>
                          {isDone ? `px (${pt.px.x}, ${pt.px.y})` : meta.hint}
                        </div>
                      </div>
                      {isDone && !isActive && (
                        <button
                          type="button"
                          onClick={() => { setBoxMode(i); setCalibMode(null); setIsEyedropperMode(false); }}
                          style={{
                            padding: '3px 8px', fontSize: 10, borderRadius: 4, fontWeight: isAuto ? 700 : 400,
                            border: `1px solid ${isAuto ? '#93c5fd' : '#e2e8f0'}`,
                            background: isAuto ? '#eff6ff' : '#fff',
                            color: isAuto ? '#2563eb' : '#475569',
                            cursor: 'pointer', flexShrink: 0,
                          }}
                        >수정</button>
                      )}
                    </div>

                    {!isDone && !isActive && (
                      <div style={{ padding: '0 10px 7px' }}>
                        <button
                          type="button"
                          onClick={() => { setBoxMode(i); setCalibMode(null); setIsEyedropperMode(false); }}
                          disabled={!imageFile}
                          style={{
                            width: '100%', padding: '5px 0', fontSize: 11, fontWeight: 600,
                            borderRadius: 6, border: `1px solid ${color}`,
                            background: '#fff', color, cursor: 'pointer', opacity: imageFile ? 1 : 0.5,
                          }}
                        >클릭하여 지정</button>
                      </div>
                    )}

                    {isActive && (
                      <div style={{ padding: '0 10px 7px' }}>
                        <div style={{ padding: '5px 8px', borderRadius: 5, background: `${color}20`, fontSize: 11, color, fontWeight: 500, lineHeight: 1.5, marginBottom: 4 }}>
                          📍 {meta.guide}
                        </div>
                        <button
                          type="button"
                          onClick={() => setBoxMode(null)}
                          style={{ width: '100%', padding: '4px 0', fontSize: 10, borderRadius: 5, border: '1px solid #e2e8f0', background: '#fff', color: '#94a3b8', cursor: 'pointer' }}
                        >취소</button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {boxValid && (
              <div style={{ marginTop: 6, padding: '5px 8px', borderRadius: 5, fontSize: 11, background: '#f0fdf4', border: '1px solid #86efac', color: '#166534' }}>
                ✓ 영역 설정 완료
              </div>
            )}
          </AccordionSection>

          {/* ── 캘리브레이션 (4점) ── */}
          <AccordionSection title="② 캘리브레이션 (4점)" open={openSections.calib} onToggle={() => toggleSection('calib')}>
            <div style={{ padding: '4px 8px 8px', borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0', fontSize: 11, color: '#64748b', lineHeight: 1.5, marginBottom: 8 }}>
              x축 tick 2개, y축 tick 2개를 클릭하고 해당 값을 입력하세요.<br />
              <b>축 끝점을 모르면</b> 보이는 아무 눈금 2개라도 OK — 선형 외삽으로 변환됨.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {calibPts.map((pt, i) => {
                const isActive = calibMode === i;
                const isDone   = pt.px !== null;
                const isAuto   = pt.autoDetected && isDone;
                const color    = CALIB_COLORS[i];
                const meta     = calibMeta[i];

                return (
                  <div
                    key={i}
                    style={{
                      borderRadius: 8,
                      border: `1px solid ${isActive ? color : isDone ? (isAuto ? '#bfdbfe' : '#d1fae5') : '#e2e8f0'}`,
                      background: isActive ? `${color}12` : isDone ? (isAuto ? '#eff6ff' : '#f0fdf4') : '#f8fafc',
                      overflow: 'hidden',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px' }}>
                      <span style={{
                        width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 11, fontWeight: 700,
                        background: isDone ? color : isActive ? color : '#e2e8f0',
                        color: isDone || isActive ? '#fff' : '#94a3b8',
                      }}>
                        {isDone ? '✓' : i + 1}
                      </span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: '#334155', display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
                          {meta.label}{autoDetectBadge(isAuto)}
                        </div>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 1 }}>
                          {isDone ? `px (${pt.px.x}, ${pt.px.y})` : meta.hint}
                        </div>
                      </div>
                      {isDone && !isActive && (
                        <button
                          type="button"
                          onClick={() => { setCalibMode(i); setBoxMode(null); setIsEyedropperMode(false); }}
                          style={{
                            padding: '3px 8px', fontSize: 10, borderRadius: 4, fontWeight: isAuto ? 700 : 400,
                            border: `1px solid ${isAuto ? '#93c5fd' : '#e2e8f0'}`,
                            background: isAuto ? '#eff6ff' : '#fff',
                            color: isAuto ? '#2563eb' : '#475569',
                            cursor: 'pointer', flexShrink: 0,
                          }}
                        >수정</button>
                      )}
                    </div>

                    {!isDone && !isActive && (
                      <div style={{ padding: '0 10px 8px' }}>
                        <button
                          type="button"
                          onClick={() => { setCalibMode(i); setBoxMode(null); setIsEyedropperMode(false); }}
                          disabled={!imageFile}
                          style={{
                            width: '100%', padding: '6px 0', fontSize: 11, fontWeight: 600,
                            borderRadius: 6, border: `1px solid ${color}`,
                            background: '#fff', color, cursor: 'pointer', opacity: imageFile ? 1 : 0.5,
                          }}
                        >클릭하여 지정</button>
                      </div>
                    )}

                    {isActive && (
                      <div style={{ padding: '0 10px 8px' }}>
                        <div style={{ padding: '6px 8px', borderRadius: 5, background: `${color}20`, fontSize: 11, color, fontWeight: 500, lineHeight: 1.5, marginBottom: 4 }}>
                          📍 {isAuto ? '자동 감지 위치를 수정합니다. ' : ''}{meta.guide}
                        </div>
                        <button
                          type="button"
                          onClick={() => setCalibMode(null)}
                          style={{ width: '100%', padding: '4px 0', fontSize: 10, borderRadius: 5, border: '1px solid #e2e8f0', background: '#fff', color: '#94a3b8', cursor: 'pointer' }}
                        >취소</button>
                      </div>
                    )}

                    {isDone && (
                      <div style={{ padding: '0 10px 8px' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ fontSize: 11, color: '#64748b', width: 30, flexShrink: 0 }}>{meta.valueLabel} =</span>
                          <input
                            type="number"
                            className="xrd-settings-input"
                            style={{ flex: 1, padding: '4px 7px', fontSize: 11, color: '#0f172a', fontWeight: 600 }}
                            value={pt.val}
                            onChange={e => setCalibPts(prev => prev.map((p, j) => j === i ? { ...p, val: e.target.value } : p))}
                            placeholder={`${meta.valueLabel} 값`}
                            step="any"
                          />
                        </label>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {calibValid && (
              <div style={{ marginTop: 6, padding: '5px 8px', borderRadius: 5, fontSize: 11, background: '#f0fdf4', border: '1px solid #86efac', color: '#166534' }}>
                ✓ 캘리브레이션 완료 (x축 2점 + y축 2점)
              </div>
            )}
          </AccordionSection>

          {/* 곡선 색상 */}
          <AccordionSection title="③ 곡선 색상" open={openSections.color} onToggle={() => toggleSection('color')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <div style={{ width: 20, height: 20, borderRadius: 4, flexShrink: 0, border: '1px solid #e2e8f0', background: colorRgb ? colorHex : '#e2e8f0' }} />
              {colorRgb ? (
                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.4 }}>
                  <div>{colorHex}</div>
                  <div style={{ color: '#94a3b8' }}>rgb({colorRgb.r}, {colorRgb.g}, {colorRgb.b})</div>
                </div>
              ) : (
                <div style={{ fontSize: 12, color: '#94a3b8' }}>색상 미선택</div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              <button
                type="button"
                className="xrd-settings-sample-btn"
                style={{
                  flex: 1, fontSize: 12, padding: '8px 0',
                  background: isDetectingColor ? '#e0e7ff' : '#2563eb',
                  color: isDetectingColor ? '#3730a3' : '#fff',
                  border: 'none',
                }}
                onClick={handleAutoDetectColor}
                disabled={!imageFile || isDetectingColor}
              >
                {isDetectingColor
                  ? <>처리 중<span style={{ animation: 'dotBlink 1.2s 0.0s ease-in-out infinite' }}>.</span><span style={{ animation: 'dotBlink 1.2s 0.4s ease-in-out infinite' }}>.</span><span style={{ animation: 'dotBlink 1.2s 0.8s ease-in-out infinite' }}>.</span></>
                  : '🎨 자동 감지'}
              </button>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                type="button"
                className="xrd-settings-sample-btn"
                style={{
                  flex: 1, fontSize: 12, padding: '8px 0',
                  background: isEyedropperMode ? '#dbeafe' : '#64748b',
                  color: isEyedropperMode ? '#2563eb' : '#fff',
                  border: isEyedropperMode ? '1px solid #93c5fd' : 'none',
                }}
                onClick={() => {
                  if (imageFile) {
                    setIsEyedropperMode(v => !v);
                    setCalibMode(null);
                    setBoxMode(null);
                  }
                }}
                disabled={!imageFile}
              >
                👁 직접 찍기
              </button>
              <button
                type="button"
                className="xrd-settings-sample-btn"
                style={{ flex: 1, fontSize: 12, padding: '8px 0', background: '#64748b' }}
                onClick={() => colorInputRef.current?.click()}
              >
                직접 선택
              </button>
            </div>
            {isEyedropperMode && (
              <div style={{ marginTop: 8, padding: '4px 8px', borderRadius: 4, background: '#eff6ff', fontSize: 11, color: '#2563eb' }}>
                이미지에서 곡선 위를 클릭하세요
              </div>
            )}
            {colorPt && (
              <div style={{ marginTop: 4, fontSize: 10, color: '#94a3b8' }}>
                색상 좌표: ({Math.round(colorPt.x)}, {Math.round(colorPt.y)})
              </div>
            )}
          </AccordionSection>
        </div>

        {/* 실행 */}
        <div className="xrd-settings-actions">
          <button type="button" className="xrd-settings-run-btn" disabled={!canRun || isLoading} onClick={handleRun}>
            {isLoading
              ? <>처리 중<span style={{ animation: 'dotBlink 1.2s 0.0s ease-in-out infinite' }}>.</span><span style={{ animation: 'dotBlink 1.2s 0.4s ease-in-out infinite' }}>.</span><span style={{ animation: 'dotBlink 1.2s 0.8s ease-in-out infinite' }}>.</span></>
              : '디지타이즈 실행'}
          </button>
          {!canRun && !isLoading && (
            <p style={{ margin: '8px 0 0', fontSize: 11, color: '#94a3b8', textAlign: 'center' }}>
              {!imageFile ? '이미지 업로드 필요'
                : !boxValid ? '영역 설정 (3점) 필요'
                : !calibValid ? '캘리브레이션 (4점) 필요'
                : '색상 선택 필요'}
            </p>
          )}
          {runError && (
            <div style={{ marginTop: 8, padding: '8px 10px', borderRadius: 6, background: '#fef2f2', border: '1px solid #fca5a5', fontSize: 12, color: '#dc2626' }}>
              {runError}
            </div>
          )}
        </div>
      </div>
    );

    return () => setToolbarContent(null);
  }, [
    setToolbarContent, mode, setMode,
    imageFile, openSections,
    boxCorners, boxMode, boxValid,
    calibPts, calibMode, calibValid,
    colorRgb, colorPt, isEyedropperMode,
    isLoading, runError, canRun,
    isDetecting, isDetectingColor, detectResult,
    handleRun, handleAutoDetect, handleAutoDetectColor, toggleSection,
  ]);

  // ── 렌더 ──────────────────────────────────────────────────────────────────
  const colorHex = colorRgb ? rgbToHex(colorRgb.r, colorRgb.g, colorRgb.b) : '#000000';

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0 }}>
      <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileInputChange} />
      <input ref={replaceInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileInputChange} />
      <input ref={colorInputRef} type="color" value={colorHex} style={{ display: 'none' }} onChange={handleColorPickerChange} />

      {!imageFile ? (
        <div
          className={`xrd-settings-dropzone${isDragOver ? ' xrd-settings-dropzone--active' : ''}`}
          style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', margin: 24, borderRadius: 12, cursor: 'pointer' }}
          onDrop={handleDrop}
          onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="xrd-settings-dropzone-icon">
            <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M3 15l5-5 4 4 3-3 6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <p className="xrd-settings-dropzone-main">XRD 패턴 이미지 업로드</p>
          <p className="xrd-settings-dropzone-hint">PNG / JPG / TIFF — 드래그하거나 클릭해서 선택</p>
        </div>
      ) : (
        <div ref={imgContainerRef} style={{ flex: 1, position: 'relative', overflow: 'hidden', background: '#f1f5f9' }}>
          <img
            ref={imgRef}
            src={imageUrl}
            alt="XRD pattern"
            onLoad={handleImageLoad}
            crossOrigin="anonymous"
            style={{ position: 'absolute', display: 'block', pointerEvents: 'none', userSelect: 'none' }}
          />
          <canvas
            ref={canvasRef}
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onClick={handleCanvasClick}
          />
          {overlayData && (
            <button
              onClick={() => setShowOverlay(v => !v)}
              style={{
                position: 'absolute', top: 8, right: 8, zIndex: 10,
                padding: '4px 10px', fontSize: 11, fontWeight: 600,
                borderRadius: 6, border: '1px solid rgba(239,68,68,0.35)',
                background: showOverlay ? 'rgba(239,68,68,0.12)' : 'rgba(255,255,255,0.9)',
                color: showOverlay ? '#dc2626' : '#64748b',
                cursor: 'pointer', backdropFilter: 'blur(4px)',
              }}
            >
              {showOverlay ? '곡선 숨기기' : '곡선 보기'}
            </button>
          )}
          <div style={{ position: 'absolute', bottom: 10, right: 10, zIndex: 10, display: 'flex', alignItems: 'center', gap: 3 }}>
            {[
              { label: '−', onClick: () => adjustZoom(-1) },
              null,
              { label: '+', onClick: () => adjustZoom(1) },
            ].map((item, idx) => item ? (
              <button key={idx} onClick={item.onClick} style={{
                width: 26, height: 26, padding: 0, fontSize: 15, fontWeight: 700,
                borderRadius: 6, border: '1px solid #e2e8f0',
                background: 'rgba(255,255,255,0.92)', color: '#334155',
                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                backdropFilter: 'blur(4px)',
              }}>{item.label}</button>
            ) : (
              <span key={idx} style={{
                fontSize: 10, fontWeight: 600, padding: '0 5px',
                background: 'rgba(255,255,255,0.92)', border: '1px solid #e2e8f0',
                borderRadius: 5, color: '#475569', minWidth: 38, textAlign: 'center',
                lineHeight: '24px', backdropFilter: 'blur(4px)',
              }}>
                {Math.round(zoomDisplay * 100)}%
              </span>
            ))}
            {zoomDisplay !== 1 && (
              <button onClick={resetZoom} style={{
                width: 26, height: 26, padding: 0, fontSize: 13,
                borderRadius: 6, border: '1px solid #e2e8f0',
                background: 'rgba(255,255,255,0.92)', color: '#334155',
                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                backdropFilter: 'blur(4px)',
              }}>↺</button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
