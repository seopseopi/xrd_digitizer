import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToolbarContentProvider } from './context/ToolbarContentContext';
import Toolbar from './components/Toolbar';
import XRDAnalyzer from './pages/XRD/XRDAnalyzer';
import './App.css';

function AppContent() {
  return (
    <ToolbarContentProvider>
      <div className="frame-changable">
        {/* 메인 콘텐츠 */}
        <div className="frame-changable-child" style={{ flex: 1, minWidth: 0 }}>
          <div id="main-contents-container">
            <div className="main-route-outlet">
              <Routes>
                <Route path="/" element={<Navigate to="/xrd" replace />} />
                <Route path="/xrd" element={<XRDAnalyzer />} />
                <Route path="*" element={<Navigate to="/xrd" replace />} />
              </Routes>
            </div>
          </div>
        </div>
        {/* 오른쪽 툴바 (파일 업로드 / 설정 패널) */}
        <div className="frame-changable-child">
          <Toolbar />
        </div>
      </div>
    </ToolbarContentProvider>
  );
}

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
