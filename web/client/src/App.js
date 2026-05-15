import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToolbarContentProvider } from './context/ToolbarContentContext';
import Toolbar from './components/Toolbar';
import XRDAnalyzer from './pages/XRD/XRDAnalyzer';
import './App.css';

function AppContent() {
  return (
    <ToolbarContentProvider>
      <div className="frame-changable xrd-standalone">
        <div className="frame-changable-child xrd-main">
          <div id="main-contents-container">
            <div className="main-route-outlet">
              <Routes>
                <Route
                  path="/xrd"
                  element={
                    <div className="tools-collection-root box-col pd0 gap10">
                      <XRDAnalyzer />
                    </div>
                  }
                />
                <Route path="/" element={<Navigate to="/xrd" replace />} />
                <Route path="*" element={<Navigate to="/xrd" replace />} />
              </Routes>
            </div>
          </div>
        </div>
        <div className="frame-changable-child xrd-toolbar">
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
