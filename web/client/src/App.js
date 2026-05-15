import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToolbarContentProvider } from './context/ToolbarContentContext';
import Toolbar from './components/Toolbar';
import XRDAnalyzer from './pages/XRD/XRDAnalyzer';
import './App.css';

function AppContent() {
  return (
    <ToolbarContentProvider>
      {/* xrd-standalone: 2-col (main + toolbar 300px) */}
      <div className="frame-changable xrd-standalone">
        <div className="frame-changable-child xrd-main">
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
