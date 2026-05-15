import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToolbarContentProvider } from './context/ToolbarContentContext';
import XRDAnalyzer from './pages/XRD/XRDAnalyzer';
import './App.css';

export default function App() {
  return (
    <Router>
      <ToolbarContentProvider>
        <div className="app-layout">
          <header className="app-header">
            <span className="app-logo">XRD Analyzer</span>
          </header>
          <main className="app-main">
            <Routes>
              <Route path="/" element={<Navigate to="/xrd" replace />} />
              <Route path="/xrd" element={<XRDAnalyzer />} />
              <Route path="*" element={<Navigate to="/xrd" replace />} />
            </Routes>
          </main>
        </div>
      </ToolbarContentProvider>
    </Router>
  );
}
