import { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/index';
import ReportsPage from './pages/reports';

function App() {
  return (
    <Router>
      <Suspense fallback={<div style={{ padding: '20px' }}>Loading...</div>}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/reports" element={<ReportsPage />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

export default App;
