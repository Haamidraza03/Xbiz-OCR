import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import AppNavbar from './components/Navbar';
import DeploymentForm from './components/DeploymentForm';
import DeploymentHistory from './components/DeploymentHistory';

const App = () => {
  return (
    <Router>
      <AppNavbar />
      <main className="container" style={{ marginTop: '100px' }}>
        <Routes>
          <Route path="/" element={<Navigate to="/deployment-form" />} />
          <Route path="/deployment-form" element={<DeploymentForm />} />
          <Route path="/deployment-history" element={<DeploymentHistory />} />
        </Routes>
      </main>
      <footer className="bg-light py-4 mt-5">
        <div className="container text-center">
          <p className="text-muted mb-0">Copyright © 2024 Deployment Tracker - Built with React & Bootstrap</p>
        </div>
      </footer>
    </Router>
  );
};

export default App;