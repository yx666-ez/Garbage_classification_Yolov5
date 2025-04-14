import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// 组件导入
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ProtectedRoute from './components/ProtectedRoute';

// 页面导入
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DetectPage from './pages/DetectPage';
import HistoryPage from './pages/HistoryPage';
import ProfilePage from './pages/ProfilePage';
import AboutPage from './pages/AboutPage';
import AdminPage from './pages/AdminPage';

// 服务导入
import { getCurrentUser } from './services/authService';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkUser = async () => {
      try {
        const currentUser = await getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Failed to get current user:', error);
      } finally {
        setLoading(false);
      }
    };

    checkUser();
  }, []);

  if (loading) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ height: '100vh' }}>
        <div className="spinner-border text-success" role="status">
          <span className="visually-hidden">加载中...</span>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="d-flex flex-column min-vh-100">
        <Navbar user={user} setUser={setUser} />
        <main className="flex-grow-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={!user ? <LoginPage setUser={setUser} /> : <Navigate to="/" />} />
            <Route path="/register" element={!user ? <RegisterPage /> : <Navigate to="/" />} />
            <Route path="/detect" element={<DetectPage user={user} />} />
            <Route 
              path="/history" 
              element={
                <ProtectedRoute user={user}>
                  <HistoryPage user={user} />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/profile" 
              element={
                <ProtectedRoute user={user}>
                  <ProfilePage user={user} setUser={setUser} />
                </ProtectedRoute>
              } 
            />
            <Route path="/about" element={<AboutPage />} />
            <Route 
              path="/admin" 
              element={
                <ProtectedRoute user={user} requiredRole="admin">
                  <AdminPage user={user} />
                </ProtectedRoute>
              } 
            />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App; 