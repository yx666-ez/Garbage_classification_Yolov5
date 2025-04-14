import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { logout } from '../services/authService';

const Navbar = ({ user, setUser }) => {
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      setUser(null);
      navigate('/');
    } catch (error) {
      console.error('登出失败:', error);
    }
  };

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container">
        <Link className="navbar-brand fw-bold" to="/">
          <i className="bi bi-recycle me-2"></i>
          垃圾分类识别系统
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <Link className="nav-link" to="/">
                <i className="bi bi-house-door me-1"></i>首页
              </Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/detect">
                <i className="bi bi-camera me-1"></i>开始识别
              </Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/about">
                <i className="bi bi-info-circle me-1"></i>关于项目
              </Link>
            </li>
            {!user ? (
              <li className="nav-item">
                <Link className="nav-link" to="/login">
                  <i className="bi bi-box-arrow-in-right me-1"></i>登录/注册
                </Link>
              </li>
            ) : (
              <li className="nav-item dropdown">
                <a
                  className="nav-link dropdown-toggle"
                  href="#"
                  id="navbarDropdown"
                  role="button"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                >
                  <i className="bi bi-person-circle me-1"></i>
                  {user.username}
                </a>
                <ul className="dropdown-menu dropdown-menu-end" aria-labelledby="navbarDropdown">
                  <li>
                    <Link className="dropdown-item" to="/profile">
                      <i className="bi bi-person me-2"></i>个人信息
                    </Link>
                  </li>
                  <li>
                    <Link className="dropdown-item" to="/history">
                      <i className="bi bi-clock-history me-2"></i>历史记录
                    </Link>
                  </li>
                  {user.isAdmin && (
                    <li>
                      <Link className="dropdown-item" to="/admin">
                        <i className="bi bi-gear me-2"></i>管理后台
                      </Link>
                    </li>
                  )}
                  <li>
                    <hr className="dropdown-divider" />
                  </li>
                  <li>
                    <button className="dropdown-item" onClick={handleLogout}>
                      <i className="bi bi-box-arrow-right me-2"></i>退出登录
                    </button>
                  </li>
                </ul>
              </li>
            )}
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 