import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer mt-auto py-3 bg-dark text-white">
      <div className="container">
        <div className="row">
          <div className="col-md-6">
            <h5>垃圾分类识别系统</h5>
            <p className="small">
              基于YOLOv5的垃圾智能分类识别系统，帮助用户准确识别垃圾类别，促进环保理念。
            </p>
          </div>
          <div className="col-md-3">
            <h5>实用链接</h5>
            <ul className="list-unstyled">
              <li><a href="/" className="text-decoration-none text-white-50">首页</a></li>
              <li><a href="/about" className="text-decoration-none text-white-50">关于项目</a></li>
              <li><a href="/detect" className="text-decoration-none text-white-50">开始识别</a></li>
            </ul>
          </div>
          <div className="col-md-3">
            <h5>联系我们</h5>
            <ul className="list-unstyled">
              <li><i className="bi bi-envelope me-2"></i>example@email.com</li>
              <li><i className="bi bi-github me-2"></i>GitHub</li>
              <li><i className="bi bi-geo-alt me-2"></i>中国</li>
            </ul>
          </div>
        </div>
        <hr />
        <div className="row">
          <div className="col-md-12 text-center">
            <p className="small mb-0">
              © {currentYear} 垃圾分类识别系统 | 保留所有权利
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 