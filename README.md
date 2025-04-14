# YOLOv5垃圾分类系统

基于YOLOv5的垃圾分类识别系统，支持四类垃圾（可回收物、有害垃圾、厨余垃圾、其他垃圾）的自动识别与分类。本系统包含前后端应用，能够实现垃圾图像上传、自动分类、历史记录查询等功能。

## 项目结构

```
yolov5_Garbage_Classification/
├── datasets/                   # 训练和测试数据集
│   ├── wangy/                  # 本地数据集
│   └── coco128/                # COCO示例数据集
├── garbage_classification/     # 后端应用
│   └── backend/
│       ├── app.py              # Flask后端主程序
│       ├── static/             # 静态资源
│       ├── templates/          # HTML模板
│       ├── uploads/            # 上传的图片存储目录
│       └── instance/           # 实例目录，包含数据库
├── garbage_classification_frontend/  # 前端React应用
│   ├── public/                 # 公共资源
│   ├── src/                    # 源代码
│   └── package.json            # 前端依赖配置
├── yolov5-6.0/                 # YOLOv5模型代码
│   ├── detect.py               # 目标检测脚本
│   ├── train.py                # 模型训练脚本
│   ├── models/                 # 模型定义
│   ├── utils/                  # 工具类
│   └── data/                   # 数据配置
├── garbage.yaml                # 垃圾分类数据集配置
├── best.pt                     # 预训练模型权重
└── run_detect.bat              # Windows下运行检测的批处理脚本
```

## 功能特性

- **垃圾图像识别**：上传图片自动识别垃圾类别（可回收物、有害垃圾、厨余垃圾、其他垃圾）
- **用户管理**：支持用户注册、登录、个人资料管理
- **历史记录**：保存检测历史，可随时查看历史检测结果
- **管理后台**：管理员可查看系统日志、用户信息、使用统计
- **响应式界面**：适配不同设备的前端界面

## 技术栈

### 前端
- React.js
- React Router
- Bootstrap 5
- Axios

### 后端
- Flask
- SQLAlchemy
- Flask-CORS
- Werkzeug

### 模型
- YOLOv5
- PyTorch
- OpenCV

## 安装与使用

### 环境要求
- Python 3.8+
- Node.js 14+
- CUDA（可选，用于GPU加速）

### 后端安装

1. 安装Python依赖
```bash
cd yolov5-6.0
pip install -r requirements.txt
cd ../garbage_classification/backend
pip install flask flask-cors flask-sqlalchemy pillow
```

2. 启动后端服务
```bash
cd garbage_classification/backend
python app.py
```

### 前端安装

1. 安装前端依赖
```bash
cd garbage_classification_frontend
npm install
```

2. 启动前端开发服务器
```bash
npm start
```

3. 构建生产版本
```bash
npm run build
```

### 使用预训练模型检测

使用批处理脚本快速运行垃圾检测：

```bash
run_detect.bat
```

或直接运行检测脚本：

```bash
cd yolov5-6.0
python detect.py --weights best.pt --source <图片路径> --img 640 --conf 0.25
```

## 模型训练

1. 准备数据集，按照YOLO格式组织数据
2. 修改garbage.yaml配置文件
3. 执行训练：

```bash
cd yolov5-6.0
python train.py --img 640 --batch 16 --epochs 100 --data ../garbage.yaml --weights yolov5s.pt
```

## 默认账户

管理员账号：
- 用户名：admin
- 密码：admin123

## 数据库

系统使用SQLite数据库，包含以下表：
- User：用户信息
- Image：上传的图像
- SystemLog：系统日志
- DetectionHistory：检测历史

## 系统截图

（这里可以添加系统界面的截图）

## 贡献

欢迎提交Issue或Pull Request来完善系统功能。

## 许可证

本项目使用YOLOv5的许可证。 