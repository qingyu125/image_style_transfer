import sys
import os

# 优化设置：关闭警告和日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭oneDNN提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 减少TensorFlow日志
import tensorflow as tf
import tensorflow_hub as hub
from qtpy.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QLabel, QFileDialog, QWidget, QProgressBar, 
                            QMessageBox, QFrame)
from qtpy.QtGui import QPixmap, QImage, QColor, QPen, QBrush
from qtpy.QtCore import Qt, QThread, Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]

class StyleTransferThread(QThread):
    """风格迁移线程，避免UI卡顿"""
    progress_updated = Signal(int)
    transfer_completed = Signal(object)
    error_occurred = Signal(str)
    
    def __init__(self, model, content_path, style_path):
        super().__init__()
        self.model = model
        self.content_path = content_path
        self.style_path = style_path
        
    def run(self):
        try:
            # 加载内容图像
            self.progress_updated.emit(10)
            content_image = self.load_image(self.content_path)
            
            # 加载风格图像
            self.progress_updated.emit(30)
            style_image = self.load_image(self.style_path)
            
            # 执行风格迁移
            self.progress_updated.emit(50)
            stylized_image = self.model(tf.constant(content_image), tf.constant(style_image))[0]
            
            # 后处理结果
            self.progress_updated.emit(80)
            stylized_image = tf.squeeze(stylized_image)
            stylized_image = tf.cast(stylized_image * 255, tf.uint8).numpy()
            
            self.progress_updated.emit(100)
            self.transfer_completed.emit(stylized_image)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def load_image(self, image_path, max_dim=512):
        """加载并调整图像大小"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        
        img = tf.image.resize(img, new_shape)
        return img[tf.newaxis, :]

class StyleTransferApp(QMainWindow):
    """风格迁移主应用窗口"""
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("图像风格迁移系统")
        self.setMinimumSize(1000, 600)
        
        # 加载预训练模型
        self.statusBar().showMessage("正在加载模型...")
        self.model = None  # 初始化模型为None
        try:
            # 尝试从本地加载模型
            self.model = hub.load('./arbitrary-image-stylization-v1-tensorflow1-256-v2/')
            self.statusBar().showMessage("模型加载成功")
        except Exception as e:
            # 如果本地加载失败，尝试从网络加载
            try:
                self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            except Exception as e:
                self.statusBar().showMessage(f"模型加载失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"无法加载模型: {str(e)}")
        
        # 创建UI组件
        self.create_widgets()
        self.create_layouts()
        self.connect_signals()
        
    def create_widgets(self):
        """创建UI组件"""
        # 按钮
        self.content_btn = QPushButton("选择内容图像")
        self.style_btn = QPushButton("选择风格图像")
        self.transfer_btn = QPushButton("开始转换")
        self.transfer_btn.setEnabled(False)  # 初始禁用，直到选择了图像
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setEnabled(False)  # 初始禁用，直到生成了结果
        
        # 图像显示区域 - 设置黑色边框
        self.content_label = QLabel("内容图像")
        self.content_label.setAlignment(Qt.AlignCenter)
        self.content_label.setMinimumSize(300, 300)
        self.content_label.setStyleSheet("""
            border: 2px solid black;
            border-radius: 0px;
            background-color: white;
            padding: 5px;
        """)
        
        self.style_label = QLabel("风格图像")
        self.style_label.setAlignment(Qt.AlignCenter)
        self.style_label.setMinimumSize(300, 300)
        self.style_label.setStyleSheet("""
            border: 2px solid black;
            border-radius: 0px;
            background-color: white;
            padding: 5px;
        """)
        
        self.result_label = QLabel("转换结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(300, 300)
        self.result_label.setStyleSheet("""
            border: 2px solid black;
            border-radius: 0px;
            background-color: white;
            padding: 5px;
        """)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # 图像路径显示
        self.content_path_label = QLabel("未选择内容图像")
        self.style_path_label = QLabel("未选择风格图像")
        
        # 当前选择的图像路径
        self.content_path = None
        self.style_path = None
        self.result_image = None
    
    def create_layouts(self):
        """创建布局"""
        # 顶部按钮布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.content_btn)
        top_layout.addWidget(self.content_path_label)
        top_layout.addWidget(self.style_btn)
        top_layout.addWidget(self.style_path_label)
        top_layout.addWidget(self.transfer_btn)
        top_layout.addWidget(self.save_btn)
        
        # 图像显示布局
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.content_label)
        images_layout.addWidget(self.style_label)
        images_layout.addWidget(self.result_label)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.progress_bar)
        
        # 设置中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def connect_signals(self):
        """连接信号和槽"""
        self.content_btn.clicked.connect(self.select_content_image)
        self.style_btn.clicked.connect(self.select_style_image)
        self.transfer_btn.clicked.connect(self.start_transfer)
        self.save_btn.clicked.connect(self.save_result)
    
    def select_content_image(self):
        """选择内容图像"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择内容图像", "", "图像文件 (*.png *.jpg *.jpeg)"
        )
        if path:
            self.content_path = path
            self.content_path_label.setText(os.path.basename(path))
            self.display_image(self.content_label, path)
            self.update_transfer_button_state()
    
    def select_style_image(self):
        """选择风格图像"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择风格图像", "", "图像文件 (*.png *.jpg *.jpeg)"
        )
        if path:
            self.style_path = path
            self.style_path_label.setText(os.path.basename(path))
            self.display_image(self.style_label, path)
            self.update_transfer_button_state()
    
    def update_transfer_button_state(self):
        """更新转换按钮状态"""
        # 检查模型是否为有效对象（非 None 且可调用）
        model_valid = self.model is not None and callable(self.model)
        content_valid = isinstance(self.content_path, str) and len(self.content_path) > 0
        style_valid = isinstance(self.style_path, str) and len(self.style_path) > 0
        enable_flag = model_valid and content_valid and style_valid
        self.transfer_btn.setEnabled(enable_flag)  # 确保 enable_flag 是布尔值
    
    def display_image(self, label, path):
        """在标签中显示图像"""
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    
    def start_transfer(self):
        """开始风格迁移"""
        if not self.model:
            QMessageBox.critical(self, "错误", "模型未加载，请检查网络连接或本地模型路径")
            return
        
        self.progress_bar.setValue(0)
        self.transfer_btn.setEnabled(False)
        self.statusBar().showMessage("正在进行风格迁移...")
        
        # 创建并启动转换线程
        self.transfer_thread = StyleTransferThread(
            self.model, self.content_path, self.style_path
        )
        self.transfer_thread.progress_updated.connect(self.update_progress)
        self.transfer_thread.transfer_completed.connect(self.on_transfer_completed)
        self.transfer_thread.error_occurred.connect(self.on_transfer_error)
        self.transfer_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def on_transfer_completed(self, result_image):
        """转换完成回调"""
        self.result_image = result_image
        self.display_numpy_image(self.result_label, result_image)
        self.transfer_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.statusBar().showMessage("风格迁移完成")
    
    def on_transfer_error(self, error_msg):
        """转换错误回调"""
        self.transfer_btn.setEnabled(True)
        self.statusBar().showMessage("风格迁移失败")
        QMessageBox.critical(self, "错误", f"风格迁移失败: {error_msg}")
    
    def display_numpy_image(self, label, image):
        """显示NumPy数组格式的图像"""
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            h, w, c = image.shape
            qimage = QImage(image.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
    
    def save_result(self):
        """保存结果图像"""
        if self.result_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果图像")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图像", "stylized_result.jpg", "图像文件 (*.png *.jpg *.jpeg)"
        )
        if path:
            try:
                # 确保图像值在0-255范围内
                img = np.clip(self.result_image, 0, 255).astype(np.uint8)
                plt.imsave(path, img)
                QMessageBox.information(self, "成功", f"结果图像已保存到: {path}")
                self.statusBar().showMessage(f"结果图像已保存到: {path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")
                self.statusBar().showMessage("保存图像失败")

def main():
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
