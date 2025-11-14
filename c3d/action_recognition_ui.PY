import gradio as gr
import torch
import cv2
import numpy as np
from torchvision import transforms
import tempfile
import os
import sys

class action_recognition_ui:
    def __init__(self, model_path='c3d/ucf101_model.pth', class_names_path='c3d/data/labels.txt', c3d_model_path='c3d/C3D_model.py'):
        """
        初始化C3D动作识别UI
        
        Args:
            model_path: 模型权重文件路径
            class_names_path: 类别名称文件路径
            c3d_model_path: C3D模型文件路径，如果不在同一目录
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.c3d_model_path = c3d_model_path
        self.class_names = []
        self.model = None
        self.device = None
        
        self._setup()
    
    def _setup(self):
        """设置模型和配置"""
        # 首先尝试导入C3D模型
        self._import_c3d_model()
        
        # 加载类别名称
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: '{self.class_names_path}' not found. Using default class names.")
            self.class_names = ['class1', 'class2', 'class3'] # 备用类别名称
        
        self.num_classes = len(self.class_names)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self._load_model()
    
    def _import_c3d_model(self):
        """导入C3D模型"""
        try:
            # 首先尝试直接导入
            from c3d.C3D_model import C3D
            self.C3D = C3D
        except ImportError:
            try:
                # 如果失败，尝试将当前目录添加到路径
                import sys
                sys.path.append('.')
                from C3D_model import C3D
                self.C3D = C3D
            except ImportError:
                try:
                    # 如果指定了自定义路径，尝试从该路径导入
                    if self.c3d_model_path:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("C3D_model", self.c3d_model_path)
                        c3d_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(c3d_module)
                        self.C3D = c3d_module.C3D
                    else:
                        raise ImportError("C3D_model not found and no custom path provided")
                except Exception as e:
                    print(f"Error: Could not import C3D model. Please ensure 'C3D_model.py' exists and contains the C3D class.")
                    print(f"Error details: {e}")
                    # 在类方法中，使用return而不是exit()
                    raise ImportError("C3D model import failed")
    
    def _load_model(self):
        """加载C3D模型"""
        # 步骤1: 创建C3D模型实例
        self.model = self.C3D(num_classes=self.num_classes, pretrained=False)
        print("Model architecture created successfully.")
        
        # 步骤2: 加载模型权重
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model weights '{self.model_path}' loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the model weights: {e}")
            raise
        
        # 步骤3: 设置模型为评估模式并移动到设备
        self.model.to(self.device)
        self.model.eval()
        print("Model set to evaluation mode and moved to the selected device.")
    
    def preprocess_video(self, video_path):
        """
        预处理视频文件为C3D模型所需的输入张量
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            预处理后的张量
        """
        # 1. 定义与训练一致的参数
        clip_len = 16
        resize_height = 128
        resize_width = 171
        crop_size = 112
        # 归一化均值 (BGR顺序)
        mean = np.array([90.0, 98.0, 102.0], dtype=np.float32)
        
        # 2. 帧采样
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 处理帧数少于所需剪辑长度的视频
        if frame_count < clip_len:
            print(f"Warning: Video frame count ({frame_count}) is less than the required {clip_len} frames. The last frame will be repeated.")
            indices = np.arange(frame_count)
            indices = np.pad(indices, (0, clip_len - frame_count), 'edge')
        else:
            # 在视频中均匀采样帧
            indices = np.linspace(0, frame_count - 1, clip_len, dtype=int)
        
        # 3. 核心预处理循环
        buffer = np.empty((clip_len, crop_size, crop_size, 3), dtype=np.float32)
        
        for i, frame_index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                # 如果读取帧失败，使用前一帧作为备用
                if i > 0:
                    frame = buffer[i - 1]
                else:
                    cap.release()
                    raise IOError("Failed to read the first frame of the video.")
            
            # 3.1 变换: 先调整大小，然后中心裁剪
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            
            # 计算中心裁剪的起始点
            start_x = (resize_width - crop_size) // 2
            start_y = (resize_height - crop_size) // 2
            
            cropped_frame = resized_frame[start_y: start_y + crop_size, start_x: start_x + crop_size, :]
            
            # 3.2 归一化: 减去BGR均值
            normalized_frame = cropped_frame.astype(np.float32) - mean
            
            buffer[i] = normalized_frame
        
        cap.release()
        
        # 4. 转置维度从 (T, H, W, C) 到 (C, T, H, W)
        buffer = buffer.transpose((3, 0, 1, 2))
        
        # 5. 转换为张量
        return torch.from_numpy(buffer)
    
    def predict_video(self, video_path):
        """
        预测视频中的动作
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含预测结果的字典
        """
        if video_path is None:
            return {"Error": 1.0, "Info": "No video uploaded"}
        
        try:
            input_tensor = self.preprocess_video(video_path)
            
            if input_tensor is None:
                return {"Error": 1.0, "Info": "Could not process the video"}
            
            # 添加批次维度并将张量移动到目标设备
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                # 应用softmax获取概率
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                # 创建类别名称和置信度分数的字典
                confidences = {self.class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
            return confidences
        
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return {"Error": 1.0, "Info": str(e)}
    
    def create_interface(self, examples=None, share=False):
        """
        创建Gradio界面
        
        Args:
            examples: 示例视频路径列表
            share: 是否生成公共链接
            
        Returns:
            Gradio界面实例
        """
        title = "C3D Human Action Recognition"
        description = """
        This is a demo for human action recognition using a C3D model.
        Please upload a short video (e.g., in .mp4 format), and the model will predict the main action in the video.
        The model currently supports the following classes: {}
        """.format(", ".join(self.class_names))
        
        if examples is None:
            examples = []
        
        demo = gr.Interface(
            fn=self.predict_video,
            inputs=gr.Video(label="Upload Video"),
            outputs=gr.Label(num_top_classes=3, label="Prediction Results"),
            title=title,
            description=description,
            examples=examples
        )
        
        return demo
    
    def launch_ui(self, examples=None, share=False, **launch_kwargs):
        """
        启动UI
        
        Args:
            examples: 示例视频路径列表
            share: 是否生成公共链接
            **launch_kwargs: 传递给launch方法的其他参数
        """
        demo = self.create_interface(examples=examples, share=share)
        demo.launch(share=share, **launch_kwargs)
