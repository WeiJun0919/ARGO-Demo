"""
CIFAR-10 图像数据集加载器 - 直接从 pickle 文件加载，不转 CSV

支持从原始 pickle 格式加载 CIFAR-10 数据：
- data_batch_1 ~ data_batch_5 (训练集)
- test_batch (测试集)
- batches.meta (元数据)

使用方式：
    from env.image_dataset_loader import CIFAR10Loader
    
    loader = CIFAR10Loader("datasets/cifar10")
    X_train, y_train = loader.load_train()  # (50000, 3072), (50000,)
    X_test, y_test = loader.load_test()      # (10000, 3072), (10000,)
    class_names = loader.get_class_names()   # ['airplane', 'automobile', ...]
"""

import os
import pickle
import numpy as np
from typing import Tuple, List, Optional


class CIFAR10Loader:
    """CIFAR-10 数据集加载器"""
    
    # CIFAR-10 类别名称
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(self, data_dir: str):
        """
        初始化加载器
        
        Args:
            data_dir: CIFAR-10 pickle 文件所在目录
        """
        self.data_dir = data_dir
    
    def _load_batch(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个 batch 文件
        
        Returns:
            images: (10000, 3072) uint8 数组，展开的像素值
            labels: (10000,) uint8 数组
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        images = data['data']  # (10000, 3072)
        labels = np.array(data['labels'])  # list -> numpy array
        
        return images, labels
    
    def load_train(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载所有训练数据
        
        Args:
            max_samples: 可选，最多加载多少样本（用于调试）
            
        Returns:
            X: (N, 3072) uint8 数组
            y: (N,) uint8 数组
        """
        all_images = []
        all_labels = []
        
        for i in range(1, 6):
            batch_path = os.path.join(self.data_dir, f'data_batch_{i}')
            images, labels = self._load_batch(batch_path)
            all_images.append(images)
            all_labels.append(labels)
        
        X = np.vstack(all_images)
        y = np.concatenate(all_labels)
        
        if max_samples is not None and len(X) > max_samples:
            idx = np.random.RandomState(42).permutation(len(X))[:max_samples]
            X = X[idx]
            y = y[idx]
        
        return X, y
    
    def load_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载测试数据
        
        Returns:
            X: (10000, 3072) uint8 数组
            y: (10000,) uint8 数组
        """
        test_path = os.path.join(self.data_dir, 'test_batch')
        return self._load_batch(test_path)
    
    def get_class_names(self) -> List[str]:
        """获取类别名称"""
        meta_path = os.path.join(self.data_dir, 'batches.meta')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f, encoding='latin1')
                return meta.get('label_names', self.CLASS_NAMES)
        return self.CLASS_NAMES
    
    @staticmethod
    def reshape_to_image(X: np.ndarray, channels: int = 3, height: int = 32, width: int = 32) -> np.ndarray:
        """
        将展开的像素数组reshape为图像
        
        Args:
            X: (N, 3072) 或 (N,) 单张图像
            channels: 通道数 (3 for RGB)
            height: 图像高度
            width: 图像宽度
            
        Returns:
            images: (N, channels, height, width) 或 (channels, height, width)
        """
        if X.ndim == 1:
            # 单张图像
            return X.reshape(channels, height, width)
        elif X.ndim == 2:
            # 多张图像
            return X.reshape(-1, channels, height, width)
        else:
            raise ValueError(f"不支持的数组维度: {X.ndim}")
    
    @staticmethod
    def to_rgb_image(X: np.ndarray) -> np.ndarray:
        """
        转换为 (N, height, width, 3) 格式用于可视化
        
        Args:
            X: (N, 3072) 或 (N, 3, 32, 32)
            
        Returns:
            images: (N, 32, 32, 3) uint8
        """
        if X.shape[1] == 3072:
            # 展开格式 -> (N, 3, 32, 32) -> (N, 32, 32, 3)
            X = X.reshape(-1, 3, 32, 32)
        
        # (N, 3, 32, 32) -> (N, 32, 32, 3)
        return X.transpose(0, 2, 3, 1)


def load_cifar10(data_dir: str, max_train_samples: Optional[int] = None) -> dict:
    """
    便捷函数：一次性加载 CIFAR-10 全部数据
    
    Args:
        data_dir: 数据目录
        max_train_samples: 最多训练样本数
        
    Returns:
        dict: 包含 train_X, train_y, test_X, test_y, class_names
    """
    loader = CIFAR10Loader(data_dir)
    
    train_X, train_y = loader.load_train(max_samples=max_train_samples)
    test_X, test_y = loader.load_test()
    class_names = loader.get_class_names()
    
    return {
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
        'class_names': class_names,
    }
