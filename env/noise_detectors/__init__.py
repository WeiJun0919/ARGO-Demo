"""
噪声检测器模块 - 可插拔的噪声检测算法

使用方式：
    from env.noise_detectors import get_noise_detector_from_config
    from config import Config
    cfg = Config()
    detector = get_noise_detector_from_config(cfg)  # 自动使用 cfg.device

    # 或手动指定设备
    from env.noise_detectors import get_noise_detector
    detector = get_noise_detector("ed2_rpt", n_features=6, device="cuda")

    # 通过名称获取检测器
    detector = get_noise_detector("ed2_rpt", n_features=6, device="cuda")
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseNoiseDetector(ABC):
    """噪声检测器基类"""
    
    @abstractmethod
    def detect(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        检测噪声样本/特征
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            
        Returns:
            noise_mask: 噪声掩码 (n_samples,) 或 (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def detect_and_correct(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测并纠正噪声
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            
        Returns:
            X_corrected: 纠正后的特征
            noise_mask: 噪声掩码
        """
        pass
    
    @abstractmethod
    def pretrain(self, X_clean: np.ndarray, y_clean: Optional[np.ndarray] = None, 
                 epochs: int = 50, **kwargs):
        """
        预训练检测器（使用干净数据）
        
        Args:
            X_clean: 干净特征
            y_clean: 干净标签
            epochs: 训练轮数
        """
        pass
    
    def finetune(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                 epochs: int = 10, **kwargs):
        """
        微调检测器（可选）
        """
        pass


class NoiseDetectorRegistry:
    """噪声检测器注册表"""
    
    _detectors = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册噪声检测器"""
        def decorator(detector_class):
            cls._detectors[name] = detector_class
            return detector_class
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """获取噪声检测器实例"""
        if name not in cls._detectors:
            raise ValueError(f"Unknown noise detector: {name}. Available: {list(cls._detectors.keys())}")
        return cls._detectors[name](**kwargs)
    
    @classmethod
    def list_detectors(cls):
        """列出所有可用的检测器"""
        return list(cls._detectors.keys())


def get_noise_detector(name: str, **kwargs) -> BaseNoiseDetector:
    """获取噪声检测器（便捷函数）"""
    return NoiseDetectorRegistry.get(name, **kwargs)


def get_noise_detector_from_config(cfg) -> BaseNoiseDetector:
    """从配置对象获取噪声检测器"""
    detector_name = getattr(cfg, 'noise_detector', 'oracle')
    
    if detector_name == 'oracle':
        return NoiseDetectorRegistry.get('oracle', 
            dirty_threshold=getattr(cfg, 'dirty_diff_threshold', 0.1))
    elif detector_name == 'ed2_rpt':
        return NoiseDetectorRegistry.get('ed2_rpt',
            n_features=cfg.n_features,
            hidden_dims=getattr(cfg, 'ed2_rpt_hidden_dims', [128, 64]),
            device=getattr(cfg, 'device', 'cpu'),
            noise_threshold=getattr(cfg, 'ed2_rpt_noise_threshold', 0.5),
            correction_scale=getattr(cfg, 'ed2_rpt_correction_scale', 0.3),
            random_state=cfg.seed,
        )
    elif detector_name == 'ide':
        return NoiseDetectorRegistry.get('ide',
            n_features=cfg.n_features,
            hidden_dims=getattr(cfg, 'ide_hidden_dims', [128, 64]),
            device=getattr(cfg, 'device', 'cpu'),
            noise_threshold=getattr(cfg, 'ide_noise_threshold', 0.5),
            random_state=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown noise detector: {detector_name}")


# 注册内置检测器
from .ed2_rpt_detector import ED2RPTDetector
from .ide_label_detector import IDELabelDetector, SimpleLabelNoiseDetector, TorchLabelNoiseDetector

NoiseDetectorRegistry._detectors['ed2_rpt'] = ED2RPTDetector
NoiseDetectorRegistry._detectors['ide'] = IDELabelDetector


class OracleNoiseDetector(BaseNoiseDetector):
    """Oracle 检测器 - 使用 ground truth（仅用于实验）"""
    
    def __init__(self, clean_reference: np.ndarray = None, 
                 dirty_threshold: float = 0.1, **kwargs):
        self.clean_reference = clean_reference
        self.dirty_threshold = dirty_threshold
    
    def set_clean_reference(self, clean_reference: np.ndarray):
        self.clean_reference = clean_reference
    
    def detect(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.clean_reference is None:
            return np.zeros(len(X), dtype=bool)
        
        noise_mask = np.abs(X - self.clean_reference) > self.dirty_threshold
        return noise_mask.any(axis=1)
    
    def detect_and_correct(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.clean_reference is None:
            return X, np.zeros_like(X, dtype=bool)
        
        noise_mask = np.abs(X - self.clean_reference) > self.dirty_threshold
        X_corrected = X.copy()
        X_corrected[noise_mask] = self.clean_reference[noise_mask]
        return X_corrected, noise_mask
    
    def pretrain(self, X_clean: np.ndarray, y_clean: Optional[np.ndarray] = None, 
                 epochs: int = 50, **kwargs):
        # Oracle 不需要预训练
        pass
    
    def finetune(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                 epochs: int = 10, **kwargs):
        pass


NoiseDetectorRegistry._detectors['oracle'] = OracleNoiseDetector
