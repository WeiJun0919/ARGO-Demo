"""
数据生成器模块 - 可插拔的数据增强算法

使用方式：
    from env.data_generators import get_data_generator
    
    # 单个生成器
    generator = get_data_generator("smote", config=cfg, rng=rng)
    
    # 多个生成器（组合）
    from env.data_generators import MultiGenerator
    generator = MultiGenerator([
        ("smote", 0.7),   # 70% 使用 SMOTE
        ("llm", 0.3),     # 30% 使用 LLM
    ], config=cfg, rng=rng)
    
    # 从配置获取
    generator = get_data_generator_from_config(cfg, rng)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd


class BaseDataGenerator(ABC):
    """数据生成器基类"""
    
    def __init__(self, config, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
    
    @abstractmethod
    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成样本。

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)
            输入特征
        y : np.ndarray, shape (n,)
            标签
        n_samples : int
            需要生成的样本数量

        Returns
        -------
        X_syn : np.ndarray
            生成的特征
        y_syn : np.ndarray
            生成的标签
        """
        pass
    
    def reset(self):
        """重置生成器状态（可选实现）"""
        pass
    
    @property
    def name(self) -> str:
        """返回生成器名称"""
        return self.__class__.__name__


class DataGeneratorRegistry:
    """数据生成器注册表 - 工厂模式"""
    
    _generators = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册数据生成器"""
        def decorator(generator_class):
            cls._generators[name] = generator_class
            return generator_class
        return decorator
    
    @classmethod
    def get(cls, name: str, config, rng: np.random.Generator, **kwargs):
        """获取数据生成器实例"""
        if name not in cls._generators:
            raise ValueError(f"Unknown data generator: {name}. Available: {list(cls._generators.keys())}")
        return cls._generators[name](config, rng, **kwargs)
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """列出所有可用的生成器"""
        return list(cls._generators.keys())
    
    @classmethod
    def create_multi(cls, configs: List[Dict], config, rng: np.random.Generator):
        """
        创建多生成器组合
        
        Parameters
        ----------
        configs : List[Dict]
            [{"name": "smote", "weight": 0.7}, {"name": "llm", "weight": 0.3}]
        """
        generators = []
        for cfg in configs:
            name = cfg.get("name")
            weight = cfg.get("weight", 1.0)
            gen = cls.get(name, config, rng)
            generators.append((gen, weight))
        return MultiGenerator(generators, config, rng)


def get_data_generator(name: str, config, rng: np.random.Generator, **kwargs) -> BaseDataGenerator:
    """获取数据生成器（便捷函数）"""
    return DataGeneratorRegistry.get(name, config, rng, **kwargs)


def get_data_generator_from_config(cfg, rng: np.random.Generator) -> BaseDataGenerator:
    """
    从配置对象获取数据生成器
    
    支持两种配置方式：
    1. 单生成器：data_generator_method = "smote"
    2. 多生成器：data_generator_methods = [{"name": "smote", "weight": 0.7}, 
                                            {"name": "llm", "weight": 0.3}]
    """
    # 优先使用多生成器配置
    multi_methods = getattr(cfg, 'data_generator_methods', None)
    if multi_methods:
        return DataGeneratorRegistry.create_multi(multi_methods, cfg, rng)
    
    # 回退到单生成器
    generator_name = getattr(cfg, 'data_generator_method', 'smote')
    return DataGeneratorRegistry.get(generator_name, cfg, rng)


# ============================================================
# 多生成器组合
# ============================================================

class MultiGenerator(BaseDataGenerator):
    """多生成器组合 - 按权重分配生成任务"""
    
    def __init__(self, generators: List[Tuple[BaseDataGenerator, float]], config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.generators = generators
        total_weight = sum(w for _, w in generators)
        self.weights = [w / total_weight for _, w in generators]
    
    def generate(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        X_all = []
        y_all = []
        
        for generator, weight in self.generators:
            n_this = int(n_samples * weight)
            if n_this > 0:
                try:
                    X_gen, y_gen = generator.generate(X, y, n_this)
                    X_all.append(X_gen)
                    y_all.append(y_gen)
                except Exception as e:
                    print(f"[MultiGenerator] {generator.name} failed: {e}")
        
        if not X_all:
            return np.array([]), np.array([])
        
        return np.vstack(X_all), np.concatenate(y_all)


class SequentialGenerator(BaseDataGenerator):
    """顺序生成器 - 按顺序使用各个生成器"""
    
    def __init__(self, generators: List[BaseDataGenerator], config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.generators = generators
    
    def generate(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        X_all = []
        y_all = []
        
        n_per_gen = n_samples // len(self.generators)
        
        for i, generator in enumerate(self.generators):
            n_this = n_per_gen
            if i == len(self.generators) - 1:
                n_this = n_samples - len(np.concatenate(y_all))  # 剩余
            
            if n_this > 0:
                try:
                    X_gen, y_gen = generator.generate(X, y, n_this)
                    X_all.append(X_gen)
                    y_all.append(y_gen)
                except Exception as e:
                    print(f"[SequentialGenerator] {generator.name} failed: {e}")
        
        if not X_all:
            return np.array([]), np.array([])
        
        return np.vstack(X_all), np.concatenate(y_all)


# ============================================================
# 注册内置生成器
# ============================================================

from .data_generator import (
    DataGenerator,
    RandomInterpolationGenerator,
    SMOTEGenerator,
    ADASYNGenerator,
    BorderlineSMOTEGenerator,
    MixedGenerator,
    MixupGenerator,
    MixupSMOTEGenerator,
)

DataGeneratorRegistry._generators['random'] = RandomInterpolationGenerator
DataGeneratorRegistry._generators['smote'] = SMOTEGenerator
DataGeneratorRegistry._generators['adasyn'] = ADASYNGenerator
DataGeneratorRegistry._generators['borderline'] = BorderlineSMOTEGenerator
DataGeneratorRegistry._generators['mixed'] = MixedGenerator
DataGeneratorRegistry._generators['mixup'] = MixupGenerator
DataGeneratorRegistry._generators['mixup_smote'] = MixupSMOTEGenerator

# VAE 生成器
try:
    from .vae_generator import VAEGenerator
    DataGeneratorRegistry._generators['vae'] = VAEGenerator
except ImportError:
    pass

# GAN 生成器
try:
    from .gan_generator import GANGenerator
    DataGeneratorRegistry._generators['gan'] = GANGenerator
except ImportError:
    pass


# LLM 生成器注册（在 llm_generator.py 中会注册）
# from .llm_generator import LLMDataGenerator
# DataGeneratorRegistry._generators['llm'] = LLMDataGenerator


# 兼容旧接口
def create_generator(config, method: str = "smote", rng: Optional[np.random.Generator] = None):
    """旧接口兼容函数"""
    if rng is None:
        rng = np.random.default_rng(config.seed)
    return get_data_generator(method, config, rng)
