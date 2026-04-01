"""
Environment 模块 - 可插拔的 PPO-HRL 数据清洗框架

主要子模块：
- noise_detectors: 噪声检测器（ED2-RPT, IDE, Oracle）
- data_generators: 数据生成器（SMOTE, ADASYN, Random, etc.）

使用示例：
    # 获取噪声检测器
    from env.noise_detectors import get_noise_detector_from_config
    from config import Config
    cfg = Config()
    detector = get_noise_detector_from_config(cfg)  # 自动使用 cfg.device

    # 或手动指定设备
    from env.noise_detectors import get_noise_detector
    detector = get_noise_detector("ed2_rpt", n_features=6, device="cuda")

    # 列出所有可用的检测器
    print(NoiseDetectorRegistry.list_detectors())  # ['ed2_rpt', 'ide', 'oracle']

    # 获取数据生成器
    from env.data_generators import get_data_generator, DataGeneratorRegistry
    generator = get_data_generator("smote", config=cfg, rng=rng)

    # 列出所有可用的生成器
    print(DataGeneratorRegistry.list_generators())  # ['random', 'smote', 'adasyn', 'borderline', 'mixed']
"""

# 导出主要类
from .noise_detectors import (
    BaseNoiseDetector,
    NoiseDetectorRegistry,
    get_noise_detector,
    get_noise_detector_from_config,
)

from .data_generators import (
    BaseDataGenerator,
    DataGeneratorRegistry,
    get_data_generator,
    get_data_generator_from_config,
)

from .text_vectorizer import TextVectorizer, create_text_vectorizer

# 图像相关模块
from .image_cleaning_env import ImageDataCleaningEnv

# 特征提取器
from .feature_extractors import ImageFeatureExtractor, create_feature_extractor

# 数据加载器
from .data_loaders import ImageDataLoader, NoisyImageDataset, load_cifar10, load_cifar100

# 分类器
from .classifiers import ImageDownstreamClassifier, create_image_classifier

__all__ = [
    # 噪声检测器
    'BaseNoiseDetector',
    'NoiseDetectorRegistry',
    'get_noise_detector',
    'get_noise_detector_from_config',
    # 数据生成器
    'BaseDataGenerator',
    'DataGeneratorRegistry',
    'get_data_generator',
    'get_data_generator_from_config',
    # 文本向量化
    'TextVectorizer',
    'create_text_vectorizer',
    # 图像相关
    'ImageDataCleaningEnv',
    'ImageFeatureExtractor',
    'create_feature_extractor',
    'ImageDataLoader',
    'NoisyImageDataset',
    'load_cifar10',
    'load_cifar100',
    'ImageDownstreamClassifier',
    'create_image_classifier',
]
