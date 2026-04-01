"""
图像特征提取器模块
"""

from env.feature_extractors.image_feature_extractor import (
    ImageFeatureExtractor,
    MODEL_CONFIGS,
    create_feature_extractor,
)

__all__ = [
    'ImageFeatureExtractor',
    'MODEL_CONFIGS',
    'create_feature_extractor',
]
