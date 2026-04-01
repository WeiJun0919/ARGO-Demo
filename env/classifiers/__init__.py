"""
图像分类器模块
"""

from env.classifiers.image_classifier import (
    ImageDownstreamClassifier,
    LightweightImageClassifier,
    create_image_classifier,
)

__all__ = [
    'ImageDownstreamClassifier',
    'LightweightImageClassifier',
    'create_image_classifier',
]
