"""
通用工具函数

提供跨数据类型（表格、文本、图像）的共享功能。
"""

import json
import numpy as np


class FourDecimalEncoder(json.JSONEncoder):
    """JSON encoder that formats all floats to 4 decimal places."""
    def encode(self, o):
        return super().encode(self._process(o))

    def _process(self, obj):
        if isinstance(obj, dict):
            return {k: self._process(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._process(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 4)
        return obj


def get_data_type(dataset_name):
    """
    根据数据集名称判断数据类型

    返回：
        'tabular': 表格数据 (adult, smartfactory)
        'text': 文本数据 (imdb)
        'image': 图像数据 (cifar10*, 图像相关)
    """
    dataset_name = dataset_name.lower()

    if 'cifar' in dataset_name or 'image' in dataset_name:
        return 'image'
    elif dataset_name == 'imdb' or 'text' in dataset_name:
        return 'text'
    else:
        return 'tabular'


def detection_threshold_to_fraction(threshold, default=0.6):
    """
    将检测阈值转为 [0, 1] 比例。

    - (0, 1]：视为比例（如 0.6 表示约 60% 分位）
    - (1, 100]：视为百分位（如 70 表示 70% 分位）
    """
    if threshold is None:
        return default
    t = float(threshold)
    if t <= 1.0:
        return float(np.clip(t, 0.0, 1.0))
    return float(np.clip(t / 100.0, 0.0, 1.0))


def get_class_ratio_str(labels, num_classes=None):
    """
    计算类别分布字符串

    参数：
        labels: 标签数组
        num_classes: 类别数（可选）

    返回：
        str: 如 "20.0%/30.0%/25.0%/25.0%"
    """
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    class_counts = np.bincount(labels.astype(int), minlength=num_classes)
    total = class_counts.sum()
    if total == 0:
        return ""

    class_pcts = class_counts / total * 100
    return "/".join([f"{p:.1f}%" for p in class_pcts])
