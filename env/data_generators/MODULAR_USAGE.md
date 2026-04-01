# ============================================================
# 数据生成器模块化使用示例
# ============================================================

# 方式一：单生成器配置（兼容旧版本）
# ============================================================
# config.py 中设置：
data_generator_method = "smote"    # 可选: random, smote, adasyn, borderline, mixed, mixup, mixup_smote, llm, llm_text


# 方式二：多生成器组合配置（推荐）
# ============================================================
# config.py 中设置：
data_generator_methods = [
    {"name": "smote", "weight": 0.7},   # 70% 使用 SMOTE
    {"name": "llm", "weight": 0.3},     # 30% 使用 LLM
]

# 这样会自动创建 MultiGenerator，每次生成时：
# - 70% 的样本由 SMOTE 生成
# - 30% 的样本由 LLM 生成


# 方式三：顺序生成器
# ============================================================
# 如果需要顺序使用多个生成器，可以在代码中创建：
from env.data_generators import SequentialGenerator, get_data_generator

generator1 = get_data_generator("smote", config, rng)
generator2 = get_data_generator("llm", config, rng)
seq_generator = SequentialGenerator([generator1, generator2], config, rng)


# ============================================================
# 添加新的数据生成器步骤
# ============================================================

# 1. 在对应的文件中创建类，继承 BaseDataGenerator
# 例如：env/data_generators/my_generator.py

from env.data_generators import BaseDataGenerator
import numpy as np

class MyCustomGenerator(BaseDataGenerator):
    """自定义数据生成器"""
    
    def __init__(self, config, rng):
        super().__init__(config, rng)
        # 初始化参数
    
    def generate(self, X, y, n_samples):
        # 实现生成逻辑
        X_new = ...  # 你的生成逻辑
        y_new = ...
        return X_new, y_new

# 2. 注册到注册表（在文件末尾添加）
from env.data_generators import DataGeneratorRegistry

DataGeneratorRegistry._generators['my_generator'] = MyCustomGenerator

# 3. 在 config.py 中使用
data_generator_method = "my_generator"

# 或者多生成器：
data_generator_methods = [
    {"name": "smote", "weight": 0.5},
    {"name": "my_generator", "weight": 0.5},
]


# ============================================================
# 噪声检测器模块化使用示例
# ============================================================

# 方式一：单个检测器
noise_detector = "ed2_rpt"  # 可选: oracle, ed2_rpt, ide

# 方式二：多检测器组合（在代码中使用）
from env.noise_detectors import get_noise_detector

detector1 = get_noise_detector("ed2_rpt", n_features=6, device="cpu")
detector2 = get_noise_detector("ide", n_features=6, device="cpu")
# 可以组合使用多个检测器

# 添加新的噪声检测器步骤：
# 1. 继承 BaseNoiseDetector
# 2. 实现 detect 和 detect_and_correct 方法
# 3. 注册到 NoiseDetectorRegistry
