"""
LLM 数据生成器 - 使用大语言模型生成合成数据

支持多种 LLM API：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (本地模型)
- OpenAI 兼容 API

功能：
- 文本数据生成（IMDB 影评）
- 表格数据生成（SmartFactory, Adult）
- 基于种子样本生成相似样本
"""

import os
import json
import re
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .data_generator import DataGenerator


# ─────────────────────────────────────────────────────────────────────────────
# LLM API 抽象层
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """LLM 后端抽象类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """调用 LLM 生成文本"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查 API 是否可用"""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API 后端"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo",
                 base_url: str = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        from openai import OpenAI
        client = self._client

        messages = [{"role": "user", "content": prompt}]
        if "system" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system"]})

        response = client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            top_p=kwargs.get("top_p", 1.0),
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API 后端"""

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307",
                 timeout: int = 60):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.timeout = timeout
        self._client = None

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        response = self._client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.content[0].text


class OllamaBackend(LLMBackend):
    """Ollama 本地模型后端"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "stream": False,
            },
            timeout=kwargs.get("timeout", 120)
        )
        data = response.json()
        return data.get("response", "")


class MockLLMBackend(LLMBackend):
    """模拟 LLM 后端（用于测试）"""

    def __init__(self, response_template: str = None):
        self.response_template = response_template or "这是生成的模拟数据。"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        # 解析 prompt 中的关键词来生成适当的响应
        if "sentiment" in prompt.lower() or "review" in prompt.lower():
            sentiments = ["positive", "negative"]
            return np.random.choice(sentiments)
        return self.response_template


# ─────────────────────────────────────────────────────────────────────────────
# LLM 数据生成器
# ─────────────────────────────────────────────────────────────────────────────

class LLMDataGenerator(DataGenerator):
    """
    使用 LLM 生成合成数据

    支持：
    - 文本数据生成（IMDB 影评）
    - 表格数据生成（SmartFactory, Adult）
    - 基于种子样本的条件生成
    """

    # 文本数据集的提示词模板
    TEXT_PROMPTS = {
        "imdb": {
            "system": "你是一个数据生成助手，擅长生成电影评论数据。",
            "generate_single": """生成一条电影评论及其情感标签。

要求：
1. 评论内容要真实自然，长度 50-200 词
2. 情感标签只能是 "positive" 或 "negative"
3. 评论应该包含具体的电影相关话题

输出格式（严格 JSON）：
{{"review": "评论内容", "sentiment": "positive/negative"}}

示例输出：
{{"review": "这部电影真是太精彩了，演员表演入木三分，剧情紧凑引人入胜。", "sentiment": "positive"}}

请生成：""",
            "generate_batch": """批量生成 {n} 条电影评论及其情感标签。

要求：
1. 每条评论长度 30-150 词
2. 情感标签只能是 "positive" 或 "negative"
3. 评论要多样化，包含不同电影类型

输出格式（严格 JSON 数组）：
[{{"review": "评论1", "sentiment": "positive"}},
 {{"review": "评论2", "sentiment": "negative"}}]

请生成 {n} 条：""",
            "generate_from_seed": """基于以下种子评论生成一条新的相似评论。

种子评论：{seed_review}
情感标签：{seed_label}

要求：
1. 生成相似但不完全相同的评论
2. 保持相同的情感倾向
3. 长度 50-200 词

输出格式（严格 JSON）：
{{"review": "新评论内容", "sentiment": "{seed_label}"}}

请生成："""
        },
        "yelp": {
            "system": "你是一个数据生成助手，擅长生成评论数据。",
            "generate_single": """生成一条评论及其情感标签。

输出格式（严格 JSON）：
{{"review": "评论内容", "sentiment": "positive/negative"}}""",
            "generate_batch": """批量生成 {n} 条评论及标签。

输出格式（严格 JSON 数组）："""
        }
    }

    # 表格数据集的提示词模板
    TABLE_PROMPTS = {
        "adult": {
            "system": "你是一个数据生成助手，擅长生成人口统计表格数据。",
            "generate_single": """生成一条人口统计数据。

特征列：age, education, occupation, hours_per_week, income
income 只能是 ">50K" 或 "<=50K"

输出格式（严格 JSON）：
{{"age": 35, "education": "Bachelors", "occupation": "Tech", "hours_per_week": 40, "income": ">50K"}}""",
            "generate_from_seed": """基于以下种子数据生成一条新数据，保持相似特征分布。

种子数据：{seed}

要求：
1. 所有数值在合理范围内变化（±20%）
2. 标签保持不变

输出格式（严格 JSON）："""
        },
        "smartfactory": {
            "system": "你是一个数据生成助手，擅长生成工业传感器数据。",
            "generate_single": """生成一条工业传感器数据。

特征列：i_w_blo_weg, o_w_blo_power, o_w_blo_voltage, i_w_bhl_weg, o_w_bhl_power, o_w_bhl_voltage
标签：0 或 1

输出格式（严格 JSON）：
{{"i_w_blo_weg": 100.5, "o_w_blo_power": 200.3, "o_w_blo_voltage": 220.1, "i_w_bhl_weg": 50.2, "o_w_bhl_power": 100.1, "o_w_bhl_voltage": 110.5, "labels": 1}}""",
            "generate_from_seed": """基于以下种子数据生成一条新数据。

种子数据：{seed}

要求：
1. 数值在合理范围内变化（±15%）
2. 标签保持不变

输出格式（严格 JSON）："""
        }
    }

    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)

        # LLM 配置
        self.llm_provider = getattr(config, "llm_provider", "openai")  # openai, anthropic, ollama, mock
        self.llm_model = getattr(config, "llm_model", "gpt-3.5-turbo")
        self.llm_api_key = getattr(config, "llm_api_key", None)
        self.llm_base_url = getattr(config, "llm_base_url", None)
        self.llm_temperature = getattr(config, "llm_temperature", 0.7)
        self.llm_max_tokens = getattr(config, "llm_max_tokens", 1000)
        self.llm_timeout = getattr(config, "llm_timeout", 60)

        # 数据集类型
        self.dataset_name = getattr(config, "dataset_name", "imdb")
        self.feature_cols = config.feature_cols
        self.label_col = config.label_col

        # 缓存和批处理
        self._cache: List[Dict] = []
        self._cache_size = getattr(config, "llm_cache_size", 50)
        self._batch_size = getattr(config, "llm_batch_size", 10)
        self._use_cache = getattr(config, "llm_use_cache", True)

        # 初始化后端
        self._backend = self._init_backend()

    def _init_backend(self) -> LLMBackend:
        """初始化 LLM 后端"""
        backends = {
            "openai": OpenAIBackend,
            "anthropic": AnthropicBackend,
            "ollama": OllamaBackend,
            "mock": MockLLMBackend,
        }

        backend_class = backends.get(self.llm_provider, MockLLMBackend)

        if self.llm_provider == "openai":
            return backend_class(
                api_key=self.llm_api_key,
                model=self.llm_model,
                base_url=self.llm_base_url,
                timeout=self.llm_timeout
            )
        elif self.llm_provider == "anthropic":
            return backend_class(
                api_key=self.llm_api_key,
                model=self.llm_model,
                timeout=self.llm_timeout
            )
        elif self.llm_provider == "ollama":
            return backend_class(
                model=self.llm_model,
                base_url=self.llm_base_url or "http://localhost:11434"
            )
        else:
            return backend_class()

    def _ensure_backend(self):
        """确保后端可用，必要时回退到 mock"""
        if not self._backend.is_available():
            print(f"[LLMDataGenerator] {self.llm_provider} 不可用，回退到 mock 模式")
            self._backend = MockLLMBackend()

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """解析 JSON 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试 JSON 数组
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                arr = json.loads(json_match.group())
                if arr and len(arr) > 0:
                    return arr[0]
            except json.JSONDecodeError:
                pass

        return None

    def _get_prompt_template(self, prompt_type: str = "generate_single") -> str:
        """获取对应数据集的提示词模板"""
        # 先检查表格数据
        if self.dataset_name in self.TABLE_PROMPTS:
            return self.TABLE_PROMPTS[self.dataset_name].get(prompt_type, "")

        # 再检查文本数据
        if self.dataset_name in self.TEXT_PROMPTS:
            return self.TEXT_PROMPTS[self.dataset_name].get(prompt_type, "")

        # 默认使用 IMDB
        return self.TEXT_PROMPTS["imdb"].get(prompt_type, "")

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        if self.dataset_name in self.TABLE_PROMPTS:
            return self.TABLE_PROMPTS[self.dataset_name].get("system", "")
        if self.dataset_name in self.TEXT_PROMPTS:
            return self.TEXT_PROMPTS[self.dataset_name].get("system", "")
        return ""

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 LLM 生成合成样本

        如果是文本数据，返回向量化的特征
        如果是表格数据，返回原始特征
        """
        self._ensure_backend()

        # 检查缓存
        if self._use_cache and len(self._cache) >= n_samples:
            cached = self._cache[:n_samples]
            self._cache = self._cache[n_samples:]
            return self._process_cached(cached)

        # 生成新数据
        new_samples = self._generate_samples(n_samples, X, y)

        # 更新缓存
        if self._use_cache:
            self._cache.extend(new_samples)

        return self._process_samples(new_samples)

    def _generate_samples(
        self,
        n_samples: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Dict]:
        """生成新样本"""
        samples = []

        # 批量生成
        for _ in range(0, n_samples, self._batch_size):
            batch_n = min(self._batch_size, n_samples - len(samples))

            if len(X) > 0:
                # 基于种子生成
                batch = self._generate_from_seeds(X, y, batch_n)
            else:
                # 全新生成
                batch = self._generate_new(batch_n)

            samples.extend(batch)

        return samples[:n_samples]

    def _generate_new(self, n: int) -> List[Dict]:
        """全新生成 n 个样本"""
        prompt_template = self._get_prompt_template("generate_batch")
        prompt = prompt_template.format(n=n)

        try:
            response = self._backend.generate(
                prompt,
                system=self._get_system_prompt(),
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )

            # 解析 JSON 数组
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                samples = json.loads(json_match.group())
                return samples[:n]

        except Exception as e:
            print(f"[LLMDataGenerator] 生成失败: {e}")

        # 回退：生成模拟数据
        return self._generate_fallback(n)

    def _generate_from_seeds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n: int
    ) -> List[Dict]:
        """基于种子样本生成"""
        samples = []

        # 检查是否为文本数据
        is_text = self.dataset_name == "imdb" or (
            len(self.feature_cols) == 1 and "text" in self.feature_cols[0].lower()
        )

        for _ in range(n):
            # 随机选择一个种子
            idx = self.rng.integers(0, len(X))
            seed_sample = X[idx]
            seed_label = y[idx]

            if is_text and len(seed_sample) > 10:
                # 文本：使用文本种子
                try:
                    # 反向向量化为文本（如果可能）
                    seed_text = self._inverse_vectorize(seed_sample)
                except:
                    seed_text = " ".join([f"word{i}" for i in range(10)])

                prompt = self._get_prompt_template("generate_from_seed")
                prompt = prompt.format(
                    seed_review=seed_text[:200],
                    seed_label="positive" if seed_label == 1 else "negative"
                )
            else:
                # 表格数据
                seed_dict = self._format_seed_dict(seed_sample, seed_label)
                prompt = self._get_prompt_template("generate_from_seed")
                prompt = prompt.format(seed=seed_dict)

            try:
                response = self._backend.generate(
                    prompt,
                    system=self._get_system_prompt(),
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens
                )

                sample = self._parse_json_response(response)
                if sample:
                    samples.append(sample)
                    continue

            except Exception as e:
                print(f"[LLMDataGenerator] 种子生成失败: {e}")

            # 回退
            samples.append(self._generate_fallback(1)[0])

        return samples

    def _generate_fallback(self, n: int) -> List[Dict]:
        """生成回退数据（当 LLM 不可用时）"""
        samples = []

        is_text = self.dataset_name == "imdb"

        for _ in range(n):
            if is_text:
                sample = {
                    "review": f"这是生成的模拟评论 {self.rng.random():.4f}",
                    "sentiment": self.rng.choice(["positive", "negative"])
                }
            elif self.dataset_name == "adult":
                sample = {
                    "age": self.rng.integers(20, 60),
                    "education": self.rng.choice(["Bachelors", "Masters", "HS-grad"]),
                    "occupation": self.rng.choice(["Tech", "Sales", "Admin"]),
                    "hours_per_week": self.rng.integers(20, 60),
                    "income": self.rng.choice([">50K", "<=50K"])
                }
            elif self.dataset_name == "smartfactory":
                sample = {
                    "i_w_blo_weg": self.rng.uniform(50, 150),
                    "o_w_blo_power": self.rng.uniform(100, 300),
                    "o_w_blo_voltage": self.rng.uniform(200, 240),
                    "i_w_bhl_weg": self.rng.uniform(30, 80),
                    "o_w_bhl_power": self.rng.uniform(50, 150),
                    "o_w_bhl_voltage": self.rng.uniform(100, 120),
                    "labels": self.rng.integers(0, 2)
                }
            else:
                # 通用回退
                sample = {"label": self.rng.integers(0, 2)}

            samples.append(sample)

        return samples

    def _inverse_vectorize(self, vector: np.ndarray) -> str:
        """将向量反转为文本（近似实现）"""
        # 这里只是一个简单的近似实现
        # 实际使用时可能需要更复杂的方法
        words = [f"word_{i}" for i in range(min(len(vector), 20))]
        return " ".join(words)

    def _format_seed_dict(self, X: np.ndarray, y: np.ndarray) -> str:
        """格式化种子样本为字典字符串"""
        data = {}

        for i, col in enumerate(self.feature_cols):
            if i < len(X):
                data[col] = float(X[i])

        data[self.label_col] = int(y) if isinstance(y, (int, float, np.number)) else y

        return json.dumps(data)

    def _process_cached(self, cached: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """处理缓存数据"""
        return self._process_samples(cached)

    def _process_samples(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """处理生成的样本，转换为特征和标签"""
        if not samples:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        for sample in samples:
            # 提取特征
            features = []
            for col in self.feature_cols:
                if col in sample:
                    val = sample[col]
                    if isinstance(val, str):
                        # 文本特征需要向量化
                        features.append(hash(val) % 1000)
                    else:
                        features.append(float(val))
                else:
                    features.append(0.0)

            # 提取标签
            label_val = sample.get(self.label_col, sample.get("label", 0))
            if isinstance(label_val, str):
                if label_val in ["positive", ">50K"]:
                    label_val = 1
                else:
                    label_val = 0

            X_list.append(features)
            y_list.append(int(label_val))

        return np.array(X_list), np.array(y_list)


class LLMTextGenerator(LLMDataGenerator):
    """专门用于文本数据的 LLM 生成器"""

    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.vectorizer = None

        # 尝试导入文本向量化器
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=config.text_max_features,
                ngram_range=config.text_ngram_range
            )
        except ImportError:
            pass

    def _process_samples(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """处理文本样本"""
        if not samples:
            return np.array([]), np.array([])

        texts = []
        labels = []

        for sample in samples:
            # 提取文本
            text = sample.get("review", sample.get("text", ""))
            texts.append(text)

            # 提取标签
            label = sample.get("sentiment", sample.get("label", 0))
            if isinstance(label, str):
                labels.append(1 if label == "positive" else 0)
            else:
                labels.append(int(label))

        # 向量化
        if self.vectorizer and len(texts) > 0:
            try:
                X = self.vectorizer.fit_transform(texts).toarray()
            except:
                # 回退到简单哈希
                X = np.array([[hash(t) % 1000 for t in texts]]).T
        else:
            X = np.array([[hash(t) % 1000 for t in texts]])

        y = np.array(labels)

        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_generator(
    config,
    method: str = "llm",
    rng: Optional[np.random.Generator] = None,
) -> DataGenerator:
    """
    工厂函数：创建 LLM 数据生成器

    Parameters
    ----------
    config : Config
        配置对象
    method : str
        生成方法：
        - "llm": 通用 LLM 生成器
        - "llm_text": 文本专用 LLM 生成器
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    DataGenerator
        LLM 生成器实例
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    generators = {
        "llm": LLMDataGenerator,
        "llm_text": LLMTextGenerator,
    }

    generator_class = generators.get(method.lower(), LLMDataGenerator)
    return generator_class(config, rng)


# ============================================================
# 注册到全局注册表（支持 get_data_generator_from_config）
# ============================================================

def _register_llm_generators():
    """将 LLM 生成器注册到 DataGeneratorRegistry"""
    try:
        from . import DataGeneratorRegistry
        DataGeneratorRegistry._generators['llm'] = LLMDataGenerator
        DataGeneratorRegistry._generators['llm_text'] = LLMTextGenerator
    except ImportError:
        # 如果 __init__ 还未加载，忽略（后续会被调用）
        pass


# 自动注册
_register_llm_generators()
