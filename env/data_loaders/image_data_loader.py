"""
图像数据集加载器 - 支持标签噪声注入

支持数据集：
- CIFAR-10: 10 类，32x32 RGB 图像
- CIFAR-100: 100 类，32x32 RGB 图像
- 自定义图像目录

噪声类型：
- symmetric: 随机标签翻转（对称噪声）
- pairflip: 成对翻转（用于二分类）
- aggrevated: 累积噪声

使用方式：
    from env.data_loaders.image_data_loader import ImageDataLoader, NoisyImageDataset

    # 加载 CIFAR-10 并注入 20% 标签噪声
    loader = ImageDataLoader(dataset_name='cifar10', data_dir='./datasets')
    dataset = loader.load_dataset(label_noise_ratio=0.2, noise_type='symmetric')

    images = dataset.images        # numpy array
    noisy_labels = dataset.noisy_labels
    true_labels = dataset.true_labels
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, Literal, List
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


class NoisyImageDataset(Dataset):
    """
    带标签噪声的图像数据集

    支持：
    - 注入对称噪声（symmetric）
    - 注入成对翻转噪声（pairflip）
    - 预计算的图像特征存储
    """

    # CIFAR-10 类别名称
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # CIFAR-100 类别名称（粗类别）
    CIFAR100_COARSE_CLASSES = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers',
        'fruit_and_vegetables', 'household_electrical_devices',
        'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
        'large_omnivores_and_herbivores', 'medium_mammals',
        'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        transform=None,
        cache_features: bool = False,
        feature_extractor=None,
    ):
        """
        参数：
        - images: 图像数据 (N, H, W, C) 或 (N, C, H, W)
        - labels: 带噪声的标签 (N,)
        - true_labels: 真实标签 (N,)，如果为 None 则等于 labels
        - transform: 图像变换
        - cache_features: 是否缓存特征
        - feature_extractor: 特征提取器
        """
        self.images = images
        self.labels = labels.astype(np.int64)
        self.true_labels = true_labels if true_labels is not None else labels.astype(np.int64)
        self.transform = transform
        self.cache_features = cache_features
        self.feature_extractor = feature_extractor

        # 缓存的特征
        self._cached_features = None

        # 噪声信息
        self.label_noise_mask = (self.labels != self.true_labels)
        self.label_noise_ratio = self.label_noise_mask.mean()

        # 元数据
        self.num_classes = len(np.unique(self.true_labels))
        self.class_names = self._get_class_names()

    def _get_class_names(self) -> List[str]:
        """获取类别名称"""
        if self.num_classes == 10:
            return self.CIFAR10_CLASSES
        elif self.num_classes == 100:
            return self.CIFAR100_COARSE_CLASSES
        else:
            return [f"class_{i}" for i in range(self.num_classes)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        """
        返回：(image, noisy_label, true_label, index)
        """
        img = self.images[idx]

        # 转换为 PIL Image
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.ndim == 1:
                # 展平的像素数组
                img = img.reshape(3, 32, 32).transpose(1, 2, 0)
            if img.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx], self.true_labels[idx], idx

    def get_raw_image(self, idx: int) -> np.ndarray:
        """获取原始图像（numpy 格式）"""
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            return img
        elif isinstance(img, Image.Image):
            return np.array(img)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def inject_noise(
        self,
        noise_ratio: float,
        noise_type: str = 'symmetric',
        random_seed: Optional[int] = None,
    ):
        """
        注入标签噪声（就地修改）

        参数：
        - noise_ratio: 噪声比例 (0.0 ~ 1.0)
        - noise_type: 噪声类型 ('symmetric', 'pairflip', 'aggrevated')
        - random_seed: 随机种子
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n = len(self.labels)
        n_noisy = int(n * noise_ratio)
        indices = np.random.choice(n, n_noisy, replace=False)
        print(f"  [DEBUG inject_noise] 注前 labels 前5: {self.labels[:5].tolist()}, true_labels 前5: {self.true_labels[:5].tolist()}")

        if noise_type == 'symmetric':
            # 对称噪声：随机翻转到其他类别
            for idx in indices:
                current_label = int(self.labels[idx])
                other_classes = [c for c in range(self.num_classes) if c != current_label]
                self.labels[idx] = np.random.choice(other_classes)

        elif noise_type == 'pairflip':
            # 成对翻转：主要用于二分类
            # 0 <-> 1 互换
            if self.num_classes == 2:
                self.labels[indices] = 1 - self.labels[indices]
            else:
                # 对于多分类，随机选一个其他类
                for idx in indices:
                    current_label = int(self.labels[idx])
                    other_classes = [c for c in range(self.num_classes) if c != current_label]
                    self.labels[idx] = np.random.choice(other_classes)

        elif noise_type == 'aggrevated':
            # 累积噪声：基于置信度的翻转
            # 假设 indices 已经是通过某种方法确定的高噪声样本
            for idx in indices:
                current_label = int(self.labels[idx])
                # 以更高概率翻转到相邻类别
                adjacent = [(current_label - 1) % self.num_classes,
                            (current_label + 1) % self.num_classes]
                self.labels[idx] = np.random.choice(adjacent)

        diff = (self.labels != self.true_labels).sum()
        print(f"  [DEBUG inject_noise] 注后 labels 前5: {self.labels[:5].tolist()}, true_labels 前5: {self.true_labels[:5].tolist()}, 噪声数: {diff}/{n} ({diff/n*100:.1f}%)")

        # 更新噪声信息
        self.label_noise_mask = (self.labels != self.true_labels)
        self.label_noise_ratio = self.label_noise_mask.mean()

    def get_feature_dim(self) -> int:
        """获取特征维度（如果有缓存）"""
        if self._cached_features is not None:
            return self._cached_features.shape[1]
        return 0


class CIFAR10Loader:
    """CIFAR-10 数据集加载器（兼容旧接口）"""

    CLASS_NAMES = NoisyImageDataset.CIFAR10_CLASSES

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _load_batch(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        images = data['data']  # (10000, 3072)
        labels = np.array(data['labels'])

        return images, labels

    def load_train(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
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
        test_path = os.path.join(self.data_dir, 'test_batch')
        return self._load_batch(test_path)

    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES

    @staticmethod
    def reshape_to_image(X: np.ndarray) -> np.ndarray:
        """将 (N, 3072) 转为 (N, 3, 32, 32)"""
        return X.reshape(-1, 3, 32, 32)

    @staticmethod
    def to_rgb_image(X: np.ndarray) -> np.ndarray:
        """转为 (N, 32, 32, 3) 用于可视化"""
        if X.shape[1] == 3072:
            X = X.reshape(-1, 3, 32, 32)
        return X.transpose(0, 2, 3, 1)


class ImageDataLoader:
    """
    统一的图像数据加载器

    支持：
    - CIFAR-10
    - CIFAR-100
    - 自定义图像目录
    """

    SUPPORTED_DATASETS = ['cifar10', 'cifar100']

    def __init__(
        self,
        dataset_name: str = 'cifar10',
        data_dir: str = './datasets',
        image_size: int = 32,
    ):
        """
        参数：
        - dataset_name: 数据集名称
        - data_dir: 数据目录
        - image_size: 图像大小（用于 transform）
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.image_size = image_size

        # 训练集的 transform
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        # 测试集的 transform
        self.test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def load_dataset(
        self,
        train: bool = True,
        label_noise_ratio: float = 0.0,
        noise_type: str = 'symmetric',
        transform=None,
        download: bool = True,
    ) -> NoisyImageDataset:
        """
        加载数据集

        参数：
        - train: 是否加载训练集
        - label_noise_ratio: 标签噪声比例
        - noise_type: 噪声类型
        - transform: 图像变换（如果为 None，使用默认）
        - download: 是否下载数据集

        返回：
        - NoisyImageDataset 实例
        """
        if self.dataset_name == 'cifar10':
            return self._load_cifar10(
                train=train,
                label_noise_ratio=label_noise_ratio,
                noise_type=noise_type,
                transform=transform,
                download=download,
            )
        elif self.dataset_name == 'cifar100':
            return self._load_cifar100(
                train=train,
                label_noise_ratio=label_noise_ratio,
                noise_type=noise_type,
                transform=transform,
                download=download,
            )
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

    def _load_cifar10(
        self,
        train: bool,
        label_noise_ratio: float,
        noise_type: str,
        transform,
        download: bool,
    ) -> NoisyImageDataset:
        """加载 CIFAR-10"""
        transform = transform if transform is not None else (
            self.train_transform if train else self.test_transform
        )

        cifar_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=train,
            transform=transform,
            download=download,
        )

        # 提取数据和标签
        images = cifar_dataset.data  # (N, 32, 32, 3)
        labels = np.array(cifar_dataset.targets)
        true_labels = labels.copy()

        # 创建数据集
        dataset = NoisyImageDataset(
            images=images,
            labels=labels,
            true_labels=true_labels,
            transform=transform,
        )

        # 注入噪声
        if label_noise_ratio > 0:
            dataset.inject_noise(
                noise_ratio=label_noise_ratio,
                noise_type=noise_type,
                random_seed=123,  # 注入噪声使用种子 123
            )

        # 更新 label_noise_mask 和 label_noise_ratio
        dataset.label_noise_mask = (dataset.labels != dataset.true_labels)
        dataset.label_noise_ratio = dataset.label_noise_mask.mean()
        print(f"  [ImageDataLoader] CIFAR-10: 注入 {label_noise_ratio*100:.1f}% "
              f"{noise_type} 标签噪声, 实际噪声比例: {dataset.label_noise_ratio*100:.1f}%")

        return dataset

    def _load_cifar100(
        self,
        train: bool,
        label_noise_ratio: float,
        noise_type: str,
        transform,
        download: bool,
    ) -> NoisyImageDataset:
        """加载 CIFAR-100"""
        transform = transform if transform is not None else (
            self.train_transform if train else self.test_transform
        )

        cifar_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=train,
            transform=transform,
            download=download,
        )

        # 提取数据和标签
        images = cifar_dataset.data  # (N, 32, 32, 3)
        labels = np.array(cifar_dataset.targets)
        true_labels = labels.copy()

        # 创建数据集
        dataset = NoisyImageDataset(
            images=images,
            labels=labels,
            true_labels=true_labels,
            transform=transform,
        )

        # 注入噪声
        if label_noise_ratio > 0:
            dataset.inject_noise(
                noise_ratio=label_noise_ratio,
                noise_type=noise_type,
                random_seed=123,  # 注入噪声使用种子 123
            )

        # 更新 label_noise_mask 和 label_noise_ratio
        dataset.label_noise_mask = (dataset.labels != dataset.true_labels)
        dataset.label_noise_ratio = dataset.label_noise_mask.mean()
        print(f"  [ImageDataLoader] CIFAR-100: 注入 {label_noise_ratio*100:.1f}% "
              f"{noise_type} 标签噪声, 实际噪声比例: {dataset.label_noise_ratio*100:.1f}%")

        return dataset

    def load_from_directory(
        self,
        directory: str,
        label_noise_ratio: float = 0.0,
        noise_type: str = 'symmetric',
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
    ) -> NoisyImageDataset:
        """
        从目录加载图像数据集（目录结构：class_name/image.jpg）

        参数：
        - directory: 图像目录
        - label_noise_ratio: 标签噪声比例
        - noise_type: 噪声类型
        - extensions: 支持的文件扩展名
        """
        from torchvision.datasets import ImageFolder

        transform = self.train_transform
        dataset = ImageFolder(
            root=directory,
            transform=transform,
        )

        # 提取数据
        images = []
        labels = []
        for img, label in dataset:
            # 将 PIL Image 转为 numpy
            images.append(np.array(img))
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        true_labels = labels.copy()

        noisy_dataset = NoisyImageDataset(
            images=images,
            labels=labels,
            true_labels=true_labels,
            transform=transform,
        )

        # 注入噪声
        if label_noise_ratio > 0:
            noisy_dataset.inject_noise(
                noise_ratio=label_noise_ratio,
                noise_type=noise_type,
                random_seed=123,  # 注入噪声使用种子 123
            )

        # 更新 label_noise_mask 和 label_noise_ratio
        noisy_dataset.label_noise_mask = (noisy_dataset.labels != noisy_dataset.true_labels)
        noisy_dataset.label_noise_ratio = noisy_dataset.label_noise_mask.mean()

        return noisy_dataset

    def create_dataloader(
        self,
        dataset: NoisyImageDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 2,
    ) -> DataLoader:
        """创建 DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


# 便捷函数
def load_cifar10(
    data_dir: str = './datasets',
    train: bool = True,
    label_noise_ratio: float = 0.0,
    noise_type: str = 'symmetric',
    transform=None,
    download: bool = True,
) -> NoisyImageDataset:
    """
    便捷函数：加载 CIFAR-10 数据集

    使用示例：
        dataset = load_cifar10(
            data_dir='./datasets',
            label_noise_ratio=0.2,
            noise_type='symmetric',
        )
    """
    loader = ImageDataLoader(dataset_name='cifar10', data_dir=data_dir)
    return loader.load_dataset(
        train=train,
        label_noise_ratio=label_noise_ratio,
        noise_type=noise_type,
        transform=transform,
        download=download,
    )


def load_cifar100(
    data_dir: str = './datasets',
    train: bool = True,
    label_noise_ratio: float = 0.0,
    noise_type: str = 'symmetric',
    transform=None,
    download: bool = True,
) -> NoisyImageDataset:
    """
    便捷函数：加载 CIFAR-100 数据集
    """
    loader = ImageDataLoader(dataset_name='cifar100', data_dir=data_dir)
    return loader.load_dataset(
        train=train,
        label_noise_ratio=label_noise_ratio,
        noise_type=noise_type,
        transform=transform,
        download=download,
    )
