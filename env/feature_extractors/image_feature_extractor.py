"""
图像特征提取器模块 - 使用预训练模型提取图像特征

支持多种预训练模型：
- ResNet50: 2048 维特征（推荐）
- ResNet18: 512 维特征
- AlexNet: 4096 维特征
- VGG16: 4096 维特征
- MobileNetV2: 1280 维特征

使用方式：
    from env.feature_extractors.image_feature_extractor import ImageFeatureExtractor

    extractor = ImageFeatureExtractor(model_name='resnet50', pretrained=True)
    features = extractor.extract(image_batch)  # shape: (batch, 2048)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Optional, Literal


# 可用的预训练模型配置
MODEL_CONFIGS = {
    'resnet50': {
        'feature_dim': 2048,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resnet18': {
        'feature_dim': 512,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'alexnet': {
        'feature_dim': 4096,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'vgg16': {
        'feature_dim': 4096,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'mobilenet_v2': {
        'feature_dim': 1280,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
}


class ImageFeatureExtractor(nn.Module):
    """
    使用预训练 CNN 模型提取图像特征

    特点：
    - 自动去掉最后的分类层，输出固定维度的特征
    - 支持多种预训练模型
    - 提供图像预处理 transform
    - 支持特征预计算和批量提取
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        trainable_layers: int = 3,
        device: Optional[torch.device] = None,
    ):
        """
        参数：
        - model_name: 模型名称 ('resnet50', 'resnet18', 'alexnet', 'vgg16', 'mobilenet_v2')
        - pretrained: 是否使用 ImageNet 预训练权重
        - freeze_backbone: 是否冻结骨干网络（只微调最后几层）
        - trainable_layers: 当 freeze_backbone=True 时，保留最后几层可训练
        - device: 计算设备
        """
        super().__init__()

        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['resnet50'])
        self.feature_dim = self.config['feature_dim']
        self.input_size = self.config['input_size']

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # 加载预训练模型
        self.backbone = self._load_backbone(model_name, pretrained)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['mean'],
                std=self.config['std'],
            )
        ])

        # 根据配置决定是否冻结
        if freeze_backbone:
            self._freeze_except_last_layers(trainable_layers)

        # 必须与 extract() 中 images.to(self.device) 一致，否则会出现
        # Input type (cuda.FloatTensor) and weight type (torch.FloatTensor)
        self.to(self.device)

    def _load_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """加载骨干网络"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # 去掉最后的 FC 层 (AvgPool 后的特征是 2048 维)
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=pretrained)
            # AlexNet: features -> avgpool -> classifier[:-1] (去掉最后的 FC)
            backbone = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten(),
                *list(model.classifier.children())[:-1],  # 去掉最后的 FC
            )

        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            # VGG16: features -> avgpool -> classifier[:-1]
            backbone = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten(),
                *list(model.classifier.children())[:-1],
            )

        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            # MobileNetV2 特征在 avgpool 之后
            backbone = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        else:
            raise ValueError(f"不支持的模型: {model_name}")

        return backbone

    def _freeze_except_last_layers(self, n_layers: int):
        """冻结除最后 n_layers 个模块外的所有参数"""
        # 首先冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 获取所有子模块
        children = list(self.backbone.children())

        # 解冻最后 n_layers 个模块
        # 对于 ResNet: children 是 [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        # 我们通常解冻 layer4 的最后几个 Bottleneck
        if len(children) > n_layers:
            for child in children[-n_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

        # 统计可训练参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [ImageFeatureExtractor] {self.model_name}: "
              f"可训练参数 {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征

        参数：
        - x: 输入图像张量 shape (batch, 3, H, W)，应该已经过 normalize

        返回：
        - features: 特征张量 shape (batch, feature_dim)
        """
        with torch.set_grad_enabled(self.training):
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
        return features

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取特征的便捷方法（自动处理设备）
        """
        images = images.to(self.device)
        with torch.no_grad():
            features = self.forward(images)
        return features.cpu()

    def extract_batch(
        self,
        image_list: list,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        从图像列表中批量提取特征

        参数：
        - image_list: 图像列表 (可以是 PIL.Image, np.ndarray, 或已处理的 tensor)
        - batch_size: 批处理大小

        返回：
        - features: 特征张量 shape (N, feature_dim)
        """
        from torch.utils.data import DataLoader, TensorDataset

        # 预处理所有图像
        processed = []
        for img in image_list:
            if isinstance(img, torch.Tensor):
                # 如果是 tensor，检查维度
                if img.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                    img = img.unsqueeze(0)
                elif img.dim() == 2:  # (H, W) -> (1, 1, H, W) 灰度图
                    img = img.unsqueeze(0).unsqueeze(0)
                    img = img.expand(3, -1, -1)  # 转为 RGB
            elif isinstance(img, nn.Module):
                # 已经处理过的
                processed.append(img if isinstance(img, torch.Tensor) else torch.from_numpy(img))
                continue
            else:
                # PIL Image 或 numpy array
                img = self.transform(img)

            if isinstance(img, torch.Tensor) and img.dim() == 3:
                img = img.unsqueeze(0)
            processed.append(img)

        # 合并为 batch
        batched = torch.cat(processed, dim=0).to(self.device)

        # 批量提取
        all_features = []
        n = len(batched)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = batched[i:i+batch_size]
                features = self.forward(batch)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    def preprocess_image(self, image):
        """
        预处理单张图像

        参数：
        - image: PIL.Image, np.ndarray, 或 torch.Tensor

        返回：
        - tensor: 预处理后的张量 (1, 3, H, W)
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)
        elif isinstance(image, nn.Module):
            # 已经是处理好的
            return image
        else:
            # PIL Image 或 numpy
            tensor = self.transform(image)
            return tensor.unsqueeze(0).to(self.device)

    @property
    def is_training(self):
        return self.training


def create_feature_extractor(
    model_name: str = 'resnet50',
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> ImageFeatureExtractor:
    """
    便捷函数：创建图像特征提取器

    参数：
    - model_name: 模型名称
    - pretrained: 是否使用预训练权重
    - device: 计算设备

    返回：
    - ImageFeatureExtractor 实例
    """
    return ImageFeatureExtractor(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
