"""
图像下游分类器 - 用于评估数据清洗效果

基于 ResNet50 提取的特征训练下游分类器，计算准确率。

使用方式：
    from env.classifiers.image_classifier import ImageDownstreamClassifier

    # 初始化分类器
    classifier = ImageDownstreamClassifier(
        feature_dim=2048,  # ResNet50 特征维度
        num_classes=10,
    )

    # 训练
    classifier.fit(features, labels)

    # 预测
    accuracy = classifier.evaluate(test_features, test_labels)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
import numpy as np


class ImageDownstreamClassifier(nn.Module):
    """
    基于图像特征的下游分类器

    用于评估数据清洗后的准确率变化。
    特征来自预训练的 ResNet50 等模型。
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        """
        参数：
        - feature_dim: 输入特征维度（ResNet50=2048）
        - num_classes: 类别数
        - hidden_dim: 隐藏层维度
        - dropout: Dropout 比例
        - device: 计算设备
        """
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.classifier(features)

    def predict(self, features: torch.Tensor) -> np.ndarray:
        """预测类别"""
        self.eval()
        features = features.to(self.device)

        with torch.no_grad():
            logits = self.forward(features)
            predictions = logits.argmax(dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, features: torch.Tensor) -> np.ndarray:
        """预测概率"""
        self.eval()
        features = features.to(self.device)

        with torch.no_grad():
            logits = self.forward(features)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        训练分类器

        参数：
        - train_features: 训练特征 (N, feature_dim)
        - train_labels: 训练标签 (N,)
        - val_features: 验证特征（可选）
        - val_labels: 验证标签（可选）
        - epochs: 训练轮数
        - batch_size: 批次大小
        - lr: 学习率
        - weight_decay: 权重衰减
        - early_stopping_patience: 早停轮数
        - verbose: 是否打印训练过程

        返回：
        - 训练历史 dict
        """
        self.train()

        # 转换为 tensor（标签必须为整型类别 id）
        train_X = torch.as_tensor(train_features, dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor(
            np.asarray(train_labels).astype(np.int64, copy=False).reshape(-1),
            dtype=torch.long,
            device=self.device,
        )

        # 创建 DataLoader
        dataset = TensorDataset(train_X, train_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 优化器
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )

        # 验证集
        val_X = (
            torch.as_tensor(val_features, dtype=torch.float32, device=self.device)
            if val_features is not None
            else None
        )
        val_y = (
            torch.as_tensor(
                np.asarray(val_labels).astype(np.int64, copy=False).reshape(-1),
                dtype=torch.long,
                device=self.device,
            )
            if val_labels is not None
            else None
        )

        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 验证
            val_acc = None
            if val_X is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(val_X)
                    val_loss = criterion(val_outputs, val_y)
                    _, val_predicted = val_outputs.max(1)
                    val_acc = val_predicted.eq(val_y).sum().item() / val_y.size(0)
                history['val_acc'].append(val_acc)

                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                          f"val_acc={val_acc:.4f}")

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

        return history

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
    ) -> float:
        """
        评估分类器准确率

        参数：
        - features: 测试特征 (N, feature_dim)
        - labels: 测试标签 (N,)

        返回：
        - 准确率
        """
        self.eval()

        X = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(labels).astype(np.int64, copy=False).reshape(-1), dtype=torch.long, device=self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                outputs = self.forward(batch_X)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        return correct / total

    def get_confusion_matrix(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """获取混淆矩阵"""
        self.eval()
        predictions = self.predict(features)

        # 简单的混淆矩阵
        n_classes = max(self.num_classes, len(np.unique(labels)))
        cm = np.zeros((n_classes, n_classes), dtype=int)

        for true_label, pred_label in zip(labels, predictions):
            cm[int(true_label), int(pred_label)] += 1

        return cm


class LightweightImageClassifier(nn.Module):
    """
    轻量级图像分类器（用于快速评估）

    不使用预训练特征，直接从原始图像训练小模型。
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        image_size: int = 32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            nn.Dropout(0.25),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            nn.Dropout(0.25),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            nn.Dropout(0.25),
        )

        # 计算特征图大小
        feature_size = 128 * (image_size // 8) * (image_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """预测类别"""
        self.eval()
        images = images.to(self.device)

        with torch.no_grad():
            logits = self.forward(images)
            predictions = logits.argmax(dim=1)

        return predictions.cpu().numpy()


# 便捷函数
def create_image_classifier(
    feature_dim: int = 2048,
    num_classes: int = 10,
    model_type: str = 'downstream',
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    创建图像分类器

    参数：
    - feature_dim: 特征维度（用于 downstream 类型）
    - num_classes: 类别数
    - model_type: 'downstream' 或 'lightweight'
    - device: 计算设备

    返回：
    - 分类器模型
    """
    if model_type == 'downstream':
        return ImageDownstreamClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            device=device,
        )
    elif model_type == 'lightweight':
        return LightweightImageClassifier(
            num_classes=num_classes,
            device=device,
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
