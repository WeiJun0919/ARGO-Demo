"""
图像 Selector Agent - 支持图像特征的 Selector

用于图像数据的标签噪声检测和样本选择。
主要针对三个动作：modify_labels, delete_samples, add_samples

对于图像数据，不需要 modify_features（图像特征噪声），
而是使用预训练的 ResNet50 提取特征。

使用方式：
    from agents.image_selector_agent import ImageSelectorAgent

    agent = ImageSelectorAgent(
        feature_dim=2048,  # ResNet50 特征维度
        num_classes=10,
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List


class ImageSelectorNet(nn.Module):
    """
    图像 Selector 网络

    输入：ResNet50 提取的 2048 维特征 + 动作 one-hot + 额外统计特征
    输出：噪声分数 + 可选的标签预测
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        n_actions: int = 3,  # 图像数据只有 3 个动作：labels, delete, add
        hidden: list = [512, 256],
        num_classes: int = 10,
    ):
        """
        参数：
        - feature_dim: 输入特征维度（ResNet50 = 2048）
        - n_actions: 动作数量
        - hidden: 隐藏层配置
        - num_classes: 类别数（用于标签预测）
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.num_classes = num_classes

        # 输入维度 = 特征 + 动作 one-hot
        input_dim = feature_dim + n_actions

        # 特征编码器
        dims = [input_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(0.2))
        self.encoder = nn.Sequential(*enc_layers)

        # 共享的隐藏表示
        self.hidden_dim = hidden[-1]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播

        参数：
        - x: 输入张量 (batch, feature_dim + n_actions)

        返回：
        - (scores, pred, hidden)
        """
        h = self.encoder(x)
        return h


class ImageSelectorNetLabels(nn.Module):
    """
    Selector for modify_labels action (Action 1).

    专门用于标签噪声检测。
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden: list = [512, 256],
        num_classes: int = 10,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        dims = [feature_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(0.2))
        self.encoder = nn.Sequential(*enc_layers)

        # 噪声分数头
        self.score_head = nn.Linear(hidden[-1], 1)

        # 标签预测头
        self.label_head = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden[-1] // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor):
        h = self.encoder(features)
        scores_raw = self.score_head(h).squeeze(-1)
        scores = torch.sigmoid(scores_raw)
        label_logits = self.label_head(h)
        return scores, label_logits, h


class ImageSelectorNetDelete(nn.Module):
    """
    Selector for delete_samples action (Action 2).

    检测低质量/有害样本。
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden: list = [512, 256],
    ):
        super().__init__()

        self.feature_dim = feature_dim

        dims = [feature_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(0.2))
        self.encoder = nn.Sequential(*enc_layers)

        self.score_head = nn.Linear(hidden[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor):
        h = self.encoder(features)
        scores_raw = self.score_head(h).squeeze(-1)
        scores = torch.sigmoid(scores_raw)
        return scores, None, h


class ImageSelectorNetAdd(nn.Module):
    """
    Selector for add_samples action (Action 3).

    从候选池选择优质样本。
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden: list = [512, 256],
    ):
        super().__init__()

        self.feature_dim = feature_dim

        dims = [feature_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(0.2))
        self.encoder = nn.Sequential(*enc_layers)

        # 质量分数头
        self.score_head = nn.Linear(hidden[-1], 1)

        # 特征增强头（可选）
        self.feat_head = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-1]),
            nn.ReLU(),
            nn.Linear(hidden[-1], feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor):
        h = self.encoder(features)
        scores_raw = self.score_head(h).squeeze(-1)
        scores = torch.sigmoid(scores_raw)
        feat_pred = self.feat_head(h)
        return scores, feat_pred, h


# 图像 Selector 网络映射
IMAGE_SELECTOR_NETS = {
    1: ImageSelectorNetLabels,  # modify_labels
    2: ImageSelectorNetDelete,   # delete_samples
    3: ImageSelectorNetAdd,     # add_samples
}

IMAGE_ACTION_NAMES = {
    1: "modify_labels",
    2: "delete_samples",
    3: "add_samples",
}


class ImageSelectorAgent:
    """
    图像数据的 Selector Agent

    支持三个动作：
    - Action 1: modify_labels - 检测和修正标签噪声
    - Action 2: delete_samples - 删除低质量样本
    - Action 3: add_samples - 添加优质样本

    注意：图像数据不使用 modify_features
    """

    def __init__(
        self,
        config,
        feature_dim: int = 2048,
        num_classes: int = 10,
    ):
        """
        参数：
        - config: 配置对象
        - feature_dim: 特征维度（ResNet50 = 2048）
        - num_classes: 类别数
        """
        self.cfg = config
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 动作数量（图像数据只有 3 个动作）
        self.n_actions = 3  # 1, 2, 3
        self.action_idx_map = {1: 0, 2: 1, 3: 2}  # 映射到网络索引

        # 设备
        self._device = torch.device(config.device)

        # 创建网络
        self.nets = {}
        self.optimizers = {}

        hidden = getattr(config, 'selector_hidden', [512, 256])

        # 为每个动作创建独立的 Selector
        for action_idx in [1, 2, 3]:
            net_idx = self.action_idx_map[action_idx]
            net_cls = IMAGE_SELECTOR_NETS[action_idx]

            if action_idx == 1:
                net = net_cls(feature_dim=feature_dim, hidden=hidden, num_classes=num_classes)
            else:
                net = net_cls(feature_dim=feature_dim, hidden=hidden)

            net = net.to(self.device)
            self.nets[action_idx] = net

            self.optimizers[action_idx] = optim.Adam(
                net.parameters(),
                lr=config.lr_selector
            )

        print(f"[ImageSelectorAgent] 初始化完成，设备: {self.device}:")
        for idx in [1, 2, 3]:
            n_params = sum(p.numel() for p in self.nets[idx].parameters())
            print(f"  - Action {idx} ({IMAGE_ACTION_NAMES[idx]}): {n_params:,} 参数")

    def build_input(
        self,
        features: np.ndarray,
        action_idx: int,
    ) -> torch.Tensor:
        """
        构建 Selector 输入

        参数：
        - features: 图像特征 (batch, feature_dim)
        - action_idx: 动作索引 (1, 2, 3)

        返回：
        - 输入张量
        """
        n = len(features)
        features_tensor = torch.FloatTensor(features).to(self.device)

        return features_tensor

    def select(
        self,
        z: torch.Tensor,
        action_idx: int,
        n_select: int,
    ):
        """
        选择样本

        参数：
        - z: 输入张量
        - action_idx: 动作索引
        - n_select: 选择数量

        返回：
        - (selected_indices, scores, pred, hidden)
        """
        n = z.shape[0]
        n_select = min(max(1, n_select), n)

        net = self.nets[action_idx]
        z = z.to(self.device)

        with torch.no_grad():
            scores, pred, hidden = net(z)

        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        # 选择分数最高的（下标始终在 [0, n)）
        _, top_idx = torch.topk(scores, n_select, largest=True, sorted=False)

        return top_idx.tolist(), scores, pred, hidden

    def forward(self, z: torch.Tensor, action_idx: int):
        """前向传播"""
        net = self.nets[action_idx]
        z = z.to(self.device)
        return net(z)

    def update(
        self,
        z: torch.Tensor,
        action_idx: int,
        selected_indices: List[int],
        reward: float,
        oracle_dirty: np.ndarray = None,
        clean_labels: np.ndarray = None,
        train_mode: str = "joint",
    ):
        """
        更新 Selector
        """
        optimizer = self.optimizers[action_idx]
        net = self.nets[action_idx]

        z = z.to(self._device)

        optimizer.zero_grad()

        scores, pred, hidden = net(z)

        loss = torch.tensor(0.0, device=z.device)

        # RL Loss
        if train_mode in ("rl", "joint") and len(selected_indices) > 0:
            sel_scores = scores[selected_indices]
            normalized_reward = torch.clamp(
                torch.tensor(reward, device=z.device),
                min=-1.0, max=1.0
            )
            rl_loss = -normalized_reward * torch.log(sel_scores.clamp(1e-7, 1 - 1e-7)).mean()
            loss = loss + self.cfg.lambda_rl * rl_loss

        # Auxiliary Loss
        if train_mode in ("aux", "joint") and oracle_dirty is not None:
            oracle_t = torch.FloatTensor(oracle_dirty).to(z.device)
            scores_safe = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
            scores_clamped = torch.clamp(scores_safe, min=0.0, max=1.0)

            if action_idx == 1 and clean_labels is not None and pred is not None:
                # modify_labels: BCE + 标签预测
                aux_bce = nn.BCELoss()(scores_clamped, oracle_t)
                loss = loss + self.cfg.lambda_aux * aux_bce

                # 标签预测损失
                clean_lbl = torch.LongTensor(clean_labels).to(z.device)
                if len(selected_indices) > 0:
                    label_logits = pred[selected_indices]
                    clean_lbl_selected = clean_lbl[selected_indices]
                    label_loss = nn.CrossEntropyLoss()(label_logits, clean_lbl_selected)
                    loss = loss + self.cfg.lambda_aux * label_loss
            else:
                aux_bce = nn.BCELoss()(scores_clamped, oracle_t)
                loss = loss + self.cfg.lambda_aux * aux_bce

        loss.backward()

        if hasattr(self.cfg, 'max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                self.cfg.max_grad_norm
            )

        optimizer.step()

        return {"loss": loss.item()}

    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'nets': {idx: net.state_dict() for idx, net in self.nets.items()},
            'optimizers': {idx: opt.state_dict() for idx, opt in self.optimizers.items()},
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        for idx, state_dict in checkpoint['nets'].items():
            self.nets[idx].load_state_dict(state_dict)
        for idx, state_dict in checkpoint['optimizers'].items():
            self.optimizers[idx].load_state_dict(state_dict)

    def eval_mode(self):
        """评估模式"""
        for net in self.nets.values():
            net.eval()

    def train_mode(self):
        """训练模式"""
        for net in self.nets.values():
            net.train()

    @property
    def device(self):
        return self._device
