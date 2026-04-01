"""
4个独立的 SelectorNet，分别用于不同的动作。
每个 Selector 针对特定任务优化，有不同的输入和输出。
"""

import torch
import torch.nn as nn


class SelectorNetFeatures(nn.Module):
    """
    Selector for modify_features action (Action 0).
    
    任务：检测特征噪声 + 预测干净特征
    输入：state + 样本特征（含熵/损失/边际等）+ action_onehot
    输出：score（噪声概率）+ feat_pred（预测的干净特征）
    """
    
    def __init__(self, input_dim: int, n_features: int, hidden: list):
        super().__init__()
        
        dims = [input_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        # Head 1: 特征噪声分数（去掉Sigmoid，使用原始分数）
        self.score_head = nn.Linear(hidden[-1], 1)
        
        # Head 2: 预测干净特征
        self.feat_head = nn.Linear(hidden[-1], n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor):
        h = self.encoder(z)
        scores_raw = self.score_head(h).squeeze(-1)
        # Sigmoid 用于选择概率，保持原始分数用于损失计算
        scores = torch.sigmoid(scores_raw)
        feat_pred = self.feat_head(h)
        return scores, feat_pred, h


class SelectorNetLabels(nn.Module):
    """
    Selector for modify_labels action (Action 1).
    
    任务：检测标签噪声 + 预测正确标签
    输入：state + 样本特征 + 预测概率 + action_onehot
    输出：score（标签错误概率）+ label_pred（预测的正确标签，0或1）
    """
    
    def __init__(self, input_dim: int, n_features: int, hidden: list):
        super().__init__()
        
        dims = [input_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        # Head 1: 标签错误分数（去掉Sigmoid）
        self.score_head = nn.Linear(hidden[-1], 1)
        
        # Head 2: 预测正确标签 (二分类)
        self.label_head = nn.Sequential(
            nn.Linear(hidden[-1], 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor):
        h = self.encoder(z)
        scores_raw = self.score_head(h).squeeze(-1)
        scores = torch.sigmoid(scores_raw)  # Sigmoid 用于选择
        label_pred = self.label_head(h).squeeze(-1)  # (N,) 概率
        return scores, label_pred, h


class SelectorNetDelete(nn.Module):
    """
    Selector for delete_samples action (Action 2).
    
    任务：检测无用/有害样本
    输入：state + 样本特征 + 损失/边际等 + action_onehot
    输出：score（应被删除的概率）
    """
    
    def __init__(self, input_dim: int, n_features: int, hidden: list):
        super().__init__()
        
        dims = [input_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        # Head: 应被删除的概率（去掉Sigmoid）
        self.score_head = nn.Linear(hidden[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor):
        h = self.encoder(z)
        scores_raw = self.score_head(h).squeeze(-1)
        scores = torch.sigmoid(scores_raw)  # Sigmoid 用于选择
        return scores, None, h  # 无预测输出


class SelectorNetAdd(nn.Module):
    """
    Selector for add_samples action (Action 3).
    
    任务：从候选池选择优质样本 + 预测增强后的特征
    输入：state + 候选样本特征 + action_onehot
    输出：score（被选中概率）+ feat_pred（增强后的特征）
    """
    
    def __init__(self, input_dim: int, n_features: int, hidden: list):
        super().__init__()
        
        dims = [input_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        # Head 1: 选择分数（去掉Sigmoid）
        self.score_head = nn.Linear(hidden[-1], 1)
        
        # Head 2: 增强后的特征（可选，用于进一步增强样本）
        self.feat_head = nn.Linear(hidden[-1], n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor):
        h = self.encoder(z)
        scores_raw = self.score_head(h).squeeze(-1)
        # Sigmoid 用于选择概率，保持原始分数用于损失计算
        scores = torch.sigmoid(scores_raw)
        feat_pred = self.feat_head(h)
        return scores, feat_pred, h


# 动作名称映射
ACTION_NAMES = {
    0: "modify_features",
    1: "modify_labels",
    2: "delete_samples",
    3: "add_samples",
}

# 网络类映射
SELECTOR_NETS = {
    0: SelectorNetFeatures,
    1: SelectorNetLabels,
    2: SelectorNetDelete,
    3: SelectorNetAdd,
}
