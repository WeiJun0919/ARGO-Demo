"""
ED2-RPT: Error Detector Denoiser + Representation Transformer

ED2: 检测哪些特征是噪声（每个特征的噪声概率）
RPT: 用预训练模型预测正确的特征值

双头设计：
- 头1 (ED2): 分类头，输出每个特征是否为噪声的概率
- 头2 (RPT): 回归头，输出修复后的特征值

预训练：用干净数据训练 RPT 预测干净特征
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class ED2RPTNetwork(nn.Module):
    """
    ED2-RPT 双头网络
    
    - ED2 head: 判断每个特征是否被污染
    - RPT head: 预测修复后的特征值
    """
    
    def __init__(self, input_dim, n_features, hidden_dims=[128, 64]):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_features = n_features
        
        # 共享特征提取层
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim),
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)
        
        # ED2 噪声检测头：输出每个特征的噪声概率
        self.ed2_head = nn.Sequential(
            nn.Linear(prev_dim, n_features),
            nn.Sigmoid(),  # 每个特征的噪声概率
        )
        
        # RPT 特征修复头：输出修复后的特征值
        self.rpt_head = nn.Sequential(
            nn.Linear(prev_dim, n_features),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) - 输入特征
            
        Returns:
            noise_prob: (batch, n_features) - 每个特征的噪声概率
            feat_correction: (batch, n_features) - 修复后的特征值
        """
        h = self.shared(x)
        noise_prob = self.ed2_head(h)
        feat_correction = self.rpt_head(h)
        return noise_prob, feat_correction


class ED2RPTDetector:
    """
    ED2-RPT 特征噪声检测与修复器
    
    使用方法：
    1. 先用干净数据预训练 (pretrain)
    2. 用当前数据检测+修复 (detect_and_correct)
    """
    
    def __init__(self, n_features, hidden_dims=[128, 64], 
                 device='cpu', noise_threshold=0.5,
                 correction_scale=0.1, random_state=42):
        """
        Parameters:
        -----------
        n_features : int
            特征数量
        hidden_dims : list
            隐藏层维度
        device : str
            设备
        noise_threshold : float
            噪声判定阈值
        correction_scale : float
            修正强度 (0-1)，1表示完全替换为预测值
        random_state : int
            随机种子
        """
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.device = device
        self.noise_threshold = noise_threshold
        self.correction_scale = correction_scale
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.is_pretrained = False
    
    def _build_model(self, input_dim):
        """构建模型"""
        torch.manual_seed(self.random_state)
        self.model = ED2RPTNetwork(
            input_dim=input_dim,
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
    
    def pretrain(self, X_clean, y_clean=None, epochs=50, batch_size=64, lr=1e-3):
        """
        用干净数据预训练 RPT 头
        
        Args:
            X_clean: 干净特征 (n_samples, n_features)
            y_clean: 标签（可选，用于条件修复）
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
        """
        X_clean = np.asarray(X_clean, dtype=np.float64)
        
        # 预处理
        X_imp = self.imputer.fit_transform(X_clean)
        X_sc = self.scaler.fit_transform(X_imp)
        
        # 转换为 tensor
        if y_clean is not None:
            y_clean = np.asarray(y_clean, dtype=np.float64)
            # 标签作为额外输入特征
            inputs = np.concatenate([X_sc, y_clean.reshape(-1, 1)], axis=1)
        else:
            inputs = X_sc
            
        self._build_model(inputs.shape[1])
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练 RPT：预测干净特征本身
        X_tensor = torch.FloatTensor(inputs).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            _, feat_correction = self.model(X_tensor)

            # 目标：预测原始干净特征
            target = torch.FloatTensor(X_sc).to(self.device)

            # MSE 损失（动态创建确保在正确设备上）
            loss = nn.MSELoss()(feat_correction, target)
            
            loss.backward()
            optimizer.step()
        
        self.is_pretrained = True
        self.model.eval()
    
    def detect_and_correct(self, X_dirty, y_labels=None, return_noise_scores=False):
        """
        检测噪声特征并修复
        
        Args:
            X_dirty: 可能有噪声的特征 (n_samples, n_features)
            y_labels: 标签（可选）
            return_noise_scores: 是否返回噪声分数
            
        Returns:
            X_corrected: 修复后的特征
            noise_mask: 噪声掩码 (n_samples, n_features)
            noise_scores: 噪声概率 (n_samples, n_features) - 仅当 return_noise_scores=True
        """
        X_dirty = np.asarray(X_dirty, dtype=np.float64)
        
        # 预处理
        X_imp = self.imputer.transform(X_dirty)
        X_sc = self.scaler.transform(X_imp)
        
        # 构建输入
        if y_labels is not None:
            y_labels = np.asarray(y_labels, dtype=np.float64)
            inputs = np.concatenate([X_sc, y_labels.reshape(-1, 1)], axis=1)
        else:
            inputs = X_sc
        
        # 转为 tensor
        X_tensor = torch.FloatTensor(inputs).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            noise_prob, feat_correction = self.model(X_tensor)
        
        noise_prob = noise_prob.cpu().numpy()
        feat_correction = feat_correction.cpu().numpy()
        
        # 反标准化得到真实尺度的预测
        feat_correction_real = self.scaler.inverse_transform(feat_correction)
        
        # 噪声掩码：概率 > 阈值的特征视为噪声
        noise_mask = noise_prob > self.noise_threshold
        
        # 混合修复：只在被检测为噪声的位置使用预测值，其他保持原始
        X_corrected = X_dirty.copy()
        for i in range(len(X_dirty)):
            for j in range(self.n_features):
                if noise_mask[i, j]:
                    # 用 scale 控制修复强度（不完全替换，保留一些原始信息）
                    X_corrected[i, j] = (
                        (1 - self.correction_scale) * X_dirty[i, j] + 
                        self.correction_scale * feat_correction_real[i, j]
                    )
        
        if return_noise_scores:
            return X_corrected, noise_mask, noise_prob
        return X_corrected, noise_mask
    
    def predict_corrections(self, X, y_labels=None):
        """
        预测修复值（不应用修复）
        
        Args:
            X: 特征
            y_labels: 标签
            
        Returns:
            corrections: 预测的修复值
        """
        X = np.asarray(X, dtype=np.float64)
        
        X_imp = self.imputer.transform(X)
        X_sc = self.scaler.transform(X_imp)
        
        if y_labels is not None:
            y_labels = np.asarray(y_labels, dtype=np.float64)
            inputs = np.concatenate([X_sc, y_labels.reshape(-1, 1)], axis=1)
        else:
            inputs = X_sc
        
        X_tensor = torch.FloatTensor(inputs).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, feat_correction = self.model(X_tensor)
        
        return feat_correction.cpu().numpy()


class SimpleED2Detector:
    """
    简化版 ED2 检测器（不需要 PyTorch）
    
    基于特征重建误差来检测噪声
    """
    
    def __init__(self, n_features, noise_threshold=0.5, random_state=42):
        self.n_features = n_features
        self.noise_threshold = noise_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.is_pretrained = False
    
    def pretrain(self, X_clean, epochs=50):
        """用干净数据训练自编码器"""
        X_clean = np.asarray(X_clean, dtype=np.float64)
        
        X_imp = self.imputer.fit_transform(X_clean)
        self.X_mean = np.mean(X_imp, axis=0)
        self.X_std = np.std(X_imp, axis=0) + 1e-8
        
        self.is_pretrained = True
    
    def detect_and_correct(self, X_dirty, return_noise_scores=False):
        """基于重建误差检测噪声"""
        X_dirty = np.asarray(X_dirty, dtype=np.float64)
        
        X_imp = self.imputer.transform(X_dirty)
        
        # 计算每个特征的重建误差（与均值的差异）
        z_scores = np.abs((X_imp - self.X_mean) / self.X_std)
        
        # 噪声分数：z-score 超过阈值越多，越可能是噪声
        noise_scores = np.minimum(z_scores / 3.0, 1.0)
        
        # 噪声掩码
        noise_mask = noise_scores > self.noise_threshold
        
        # 修复：用均值替换噪声特征
        X_corrected = X_imp.copy()
        X_corrected[noise_mask] = self.X_mean[noise_mask]
        
        if return_noise_scores:
            return X_corrected, noise_mask, noise_scores
        return X_corrected, noise_mask
