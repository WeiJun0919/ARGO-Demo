"""
VAE 数据生成器 - 变分自编码器

使用变分自编码器（Variational Autoencoder）学习数据的潜在分布，
然后从潜在空间采样生成新的合成样本。

对于分类任务，使用条件 VAE (CVAE) 来根据类别生成样本。
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict, Any
import warnings

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = None
    optim = None
    DataLoader = None

from . import BaseDataGenerator, DataGeneratorRegistry


class VAEEncoder(nn.Module):
    """VAE 编码器"""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE 解码器"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class ConditionalVAE(nn.Module):
    """条件 VAE (CVAE) - 支持按类别生成"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_classes: int,
        hidden_dims: list = [256, 128],
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # 编码器：输入 + 类别嵌入
        self.encoder = VAEEncoder(input_dim + n_classes, latent_dim, hidden_dims)
        
        # 解码器：潜在向量 + 类别嵌入
        decoder_input_dim = latent_dim + n_classes
        self.decoder = VAEDecoder(decoder_input_dim, input_dim, hidden_dims)
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(n_classes, n_classes)
    
    def encode(self, x, y):
        """编码到潜在空间"""
        # 将类别转换为 one-hot 并与输入拼接
        y_onehot = torch.zeros(y.size(0), self.n_classes).to(x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        
        xy = torch.cat([x, y_onehot], dim=1)
        mu, logvar = self.encoder(xy)
        return mu, logvar
    
    def decode(self, z, y):
        """从潜在空间解码"""
        y_onehot = torch.zeros(y.size(0), self.n_classes).to(z.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        
        zy = torch.cat([z, y_onehot], dim=1)
        return self.decoder(zy)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """VAE 损失函数 = 重构损失 + KL 散度"""
    # BCE 重构损失
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


@DataGeneratorRegistry.register("vae")
class VAEGenerator(BaseDataGenerator):
    """
    VAE 数据生成器
    
    使用变分自编码器学习数据分布并生成新样本。
    支持条件生成（按类别生成）。
    
    参数：
    - latent_dim: 潜在空间维度
    - hidden_dims: 隐藏层维度列表
    - epochs: 训练轮数
    - batch_size: 批次大小
    - learning_rate: 学习率
    - beta: KL 散度权重
    - use_cvae: 是否使用条件 VAE
    """
    
    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for VAEGenerator")
        
        # VAE 参数
        self.latent_dim = getattr(config, 'vae_latent_dim', 32)
        self.hidden_dims = getattr(config, 'vae_hidden_dims', [256, 128])
        self.epochs = getattr(config, 'vae_epochs', 100)
        self.batch_size = getattr(config, 'vae_batch_size', 64)
        self.lr = getattr(config, 'vae_lr', 1e-3)
        self.beta = getattr(config, 'vae_beta', 1.0)
        self.use_cvae = getattr(config, 'vae_use_cvae', True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_fitted = False
        self.n_classes = 2  # 默认二分类
        self.input_dim = None
        
        # 缓存
        self._class_mean_latent = {}
    
    def _build_model(self, input_dim: int, n_classes: int):
        """构建模型"""
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        if self.use_cvae:
            self.model = ConditionalVAE(
                input_dim=input_dim,
                latent_dim=self.latent_dim,
                n_classes=n_classes,
                hidden_dims=self.hidden_dims,
            ).to(self.device)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, input_dim),
            ).to(self.device)
            self.model.encode = lambda x: (torch.zeros(x.size(0), self.latent_dim).to(x.device), 
                                           torch.zeros(x.size(0), self.latent_dim).to(x.device))
            self.model.decode = lambda z: self.model(z)
            self.model.reparameterize = lambda mu, logvar: mu
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def _train_vae(self, X: np.ndarray, y: np.ndarray):
        """训练 VAE"""
        n_samples, input_dim = X.shape
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        
        # 获取类别
        classes = np.unique(y)
        self.n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[c] for c in y])
        
        # 构建模型
        self._build_model(input_dim, self.n_classes)
        
        # 转换为 PyTorch 张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, n_samples), shuffle=True)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_cvae:
                    recon, mu, logvar = self.model(batch_x, batch_y)
                    loss, recon_loss, kl_loss = vae_loss(recon, batch_x, mu, logvar, self.beta)
                else:
                    recon = self.model(batch_x)
                    mu, logvar = self.model.encode(batch_x)
                    loss, recon_loss, kl_loss = vae_loss(recon, batch_x, mu, logvar, self.beta)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        # 计算每个类别的潜在空间均值（用于条件生成）
        self._compute_class_latent(X, y_idx)
        
        self.is_fitted = True
    
    def _compute_class_latent(self, X: np.ndarray, y: np.ndarray):
        """计算每个类别的潜在空间均值"""
        self.model.eval()
        
        X_tensor = torch.tensor(X.astype(np.float32), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            if self.use_cvae:
                mu, _ = self.model.encode(X_tensor, y_tensor)
            else:
                mu, _ = self.model.encode(X_tensor)
        
        mu = mu.cpu().numpy()
        
        for c in range(self.n_classes):
            mask = y == c
            if mask.sum() > 0:
                self._class_mean_latent[c] = mu[mask].mean(axis=0)
    
    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成样本"""
        if not _HAS_TORCH:
            return np.array([]), np.array([])
        
        if len(X) < 10:
            return np.array([]), np.array([])
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 训练 VAE（如果还没训练）
        if not self.is_fitted:
            try:
                self._train_vae(X, y)
            except Exception as e:
                warnings.warn(f"VAE training failed: {e}")
                return np.array([]), np.array([])
        
        self.model.eval()
        
        classes = np.unique(y)
        n_classes = len(classes)
        
        # 每个类别生成的样本数
        samples_per_class = n_samples // n_classes
        if samples_per_class == 0:
            samples_per_class = 1
        
        X_syn = []
        y_syn = []
        
        with torch.no_grad():
            for c in classes:
                # 从该类别的潜在空间均值附近采样
                if c in self._class_mean_latent:
                    mean_latent = self._class_mean_latent[c]
                else:
                    mean_latent = np.zeros(self.latent_dim)
                
                # 采样潜在向量
                n_gen = min(samples_per_class, n_samples - len(y_syn))
                for _ in range(n_gen):
                    z = mean_latent + self.rng.normal(0, 0.5, self.latent_dim).astype(np.float32)
                    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
                    y_tensor = torch.tensor([c], dtype=torch.long).to(self.device)
                    
                    if self.use_cvae:
                        x_gen = self.model.decode(z_tensor, y_tensor)
                    else:
                        x_gen = self.model.decode(z_tensor)
                    
                    X_syn.append(x_gen.cpu().numpy()[0])
                    y_syn.append(c)
        
        if len(X_syn) == 0:
            return np.array([]), np.array([])
        
        return np.array(X_syn), np.array(y_syn)
    
    def reset(self):
        """重置生成器状态"""
        self.is_fitted = False
        self.model = None
        self._class_mean_latent = {}
