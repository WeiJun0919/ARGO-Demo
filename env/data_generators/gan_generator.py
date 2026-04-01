"""
GAN 数据生成器 - 生成对抗网络

使用生成对抗网络（GAN）学习数据分布并生成新合成样本。
使用条件 GAN (CGAN) 来按类别生成样本。

支持两种 GAN 架构：
1. Vanilla GAN (MLP)
2. WGAN-GP (Wasserstein GAN with Gradient Penalty)
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


class Generator(nn.Module):
    """GAN 生成器"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = [256, 128], n_classes: int = 0):
        super().__init__()
        
        self.n_classes = n_classes
        input_dim = latent_dim + n_classes if n_classes > 0 else latent_dim
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z, y=None):
        if y is not None and self.n_classes > 0:
            # 条件生成
            y_onehot = torch.zeros(y.size(0), self.n_classes).to(z.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            zy = torch.cat([z, y_onehot], dim=1)
            return self.model(zy)
        return self.model(z)


class Discriminator(nn.Module):
    """GAN 判别器"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128], n_classes: int = 0):
        super().__init__()
        
        self.n_classes = n_classes
        input_dim = input_dim + n_classes if n_classes > 0 else input_dim
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y=None):
        if y is not None and self.n_classes > 0:
            # 条件判别
            y_onehot = torch.zeros(y.size(0), self.n_classes).to(x.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            xy = torch.cat([x, y_onehot], dim=1)
            return self.model(xy)
        return self.model(x)


@DataGeneratorRegistry.register("gan")
class GANGenerator(BaseDataGenerator):
    """
    GAN 数据生成器
    
    使用生成对抗网络学习数据分布并生成新样本。
    支持条件生成（按类别生成）。
    
    参数：
    - latent_dim: 潜在空间维度
    - hidden_dims: 隐藏层维度列表
    - epochs: 训练轮数
    - batch_size: 批次大小
    - lr: 学习率
    - use_wgan: 是否使用 WGAN-GP
    - use_cgan: 是否使用条件 GAN
    - discriminator_steps: 判别器训练步数（每生成器1步）
    """
    
    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for GANGenerator")
        
        # GAN 参数
        self.latent_dim = getattr(config, 'gan_latent_dim', 64)
        self.hidden_dims = getattr(config, 'gan_hidden_dims', [256, 128])
        self.epochs = getattr(config, 'gan_epochs', 100)
        self.batch_size = getattr(config, 'gan_batch_size', 64)
        self.lr = getattr(config, 'gan_lr', 1e-3)
        self.use_wgan = getattr(config, 'gan_use_wgan', False)
        self.use_cgan = getattr(config, 'gan_use_cgan', True)
        self.d_steps = getattr(config, 'gan_d_steps', 5)  # 判别器训练步数
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.discriminator = None
        self.is_fitted = False
        self.n_classes = 2
        self.input_dim = None
        
        # 缓存各类别的噪声样本
        self._class_generated = {}
    
    def _build_models(self, input_dim: int, n_classes: int):
        """构建生成器和判别器"""
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # 生成器
        self.generator = Generator(
            latent_dim=self.latent_dim,
            output_dim=input_dim,
            hidden_dims=self.hidden_dims,
            n_classes=n_classes if self.use_cgan else 0,
        ).to(self.device)
        
        # 判别器
        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            n_classes=n_classes if self.use_cgan else 0,
        ).to(self.device)
        
        # 优化器
        if self.use_wgan:
            self.opt_g = optim.RMSprop(self.generator.parameters(), lr=self.lr)
            self.opt_d = optim.RMSprop(self.discriminator.parameters(), lr=self.lr)
        else:
            self.opt_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def _train_gan(self, X: np.ndarray, y: np.ndarray):
        """训练 GAN"""
        n_samples, input_dim = X.shape
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        
        # 获取类别
        classes = np.unique(y)
        self.n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[c] for c in y])
        
        # 构建模型
        self._build_models(input_dim, self.n_classes)
        
        # 转换为 PyTorch 张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, n_samples), shuffle=True)
        
        # 标签
        real_labels = torch.ones(self.batch_size, 1).to(self.device)
        fake_labels = torch.zeros(self.batch_size, 1).to(self.device)
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size = batch_x.size(0)
                
                # ---------------------
                # 训练判别器
                # ---------------------
                self.opt_d.zero_grad()
                
                # 真实样本
                if self.use_cgan:
                    real_output = self.discriminator(batch_x, batch_y[:batch_size])
                else:
                    real_output = self.discriminator(batch_x)
                
                if self.use_wgan:
                    d_loss_real = real_output.mean()
                else:
                    d_loss_real = self.bce_loss(real_output, real_labels[:batch_size])
                
                # 生成样本
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                if self.use_cgan:
                    fake_x = self.generator(z, batch_y[:batch_size])
                else:
                    fake_x = self.generator(z)
                
                if self.use_cgan:
                    fake_output = self.discriminator(fake_x.detach(), batch_y[:batch_size])
                else:
                    fake_output = self.discriminator(fake_x.detach())
                
                if self.use_wgan:
                    d_loss_fake = fake_output.mean()
                    d_loss = d_loss_fake - d_loss_real
                    
                    # WGAN-GP 梯度惩罚
                    if self.use_wgan:
                        alpha = torch.rand(batch_size, 1).to(self.device)
                        interpolates = alpha * batch_x + (1 - alpha) * fake_x
                        interpolates.requires_grad_(True)
                        if self.use_cgan:
                            disc_interpolates = self.discriminator(interpolates, batch_y[:batch_size])
                        else:
                            disc_interpolates = self.discriminator(interpolates)
                        gradients = torch.autograd.grad(
                            outputs=disc_interpolates,
                            inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        d_loss += 10 * gradient_penalty
                else:
                    d_loss_fake = self.bce_loss(fake_output, fake_labels[:batch_size])
                    d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.opt_d.step()
                
                # ---------------------
                # 训练生成器
                # ---------------------
                if not self.use_wgan or epoch % self.d_steps == 0:
                    self.opt_g.zero_grad()
                    
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)
                    if self.use_cgan:
                        fake_x = self.generator(z, batch_y[:batch_size])
                    else:
                        fake_x = self.generator(z)
                    
                    if self.use_cgan:
                        fake_output = self.discriminator(fake_x, batch_y[:batch_size])
                    else:
                        fake_output = self.discriminator(fake_x)
                    
                    if self.use_wgan:
                        g_loss = -fake_output.mean()
                    else:
                        g_loss = self.bce_loss(fake_output, real_labels[:batch_size])
                    
                    g_loss.backward()
                    self.opt_g.step()
        
        self.is_fitted = True
    
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
        
        # 训练 GAN（如果还没训练）
        if not self.is_fitted:
            try:
                self._train_gan(X, y)
            except Exception as e:
                warnings.warn(f"GAN training failed: {e}")
                return np.array([]), np.array([])
        
        self.generator.eval()
        
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
                # 采样噪声并生成
                n_gen = min(samples_per_class, n_samples - len(y_syn))
                
                z = torch.randn(n_gen, self.latent_dim).to(self.device)
                y_tensor = torch.tensor([c] * n_gen, dtype=torch.long).to(self.device)
                
                if self.use_cgan:
                    x_gen = self.generator(z, y_tensor)
                else:
                    x_gen = self.generator(z)
                
                X_syn.append(x_gen.cpu().numpy())
                y_syn.extend([c] * n_gen)
        
        if len(X_syn) == 0:
            return np.array([]), np.array([])
        
        return np.vstack(X_syn), np.array(y_syn)
    
    def reset(self):
        """重置生成器状态"""
        self.is_fitted = False
        self.generator = None
        self.discriminator = None
