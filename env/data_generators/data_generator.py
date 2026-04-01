"""
数据生成器模块 - 方案一（虚拟数据/合成数据）

提供多种生成方法来扩充候选池：
- RandomInterpolation: 简单的随机样本间插值
- SMOTE: 合成少数类过采样技术
- ADASYN: 自适应合成采样
- BorderlineSMOTE: 边界SMOTE
- VAE: 变分自编码器，在隐空间采样生成
- GAN: 生成对抗网络，Generator vs Discriminator 对抗训练
- Mixup: 两个样本按 Beta 分布权重混合
- MixupSMOTE: SMOTE + Mixup 混合
- Mixed: 随机选择上述方法之一
- LLM: 大语言模型生成

VAE 和 GAN 是真正的深度生成模型，通过学习数据的隐分布来生成新样本。
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class DataGenerator:
    """
    数据生成器基类，提供统一的接口来生成合成样本。
    """

    def __init__(self, config, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成样本。

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)
            输入特征
        y : np.ndarray, shape (n,)
            标签
        n_samples : int
            需要生成的样本数量

        Returns
        -------
        X_syn : np.ndarray
            生成的特征
        y_syn : np.ndarray
            生成的标签
        """
        raise NotImplementedError


class RandomInterpolationGenerator(DataGenerator):
    """
    随机插值生成器。

    在同类样本对之间进行随机线性插值生成新样本。
    这是一种最简单的生成方法，不依赖任何外部库。
    random_balance=True 时按类别比例生成，避免全部为少数类。
    """

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用随机插值生成合成样本。

        random_balance=True: 按类别比例生成，两类都有。
        random_balance=False: 仅对少数类插值（原行为）。
        """
        balance = getattr(self.cfg, "random_balance", False)
        classes, counts = np.unique(y, return_counts=True)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        valid_classes = [c for c in classes if len(class_indices[c]) >= 2]

        if not valid_classes:
            return np.array([]), np.array([])

        if balance and len(valid_classes) >= 2:
            # 按原始类别比例分配 n_samples
            probs = counts / counts.sum()
            n_per_class = []
            remainder = n_samples
            for k, c in enumerate(classes):
                if k == len(classes) - 1:
                    n_per_class.append(remainder)
                else:
                    nk = max(0, int(round(n_samples * probs[k])))
                    n_per_class.append(min(nk, remainder))
                    remainder -= n_per_class[-1]
        else:
            minority_class = classes[np.argmin(counts)]
            n_per_class = [n_samples if c == minority_class else 0 for c in classes]

        cat_idx = getattr(self.cfg, "categorical_indices", None) or []

        X_syn = []
        y_syn = []
        for c, n_c in zip(classes, n_per_class):
            if n_c <= 0 or c not in valid_classes:
                continue
            idx = class_indices[c]
            for _ in range(n_c):
                i1, i2 = self.rng.choice(len(idx), size=2, replace=False)
                ii1, ii2 = idx[i1], idx[i2]
                alpha = self.rng.uniform(0.1, 0.9)
                x_new = np.zeros(X.shape[1], dtype=float)
                for j in range(X.shape[1]):
                    if j in cat_idx:
                        x_new[j] = X[ii1, j] if self.rng.random() < 0.5 else X[ii2, j]
                    else:
                        x_new[j] = X[ii1, j] * alpha + X[ii2, j] * (1 - alpha)
                X_syn.append(x_new)
                y_syn.append(c)

        if not X_syn:
            return np.array([]), np.array([])
        return np.array(X_syn), np.array(y_syn)


class SMOTEGenerator(DataGenerator):
    """
    SMOTE生成器。

    使用imbalanced-learn库的SMOTE进行合成样本生成。
    当数据已平衡时，对每个类分别生成，保持类别比例。
    """

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用SMOTE生成合成样本。"""
        if not _HAS_IMBLEARN:
            return np.array([]), np.array([])

        n_orig = len(X)
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            return np.array([]), np.array([])

        k = min(getattr(self.cfg, "smote_k", 5), n_orig - 1, 4)
        if k < 1:
            return np.array([]), np.array([])

        seed = self.cfg.seed
        if hasattr(self, 'step_count'):
            seed = int(seed + self.step_count)

        try:
            smote = SMOTE(
                k_neighbors=k,
                sampling_strategy="minority",
                random_state=seed,
            )
            X_res, y_res = smote.fit_resample(X, y)
        except Exception:
            return np.array([]), np.array([])

        n_syn = len(X_res) - n_orig
        if n_syn <= 0:
            # 数据已平衡：对每个类单独生成，保持原始比例
            return self._generate_balanced(X, y, n_samples, k, seed)

        X_syn = X_res[n_orig:]
        y_syn = y_res[n_orig:]

        if n_syn > n_samples:
            idx = self.rng.choice(n_syn, size=n_samples, replace=False)
            X_syn = X_syn[idx]
            y_syn = y_syn[idx]

        return X_syn, y_syn

    def _generate_balanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        k: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """数据已平衡时，按原始比例对每个类单独生成 SMOTE 样本。"""
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        X_syn, y_syn = [], []
        for c, p in zip(classes, probs):
            n_c = max(1, int(round(n_samples * p)))
            mask = y == c
            X_c = X[mask]

            if len(X_c) < k + 1:
                X_syn.extend(X_c[:1].tolist() * n_c)
                y_syn.extend([c] * n_c)
                continue

            try:
                smote = SMOTE(
                    k_neighbors=min(k, len(X_c) - 1),
                    sampling_strategy="minority",
                    random_state=seed,
                )
                X_c_res, _ = smote.fit_resample(X_c, np.full(len(X_c), c))
                X_syn.extend(X_c_res[len(X_c):].tolist())
                y_syn.extend([c] * (len(X_c_res) - len(X_c)))
            except Exception:
                X_syn.extend(X_c[:n_c].tolist())
                y_syn.extend([c] * n_c)

        if not X_syn:
            return np.array([]), np.array([])

        idx = self.rng.permutation(len(X_syn))
        return np.array(X_syn)[idx][:n_samples], np.array(y_syn)[idx][:n_samples]


class ADASYNGenerator(DataGenerator):
    """
    ADASYN生成器。

    自适应合成采样，根据样本密度自适应生成更多合成样本。
    会在边界区域生成更多样本。
    """

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用ADASYN生成合成样本。"""
        if not _HAS_IMBLEARN:
            return np.array([]), np.array([])

        n_orig = len(X)
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            return np.array([]), np.array([])

        k = min(getattr(self.cfg, "smote_k", 5), n_orig - 1, 4)  # 确保k不超过少数类样本数-1
        if k < 1:
            return np.array([]), np.array([])

        # 获取随机种子
        seed = self.cfg.seed
        if hasattr(self, 'step_count'):
            seed = int(seed + self.step_count)

        try:
            adasyn = ADASYN(
                n_neighbors=k,
                random_state=seed,
            )
            X_res, y_res = adasyn.fit_resample(X, y)
        except Exception:
            return np.array([]), np.array([])

        n_syn = len(X_res) - n_orig
        if n_syn <= 0:
            return np.array([]), np.array([])

        X_syn = X_res[n_orig:]
        y_syn = y_res[n_orig:]

        if n_syn > n_samples:
            idx = self.rng.choice(n_syn, size=n_samples, replace=False)
            X_syn = X_syn[idx]
            y_syn = y_syn[idx]

        return X_syn, y_syn


class BorderlineSMOTEGenerator(DataGenerator):
    """
    BorderlineSMOTE生成器。

    专注于边界样本的SMOTE变体，只在少数类的边界样本附近进行插值。
    边界样本定义为：其K近邻中多数类样本数量 > K/2 但 < K。
    """

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用BorderlineSMOTE生成合成样本。"""
        if not _HAS_IMBLEARN:
            return np.array([]), np.array([])

        n_orig = len(X)
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            return np.array([]), np.array([])

        k = min(getattr(self.cfg, "smote_k", 5), n_orig - 1, 4)  # 确保k不超过少数类样本数-1
        if k < 1:
            return np.array([]), np.array([])

        # 获取随机种子
        seed = self.cfg.seed
        if hasattr(self, 'step_count'):
            seed = int(seed + self.step_count)

        try:
            bsmote = BorderlineSMOTE(
                k_neighbors=k,
                kind="borderline-1",
                random_state=seed,
            )
            X_res, y_res = bsmote.fit_resample(X, y)
        except Exception:
            return np.array([]), np.array([])

        n_syn = len(X_res) - n_orig
        if n_syn <= 0:
            return np.array([]), np.array([])

        X_syn = X_res[n_orig:]
        y_syn = y_res[n_orig:]

        if n_syn > n_samples:
            idx = self.rng.choice(n_syn, size=n_samples, replace=False)
            X_syn = X_syn[idx]
            y_syn = y_syn[idx]

        return X_syn, y_syn


class MixupGenerator(DataGenerator):
    """
    Mixup数据增强生成器。

    Mixup是一种数据增强技术，通过对两个随机样本进行线性插值来生成新样本：
    x_new = λ * x_i + (1 - λ) * x_j
    y_new = λ * y_i + (1 - λ) * y_j

    其中λ来自Beta(α, α)分布。

    Mixup的优势：
    - 增强模型的泛化能力
    - 提高对对抗样本的鲁棒性
    - 减少对错误标签的记忆
    """

    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.alpha = getattr(config, "mixup_alpha", 0.2)
        self.mixup_k = getattr(config, "mixup_k", 1)
        self.same_class_only = getattr(config, "mixup_same_class", True)
        self.cat_idx = getattr(config, "categorical_indices", None)

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Mixup生成合成样本。

        对于每个新样本，随机选择两个样本进行混合。
        mixup_same_class=True 时仅同类混合，适合表格数据。
        """
        n_orig = len(X)
        
        if n_orig < 2:
            return np.array([]), np.array([])

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 计算类别
        classes, counts = np.unique(y, return_counts=True)
        
        if len(classes) < 2 or self.same_class_only:
            # 同类混合：按类别分组插值，保证标签一致
            return self._generate_same_class_mixup(X, y, n_samples, classes)
        
        X_syn = []
        y_syn = []
        
        # 计算Beta分布的λ值
        lam = self.rng.beta(self.alpha, self.alpha, size=n_samples)
        
        for i in range(n_samples):
            # 随机选择两个不同的样本
            idx_i = self.rng.integers(0, n_orig)
            idx_j = self.rng.integers(0, n_orig)
            
            while idx_j == idx_i:
                idx_j = self.rng.integers(0, n_orig)
            
            lambda_val = lam[i]
            
            # Mixup特征混合
            x_new = lambda_val * X[idx_i] + (1 - lambda_val) * X[idx_j]
            
            # 标签混合（对于分类，使用硬标签或软标签）
            y_i = y[idx_i]
            y_j = y[idx_j]
            
            # 对于分类任务，使用随机硬标签或软标签
            if getattr(self.cfg, 'mixup_soft_label', False):
                # 软标签混合
                y_new = lambda_val * y_i + (1 - lambda_val) * y_j
            else:
                # 随机选择硬标签（更常用）
                y_new = y_i if self.rng.random() < lambda_val else y_j
            
            X_syn.append(x_new)
            y_syn.append(y_new)
        
        return np.array(X_syn), np.array(y_syn)

    def _generate_same_class_mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        classes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """同类 Mixup：仅在同类样本间插值，保持标签一致。

        - 数值列：线性插值
        - 分类列（ordinal 编码）：最近邻复制（避免产生非法类别值）
        """
        class_indices = {c: np.where(y == c)[0] for c in classes}
        valid_classes = [c for c in classes if len(class_indices[c]) >= 2]
        if not valid_classes:
            return np.array([]), np.array([])

        cat_idx = self.cat_idx or []

        X_syn = []
        y_syn = []
        lam = self.rng.beta(self.alpha, self.alpha, size=n_samples)

        for i in range(n_samples):
            c = self.rng.choice(valid_classes)
            idx = class_indices[c]
            idx_i, idx_j = self.rng.choice(len(idx), size=2, replace=False)
            i1, i2 = idx[idx_i], idx[idx_j]
            lambda_val = lam[i]

            x_new = np.zeros(X.shape[1], dtype=float)
            for j in range(X.shape[1]):
                if j in cat_idx:
                    x_new[j] = X[i1, j] if self.rng.random() < 0.5 else X[i2, j]
                else:
                    x_new[j] = lambda_val * X[i1, j] + (1 - lambda_val) * X[i2, j]

            X_syn.append(x_new)
            y_syn.append(c)

        return np.array(X_syn), np.array(y_syn)

    def _generate_same_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """当只有一个类别时的生成方法（类似RandomInterpolation）"""
        n = len(X)
        X_syn = []
        y_syn = []
        
        for _ in range(n_samples):
            idx1 = self.rng.integers(0, n)
            idx2 = self.rng.integers(0, n)
            
            while idx2 == idx1:
                idx2 = self.rng.integers(0, n)
            
            alpha = self.rng.uniform(0.1, 0.9)
            x_new = X[idx1] * alpha + X[idx2] * (1 - alpha)
            
            X_syn.append(x_new)
            y_syn.append(y[idx1])
        
        return np.array(X_syn), np.array(y_syn)


class MixupSMOTEGenerator(DataGenerator):
    """
    Mixup + SMOTE 混合生成器。

    先使用SMOTE生成样本，然后对SMOTE生成的样本和原始样本应用Mixup。
    这样可以结合两种方法的优势。
    """

    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.smote = SMOTEGenerator(config, rng)
        self.mixup = MixupGenerator(config, rng)
        self.mix_prob = getattr(config, "mixup_smote_prob", 0.5)  # Mixup应用概率

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用Mixup+SMOTE生成合成样本。"""
        # 首先用SMOTE生成一些样本
        n_smote = int(n_samples * 0.5)
        X_smote, y_smote = self.smote.generate(X, y, n_smote)
        
        if len(X_smote) == 0:
            # 如果SMOTE失败，回退到Mixup
            return self.mixup.generate(X, y, n_samples)
        
        # 合并原始样本和SMOTE样本
        X_combined = np.vstack([X, X_smote])
        y_combined = np.hstack([y, y_smote])
        
        # 剩余样本用Mixup生成
        n_mixup = n_samples - len(X_smote)
        X_mixup, y_mixup = self.mixup.generate(X_combined, y_combined, n_mixup)
        
        # 合并结果
        X_syn = np.vstack([X_smote, X_mixup]) if len(X_mixup) > 0 else X_smote
        y_syn = np.hstack([y_smote, y_mixup]) if len(y_mixup) > 0 else y_smote
        
        return X_syn, y_syn


class MixedGenerator(DataGenerator):
    """
    混合生成器。

    结合多种生成方法，每次随机选择一种方法使用。
    可以增加生成样本的多样性。
    """

    def __init__(self, config, rng: np.random.Generator, generators=None):
        super().__init__(config, rng)
        self.generators = generators or [
            RandomInterpolationGenerator(config, rng),
            SMOTEGenerator(config, rng),
        ]

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机选择一种生成器生成样本。"""
        generator = self.rng.choice(self.generators)
        return generator.generate(X, y, n_samples)


# ============================================================
# VAE 生成器
# ============================================================

class VAEGenerator(DataGenerator):
    """
    Variational Autoencoder (VAE) 生成器。

    VAE 通过学习数据的隐分布来生成新样本：
    1. Encoder 将输入 x 编码为隐空间的均值和方差
    2. 从隐分布中采样 z
    3. Decoder 从 z 重构回 x

    生成时只需从隐空间采样并通过 Decoder。
    """

    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.latent_dim = getattr(config, "vae_latent_dim", 16)
        self.hidden_dim = getattr(config, "vae_hidden_dim", 128)
        self.epochs = getattr(config, "vae_epochs", 100)
        self.batch_size = getattr(config, "vae_batch_size", 64)
        self.lr = getattr(config, "vae_lr", 0.001)
        self._vae = None
        self._is_fitted = False

    def _build_vae(self, input_dim: int, device: torch.device):
        """构建 VAE 模型。"""
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dim):
                super().__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def reparameterize(self, mu, logvar):
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                return mu

            def forward(self, x):
                h = self.encoder(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                z = self.reparameterize(mu, logvar)
                return self.decoder(z), mu, logvar

        return VAE(input_dim, self.latent_dim, self.hidden_dim).to(device)

    def _train_vae(self, X: np.ndarray) -> Optional[torch.nn.Module]:
        """训练 VAE。"""
        if not _HAS_TORCH:
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X.shape[1]

        # 确保数据足够
        if len(X) < 10:
            return None

        # 标准化数据
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std

        X_tensor = torch.FloatTensor(X_norm)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)

        vae = self._build_vae(input_dim, device)
        optimizer = optim.Adam(vae.parameters(), lr=self.lr)

        vae.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()

                recon, mu, logvar = vae(batch_x)
                recon_loss = nn.functional.mse_loss(recon, batch_x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.1 * kl_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"  [VAE] Epoch {epoch}/{self.epochs}, Loss: {total_loss / len(X):.4f}")

        vae.eval()
        self._vae = vae
        self._X_mean = X_mean
        self._X_std = X_std
        self._device = device
        return vae

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用 VAE 生成合成样本。"""
        if not _HAS_TORCH:
            return np.array([]), np.array([])

        # 训练 VAE（如需要）
        if self._vae is None:
            self._train_vae(X)

        if self._vae is None:
            return np.array([]), np.array([])

        # 找到少数类
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]

        # 从隐空间采样并生成
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self._device)
            X_gen_norm = self._vae.decoder(z).cpu().numpy()

        # 反标准化
        X_gen = X_gen_norm * self._X_std + self._X_mean

        return X_gen, np.full(n_samples, minority_class)


# ============================================================
# GAN 生成器
# ============================================================

class GANGenerator(DataGenerator):
    """
    Generative Adversarial Network (GAN) 生成器。

    GAN 由两个网络组成：
    1. Generator (G): 从随机噪声生成假样本
    2. Discriminator (D): 区分真样本和假样本

    两者对抗训练，最终 G 能生成接近真实分布的样本。
    """

    def __init__(self, config, rng: np.random.Generator):
        super().__init__(config, rng)
        self.latent_dim = getattr(config, "gan_latent_dim", 32)
        self.hidden_dim = getattr(config, "gan_hidden_dim", 128)
        self.epochs = getattr(config, "gan_epochs", 200)
        self.batch_size = getattr(config, "gan_batch_size", 64)
        self.lr = getattr(config, "gan_lr", 0.0002)
        self._generator = None
        self._discriminator = None
        self._X_mean = None
        self._X_std = None
        self._device = None

    def _build_networks(self, input_dim: int, latent_dim: int, hidden_dim: int, device: torch.device):
        """构建 Generator 和 Discriminator。"""
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, input_dim),
                )

            def forward(self, z):
                return self.net(z)

        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        return Generator().to(device), Discriminator().to(device)

    def _train_gan(self, X: np.ndarray) -> bool:
        """训练 GAN。"""
        if not _HAS_TORCH:
            return False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X.shape[1]

        if len(X) < 10:
            return False

        # 标准化
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._X_mean) / self._X_std

        X_tensor = torch.FloatTensor(X_norm)
        dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)

        generator, discriminator = self._build_networks(input_dim, self.latent_dim, self.hidden_dim, device)
        optimizer_G = optim.Adam(generator.parameters(), lr=self.lr)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=self.lr)

        criterion = nn.BCELoss()

        generator.train()
        discriminator.train()

        for epoch in range(self.epochs):
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                batch_size_actual = batch_x.size(0)

                real_labels = torch.ones(batch_size_actual, 1).to(device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(device)

                # 训练 Discriminator
                optimizer_D.zero_grad()
                outputs = discriminator(batch_x)
                d_loss_real = criterion(outputs, real_labels)

                z = torch.randn(batch_size_actual, self.latent_dim).to(device)
                fake_x = generator(z)
                outputs = discriminator(fake_x.detach())
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # 训练 Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size_actual, self.latent_dim).to(device)
                fake_x = generator(z)
                outputs = discriminator(fake_x)
                g_loss = criterion(outputs, real_labels)

                g_loss.backward()
                optimizer_G.step()

            if epoch % 50 == 0:
                print(f"  [GAN] Epoch {epoch}/{self.epochs}, D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        self._generator = generator
        self._discriminator = discriminator
        self._device = device
        return True

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用 GAN 生成合成样本。"""
        if not _HAS_TORCH:
            return np.array([]), np.array([])

        if self._generator is None:
            if not self._train_gan(X):
                return np.array([]), np.array([])

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]

        self._generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self._device)
            X_gen_norm = self._generator(z).cpu().numpy()

        X_gen = X_gen_norm * self._X_std + self._X_mean

        return X_gen, np.full(n_samples, minority_class)


def create_generator(
    config,
    method: str = "smote",
    rng: Optional[np.random.Generator] = None,
) -> DataGenerator:
    """
    工厂函数：根据配置创建相应的生成器。

    Parameters
    ----------
    config : Config
        配置对象
    method : str
        生成方法，可选：'random', 'smote', 'adasyn', 'borderline', 'mixed', 'mixup', 'mixup_smote', 'vae', 'gan', 'llm', 'llm_text'
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    DataGenerator
        生成器实例
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    generators = {
        "random": RandomInterpolationGenerator,
        "smote": SMOTEGenerator,
        "adasyn": ADASYNGenerator,
        "borderline": BorderlineSMOTEGenerator,
        "mixed": MixedGenerator,
        "mixup": MixupGenerator,
        "mixup_smote": MixupSMOTEGenerator,
        "vae": VAEGenerator,
        "gan": GANGenerator,
    }

    # LLM 生成器
    llm_methods = ["llm", "llm_text"]
    if method.lower() in llm_methods:
        try:
            from .llm_generator import create_llm_generator
            return create_llm_generator(config, method, rng)
        except ImportError as e:
            print(f"[create_generator] LLM 生成器导入失败: {e}")
            print("[create_generator] 回退到 SMOTE 生成器")
            method = "smote"

    generator_class = generators.get(method.lower(), SMOTEGenerator)

    if generator_class == MixedGenerator:
        return generator_class(config, rng, generators=[
            RandomInterpolationGenerator(config, rng),
            SMOTEGenerator(config, rng),
        ])

    return generator_class(config, rng)
