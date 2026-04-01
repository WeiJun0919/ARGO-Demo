"""
图像数据清洗环境 - 使用 HRL 进行图像标签噪声清洗

支持：
- 使用预训练 ResNet50 提取图像特征
- 三种动作：modify_labels, delete_samples, add_samples
- 下游分类器评估准确率

使用方式：
    from env.image_cleaning_env import ImageDataCleaningEnv

    env = ImageDataCleaningEnv(config)
    state = env.reset()

    # 选择动作
    action_idx = 1  # modify_labels

    # 获取候选
    cand = env.get_candidates(action_idx)

    # 执行动作
    state, reward, done, info = env.step(action_idx, selected_indices)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple, List

from env.feature_extractors import ImageFeatureExtractor, create_feature_extractor
from env.data_loaders import ImageDataLoader, NoisyImageDataset
from env.classifiers import ImageDownstreamClassifier, create_image_classifier


class ResNetCIFARFinetuner:
    """
    端到端 fine-tune 评估器——直接 fine-tune 完整 ResNet50（适配 CIFAR-10），
    每个 iteration 后在当前工作集上训练，在完整干净测试集上评估。

    相比冻结特征 + 随机 MLP 的方案：
    - ResNet50 骨干网络参与训练，真正适配 CIFAR-10 图片分布
    - 只训练最后一层的权重，收敛快、开销小
    - 测试集保持不变（干净标签），准确率直接反映清洗质量
    - 支持模型复用，可跨 episode 保持训练状态以加速
    """

    def __init__(
        self,
        num_classes: int = 10,
        device: torch.device = None,
        quick_epochs: int = 5,
        full_epochs: int = 30,
        lr: float = 0.05,
        batch_size: int = 128,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.num_classes = num_classes
        self.quick_epochs = quick_epochs
        self.full_epochs = full_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.test_images = None
        self.test_labels = None
        self.best_accuracy = 0.0

        # ── 模型复用支持 ──────────────────────────────────────────
        self._cached_model = None      # 复用的模型
        self._cached_optimizer = None  # 复用的优化器
        self._cached_scheduler = None # 复用的学习率调度器
        self._last_trained_on = None  # 上次训练的索引（用于判断是否需要重训）
        self._reuse_enabled = True     # 是否启用复用

    def set_test_set(self, images: np.ndarray, labels: np.ndarray):
        """传入完整测试集 raw 图像（uint8 numpy），用于评估。"""
        self.test_images = images  # (N, 32, 32, 3), uint8
        self.test_labels = np.asarray(labels, dtype=np.int64).clip(0, self.num_classes - 1)

    def _build_model(self) -> nn.Module:
        """
        构建适配 CIFAR-10 的 ResNet50（按 He et al. 2015 标准修改）：
        - 第一个 7×7 stride-2 conv → 3×3 stride-1 conv（适应 32×32 小图）
        - 去掉最后的 MaxPool（保留更多空间信息）
        - 保留 ImageNet 预训练权重（backbone 不再冻结，只训 FC 层）
        """
        import torchvision.models as models

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 修改第一层：7×7 stride2 → 3×3 stride1
        old_conv1 = model.conv1
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            model.conv1.weight[:, :] = old_conv1.weight[:, :, 2:5, 2:5]

        # 去掉 MaxPool（CIFAR-10 用小图）
        model.maxpool = nn.Identity()

        # 替换最后的 FC 层
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)

        return model.to(self.device)

    def _numpy_to_tensor(self, images: np.ndarray, labels: np.ndarray,
                          augment: bool) -> TensorDataset:
        """把 raw numpy 图像转为 TensorDataset。"""
        from torchvision import transforms
        from PIL import Image as PILImage

        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # CIFAR-10 专用归一化参数
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ])

        img_list = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            pil = PILImage.fromarray(img)
            img_list.append(transform(pil))

        X = torch.stack(img_list).to(self.device)
        y = torch.LongTensor(labels).to(self.device)
        return TensorDataset(X, y)

    def _collate_eval(self) -> TensorDataset:
        return self._numpy_to_tensor(self.test_images, self.test_labels, augment=False)

    def _collate_train(self, indices: np.ndarray, images: np.ndarray,
                       labels: np.ndarray, augment: bool = True) -> TensorDataset:
        imgs = images[indices]
        lbls = np.asarray(labels, dtype=np.int64).clip(0, self.num_classes - 1)[indices]
        return self._numpy_to_tensor(imgs, lbls, augment=augment)

    def _train_and_eval(self, train_indices: np.ndarray,
                        train_images: np.ndarray, train_labels: np.ndarray,
                        epochs: int, force_new_model: bool = False) -> float:
        """训练后在完整测试集上评估。
        
        参数：
        - train_indices: 训练样本索引
        - train_images: 原始图像
        - train_labels: 标签
        - epochs: 训练轮数
        - force_new_model: 是否强制新建模型（禁用复用）
        """
        # ── 检查是否复用模型 ──────────────────────────────────────
        # 如果数据没变且有缓存模型，则复用（只训练几个 epoch）
        can_reuse = (
            self._reuse_enabled 
            and not force_new_model 
            and self._cached_model is not None
            and self._last_trained_on is not None
            and len(train_indices) == len(self._last_trained_on)
            and np.array_equal(np.sort(train_indices), np.sort(self._last_trained_on))
        )
        
        if can_reuse:
            # 复用模型，少量微调
            model = self._cached_model
            optimizer = self._cached_optimizer
            scheduler = self._cached_scheduler
            resume_epochs = epochs
            # 恢复学习率（因为 CosineAnnealing 可能降到很低）
            for pg in optimizer.param_groups:
                pg['lr'] = self.lr * 0.1  # 从一个较高的值开始微调
        else:
            # 新建模型
            model = self._build_model()
            
            # 解冻所有层，但使用 SGD + momentum（CIFAR 效果更好）
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            resume_epochs = epochs
            
            # 缓存模型供下次复用
            self._cached_model = model
            self._cached_optimizer = optimizer
            self._cached_scheduler = scheduler
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 训练时启用数据增强
        augment = True
        train_ds = self._collate_train(train_indices, train_images, train_labels, augment=augment)
        train_loader = DataLoader(
            train_ds, batch_size=min(self.batch_size, len(train_ds)),
            shuffle=True, drop_last=True,
        )
        eval_ds = self._collate_eval()
        eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)

        best_acc = 0.0
        for epoch in range(resume_epochs):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                # 梯度裁剪，防止 backbone 梯度过大
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in eval_loader:
                    out = model(bx)
                    _, p = out.max(1)
                    correct += p.eq(by).sum().item()
                    total += by.size(0)
            best_acc = correct / max(total, 1)

        # ── 保存训练状态供复用 ──────────────────────────────────
        if can_reuse or force_new_model:
            self._last_trained_on = train_indices.copy()
        
        # 清理（仅当强制新建且不复用时）
        if force_new_model and not can_reuse:
            del model
            torch.cuda.empty_cache()
        
        return best_acc

    def evaluate(self, train_indices: np.ndarray,
                 train_images: np.ndarray, train_labels: np.ndarray,
                 quick: bool = False, force_new_model: bool = False) -> float:
        """Fine-tune 并评估。quick=True 用 few epochs（快速反馈）。"""
        if self.test_images is None or len(train_indices) < 20:
            return 0.0
        epochs = self.quick_epochs if quick else self.full_epochs
        labels = np.asarray(train_labels, dtype=np.int64).clip(0, self.num_classes - 1)

        rng = np.random.RandomState(777)
        perm = rng.permutation(len(train_indices))
        val_n = max(1, int(len(perm) * 0.1))
        train_idx = train_indices[perm[val_n:]]

        return self._train_and_eval(train_idx, train_images, labels, epochs, force_new_model)


class ImageAccuracyTracker:
    """
    包装 ResNetCIFARFinetuner，提供与旧接口兼容的 evaluate_and_track()。
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        quick_epochs: int = 3,
        full_epochs: int = 20,
    ):
        self.device = device
        self.num_classes = num_classes
        self.finetuner = ResNetCIFARFinetuner(
            num_classes=num_classes, device=device,
            quick_epochs=quick_epochs, full_epochs=full_epochs,
        )
        self.finetuner.set_test_set(test_images, test_labels)
        self.best_accuracy = 0.0

    def evaluate_and_track(
        self,
        train_indices: np.ndarray,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        quick: bool = False,
        force_new_model: bool = False,
    ) -> float:
        if len(train_indices) < 20:
            return 0.0
        acc = self.finetuner.evaluate(train_indices, train_images, train_labels, 
                                      quick=quick, force_new_model=force_new_model)
        if not quick and acc > self.best_accuracy:
            self.best_accuracy = acc
        return acc

    def reset_to_initial(self):
        """重置跟踪器，清除缓存的模型"""
        self.best_accuracy = 0.0
        # 清除缓存的模型
        if self.finetuner is not None:
            self.finetuner._cached_model = None
            self.finetuner._cached_optimizer = None
            self.finetuner._cached_scheduler = None
            self.finetuner._last_trained_on = None


class ImageDataCleaningEnv:
    """
    图像数据清洗环境

    用于 RL 智能体进行图像标签噪声清洗。
    """

    def __init__(self, config):
        """
        参数：
        - config: 配置对象
        """
        self.cfg = config
        self.device = torch.device(config.device)
        self.sel_agent = None  # 由 train 中 set_sel_agent 绑定，与 MultiSelector 环境接口一致

        # 加载数据集
        self._load_dataset()

        # 初始化特征提取器
        self._setup_feature_extractor()

        # 预计算特征
        self._precompute_features()

        # 预计算测试集特征（用于参考评估集）
        self._precompute_test_features()

        # 子集参考（可选，用于调试）；tracker 使用完整测试集评估
        self._setup_reference_sets()

        # 初始化专用准确率跟踪器
        self._setup_accuracy_tracker()

        # 当前数据索引
        self.current_indices = np.arange(len(self.dataset))
        self.n_steps = 0
        self.step_count = 0

        # 与 DataCleaningEnvMultiSelector 兼容的属性
        self.action_counts = np.zeros(5)  # 5 个动作的计数
        self.current_acc = 0.0
        self.best_acc = 0.0

        # ── 评估缓存（加速训练）──────────────────────────────────────────
        # 评估频率控制：从 config 读取
        self._eval_every_n = getattr(config, 'image_eval_every_n_steps', 9999)
        self._last_eval_step = 0
        self._cached_acc = 0.0  # 缓存的准确率
        self._acc_is_fresh = False  # 缓存是否有效

        # ── 准确率跟踪器 epochs 设置────────────────────────────────────
        self._quick_epochs = getattr(config, 'image_eval_epochs_quick', 5)
        self._full_epochs = getattr(config, 'image_eval_epochs_full', 20)

        # 确保 config 有 action_names 属性（用于 info 输出）
        if not hasattr(self.cfg, 'action_names'):
            self.cfg.action_names = ['modify_features', 'modify_labels', 'delete_samples', 'add_samples', 'no_op']

        print(f"[ImageDataCleaningEnv] 初始化完成:")
        print(f"  - 训练样本数: {len(self.dataset)}")
        print(f"  - 特征维度: {self.feature_dim}")
        print(f"  - 标签噪声比例: {self.label_noise_ratio*100:.1f}%")
        print(f"  - 设备: {self.device}")

    def _load_dataset(self):
        """加载数据集"""
        # 确定 torchvision 内部数据集名称
        # cifar10_resnet50 → cifar10（因为 torchvision.datasets.CIFAR10 内部就叫 cifar10）
        _dn = self.cfg.dataset_name.lower()
        if 'cifar10' in _dn:
            self.dataset_name = 'cifar10'
        elif 'cifar100' in _dn:
            self.dataset_name = 'cifar100'
        else:
            self.dataset_name = _dn

        # 创建数据加载器（使用 torchvision 标准目录名）
        loader = ImageDataLoader(
            dataset_name=self.dataset_name,
            data_dir=self.cfg.image_data_dir,  # 默认 "./datasets"
            image_size=getattr(self.cfg, 'image_height', 32),
        )

        # 加载训练集和测试集
        self.raw_dataset = loader.load_dataset(
            train=True,
            label_noise_ratio=getattr(self.cfg, 'label_noise_ratio', 0.2),
            noise_type=getattr(self.cfg, 'noise_type', 'symmetric'),
            download=True,
        )

        max_n = getattr(self.cfg, "max_train_samples", None)
        if max_n is not None and len(self.raw_dataset) > max_n:
            n_full = len(self.raw_dataset)
            self.raw_dataset = self._subset_noisy_dataset(
                self.raw_dataset, max_n, seed=getattr(self.cfg, "seed", 42)
            )
            print(
                f"  - 训练集子采样: {len(self.raw_dataset)}/{n_full} "
                f"(max_train_samples={max_n})"
            )

        # 保存原始数据
        self.images = self.raw_dataset.images
        self.true_labels = self.raw_dataset.true_labels
        self.noisy_labels = self.raw_dataset.labels
        self.num_classes = self.raw_dataset.num_classes

        # 标签噪声信息
        self.label_noise_mask = self.raw_dataset.label_noise_mask
        self.label_noise_ratio = self.raw_dataset.label_noise_ratio

        # 数据集
        self.dataset = self.raw_dataset

        # 测试集（无噪声）；训练集已 download，避免重复「Files already downloaded」提示
        self.test_dataset = loader.load_dataset(
            train=False,
            label_noise_ratio=0.0,
            download=False,
        )
        self.test_images = self.test_dataset.images
        self.test_true_labels = self.test_dataset.true_labels

        print(f"  - 数据集: {self.dataset_name}")
        print(f"  - 类别数: {self.num_classes}")
        # 调试：确认噪声注入
        n_noisy = int((self.noisy_labels != self.true_labels).sum())
        print(f"  - DEBUG: 实际噪声 {n_noisy}/{len(self.noisy_labels)} ({n_noisy/len(self.noisy_labels)*100:.1f}%)")

    @staticmethod
    def _subset_noisy_dataset(dataset: NoisyImageDataset, n: int, seed: int = 42):
        """从 NoisyImageDataset 中固定随机种子子采样 n 条（用于加速调试）。"""
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(dataset))[:n]
        return NoisyImageDataset(
            images=np.asarray(dataset.images)[idx],
            labels=np.asarray(dataset.labels)[idx],
            true_labels=np.asarray(dataset.true_labels)[idx],
            transform=dataset.transform,
        )

    def set_sel_agent(self, sel_agent):
        """与 DataCleaningEnvMultiSelector 对齐：绑定 Selector（图像管线可选使用）。"""
        self.sel_agent = sel_agent

    def _setup_feature_extractor(self):
        """初始化特征提取器"""
        model_name = getattr(
            self.cfg, 'feature_extractor_model', 'resnet50'
        )
        pretrained = getattr(self.cfg, 'extractor_pretrained', True)

        self.feature_extractor = create_feature_extractor(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.feature_extractor.eval()

        self.feature_dim = self.feature_extractor.feature_dim
        print(f"  - 特征提取器: {model_name}, 特征维度: {self.feature_dim}")

    def _precompute_features(self):
        """预计算所有图像的特征"""
        print(f"  - 预计算图像特征...")

        # 创建 DataLoader
        loader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        all_features = []
        all_images = []  # 保存原始图像用于后续操作

        self.feature_extractor.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0]  # (batch, 3, H, W)
                imgs = imgs.to(self.device)

                # 调整大小以适应预训练模型
                if imgs.shape[1] != 3 or imgs.shape[2] != 224:
                    # 需要 resize
                    imgs_resized = torch.nn.functional.interpolate(
                        imgs, size=(224, 224), mode='bilinear', align_corners=False
                    )
                else:
                    imgs_resized = imgs

                feats = self.feature_extractor.extract(imgs_resized)
                all_features.append(feats)

                # 保存原始图像
                all_images.extend(batch[0].numpy())

        self.features = torch.cat(all_features, dim=0).numpy()
        self.images_tensor = torch.tensor(
            np.array(all_images), dtype=torch.float32
        ) / 255.0 if not isinstance(all_images[0], torch.Tensor) else torch.stack(all_images)

        print(f"  - 特征预计算完成: {self.features.shape}")

    def _precompute_test_features(self):
        """预计算测试集图像的特征（用于参考集评估）"""
        print(f"  - 预计算测试集特征...")

        loader = DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

        all_test_features = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0].to(self.device)
                if imgs.shape[1] != 3 or imgs.shape[2] != 224:
                    imgs_resized = torch.nn.functional.interpolate(
                        imgs, size=(224, 224), mode='bilinear', align_corners=False
                    )
                else:
                    imgs_resized = imgs
                feats = self.feature_extractor.extract(imgs_resized)
                all_test_features.append(feats)

        self.test_features = torch.cat(all_test_features, dim=0).numpy()
        print(f"  - 测试集特征预计算完成: {self.test_features.shape}")

    def _setup_accuracy_tracker(self):
        """初始化端到端 fine-tune 准确率跟踪器"""
        self.acc_tracker = ImageAccuracyTracker(
            num_classes=self.num_classes,
            device=self.device,
            test_images=self.test_dataset.images,
            test_labels=self.test_true_labels,
            quick_epochs=3,
            full_epochs=20,
        )
        # 预测用分类器（冻结特征提取 + 轻量 MLP，用于 get_candidates 的不确定性估计）
        self.classifier = ImageDownstreamClassifier(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=512,
            dropout=0.3,
            device=self.device,
        )

    def train_classifier_for_detection(self, epochs: int = 20, batch_size: int = 128):
        """
        在检测阶段之前训练分类器，使其能可靠识别噪声样本。

        使用 true_labels（ground-truth）训练，避免 label_noise 干扰；
        训练后分类器在干净测试集上应达到 ~85%+ 准确率，
        其预测与 noisy_labels 的不一致才能有效反映噪声。
        """
        print(f"\n  [TrainClassifier] 在 true_labels 上训练 {epochs} epochs...")
        try:
            self.classifier.fit(
                train_features=self.features,
                train_labels=self.true_labels,
                epochs=epochs,
                batch_size=batch_size,
                lr=1e-3,
                early_stopping_patience=5,
                verbose=False,
            )
            # 在测试集上验证
            test_feats = self.test_features
            with torch.no_grad():
                pred = self.classifier.predict(
                    torch.FloatTensor(test_feats).to(self.device)
                )
            test_acc = (pred == self.test_true_labels).mean()
            print(f"  [TrainClassifier] 分类器测试准确率: {test_acc:.4f}")
        except Exception as e:
            print(f"  [TrainClassifier] 训练失败: {e}, 继续用未训练分类器")

    def _setup_reference_sets(self):
        """设置参考集（使用 ResNet50 特征，而非原始图像）"""
        n_ref = min(2000, len(self.test_features))
        indices = np.random.RandomState(42).permutation(len(self.test_features))[:n_ref]

        self.X_ref = self.test_features[indices]
        self.y_ref = self.test_true_labels[indices]

    def _state(self) -> np.ndarray:
        """
        构建 RL 状态

        状态 = 数据集统计特征
        """
        # 获取当前数据的特征
        current_features = self.features[self.current_indices]
        current_labels = self.noisy_labels[self.current_indices]

        # 特征统计
        mean_feat = current_features.mean(axis=0)
        std_feat = current_features.std(axis=0)

        # 标签噪声估计
        current_noise_mask = self.label_noise_mask[self.current_indices]
        estimated_noise_ratio = current_noise_mask.mean()

        # 类别分布
        class_dist = np.bincount(
            current_labels,
            minlength=self.num_classes
        ).astype(float) / max(1, len(current_labels))

        # 组装状态
        state = np.concatenate([
            mean_feat,                    # feature_dim
            std_feat,                    # feature_dim
            [estimated_noise_ratio],     # 1
            [len(self.current_indices)], # 1
            class_dist,                  # num_classes
        ])

        return state.astype(np.float32)

    @staticmethod
    def _sanitize_local_indices(selected_indices: List[int], pool_len: int) -> List[int]:
        """
        Selector / topk 给出的下标必须在当前工作集长度内 [0, pool_len)。
        若出现等于 pool_len 的越界值（典型 off-by-one），直接丢弃非法项。
        """
        if pool_len <= 0 or not selected_indices:
            return []
        arr = np.asarray(selected_indices, dtype=np.intp)
        arr = arr[(arr >= 0) & (arr < pool_len)]
        return arr.tolist()

    def get_candidates(self, action_idx: int) -> Optional[Dict]:
        """
        获取候选样本

        对于图像数据：
        - Action 1 (modify_labels): 所有样本都可以考虑修改标签
        - Action 2 (delete_samples): 所有样本都可以考虑删除
        - Action 3 (add_samples): 从增强池获取候选
        """
        n = len(self.current_indices)

        if n == 0:
            return None

        # 获取当前数据的特征
        features = self.features[self.current_indices]
        labels = self.noisy_labels[self.current_indices]

        # 候选信息
        u = np.zeros((n, 3), dtype=np.float32)

        # 如果有分类器预测，添加不确定性信息
        if self.classifier is not None:
            with torch.no_grad():
                feats_tensor = torch.FloatTensor(features).to(self.device)
                probs = self.classifier.predict_proba(feats_tensor)

            # 预测的熵（不确定性）
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)

            # 预测概率
            pred_labels = probs.argmax(axis=1)

            # 是否与当前标签一致
            label_correct = (pred_labels == labels).astype(float)

            u[:, 0] = entropy
            u[:, 1] = label_correct
            u[:, 2] = probs.max(axis=1)  # 置信度

        return {
            "indices": self.current_indices.copy(),
            "features": features,     # 兼容 image_selector_agent
            "labels": labels,        # 兼容 image_selector_agent
            "X": features,           # 兼容 train_multi_selector_v2.py
            "y": labels,             # 兼容 train_multi_selector_v2.py
            "u": u,
            "n": n,
            "is_noisy": self.label_noise_mask[self.current_indices].copy(),
        }

    def step(
        self,
        action_idx: int,
        selected_indices: List[int],
        user_feedback: str = None,
        selector_pred: np.ndarray = None,
        force_eval: bool = False,
    ) -> Tuple:
        """
        执行动作

        参数：
        - action_idx: 动作索引 (1=modify_labels, 2=delete, 3=add)
        - selected_indices: 选中的样本索引
        - user_feedback: 用户反馈
        - selector_pred: Selector 预测的值
        - force_eval: 强制评估（忽略缓存）

        返回：
        - (next_state, reward, done, info)
        """
        self.step_count += 1

        pool_len = len(self.current_indices)
        selected_indices = self._sanitize_local_indices(
            list(selected_indices) if selected_indices is not None else [],
            pool_len,
        )

        # 删除会缩短 current_indices，奖励统计必须在删除前用当前索引计算
        noise_deleted_pre = None
        if action_idx == 2 and len(selected_indices) > 0:
            sel = np.asarray(selected_indices, dtype=np.intp)
            del_global = self.current_indices[sel]
            noise_deleted_pre = int(self.label_noise_mask[del_global].sum())

        # 执行动作
        if action_idx == 1:  # modify_labels
            self._modify_labels(selected_indices, selector_pred)
        elif action_idx == 2:  # delete_samples
            self._delete_samples(selected_indices)
        elif action_idx == 3:  # add_samples
            self._add_samples(selected_indices)

        # ── 准确率评估（支持缓存加速）──────────────────────────────
        # 条件：force_eval=True 或 达到评估间隔
        should_eval = force_eval or (self.step_count - self._last_eval_step >= self._eval_every_n)
        
        if should_eval:
            prev_acc = self._evaluate_accuracy(quick=True)
            self._last_eval_step = self.step_count
            self._cached_acc = prev_acc
            self._acc_is_fresh = True
        else:
            # 使用缓存的准确率
            prev_acc = self._cached_acc
            self._acc_is_fresh = False

        # 计算选中样本中的噪声数量（用于 info 记录）
        # 注意：selected_indices 可能在动作执行后变得无效，需要重新获取
        if action_idx == 3:  # add_samples 后重新获取当前索引
            self._last_selected_noise_count = 0
            self._last_selected_total = len(selected_indices)
        elif len(selected_indices) > 0:
            sel = np.asarray(selected_indices, dtype=np.intp)
            # 再次验证索引有效性
            valid_mask = sel < len(self.current_indices)
            sel = sel[valid_mask]
            if len(sel) > 0:
                gidx = self.current_indices[sel]
                self._last_selected_noise_count = int(self.label_noise_mask[gidx].sum())
                self._last_selected_total = len(sel)
            else:
                self._last_selected_noise_count = 0
                self._last_selected_total = 0
        else:
            self._last_selected_noise_count = 0
            self._last_selected_total = 0
        self._last_action_idx = action_idx

        # 计算奖励（基于训前准确率，延迟 1 step 才能感知变化）
        reward = self._compute_reward(
            action_idx, selected_indices, prev_acc,
            noise_deleted_pre=noise_deleted_pre,
        )

        # 检查是否完成
        done = self.step_count >= self.cfg.max_steps_per_episode

        # 获取新状态
        next_state = self._state()

        # 更新 best_acc
        if prev_acc > self.best_acc:
            self.best_acc = prev_acc

        # 信息（不重复训练，只返回当前 snapshot）
        info = self._get_info()
        info["accuracy"] = prev_acc  # 对齐：用训练后的快照，不额外再训
        info["best_accuracy"] = self.best_acc
        info["acc_is_fresh"] = self._acc_is_fresh  # 标记是否为新评估
        self.current_acc = prev_acc

        return next_state, reward, done, info

    def _modify_labels(self, selected_indices: List[int], pred: np.ndarray = None):
        """
        修改标签

        使用分类器预测的标签替换当前标签
        """
        if len(selected_indices) == 0:
            return

        # 验证并过滤非法索引
        sel = np.asarray(selected_indices, dtype=np.intp)
        valid_mask = sel < len(self.current_indices)
        sel = sel[valid_mask]
        if len(sel) == 0:
            return

        # 获取当前数据的索引
        current_idx = self.current_indices[sel]

        if pred is not None:
            # Selector 可能传入 (N,) 类别索引，或 (N, num_classes) logits
            new_labels = pred
            if isinstance(new_labels, torch.Tensor):
                new_labels = new_labels.detach().cpu().numpy()
            new_labels = np.asarray(new_labels)
            if new_labels.ndim == 2:
                new_labels = new_labels.argmax(axis=1)
            new_labels = new_labels.astype(np.int64).reshape(-1)
        else:
            # 使用分类器预测
            features = self.features[current_idx]
            with torch.no_grad():
                feats_tensor = torch.FloatTensor(features).to(self.device)
                pred_labels = self.classifier.predict(feats_tensor)
            new_labels = pred_labels

        # 更新标签
        self.noisy_labels[current_idx] = new_labels

    def _delete_samples(self, selected_indices: List[int]):
        """删除样本"""
        if len(selected_indices) == 0:
            return

        # 从当前数据中删除
        mask = np.ones(len(self.current_indices), dtype=bool)
        mask[selected_indices] = False
        self.current_indices = self.current_indices[mask]

    def _add_samples(self, selected_indices: List[int]):
        """
        添加样本

        从测试集或外部数据源添加样本
        """
        if len(selected_indices) == 0:
            return

        # 这里可以从测试集或额外数据源添加
        # 简化：暂时不实现
        pass

    def _compute_reward(
        self,
        action_idx: int,
        selected_indices: List[int],
        prev_acc: float = 0.0,
        noise_deleted_pre: Optional[int] = None,
    ) -> float:
        """
        计算奖励。

        基于 oracle 信息（true_labels / label_noise_mask）给出稠密奖励：
        - action 1 (modify_labels): 正确修正的标签数 / 总选择数 * 0.1
        - action 2 (delete): 删除的噪声样本数 / 总选择数 * 0.1
        - action 3 (add) / action 4 (no_op): 0
        """
        if len(selected_indices) == 0 or action_idx == 4:
            return 0.0

        if action_idx == 1:  # modify_labels
            sel = np.asarray(selected_indices, dtype=np.intp)
            # 验证索引有效性
            valid_mask = sel < len(self.current_indices)
            sel = sel[valid_mask]
            if len(sel) == 0:
                return 0.0
            gidx = self.current_indices[sel]
            correct_fixes = (self.noisy_labels[gidx] == self.true_labels[gidx]).sum()
            return float(correct_fixes) / max(1, len(sel)) * 0.1

        elif action_idx == 2:  # delete
            if noise_deleted_pre is None:
                return 0.0
            sel = np.asarray(selected_indices, dtype=np.intp)
            valid_mask = sel < len(self.current_indices)
            return float(noise_deleted_pre) / max(1, valid_mask.sum()) * 0.1

        return 0.0

    def _evaluate_accuracy(self, quick: bool = False, epochs: int = None, force_new_model: bool = False) -> float:
        """
        Fine-tune ResNet50 并在完整干净测试集上评估。

        quick=True  → 使用 quick_epochs（默认5，用于快速反馈）
        quick=False → 使用 full_epochs（默认20，用于 episode 最终记录）
        epochs 优先级高于 quick（用于 episode 结束时的完整评估）
        force_new_model=True → 强制新建模型（用于 episode 开始的完整评估）
        """
        if len(self.current_indices) < 20:
            return 0.0

        # 确定 epochs
        if epochs is not None:
            target_epochs = epochs
        elif quick:
            target_epochs = self._quick_epochs
        else:
            target_epochs = self._full_epochs

        try:
            return self.acc_tracker.evaluate_and_track(
                self.current_indices,
                self.images,               # raw uint8 numpy (N, 32, 32, 3)
                self.noisy_labels,
                quick=(target_epochs <= self._quick_epochs),
                force_new_model=force_new_model,
            )
        except Exception as e:
            print(f"  [Warning] accuracy evaluation failed: {e}")
            return 0.0

    def _get_info(self) -> Dict:
        """获取信息（accuracy 已在上层 step 中更新，此处只返回统计数据）
        
        统一格式：与 DataCleaningEnvMultiSelector 的 info 格式保持一致
        """
        current_noise_mask = self.label_noise_mask[self.current_indices]

        # 计算类别比例字符串（与 adult 数据集格式一致）
        class_dist = np.bincount(
            self.noisy_labels[self.current_indices],
            minlength=self.num_classes,
        )
        class_counts = class_dist.sum()
        class_pcts = (class_dist / max(1, class_counts) * 100).round(1)
        class_ratio_str = "/".join([f"{p:.1f}%" for p in class_pcts])

        # 特征噪声比例（图像数据没有特征噪声，设为0）
        feature_noise_ratio = 0.0

        # 缺失率（图像数据没有缺失值，设为0）
        missing_rate = 0.0

        # 计算选中样本中的噪声数量（从 step 方法获取）
        selected_noise_count = getattr(self, '_last_selected_noise_count', 0)
        selected_total = getattr(self, '_last_selected_total', 0)
        selected_noise_ratio = selected_noise_count / selected_total if selected_total > 0 else 0.0

        return {
            "n_samples": len(self.current_indices),
            "accuracy": self.current_acc,
            "best_accuracy": getattr(self, 'best_acc', self.current_acc),
            "action_name": getattr(self.cfg, 'action_names', [''])[getattr(self, '_last_action_idx', 0)] if hasattr(self.cfg, 'action_names') else '',
            "class_ratio": class_ratio_str,
            "label_noise_ratio": float(current_noise_mask.mean()),
            "feature_noise_ratio": feature_noise_ratio,
            "missing_rate": missing_rate,
            "class_distribution": class_dist.tolist(),
            "user_feedback": "accept",
            "rejected": False,
            "selected_noise_count": selected_noise_count,
            "selected_noise_ratio": selected_noise_ratio,
        }

    def reset(self) -> np.ndarray:
        """重置环境（仅恢复索引与步数，不重新下载/提特征，避免重复日志与开销）"""
        self.current_indices = np.arange(len(self.dataset))
        self.n_steps = 0
        self.step_count = 0
        self.current_acc = 0.0

        # 重置准确率跟踪器（每个 episode 重新训练）
        if hasattr(self, 'acc_tracker'):
            self.acc_tracker.reset_to_initial()

        # 重置评估缓存
        self._last_eval_step = 0
        self._cached_acc = 0.0
        self._acc_is_fresh = False

        return self._state()

    @property
    def current_data(self):
        """获取当前数据"""
        return {
            'images': self.images[self.current_indices],
            'labels': self.noisy_labels[self.current_indices],
            'features': self.features[self.current_indices],
        }

    @property
    def state_dim(self) -> int:
        """状态维度"""
        return 2 * self.feature_dim + 2 + self.num_classes

    def close(self):
        """关闭环境"""
        pass
