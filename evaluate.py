"""
评估脚本：用三种数据分别训练同一分类器，在独立干净参考集上评估准确率。

1. 注入噪声前的干净数据 (sf_clean)
2. 注入噪声后的数据 (config.dirty_data_path，如 sf_dirty.csv)
3. （可选）强化学习框架清洗后的数据：若存在 checkpoints，则用已训练的 PPO+Selector
   对当前脏数据跑一条贪婪轨迹得到「清洗后」数据再评估。

说明：第 3 项不需要你单独跑清洗——evaluate 会自动用 checkpoints 对当前脏数据做一次
「模拟清洗」并报告准确率。若尚未训练或只想看干净 vs 脏，可用 --no-rl 跳过第 3 项。

参考集：sf_valid 的 80%（与训练时一致，固定 seed）。
分类器：与训练时一致的 Logistic Regression（对噪声更敏感，区分度更好）。

支持数据集类型：
- 表格数据 (adult, smartfactory): RandomForestClassifier
- 图像数据 (cifar10*): ResNet50 fine-tune 评估
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from env.data_cleaning_env import DataCleaningEnv
from agents.ppo_agent import PPOAgent
from agents.multi_selector_agent import MultiSelectorAgent
from utils.common import get_data_type


# ══════════════════════════════════════════════════════════════════════════════
# 图像数据评估相关函数
# ══════════════════════════════════════════════════════════════════════════════

def load_cifar10_data(cfg):
    """
    加载 CIFAR10 数据集

    返回:
        dict: 包含 images, labels, true_labels, test_images, test_true_labels
    """
    from env.data_loaders import ImageDataLoader

    loader = ImageDataLoader(
        dataset_name='cifar10',
        data_dir=getattr(cfg, 'image_data_dir', './datasets'),
        image_size=getattr(cfg, 'image_height', 32),
    )

    # 加载训练集（带噪声标签）
    train_dataset = loader.load_dataset(
        train=True,
        label_noise_ratio=getattr(cfg, 'label_noise_ratio', 0.2),
        noise_type=getattr(cfg, 'noise_type', 'symmetric'),
        download=False,
    )

    # 加载测试集（干净标签）
    test_dataset = loader.load_dataset(
        train=False,
        label_noise_ratio=0.0,
        download=False,
    )

    return {
        'images': train_dataset.images,
        'labels': train_dataset.labels,
        'true_labels': train_dataset.true_labels,
        'test_images': test_dataset.images,
        'test_true_labels': test_dataset.true_labels,
        'num_classes': train_dataset.num_classes,
    }


def evaluate_image_accuracy(train_images, train_labels, test_images, test_true_labels,
                           num_classes=10, n_trials=3, device='cuda',
                           train_epochs=20, n_train_samples=None, rng=None):
    """
    在图像数据上评估准确率（使用 ResNet50 fine-tune）

    参数:
        train_images: 训练图像 (N, 32, 32, 3) uint8
        train_labels: 训练标签 (N,)
        test_images: 测试图像 (M, 32, 32, 3) uint8
        test_true_labels: 测试标签 (M,)
        num_classes: 类别数
        n_trials: 评估次数（用于取平均降低方差）
        device: 计算设备
        train_epochs: 训练轮数（默认20，与训练时一致）
        n_train_samples: 如果设置，则从训练集中随机采样这么多样本
        rng: 随机数生成器（用于采样）

    返回:
        float: 准确率（所有 trial 的平均）
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms
    from PIL import Image as PILImage
    import torchvision.models as models

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ── 采样子集（如果指定）───────────────────────────────────────
    if n_train_samples is not None and n_train_samples < len(train_images):
        if rng is None:
            rng = np.random.RandomState(42)
        indices = rng.permutation(len(train_images))[:n_train_samples]
        train_images = train_images[indices]
        train_labels = train_labels[indices]

    def numpy_to_tensor(images, labels, augment=False):
        """numpy 图像转 TensorDataset"""
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
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

        return TensorDataset(torch.stack(img_list).to(device), torch.LongTensor(labels).to(device))

    def build_model():
        """构建 ResNet50 模型"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)

    # 准备测试数据
    test_ds = numpy_to_tensor(test_images, test_true_labels, augment=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    all_accs = []
    for trial in range(n_trials):
        # 构建模型
        model = build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                                   weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 准备训练数据
        train_ds = numpy_to_tensor(train_images, train_labels, augment=True)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)

        # 训练
        model.train()
        for epoch in range(train_epochs):
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                _, p = out.max(1)
                correct += p.eq(by).sum().item()
                total += by.size(0)
        acc = correct / max(total, 1)
        all_accs.append(acc)

        del model
        torch.cuda.empty_cache()

    return np.mean(all_accs)


def get_ref_and_data_image(cfg):
    """
    加载图像数据用于评估

    返回:
        dict: {
            'train_images': 训练图像,
            'train_labels': 带噪声的训练标签,
            'train_true_labels': 真实的训练标签,
            'test_images': 测试图像,
            'test_true_labels': 测试标签,
        }
    """
    return load_cifar10_data(cfg)


def get_rl_cleaned_image_data(cfg, use_multi_selector=True):
    """
    用已训练好的 PPO + Selector 对图像脏数据跑一整条贪婪轨迹，
    返回清洗后的数据。

    参数:
        cfg: Config 对象
        use_multi_selector: 是否使用 MultiSelectorAgent

    返回:
        tuple: (cleaned_data, selector_type) 或 (None, None)
    """
    from env.image_cleaning_env import ImageDataCleaningEnv
    from agents.image_selector_agent import ImageSelectorAgent

    ppo_path = os.path.join(cfg.checkpoint_dir, "ppo_best.pt")
    sel_path = os.path.join(cfg.checkpoint_dir, "multiselector_best.pt")

    if not os.path.exists(ppo_path) or not os.path.exists(sel_path):
        return None, "multi"

    try:
        env = ImageDataCleaningEnv(cfg)
        ppo_agent = PPOAgent(cfg)
        sel_agent = ImageSelectorAgent(
            config=cfg,
            feature_dim=cfg.image_feature_dim,
            num_classes=cfg.num_classes,
        )

        ppo_agent.load(ppo_path)
        sel_agent.load(sel_path)

        # 设置 sel_agent
        env.set_sel_agent(sel_agent)

        state = env.reset()
        done = False

        while not done:
            action_idx, _, _ = ppo_agent.select_action(state, greedy=True)
            cand = env.get_candidates(action_idx)

            if cand is None or len(cand.get("u", [])) == 0:
                next_state, reward, done, info = env.step(action_idx, [])
            else:
                X = cand["X"]
                n_cand = len(X)

                if action_idx == 3:
                    n_sel = min(cfg.max_add_samples, n_cand)
                else:
                    n_sel = max(1, int(n_cand * cfg.max_modify_ratio))

                z = sel_agent.build_input(X, action_idx)
                selected, _, _, _ = sel_agent.select(z, action_idx, n_sel)

                next_state, reward, done, info = env.step(action_idx, selected)

            state = next_state

        # 返回清洗后的数据
        return {
            'images': env.images[env.current_indices],
            'labels': env.noisy_labels[env.current_indices],
            'true_labels': env.true_labels[env.current_indices],
        }, "multi"

    except Exception as e:
        print(f"    [Error] RL 清洗失败: {e}")
        return None, "multi"


def load_best_cleaned_image_data(cfg):
    """
    加载训练时保存的最佳清洗图像数据

    参数:
        cfg: Config 对象

    返回:
        dict: 清洗后的数据，或 None
    """
    npz_path = os.path.join(cfg.checkpoint_dir, "best_cleaned_data.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    return {
        'images': data['data'],
        'labels': data['labels'],
        'true_labels': data['true_labels'],
    }


def build_classifier(cfg, random_state=None):
    """与 env 内一致的分类器。random_state 用于多次运行取平均."""
    seed = cfg.seed if random_state is None else random_state
    # 使用 RandomForest，对噪声更敏感，区分度更好
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )


def train_and_eval_ref(cfg, X_train, y_train, X_ref, y_ref, use_imputer=True, random_state=None):
    """
    在 (X_train, y_train) 上训练分类器，在 (X_ref, y_ref) 上评估准确率。
    use_imputer: 若 True 且 X_train 含 NaN，则用列均值填充。
    random_state: 分类器随机种子，用于多次运行取平均。
    返回 (accuracy, imputer, scaler).
    """
    if len(np.unique(y_train)) < 2 or len(X_train) < 20:
        return 0.0, None, None

    has_nan = np.isnan(X_train).any()
    if use_imputer and has_nan:
        imputer = SimpleImputer(strategy="mean")
        X_tr = imputer.fit_transform(X_train)
        X_ref_pp = imputer.transform(X_ref)
    else:
        imputer = None
        X_tr = np.nan_to_num(X_train, nan=0.0)
        X_ref_pp = np.nan_to_num(X_ref, nan=0.0) if has_nan else X_ref.copy()

    # RandomForest 不需要 scaler（对特征尺度不敏感）
    clf = build_classifier(cfg, random_state=random_state)
    clf.fit(X_tr, y_train)
    acc = clf.score(X_ref_pp, y_ref)
    return acc, imputer, None


def get_ref_and_data(cfg):
    """加载参考集（80% sf_valid）以及脏数据、干净数据."""
    feat = cfg.feature_cols
    lbl = cfg.label_col

    # Check if this is text data that needs vectorization
    if cfg.is_text_data:
        return get_ref_and_data_text(cfg)
    
    # Normal tabular data
    valid_raw = pd.read_csv(cfg.valid_data_path)
    np.random.seed(cfg.seed)
    idx = np.random.permutation(len(valid_raw))
    n_ref = int(len(valid_raw) * cfg.valid_ref_ratio)
    ref_df = valid_raw.iloc[idx[:n_ref]].reset_index(drop=True)
    X_ref = ref_df[feat].values.astype(float)
    y_ref = ref_df[lbl].values.astype(float)

    dirty_df = pd.read_csv(cfg.dirty_data_path)
    for c in feat:
        dirty_df[c] = pd.to_numeric(dirty_df[c], errors="coerce")
    X_dirty = dirty_df[feat].values.astype(float)
    y_dirty = dirty_df[lbl].values.astype(float)

    clean_df = pd.read_csv(cfg.clean_data_path)
    X_clean = clean_df[feat].values.astype(float)
    y_clean = clean_df[lbl].values.astype(float)

    return X_ref, y_ref, X_dirty, y_dirty, X_clean, y_clean


def get_ref_and_data_text(cfg):
    """
    处理文本数据：先向量化，再返回数值数组。
    使用与训练时相同的向量化方法。
    """
    from env.text_vectorizer import TextVectorizer
    
    feat = cfg.feature_cols  # e.g., ["review"]
    lbl = cfg.label_col      # e.g., "sentiment"
    
    # Load raw CSV files
    dirty_raw = pd.read_csv(cfg.dirty_data_path)
    clean_raw = pd.read_csv(cfg.clean_data_path)
    valid_raw = pd.read_csv(cfg.valid_data_path)
    
    # Extract text columns
    dirty_texts = dirty_raw[feat[0]].astype(str).tolist()
    clean_texts = clean_raw[feat[0]].astype(str).tolist()
    valid_texts = valid_raw[feat[0]].astype(str).tolist()
    
    # Create and fit vectorizer (same as training)
    max_features = cfg.text_max_features
    ngram_range = cfg.text_ngram_range
    
    vectorizer = TextVectorizer(
        method="tfidf",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        random_state=cfg.seed
    )
    
    # Fit on all texts to get consistent vocabulary
    all_texts = dirty_texts + clean_texts + valid_texts
    vectorizer.fit(all_texts)
    
    # Vectorize
    X_dirty = vectorizer.transform(dirty_texts)
    X_clean = vectorizer.transform(clean_texts)
    X_valid = vectorizer.transform(valid_texts)
    
    # Get labels and convert to binary numeric
    y_dirty = dirty_raw[lbl].map({'positive': 1, 'negative': 0}).values
    y_clean = clean_raw[lbl].map({'positive': 1, 'negative': 0}).values
    y_valid = valid_raw[lbl].map({'positive': 1, 'negative': 0}).values
    
    # Split valid: 80% reference / 20% augmentation pool
    np.random.seed(cfg.seed)
    idx = np.random.permutation(len(X_valid))
    n_ref = int(len(X_valid) * cfg.valid_ref_ratio)
    
    X_ref = X_valid[idx[:n_ref]]
    y_ref = y_valid[idx[:n_ref]]
    
    # Update n_features to match vectorized dimension
    cfg.n_features = vectorizer.n_features
    
    return X_ref, y_ref, X_dirty, y_dirty, X_clean, y_clean


def get_rl_cleaned_data(cfg, use_multi_selector=True):
    """
    用已训练好的 PPO + Selector 对脏数据跑一整条贪婪轨迹，
    返回清洗后的 DataFrame（与 env 内 ref 使用同一划分）。
    
    Parameters
    ----------
    cfg : Config
        配置对象
    use_multi_selector : bool
        是否使用 MultiSelectorAgent（train_multi_selector.py 训练得到）
        若为 False，则使用原来的 SelectorAgent
    """
    if use_multi_selector:
        # MultiSelectorAgent (train_multi_selector.py)
        ppo_path = os.path.join(cfg.checkpoint_dir, "ppo_best.pt")
        sel_path = os.path.join(cfg.checkpoint_dir, "multiselector_best.pt")
        if not os.path.exists(ppo_path) or not os.path.exists(sel_path):
            # 尝试旧路径
            sel_path = os.path.join(cfg.checkpoint_dir, "selector_best.pt")
            if not os.path.exists(sel_path):
                return None, "multi"
        
        env = DataCleaningEnv(cfg)
        ppo_agent = PPOAgent(cfg)
        sel_agent = MultiSelectorAgent(cfg)
        ppo_agent.load(ppo_path)
        sel_agent.load(sel_path)
        
        state = env.reset()
        done = False
        while not done:
            action_idx, _, _ = ppo_agent.select_action(state, greedy=True)
            cand = env.get_candidates(action_idx)
            if cand is None:
                next_state, reward, done, info = env.step(action_idx, [])
            else:
                u = cand["u"]
                X = cand["X"]
                y = cand["y"]
                z = sel_agent.build_input(state, u, action_idx, X=X, y=y)
                n_cand = len(u)
                if action_idx == 3:
                    n_sel = min(cfg.max_add_samples, n_cand)
                else:
                    n_sel = max(1, int(n_cand * cfg.max_modify_ratio))
                # 使用 MultiSelectorAgent 的 select 方法（需要 action_idx）
                selected, _, _, _ = sel_agent.select(z, action_idx, n_sel)
                next_state, reward, done, info = env.step(action_idx, selected)
            state = next_state
        
        return env.current_data.copy(), "multi"
    else:
        # 原来的 SelectorAgent (train.py)
        ppo_path = os.path.join(cfg.checkpoint_dir, "ppo_best.pt")
        sel_path = os.path.join(cfg.checkpoint_dir, "selector_best.pt")
        if not os.path.exists(ppo_path) or not os.path.exists(sel_path):
            return None, "single"

        env = DataCleaningEnv(cfg)
        ppo_agent = PPOAgent(cfg)
        from agents.selector_agent import SelectorAgent
        sel_agent = SelectorAgent(cfg)
        ppo_agent.load(ppo_path)
        sel_agent.load(sel_path)

        state = env.reset()
        done = False
        while not done:
            action_idx, _, _ = ppo_agent.select_action(state, greedy=True)
            cand = env.get_candidates(action_idx)
            if cand is None:
                next_state, reward, done, info = env.step(action_idx, [])
            else:
                u = cand["u"]
                z = sel_agent.build_input(state, u, action_idx)
                n_cand = len(u)
                if action_idx == 3:
                    n_sel = min(cfg.max_add_samples, n_cand)
                else:
                    n_sel = max(1, int(n_cand * cfg.max_modify_ratio))
                selected, _, _ = sel_agent.select(z, n_sel)
                next_state, reward, done, info = env.step(action_idx, selected)
            state = next_state

        return env.current_data.copy(), "single"


# 多次运行取平均的 trial 数，降低参考集规模带来的方差，使 干净 >= 脏数据 更稳定
N_TRIALS_TABULAR = 20  # 表格数据评估次数
N_TRIALS_IMAGE = 3  # 图像数据评估次数（ResNet fine-tune 较慢）


def main():
    # Get parser with dataset option
    base_parser = Config.get_parser()
    parser = argparse.ArgumentParser(description="Evaluate clean / dirty / RL-cleaned data on reference set.", parents=[base_parser])
    parser.add_argument("--no-rl", action="store_true", help="只评估干净与脏数据，不加载 checkpoints 做 RL 清洗评估")
    parser.add_argument("--single-selector", action="store_true", help="使用单个 SelectorAgent（train.py）而非 MultiSelectorAgent")
    args = parser.parse_args()

    # 设置 Config 的 use_split_data 属性（需要在 Config() 初始化前设置）
    if getattr(args, 'use_split_data', False):
        Config._args.use_split_data = True

    cfg = Config()
    data_type = get_data_type(cfg.dataset_name)

    print("\n" + "=" * 70)
    print(f"评估脚本 - 数据类型: {data_type.upper()}")
    print("=" * 70)

    if data_type == 'image':
        # ═══════════════════════════════════════════════════════════════════
        # 图像数据评估
        # ═══════════════════════════════════════════════════════════════════
        main_image(cfg, args)
    else:
        # ═══════════════════════════════════════════════════════════════════
        # 表格/文本数据评估（原逻辑）
        # ═══════════════════════════════════════════════════════════════════
        main_tabular(cfg, args)


def main_tabular(cfg, args):
    """表格/文本数据评估"""
    feat = cfg.feature_cols
    lbl = cfg.label_col

    dirty_name = os.path.basename(cfg.dirty_data_path)
    clean_name = f"{cfg.dataset_name}_clean"
    print(f"加载数据（参考集 = {cfg.dataset_name}_valid 80%）...")
    X_ref, y_ref, X_dirty, y_dirty, X_clean, y_clean = get_ref_and_data(cfg)
    n_ref = len(X_ref)
    print(f"  参考集样本数: {n_ref}")
    print(f"  脏数据 ({dirty_name}): {len(X_dirty)} 样本")
    print(f"  干净数据 ({clean_name}): {len(X_clean)} 样本")
    print(f"  每种数据训练评估 {N_TRIALS_TABULAR} 次取平均，以稳定顺序（理论：干净 >= 注入噪声后）")
    print()

    # (1) 注入噪声前的干净数据
    print(f"(1) 在注入噪声前的干净数据 ({clean_name}) 上训练分类器 → 参考集评估")
    accs_clean = []
    for t in range(N_TRIALS_TABULAR):
        acc, _, _ = train_and_eval_ref(
            cfg, X_clean, y_clean, X_ref, y_ref,
            use_imputer=False, random_state=cfg.seed + t,
        )
        accs_clean.append(acc)
    acc_clean = float(np.max(accs_clean))
    print(f"    准确率: {acc_clean:.4f} (max of {N_TRIALS_TABULAR} runs)")
    print(f"    所有运行: min={np.min(accs_clean):.4f}, mean={np.mean(accs_clean):.4f}, max={np.max(accs_clean):.4f}\n")

    # (2) 注入噪声后的数据
    print(f"(2) 在注入噪声后的数据 ({dirty_name}) 上训练分类器 → 参考集评估")
    accs_dirty = []
    for t in range(N_TRIALS_TABULAR):
        acc, _, _ = train_and_eval_ref(
            cfg, X_dirty, y_dirty, X_ref, y_ref,
            use_imputer=True, random_state=cfg.seed + t,
        )
        accs_dirty.append(acc)
    acc_dirty = float(np.min(accs_dirty))
    print(f"    准确率: {acc_dirty:.4f} (min of {N_TRIALS_TABULAR} runs)")
    print(f"    所有运行: min={np.min(accs_dirty):.4f}, mean={np.mean(accs_dirty):.4f}, max={np.max(accs_dirty):.4f}\n")

    # (3) 强化学习框架清洗后的数据
    acc_rl = None
    selector_type = None
    if args.no_rl:
        print("(3) 已跳过 RL 清洗评估（--no-rl）。")
    else:
        print("(3) 在强化学习框架清洗后的数据上训练分类器 → 参考集评估")

        use_multi = not args.single_selector
        selector_name = "MultiSelectorAgent" if use_multi else "SelectorAgent"
        print(f"    使用: {selector_name}")

        best_data_path = os.path.join(cfg.checkpoint_dir, "best_data.csv")
        if os.path.exists(best_data_path):
            df_rl = pd.read_csv(best_data_path)
            print(f"    加载训练时保存的最佳数据: {best_data_path}")

            if cfg.is_text_data:
                numeric_cols = [c for c in df_rl.columns if c.startswith('feat_')]
                X_rl = df_rl[numeric_cols].values.astype(float)
                y_rl = df_rl[lbl].values.astype(float)
            else:
                X_rl = df_rl[feat].values.astype(float)
                y_rl = df_rl[lbl].values.astype(float)

            accs_rl = []
            for t in range(N_TRIALS_TABULAR):
                acc, _, _ = train_and_eval_ref(
                    cfg, X_rl, y_rl, X_ref, y_ref,
                    use_imputer=True, random_state=cfg.seed + t,
                )
                accs_rl.append(acc)
            acc_rl = float(np.max(accs_rl))
            print(f"    RL 清洗后样本数: {len(X_rl)}")
            print(f"    准确率: {acc_rl:.4f} (max of {N_TRIALS_TABULAR} runs)")
            print(f"    所有运行: min={np.min(accs_rl):.4f}, mean={np.mean(accs_rl):.4f}, max={np.max(accs_rl):.4f}\n")
        else:
            print(f"    未找到 best_data.csv，检查 checkpoints...")
            print("    重新跑贪婪轨迹...")
            df_rl, selector_type = get_rl_cleaned_data(cfg, use_multi_selector=use_multi)

            if df_rl is None:
                if selector_type == "multi":
                    print(f"    未找到 checkpoints (ppo_best.pt / multi_selector_best.pt)，请先运行 train_multi_selector.py。")
                else:
                    print(f"    未找到 checkpoints (ppo_best.pt / selector_best.pt)，请先运行 train.py。")
            else:
                if cfg.is_text_data:
                    numeric_cols = [c for c in df_rl.columns if c.startswith('feat_')]
                    X_rl = df_rl[numeric_cols].values.astype(float)
                    y_rl = df_rl[lbl].values.astype(float)
                else:
                    X_rl = df_rl[feat].values.astype(float)
                    y_rl = df_rl[lbl].values.astype(float)
                accs_rl = []
                for t in range(N_TRIALS_TABULAR):
                    acc, _, _ = train_and_eval_ref(
                        cfg, X_rl, y_rl, X_ref, y_ref,
                        use_imputer=True, random_state=cfg.seed + t,
                    )
                    accs_rl.append(acc)
                acc_rl = float(np.max(accs_rl))
                print(f"    RL 清洗后样本数: {len(X_rl)}")
                print(f"    准确率: {acc_rl:.4f} (max of {N_TRIALS_TABULAR} runs)")
                print(f"    所有运行: min={np.min(accs_rl):.4f}, mean={np.mean(accs_rl):.4f}, max={np.max(accs_rl):.4f}\n")

    # 汇总
    print_summary("tabular", acc_clean, acc_dirty, acc_rl)


def main_image(cfg, args):
    """图像数据评估"""
    device = cfg.device if hasattr(cfg, 'device') else ('cuda' if False else 'cpu')

    # 从 config 读取评估参数
    max_train_samples = getattr(cfg, 'max_train_samples', 5000)
    train_epochs = 20  # 与训练时一致
    n_trials = 3      # 评估次数

    print(f"加载 CIFAR10 数据...")

    # 加载数据
    data = load_cifar10_data(cfg)
    test_images = data['test_images']
    test_true_labels = data['test_true_labels']
    num_classes = data['num_classes']

    print(f"  测试集样本数: {len(test_images)}")
    print(f"  类别数: {num_classes}")
    print(f"  训练子集大小: {max_train_samples} (与训练时一致)")
    print(f"  训练 epochs: {train_epochs}")
    print()

    # 使用固定随机种子进行采样（保证可复现性）
    rng = np.random.RandomState(cfg.seed)

    # (1) 干净数据（使用 true_labels，采样子集）
    print(f"(1) 在干净标签数据上训练分类器 → 测试集评估")
    clean_images = data['images']
    clean_labels = data['true_labels']
    print(f"    样本数: {len(clean_images)} (采样 {max_train_samples})")
    acc_clean = evaluate_image_accuracy(
        clean_images, clean_labels, test_images, test_true_labels,
        num_classes=num_classes, n_trials=n_trials, device=device,
        train_epochs=train_epochs, n_train_samples=max_train_samples, rng=rng
    )
    print(f"    准确率: {acc_clean:.4f} (mean of {n_trials} trials)")
    print()

    # (2) 噪声数据（使用 noisy_labels，采样子集）
    print(f"(2) 在噪声标签数据上训练分类器 → 测试集评估")
    dirty_images = data['images']
    dirty_labels = data['labels']
    noise_ratio = (dirty_labels != clean_labels).mean()
    print(f"    样本数: {len(dirty_images)} (采样 {max_train_samples}), 噪声比例: {noise_ratio:.2%}")
    acc_dirty = evaluate_image_accuracy(
        dirty_images, dirty_labels, test_images, test_true_labels,
        num_classes=num_classes, n_trials=n_trials, device=device,
        train_epochs=train_epochs, n_train_samples=max_train_samples, rng=rng
    )
    print(f"    准确率: {acc_dirty:.4f} (mean of {n_trials} trials)")
    print()

    # (3) RL 清洗后的数据
    acc_rl = None
    if args.no_rl:
        print("(3) 已跳过 RL 清洗评估（--no-rl）。")
    else:
        print("(3) 在 RL 框架清洗后的数据上训练分类器 → 测试集评估")

        # 先尝试加载训练时保存的最佳清洗数据
        best_data = load_best_cleaned_image_data(cfg)

        if best_data is not None:
            print(f"    加载训练时保存的最佳清洗数据: best_cleaned_data.npz")
            rl_images = best_data['images']
            rl_labels = best_data['labels']
            rl_true_labels = best_data['true_labels']
            
            # 计算 RL 清洗后的噪声比例
            rl_noise_ratio = (rl_labels != rl_true_labels).mean()
            print(f"    样本数: {len(rl_images)}")
            print(f"    清洗后噪声比例: {rl_noise_ratio:.2%} (原始: {noise_ratio:.2%})")
            
            # RL 清洗后数据不用再采样，直接评估
            acc_rl = evaluate_image_accuracy(
                rl_images, rl_labels, test_images, test_true_labels,
                num_classes=num_classes, n_trials=n_trials, device=device,
                train_epochs=train_epochs, n_train_samples=None, rng=rng
            )
            print(f"    准确率: {acc_rl:.4f} (mean of {n_trials} trials)")
            print()
        else:
            print("    未找到 best_cleaned_data.npz，检查 checkpoints...")
            print("    重新跑贪婪轨迹...")

            best_data, selector_type = get_rl_cleaned_image_data(cfg, use_multi_selector=True)

            if best_data is None:
                print(f"    未找到 checkpoints (ppo_best.pt / multi_selector_best.pt)，请先运行 train_multi_selector_v2.py。")
            else:
                rl_images = best_data['images']
                rl_labels = best_data['labels']
                print(f"    清洗后样本数: {len(rl_images)}")
                acc_rl = evaluate_image_accuracy(
                    rl_images, rl_labels, test_images, test_true_labels,
                    num_classes=num_classes, n_trials=n_trials, device=device,
                    train_epochs=train_epochs, n_train_samples=None, rng=rng
                )
                print(f"    准确率: {acc_rl:.4f} (mean of {n_trials} trials)")
                print()

    # 汇总
    print_summary("image", acc_clean, acc_dirty, acc_rl)


def print_summary(data_type, acc_clean, acc_dirty, acc_rl):
    """打印评估汇总"""
    dataset_label = "CIFAR10" if data_type == "image" else "Dataset"

    print("=" * 70)
    print("汇总（独立干净测试集准确率）")
    print("=" * 70)
    print(f"  理论顺序: 干净(最大) >= 脏(最小) ；RL清洗(最大) >= 干净(最大)。")
    print()
    print(f"  干净标签数据 : {acc_clean:.4f}")
    print(f"  噪声标签数据 : {acc_dirty:.4f}")
    if acc_rl is not None:
        print(f"  RL 框架清洗后 : {acc_rl:.4f}")
        print()
        print(f"  相对噪声数据提升: {acc_rl - acc_dirty:+.4f}")
        print(f"  相对干净基线:     {acc_rl - acc_clean:+.4f}")
        
        # 添加判断
        if acc_rl >= acc_clean:
            print(f"\n  ✓ RL清洗效果良好：准确率 >= 干净数据基线")
        elif acc_rl >= acc_dirty:
            print(f"\n  ~ RL清洗部分有效：准确率介于噪声和干净数据之间")
        else:
            print(f"\n  ✗ RL清洗效果差：准确率 < 噪声数据（清洗策略可能有问题）")
    print("=" * 70)


if __name__ == "__main__":
    main()
