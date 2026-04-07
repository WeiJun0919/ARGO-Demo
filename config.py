import os
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configurations
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "smartfactory": {
        "feature_cols": [
            "i_w_blo_weg", "o_w_blo_power", "o_w_blo_voltage",
            "i_w_bhl_weg", "o_w_bhl_power", "o_w_bhl_voltage",
        ],
        "label_col": "labels",
        "n_features": 6,
        "min_dataset_size": 500,
        "state_dim": 22,
        "sample_feature_dim": 10,  # 包含oracle_dirty: 3 + 6 + 1 = 10
        "selector_input_dim": 37,  # state_dim(22) + sample_feature_dim(10) + n_actions(5)
        "dirty_diff_threshold": 1.0,
        "file_prefix": "sf",
    },
    "adult": {
        # 与 datasets/adult/*.csv 表头一致（14 个特征列，不含标签 income）
        "feature_cols": [
            "age", "workclass", "fnlwgt", "education", "educational-num",
            "marital-status", "occupation", "relationship", "race", "gender",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
        ],
        "label_col": "income",
        "n_features": 14,
        "min_dataset_size": 200,
        # state_dim = 5 + 2*n_features + n_actions = 5 + 28 + 5 = 38
        "state_dim": 38,
        # sample_feature_dim = 3 + n_features + oracle_dirty = 3 + 14 + 1 = 18
        "sample_feature_dim": 18,
        # selector_input_dim = state_dim + sample_feature_dim + n_actions = 61
        "selector_input_dim": 61,
        "dirty_diff_threshold": 5.0,
        "file_prefix": "adult",
    },
    "imdb": {
        "feature_cols": ["review"],
        "label_col": "sentiment",
        "n_features": 5000,
        "num_classes": 2,
        "min_dataset_size": 200,
        # state_dim = 5 + n_features + n_features + n_actions = 5 + 5000 + 5000 + 5 = 10010
        "state_dim": 10010,
        "sample_feature_dim": 104,  # 包含oracle_dirty: 3 + 100 + 1 = 104
        "selector_input_dim": 10010 + 104 + 5,  # state_dim + sample_feature_dim + n_actions
        "dirty_diff_threshold": 1.0,
        "file_prefix": "IMDB",
        "is_text_data": True,
        "text_max_features": 5000,
        "text_ngram_range": (1, 2),
        "text_min_df": 2,
    },
    "cifar10": {
        "feature_cols": [f"pixel_{i}" for i in range(3072)],
        "label_col": "label",
        "n_features": 3072,
        "min_dataset_size": 500,
        "state_dim": 6154,
        "sample_feature_dim": 3076,  # 包含oracle_dirty: 3 + 3072 + 1 = 3076
        "selector_input_dim": 9239,  # state_dim(6154) + sample_feature_dim(3076) + n_actions(5)
        "dirty_diff_threshold": 1.0,
        "file_prefix": "cifar10",
        "is_image_data": True,
        "image_channels": 3,
        "image_height": 32,
        "image_width": 32,
        "use_feature_extractor": True,  # 是否使用预训练特征提取器
        "feature_extractor_model": "resnet50",  # 特征提取器模型
        "image_feature_dim": 2048,  # ResNet50 特征维度
    },
    "cifar10_dirty": {
        "feature_cols": [f"pixel_{i}" for i in range(3072)],
        "label_col": "label",
        "n_features": 3072,
        "min_dataset_size": 500,
        "state_dim": 6154,
        "sample_feature_dim": 3076,  # 包含oracle_dirty: 3 + 3072 + 1 = 3076
        "selector_input_dim": 9239,  # state_dim(6154) + sample_feature_dim(3076) + n_actions(5)
        "dirty_diff_threshold": 1.0,
        "file_prefix": "cifar10",
        "is_image_data": True,
        "image_channels": 3,
        "image_height": 32,
        "image_width": 32,
        'class_num': 10,
        "use_feature_extractor": True,
        "feature_extractor_model": "resnet50",
        "image_feature_dim": 2048,
    },
    # 图像数据集：使用 ResNet50 特征提取器
    "cifar10_resnet50": {
        "feature_cols": None,  # 不使用原始像素
        "label_col": "label",
        "n_features": 2048,  # ResNet50 特征维度
        "min_dataset_size": 500,
        "state_dim": 4108,  # 2*2048 (mean/std) + 2 (noise ratios) + 10 (class dist)
        "sample_feature_dim": 2051,  # 2048 (feature) + 3 (action-specific features)
        "selector_input_dim": 4108,  # state_dim + sample_feature_dim + n_actions
        "dirty_diff_threshold": 1.0,
        "file_prefix": "cifar10",
        "is_image_data": True,
        "use_feature_extractor": True,
        "feature_extractor_model": "resnet50",
        "image_feature_dim": 2048,
        "image_channels": 3,
        "image_height": 32,
        "image_width": 32,
        "num_classes": 10,
        "label_noise_ratio": 0.2,  # 默认 20% 标签噪声
        "noise_type": "symmetric",
        # 训练集子集大小（None=全量 50000）；用于快速调试
        "max_train_samples": 5000,
    },
}


def get_dataset_parser():
    """Create argument parser for dataset selection."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="adult",
        choices=list(DATASET_CONFIGS.keys()),
        help=f"Dataset to use: {', '.join(DATASET_CONFIGS.keys())} (default: adult)",
    )
    parser.add_argument(
        "--use-split-data",
        action="store_true",
        help="Use split/D_clean.csv and split/D_dirty.csv (smartfactory); valid 仍用 sf_valid.csv",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Main Config class
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Parse command line arguments once
    _parser = get_dataset_parser()
    _args, _ = _parser.parse_known_args()
    
    def __init__(self, dataset=None):
        """Initialize Config with optional dataset override."""
        # Use provided dataset or fall back to command-line/default
        if dataset is not None:
            self.dataset_name = dataset
        else:
            self.dataset_name = Config._args.dataset
        
        # ── Paths ──────────────────────────────────────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get file prefix for this dataset
        _ds_cfg = DATASET_CONFIGS[self.dataset_name]
        _file_prefix = _ds_cfg.get("file_prefix", self.dataset_name)
        
        # 是否使用 split 文件夹（D_clean / D_dirty 分离），用于图片中的两阶段训练
        self.use_split_data = getattr(Config._args, 'use_split_data', False)
        if self.use_split_data and self.dataset_name == "smartfactory":
            split_dir = os.path.join(base_dir, "datasets", self.dataset_name, "split")
            self.dirty_data_path = os.path.join(split_dir, "D_dirty.csv")
            self.clean_data_path = os.path.join(split_dir, "D_clean.csv")
            self.oracle_data_path = self.clean_data_path
        else:
            # 对于 IMDB 等数据集，目录名可能大小写不同，尝试匹配
            _dataset_lower = self.dataset_name.lower()
            _dataset_dir = self.dataset_name  # 默认使用原始名称
            _base_ds_dir = os.path.join(base_dir, "datasets")
            
            # 尝试找到匹配的数据集目录（忽略大小写）
            if os.path.isdir(_base_ds_dir):
                for _d in os.listdir(_base_ds_dir):
                    if _d.lower() == _dataset_lower:
                        _dataset_dir = _d
                        break
            
            self.dirty_data_path = os.path.join(_base_ds_dir, _dataset_dir, f"{_file_prefix}_dirty.csv")
            self.clean_data_path = os.path.join(_base_ds_dir, _dataset_dir, f"{_file_prefix}_clean.csv")
            self.oracle_data_path = self.clean_data_path
            self.valid_data_path = os.path.join(_base_ds_dir, _dataset_dir, f"{_file_prefix}_valid.csv")
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints", self.dataset_name)
        self.best_data_path = None  # 由 train.py 设置为 checkpoints/best_data.csv，env 在达到 best 的当步写入

        # ── Data ───────────────────────────────────────────────────────────────
        # Load dataset-specific configuration
        self.feature_cols = _ds_cfg["feature_cols"]
        self.label_col = _ds_cfg["label_col"]
        self.n_features = _ds_cfg["n_features"]
        self.valid_ref_ratio = 0.5   # 增大参考集以减少方差
        self.dirty_diff_threshold = _ds_cfg["dirty_diff_threshold"]

        # Text data settings
        self.is_text_data = _ds_cfg.get("is_text_data", False)
        self.text_max_features = _ds_cfg.get("text_max_features", 500)
        self.text_ngram_range = _ds_cfg.get("text_ngram_range", (1, 2))

        # Image data settings
        self.is_image_data = _ds_cfg.get("is_image_data", False)
        self.image_channels = _ds_cfg.get("image_channels", 3)
        self.image_height = _ds_cfg.get("image_height", 32)
        self.image_width = _ds_cfg.get("image_width", 32)

        # Feature extractor settings (for image data)
        self.use_feature_extractor = _ds_cfg.get("use_feature_extractor", False)
        self.feature_extractor_model = _ds_cfg.get("feature_extractor_model", "resnet50")
        self.image_feature_dim = _ds_cfg.get("image_feature_dim", 2048)
        self.num_classes = _ds_cfg.get("num_classes", 10)
        self.label_noise_ratio = _ds_cfg.get("label_noise_ratio", 0.2)
        self.noise_type = _ds_cfg.get("noise_type", "symmetric")
        self.max_train_samples = _ds_cfg.get("max_train_samples", None)
        
        # ── Override paths for image data (PNG directory mode) ──────────────
        _is_image = _ds_cfg.get("is_image_data", False)
        _png_dir = os.path.join(base_dir, "datasets", "cifar10_images")
        
        if _is_image and os.path.isdir(_png_dir):
            print(f"  [Config] 检测到 PNG 图片目录: {_png_dir}")
            self.dirty_data_path = _png_dir
            self.clean_data_path = _png_dir
            self.valid_data_path = _png_dir
            self.oracle_data_path = _png_dir
        
        # 对于图像数据，使用目录（PNG文件）或 pickle 文件
        if self.is_image_data:
            # 检查 datasets/cifar10_images 目录是否存在
            png_dir = os.path.join(base_dir, "datasets", "cifar10_images")
            if os.path.isdir(png_dir):
                # 使用 PNG 文件目录
                self.image_data_dir = png_dir
                print(f"  [Config] 检测到 PNG 图片目录: {png_dir}")
            else:
                # torchvision.datasets.CIFAR10/CIFAR100 会在 datasets/ 下创建自己的子目录
                # 例如: datasets/cifar-10-batches-py/
                # 所以这里只设置到 datasets/ 这一层
                self.image_data_dir = os.path.join(base_dir, "datasets")
            
            # 图片数据路径可以是目录或 pickle 文件
            # dirty/clean/valid 子目录用于 PNG 模式
            # pickle 文件直接是路径
            _prefix = _file_prefix.replace("cifar10", "cifar10")  # 保持前缀一致
            
            # 检查是否存在子目录模式
            if os.path.isdir(os.path.join(self.image_data_dir, "airplane")):
                # PNG 目录模式: datasets/cifar10_images/airplane/...
                self.dirty_data_path = self.image_data_dir  # dirty, clean, valid 共享父目录
                self.clean_data_path = self.image_data_dir
                self.valid_data_path = self.image_data_dir
            else:
                # pickle 文件模式: datasets/cifar10/xxx.pickle
                self.dirty_data_path = os.path.join(self.image_data_dir, f"{_file_prefix}_dirty")
                self.clean_data_path = os.path.join(self.image_data_dir, f"{_file_prefix}_clean")
                self.valid_data_path = os.path.join(self.image_data_dir, f"{_file_prefix}_valid")

        # ── Environment ────────────────────────────────────────────────────────
        self.min_dataset_size = _ds_cfg["min_dataset_size"]

        # ── State representation ────────────────────────────────────────────────
        self.state_dim = _ds_cfg["state_dim"]
        self.sample_feature_dim = _ds_cfg["sample_feature_dim"]
        self.selector_input_dim = _ds_cfg["selector_input_dim"]

    # ── Action space ───────────────────────────────────────────────────────
    # 0: modify_features  1: modify_labels  2: delete_samples
    # 3: add_samples      4: no_op
    n_actions = 5
    action_names = [
        "modify_features", "modify_labels",
        "delete_samples", "add_samples", "no_op",
    ]

    # ── Environment ────────────────────────────────────────────────────────
    max_steps_per_episode = 10
    # 保守清洗：降低每次修改比例，避免错误累积
    # 方案3: 进一步降低修改比例，实现更保守的清洗策略
    max_modify_ratio = 0.01   # 每次最多修改1%的样本（原0.05）
    max_delete_ratio = 0.01   # 每次最多删除1%的样本（原0.03）
    max_add_samples = 10       # 每次最多添加10个样本（原20）

    # ── Oracle 配置 ───────────────────────────────────────────────────────
    # 方案4: 控制Oracle的使用
    # True: 使用Oracle辅助训练（泄露答案，不现实）
    # False: 不使用Oracle，更接近实际应用
    use_oracle = False  # 默认关闭Oracle，更符合实际应用场景

    # 当 use_oracle=False 时，是否从u向量中移除oracle_dirty特征
    # 移除后 Selector 输入不再包含 oracle_dirty
    # 注意：目前设为True以保持维度一致，避免维度不匹配错误
    use_oracle_in_u = True  # 暂时设为True避免报错，后续可改为False测试

    # 当 use_oracle=False 时，是否禁用辅助损失中的Oracle监督
    # 禁用后 Selector 只能通过 RL reward 学习
    use_oracle_in_aux_loss = True  # 启用Oracle监督，让Selector学习识别噪声模式
    # 增加样本候选池：True=用 SMOTE 从当前数据生成合成样本；False=用 valid 20% 的预置池
    use_smote_for_add = False

    # ── Module Selection ─────────────────────────────────────────────────────
    # 选择要使用的噪声检测器（可多选，用逗号分隔，如 "ed2_rpt,ide"）
    # 可选：oracle, ed2_rpt, ide
    # - oracle: 使用 ground truth（实验用）
    # - ed2_rpt: ED2-RPT 特征噪声检测
    # - ide: IDE 标签噪声检测
    noise_detector = "ed2_rpt"
    
    # ── 数据生成器配置 ─────────────────────────────────────────────────────
    # 方式一：单个生成器（兼容旧版本）
    # 可选：random, smote, adasyn, borderline, mixed, mixup, mixup_smote, llm, llm_text
    data_generator_method = "smote"
    
    # 方式二：多生成器组合（推荐）
    # 格式：[{"name": "smote", "weight": 0.7}, {"name": "llm", "weight": 0.3}]
    # 当此项非空时，优先使用此项
    data_generator_methods = None
    
    # ── LLM 数据生成配置 ─────────────────────────────────────────────────────
    # LLM 提供商: openai, anthropic, ollama, mock
    llm_provider = "openai"
    llm_model = "gpt-3.5-turbo"           # 模型名称
    llm_api_key = None                    # API Key (也支持环境变量 OPENAI_API_KEY / ANTHROPIC_API_KEY)
    llm_base_url = None                   # 自定义 API 地址 (用于代理或兼容 API)
    llm_temperature = 0.7                  # 生成温度 (0-2)
    llm_max_tokens = 1000                  # 最大 token 数
    llm_timeout = 60                       # 请求超时(秒)
    llm_batch_size = 10                    # 批量生成数量
    llm_cache_size = 50                    # 缓存数量
    llm_use_cache = True                   # 是否使用缓存
    smote_k = 5
    # VAE 参数
    vae_latent_dim = 32
    vae_hidden_dims = [256, 128]
    vae_epochs = 100
    vae_batch_size = 64
    vae_lr = 1e-3
    vae_beta = 1.0
    vae_use_cvae = True

    # GAN 参数
    gan_latent_dim = 64
    gan_hidden_dims = [256, 128]
    gan_epochs = 100
    gan_batch_size = 64
    gan_lr = 1e-3
    gan_use_wgan = False
    gan_use_cgan = True
    gan_d_steps = 5
    # Mixup 参数
    mixup_alpha = 0.2       # Beta分布参数
    mixup_soft_label = False  # 是否使用软标签混合
    mixup_k = 1              # 每个样本与K个近邻混合
    mixup_smote_prob = 0.5   # MixupSMOTE中Mixup应用概率
    # 每步生成的 SMOTE 候选数量 = min(max_add_samples * smote_candidates_mult, smote_max_candidates)
    smote_candidates_mult = 4
    smote_max_candidates = 120
    # Safety rollback: revert if consecutive negative-reward steps exceed this
    rollback_patience = 5

    # 用户反馈配置
    # 模式: "auto"(自动), "accept"(全部接受), "reject"(全部拒绝)
    # 训练时建议用 "accept"，评估时可用 "auto" 模拟用户
    user_feedback_mode = "accept"
    user_feedback_accuracy_threshold = 0.5  # 自动判断阈值
    reject_penalty = -0.1  # 拒绝惩罚

    # Network architecture
    actor_hidden = [256, 128]
    selector_hidden = [256, 128]
    # Mini-batch size for selector backward pass (keeps training fast)
    selector_train_batch = 512

    # ── PPO hyperparameters ────────────────────────────────────────────────
    lr_ppo = 1e-4  # 降低学习率，更稳定
    gamma = 0.99
    gae_lambda = 0.95
    ppo_clip = 0.2
    ppo_epochs = 4
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5

    # ── Selector hyperparameters ───────────────────────────────────────────
    lr_selector = 3e-4  # 降低学习率（1e-3 → 3e-4），更稳定
    # Loss weights: total = λ_rl·L_RL + λ_aux·L_aux + λ_div·L_div + λ_contrastive·L_contrastive
    lambda_rl = 1.0
    lambda_aux = 0.5
    lambda_div = 0.05
    lambda_contrastive = 0.0   # 对比学习损失系数（暂时禁用）
    contrastive_temperature = 0.5  # 对比学习温度系数（增大，避免梯度消失）

    # ── Training phases (number of episodes) ──────────────────────────────
    n_warmup = 30           # Phase 1: selector aux-loss only, constructive HL actions（增加）
    n_freeze_hl = 10        # Phase 2: heuristic HL action, selector RL + aux（增加）
    n_joint = 10          # Phase 3: end-to-end HRL

    # ── Reward ─────────────────────────────────────────────────────────────
    reward_scale = 20.0     # scale delta-accuracy → reward（增大以强调准确率提升）
    
    # Reward balancing: prevent over-deletion and encourage diverse actions
    # 删除惩罚：删除样本数占总数据比例越大，惩罚越重
    # 降低删除惩罚，鼓励删除噪声样本
    delete_penalty_scale = 0.3   # 删除惩罚系数（原0.5 → 0.3）
    # 保留率奖励：保留更多数据获得额外奖励
    # 降低保留奖励权重，减少"不作为"也能得高奖励的问题
    preserve_reward_scale = 0.1   # 保留率奖励系数（原0.3 → 0.1）
    # 探索奖励：只要采取了有效动作（不是no_op且有选中样本），给一个正奖励
    exploration_bonus = 0.15   # 探索奖励系数，增大以鼓励探索（0.10 → 0.15）
    # 动作多样性奖励：鼓励选择与历史不同的动作
    diversity_bonus = 0.01   # 多样性奖励系数

    # 动作平衡奖励：鼓励探索不同的动作
    action_bonus = {
        "modify_features": 0.05,   # 奖励修改特征
        "modify_labels": 0.05,     # 奖励修改标签
        "delete_samples": 0.03,    # 奖励删除（删除噪声是正确的）
        "add_samples": 0.05,       # 奖励添加新样本
        "no_op": -0.50,           # 大幅增加不操作惩罚，迫使智能体必须动作
    }

    # ── Classifier ──────────────────────────────────────────────────────────
    # 训练时 env 内用哪个分类器算 reward（logistic 快 ~1s/步 → ~20min；mlp 慢 ~2h）
    env_classifier_type = "logistic"   # "logistic" 训练快；"mlp" 与评估一致但慢
    # env_classifier_type = "image"   # "image" 使用图像分类器；"logistic" 使用逻辑回归分类器
    # 评估脚本 evaluate.py 用 classifier_type 做三种数据对比（建议保持 mlp）
    classifier_type = "mlp"   # "mlp" | "logistic"

    # ── 图像评估优化 ───────────────────────────────────────────────────────
    # 评估频率：每 N 步评估一次（设为 9999 表示只在 episode 结束时评估）
    # - 1: 每步评估（慢但准确）
    # - 9999: 只在 episode 结束时评估（快，推荐）
    image_eval_every_n_steps = 9999
    image_eval_epochs_quick = 5    # 快速评估的 epochs
    image_eval_epochs_full = 20   # episode 结束时评估的 epochs

    # ── 标签修改置信度阈值 ───────────────────────────────────────────
    # 只有当分类器置信度 >= 此阈值时才修改标签
    # 阈值越高越保守，但可能漏掉一些噪声
    modify_label_confidence_threshold = 0.85
    classifier_max_iter = 500
    # MLP: hidden layers and early stopping
    mlp_hidden = (128, 64)
    mlp_early_stopping = True
    mlp_val_fraction = 0.1

    # ── Misc ───────────────────────────────────────────────────────────────
    seed = 42
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    log_interval = 10
    save_interval = 50

    # ── ED2-RPT Feature Noise Detection ──────────────────────────────────────
    # 使用 ED2-RPT 方法检测和修复特征噪声
    use_ed2_rpt = True              # 是否使用 ED2-RPT
    ed2_rpt_pretrain_epochs = 50   # 预训练轮数
    ed2_rpt_hidden_dims = [128, 64] # 隐藏层维度
    ed2_rpt_noise_threshold = 0.5  # 噪声判定阈值
    ed2_rpt_correction_scale = 0.3 # 修正强度 (0-1)
    ed2_rpt_lr = 1e-3              # 学习率
    ed2_rpt_batch_size = 64        # 批次大小
    # 在线微调参数
    ed2_rpt_online_finetune = False # 禁用在线微调，加速训练
    ed2_rpt_finetune_interval = 20  # 每隔多少步微调一次
    ed2_rpt_finetune_epochs = 5     # 每次微调的轮数
    
    # ── 噪声检测阈值（用于三阶段训练）────────────────────────────────────
    # 阈值设为接近实际噪声比例（约 20%），提高检测精确率
    detection_noise_threshold = 20   # 高置信度噪声阈值百分位 (约 20% 接近实际噪声比例)
    detection_clean_threshold = 5   # 高置信度干净阈值百分位 (只标记确定干净的样本)
    detection_min_gap = 0.05          # 阈值之间的最小间隔

    # ── IDE Label Noise Detection ─────────────────────────────────────────
    # 使用简化版检测器还是完整版 IDE
    use_simple_label_detector = True
    # 简化版参数
    label_noise_ratio = 0.3   # 估计的噪声比例
    # 完整版 IDE 参数
    ide_clf_type = 'mlp'       # 'mlp' or 'logistic'
    ide_hidden = (128, 64)     # MLP 隐藏层大小 - 增加容量
    ide_n_iterations = 1       # 迭代次数 - 减少避免误差累积
    ide_confidence_threshold = 0.95  # 修复阈值 - 提高精确率
    ide_noise_ratio_estimate = 0.3  # 噪声比例估计

    # ── Text Data Settings ───────────────────────────────────────────────
    is_text_data = False       # 是否为文本数据
    text_max_features = 500    # 文本向量化最大特征数
    text_ngram_range = (1, 2)  # N-gram 范围
    text_vectorizer = "tfidf"   # 向量化方法: tfidf, count

    # ── Image Data Settings ─────────────────────────────────────────────
    # 预训练特征提取器
    use_pretrained_extractor = True      # 是否使用预训练特征提取器
    extractor_model = "resnet50"         # 特征提取器模型: resnet50, resnet18, alexnet, vgg16, mobilenet_v2
    extractor_pretrained = True           # 是否使用 ImageNet 预训练权重
    extractor_freeze_backbone = True      # 是否冻结骨干网络
    extractor_trainable_layers = 3        # 解冻最后几层

    # 下游分类器
    use_downstream_classifier = True     # 是否使用下游分类器评估准确率
    downstream_classifier_type = "mlp"   # mlp 或 lightweight
    downstream_hidden_dim = 512          # 隐藏层维度
    downstream_epochs = 50               # 训练轮数
    downstream_batch_size = 64           # 批次大小
    downstream_lr = 1e-3                # 学习率
    downstream_early_stopping = 5        # 早停轮数

    # 图像增强（用于 modify_features）
    image_augmentation = True            # 是否使用图像增强
    augmentation_types = ["flip", "crop", "color"]  # 增强类型

    # 数据加载器
    image_data_dir = "./datasets"         # 图像数据目录
    image_download = True                # 是否自动下载数据集

    @classmethod
    def get_parser(cls):
        """返回完整的命令行参数解析器（包含数据集选择）"""
        return cls._parser

    @classmethod
    def list_datasets(cls):
        """列出所有可用的数据集"""
        return list(DATASET_CONFIGS.keys())
