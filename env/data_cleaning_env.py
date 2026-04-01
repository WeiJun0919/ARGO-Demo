"""
Data Cleaning RL Environment (PPO-HRL)

High-level action space:
  0 – modify_features : Oracle-fix dirty features of selected samples
  1 – modify_labels   : Flip label of selected samples
  2 – delete_samples  : Remove selected samples from dataset
  3 – add_samples     : Add selected samples from augmentation pool
  4 – no_op           : Do nothing

State vector (dim = 22):
  [n_samples_ratio, class_ratio, accuracy, recent_reward, dirty_ratio,
   feat_means×6, feat_stds×6, action_counts_norm×5]

Per-sample features u_i (dim = 10):
  [entropy, loss, margin, norm_features×6, oracle_dirty]
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# 使用新的模块化导入
from env.data_generators import create_generator, DataGenerator, get_data_generator_from_config
from env.text_vectorizer import TextVectorizer, create_text_vectorizer
from env.noise_detectors import (
    get_noise_detector_from_config,
    IDELabelDetector,
    SimpleLabelNoiseDetector,
    ED2RPTDetector,
)
from env.image_dataset_loader import CIFAR10Loader

from .image_classifier import ImageClassifier

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False


class DataCleaningEnv:
    def __init__(self, config):
        self.cfg = config
        self._cat_cols = []
        self._num_cols = []
        self._ord_enc = None
        self._load_data()

    def _float_matrix(self, df):
        """将当前特征列编码为 (n, n_features) 浮点矩阵，列顺序与 cfg.feature_cols 一致。"""
        feat = self.cfg.feature_cols
        if len(df) == 0:
            return np.empty((0, len(feat)), dtype=float)
        if not getattr(self, "_cat_cols", None):
            return df[feat].to_numpy(dtype=float, copy=True)
        out = np.empty((len(df), len(feat)), dtype=float)
        cat_part = self._ord_enc.transform(
            df[self._cat_cols].astype(str).fillna("__missing__")
        )
        for j, col in enumerate(feat):
            if col in self._cat_cols:
                k = self._cat_cols.index(col)
                out[:, j] = cat_part[:, k]
            else:
                out[:, j] = pd.to_numeric(df[col], errors="coerce")
        return out

    def encode_features_to_float(self, df):
        """将 DataFrame 的特征列编码为 float32 numpy 数组（用于噪声检测）。"""
        feat = self.cfg.feature_cols
        if len(df) == 0:
            return np.empty((0, len(feat)), dtype=np.float32)
        if not getattr(self, "_cat_cols", None):
            return df[feat].to_numpy(dtype=np.float32, copy=True)
        out = np.empty((len(df), len(feat)), dtype=np.float32)
        cat_part = self._ord_enc.transform(
            df[self._cat_cols].astype(str).fillna("__missing__")
        )
        for j, col in enumerate(feat):
            if col in self._cat_cols:
                k = self._cat_cols.index(col)
                out[:, j] = cat_part[:, k]
            else:
                out[:, j] = pd.to_numeric(df[col], errors="coerce")
        return out

    def decode_selector_feat_value(self, col, val):
        """将 Selector 输出的浮点预测转为可写入 DataFrame / 展示的值。
        
        对于数值型特征，会使用 scaler 的 mean 和 scale 进行反标准化。
        """
        if col not in getattr(self, "_cat_cols", []):
            # 数值型特征：需要进行反标准化
            scaler = getattr(self, "scaler", None)
            cfg = getattr(self, "cfg", None)
            feat_cols = getattr(cfg, "feature_cols", None) if cfg else None
            if scaler is not None and feat_cols and col in feat_cols:
                try:
                    feat_idx = feat_cols.index(col)
                    # 使用 scaler 的 mean 和 scale 进行反标准化
                    # scaler.mean_[feat_idx] 是该特征的均值
                    # scaler.scale_[feat_idx] 是该特征的标准差
                    mean_val = scaler.mean_[feat_idx]
                    scale_val = scaler.scale_[feat_idx]
                    # 反标准化：original = standardized * scale + mean
                    original_val = float(val) * scale_val + mean_val
                    return float(original_val)
                except Exception:
                    return float(val)
            return float(val)
        k = self._cat_cols.index(col)
        cats = self._ord_enc.categories_[k]
        code = int(np.clip(np.round(float(val)), 0, len(cats) - 1))
        out = cats[code]
        if isinstance(out, bytes):
            return out.decode("utf-8", errors="replace")
        if out == "__missing__":
            return np.nan
        return out

    def _synthetic_float_matrix_to_df(self, X_syn, y_syn):
        """将生成器输出的浮点矩阵还原为与 current_data 列类型一致的 DataFrame。"""
        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col
        rows = []
        for i in range(len(X_syn)):
            d = {lbl: float(y_syn[i])}
            for j, col in enumerate(feat):
                v = float(X_syn[i, j])
                if col in getattr(self, "_cat_cols", []):
                    d[col] = self.decode_selector_feat_value(col, v)
                else:
                    d[col] = v
            rows.append(d)
        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Image data loading helpers
    # ──────────────────────────────────────────────────────────────────────
    
    def _load_image_batch(self, filepath: str) -> tuple:
        """
        从 pickle 文件加载 CIFAR-10 图像批次
        
        Returns:
            X: (N, 3072) uint8 数组
            y: (N,) uint8 数组
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data['data'], np.array(data['labels'])
    
    def _load_images_from_png_dir(self, base_dir: str) -> tuple:
        """
        从 PNG 文件目录加载 CIFAR-10 图片数据
        
        目录结构：
            base_dir/
                airplane/
                    airplane_00000.png
                    airplane_00001.png
                    ...
                automobile/
                bird/
                cat/
                deer/
                dog/
                frog/
                horse/
                ship/
                truck/
        
        Returns:
            X: (N, 3072) uint8 数组
            y: (N,) uint8 数组
        """
        from PIL import Image
        import os
        
        # CIFAR-10 类别顺序（与原始 CIFAR-10 一致）
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        all_images = []
        all_labels = []
        
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # 获取该类别下所有 PNG 文件
            png_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
            
            for png_file in png_files:
                img_path = os.path.join(class_dir, png_file)
                try:
                    # 加载图片并转换为 RGB
                    img = Image.open(img_path).convert('RGB')
                    # 转换为 numpy 数组并展平
                    img_array = np.array(img, dtype=np.uint8)
                    img_flat = img_array.flatten()  # (3072,)
                    
                    all_images.append(img_flat)
                    all_labels.append(label_idx)
                except Exception as e:
                    print(f"  [Warning] 加载图片失败: {png_path}, 错误: {e}")
                    continue
        
        if len(all_images) == 0:
            raise ValueError(f"未找到任何图片文件: {base_dir}")
        
        return np.array(all_images), np.array(all_labels)
    
    def _load_data(self):
        # ── Check if image data (PNG directory mode) ─────────────────────────
        is_image_data = getattr(self.cfg, 'is_image_data', False)
        png_dir = getattr(self.cfg, 'dirty_data_path', None)
        is_png_mode = is_image_data and png_dir and os.path.isdir(png_dir)
        
        if is_png_mode:
            # PNG 图片模式：直接从目录加载
            # 加载 dirty (带噪声标签)
            X_dirty, y_dirty = self._load_images_from_png_dir(png_dir)
            
            # 检查是否有独立的 clean 和 valid 数据
            # 暂时使用 dirty 数据作为所有数据（单数据集模式）
            X_clean = X_dirty.copy()
            y_clean = y_dirty.copy()
            X_valid = X_dirty.copy()
            y_valid = y_dirty.copy()
            
            print(f"  [Image PNG Mode] 加载了 {len(X_dirty)} 张图片")
            
            # 设置标志，跳过后续 CSV 加载
            self._dirty_truncated = None
            self.clean_oracle_X = X_clean
            self.clean_oracle_y = y_clean
            self._oracle_loaded = True
        else:
            # CSV 模式
            oracle_raw = pd.read_csv(self.cfg.oracle_data_path)
            if hasattr(self.cfg, 'dirty_data_path'):
                dirty_raw_check = pd.read_csv(self.cfg.dirty_data_path)
                if len(dirty_raw_check) != len(oracle_raw):
                    print(f"  [Info] Dirty({len(dirty_raw_check)}) vs Oracle({len(oracle_raw)}) 行数不一致，")
                    print(f"         它们是完全不同的数据集，将独立运行")
                self._dirty_truncated = dirty_raw_check
            else:
                self._dirty_truncated = None
            self.clean_oracle = oracle_raw
            self._oracle_loaded = False
        
        # ── Check if image data (should be handled at _load_data entry) ──────────
        is_image_data = getattr(self.cfg, 'is_image_data', False)
        
        if is_png_mode:
            # PNG 模式：数据已在前面加载，跳过重复加载
            # 直接使用已加载的数据创建 DataFrame
            n_samples = len(X_dirty)
            numeric_cols = [f"pixel_{i}" for i in range(self.cfg.n_features)]
            lbl = self.cfg.label_col
            
            # Create DataFrames for compatibility
            self.dirty_raw = pd.DataFrame(X_dirty, columns=numeric_cols)
            self.dirty_raw[lbl] = y_dirty
            
            self.clean_oracle = pd.DataFrame(X_clean, columns=numeric_cols)
            self.clean_oracle[lbl] = y_clean
            
            valid_df = pd.DataFrame(X_valid, columns=numeric_cols)
            valid_df[lbl] = y_valid
            
            # Update feature columns
            feat = numeric_cols
            self.cfg.feature_cols = numeric_cols
            
            # Oracle dirty detection: label noise only for images (feature noise is complex)
            self._oracle_feat_dirty = np.zeros((n_samples, self.cfg.n_features), dtype=bool)
            self._oracle_sample_dirty = (y_dirty != y_clean).astype(float)
            
            # Reference and augmentation pool
            # 使用 valid 数据作为增强池（图像数据没有额外的增强池）
            self.ref_data = valid_df.copy()
            self.aug_pool_init = valid_df.copy()
            self.X_ref = self.ref_data[feat].values.astype(float)
            self.y_ref = self.ref_data[lbl].values.astype(float)
            self._clean_feats_arr = self.clean_oracle[feat].values.astype(float)
            
            # Set image mode flag
            self._is_image_mode = True
            self._is_text_mode = False
            self._text_vectorizer = None
            return
        
        elif is_image_data:
            # Pickle 文件模式：从 pickle 加载
            dirty_path = self.cfg.dirty_data_path
            clean_path = self.cfg.clean_data_path
            valid_path = self.cfg.valid_data_path
            
            X_dirty, y_dirty = self._load_image_batch(dirty_path)
            X_clean, y_clean = self._load_image_batch(clean_path)
            X_valid, y_valid = self._load_image_batch(valid_path)
            
            # Store as numpy arrays directly (no DataFrame needed for images)
            self._image_dirty = X_dirty.astype(np.float32) / 255.0  # Normalize to [0, 1]
            self._image_clean = X_clean.astype(np.float32) / 255.0
            self._image_valid = X_valid.astype(np.float32) / 255.0
            
            # Convert to feature arrays for compatibility with existing logic
            n_samples = len(X_dirty)
            numeric_cols = [f"pixel_{i}" for i in range(self.cfg.n_features)]
            lbl = self.cfg.label_col
            
            # Create DataFrames for compatibility
            self.dirty_raw = pd.DataFrame(X_dirty, columns=numeric_cols)
            self.dirty_raw[lbl] = y_dirty
            
            self.clean_oracle = pd.DataFrame(X_clean, columns=numeric_cols)
            self.clean_oracle[lbl] = y_clean
            
            valid_df = pd.DataFrame(X_valid, columns=numeric_cols)
            valid_df[lbl] = y_valid
            
            # Update feature columns
            feat = numeric_cols
            self.cfg.feature_cols = numeric_cols
            
            # Oracle dirty detection: label noise only for images (feature noise is complex)
            self._oracle_feat_dirty = np.zeros((n_samples, self.cfg.n_features), dtype=bool)
            self._oracle_sample_dirty = (y_dirty != y_clean).astype(float)
            
            # Reference and augmentation pool
            # 使用 valid 数据作为增强池
            self.ref_data = valid_df.copy()
            self.aug_pool_init = valid_df.copy()
            self.X_ref = self.ref_data[feat].values.astype(float)
            self.y_ref = self.ref_data[lbl].values.astype(float)
            self._clean_feats_arr = self.clean_oracle[feat].values.astype(float)
            
            # Set image mode flag
            self._is_image_mode = True
            self._is_text_mode = False
            self._text_vectorizer = None
            return
        
        # ── Tabular data: load CSV files ──────────────────────────────────────
        # Use pre-truncated dirty if available
        if self._dirty_truncated is not None:
            self.dirty_raw = self._dirty_truncated
        else:
            self.dirty_raw = pd.read_csv(self.cfg.dirty_data_path)
        valid_raw = pd.read_csv(self.cfg.valid_data_path)

        # Use pre-loaded oracle (already truncated in _load_data entry)
        # Note: clean_oracle is already set at the beginning of _load_data()

        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col

        # ── Check if text data and vectorize ─────────────────────────────────
        is_text_data = getattr(self.cfg, 'is_text_data', False)
        
        if is_text_data and len(feat) == 1:
            # Text data: use TF-IDF vectorization
            self._is_text_mode = True
            self._text_column = feat[0]
            
            # Get text data
            dirty_texts = self.dirty_raw[self._text_column].fillna("").astype(str).tolist()
            clean_texts = self.clean_oracle[self._text_column].fillna("").astype(str).tolist()
            valid_texts = valid_raw[self._text_column].fillna("").astype(str).tolist()
            
            # Create and fit vectorizer on all texts
            max_features = getattr(self.cfg, 'text_max_features', 500)
            ngram_range = getattr(self.cfg, 'text_ngram_range', (1, 2))
            
            self._text_vectorizer = TextVectorizer(
                method="tfidf",
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                random_state=self.cfg.seed
            )
            
            # Fit on all texts to get consistent vocabulary
            all_texts = dirty_texts + clean_texts + valid_texts
            self._text_vectorizer.fit(all_texts)
            
            # Update n_features to reflect vectorized dimension
            self.cfg.n_features = self._text_vectorizer.n_features
            
            # Use pre-truncated oracle, but extract features using vectorizer
            X_dirty = self._text_vectorizer.transform(dirty_texts)
            X_clean = self._text_vectorizer.transform(clean_texts)
            X_valid = self._text_vectorizer.transform(valid_texts)

            # Convert to DataFrame with numeric column names
            numeric_cols = [f"feat_{i}" for i in range(self.cfg.n_features)]
            
            # Get labels and convert to binary numeric
            y_dirty = self.dirty_raw[lbl].map({'positive': 1, 'negative': 0}).values
            y_clean = self.clean_oracle[lbl].map({'positive': 1, 'negative': 0}).values
            y_valid = valid_raw[lbl].map({'positive': 1, 'negative': 0}).values
            
            # Update dirty_raw with vectorized features
            self.dirty_raw = pd.DataFrame(X_dirty, columns=numeric_cols)
            self.dirty_raw[lbl] = y_dirty
            
            # Keep clean_oracle as is (already truncated at entry), but update features
            # Reconstruct with vectorized features but same labels
            clean_df = pd.DataFrame(X_clean, columns=numeric_cols)
            clean_df[lbl] = y_clean
            self.clean_oracle = clean_df
            
            valid_df = pd.DataFrame(X_valid, columns=numeric_cols)
            valid_df[lbl] = y_valid
            
            # Update feature columns to use numeric names
            feat = numeric_cols
            self.cfg.feature_cols = numeric_cols
            
            # For oracle dirty detection, text noise is simulated at label level
            # (text modification is complex, so we use label noise as proxy)
            self._oracle_feat_dirty = np.zeros((len(self.dirty_raw), self.cfg.n_features), dtype=bool)
            
            # Detect label noise (dirty samples are those where label differs from clean)
            self._oracle_sample_dirty = (y_dirty != y_clean).astype(float)
            
            # Store reference and pool
            self.ref_data = valid_df.copy()
            self.aug_pool_init = pd.DataFrame(columns=numeric_cols + [lbl])
            self.X_ref = self.ref_data[feat].values.astype(float)
            self.y_ref = self.ref_data[lbl].values.astype(float)
            self._clean_feats_arr = self.clean_oracle[feat].values.astype(float)
            return
        
        # ── Tabular data: original logic ──────────────────────────────────────────
        self._is_text_mode = False
        self._text_vectorizer = None
        
        # ── Numeric → float; categorical (object/string) → str + OrdinalEncoder ──
        self._cat_cols = [
            c
            for c in feat
            if c in self.clean_oracle.columns
            and (
                pd.api.types.is_object_dtype(self.clean_oracle[c])
                or pd.api.types.is_string_dtype(self.clean_oracle[c])
            )
        ]
        self._num_cols = [c for c in feat if c not in self._cat_cols]

        for col in self._num_cols:
            self.dirty_raw[col] = pd.to_numeric(self.dirty_raw[col], errors="coerce")
            valid_raw[col] = pd.to_numeric(valid_raw[col], errors="coerce")
            self.clean_oracle[col] = pd.to_numeric(self.clean_oracle[col], errors="coerce")

        for col in self._cat_cols:
            self.dirty_raw[col] = self.dirty_raw[col].fillna("__missing__").astype(str)
            valid_raw[col] = valid_raw[col].fillna("__missing__").astype(str)
            self.clean_oracle[col] = self.clean_oracle[col].fillna("__missing__").astype(str)

        if self._cat_cols:
            self._ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=float,
            )
            self._ord_enc.fit(self.clean_oracle[self._cat_cols].astype(str))
        else:
            self._ord_enc = None

        # ── Reference / Augmentation pool ───────
        np.random.seed(self.cfg.seed)
        # 使用 split 时：ref = sf_valid（用于 reward），aug_pool = D_clean（添加候选池）
        if getattr(self.cfg, 'use_split_data', False):
            print(f"  [Split Mode] Loading D_dirty ({len(self.dirty_raw)}) + D_clean")
            # D_dirty -> dirty_raw
            # D_clean -> clean_oracle
            # sf_valid -> ref_data + aug_pool (or just ref_data)
            idx = np.random.permutation(len(valid_raw))
            n_ref = int(len(valid_raw) * self.cfg.valid_ref_ratio)
            self.ref_data = valid_raw.iloc[idx[:n_ref]].reset_index(drop=True)
            self.aug_pool_init = self.clean_oracle.copy().reset_index(drop=True)
            print(f"  [Split Mode] aug_pool = D_clean ({len(self.aug_pool_init)} rows)")
        else:
            idx = np.random.permutation(len(valid_raw))
            n_ref = int(len(valid_raw) * self.cfg.valid_ref_ratio)
            self.ref_data = valid_raw.iloc[idx[:n_ref]].reset_index(drop=True)
            self.aug_pool_init = valid_raw.iloc[idx[n_ref:]].reset_index(drop=True)

        # Reference arrays (fixed throughout training)
        self.X_ref = self._float_matrix(self.ref_data)
        self.y_ref = self.ref_data[lbl].values.astype(float)

        # ── Pre-extract clean feature array for fast numpy indexing ──────
        self._clean_feats_arr = self._float_matrix(self.clean_oracle)
        
        # 缓存特征标准差（用于动态阈值）
        self._feat_std = np.nanstd(self._clean_feats_arr, axis=0)
        self._feat_std = np.where(self._feat_std > 0, self._feat_std, 1.0)

        # ── Oracle dirty mask (per original sample × feature) ─────────────
        # 处理 dirty 和 oracle 完全独立的情况
        dirty_feats = self._float_matrix(self.dirty_raw)
        clean_feats = self._clean_feats_arr

        n_dirty = len(dirty_feats)
        n_clean = len(clean_feats)

        if n_dirty != n_clean:
            # dirty 和 oracle 长度不一致，认为是完全不同的数据集
            # 不再使用 oracle 判断 dirty 的噪声，而是使用检测器
            print(f"  [Info] Dirty({n_dirty}) vs Oracle({n_clean})，它们是完全不同的数据集")
            print(f"         将使用 ED2/IDE 检测器来检测噪声，而非 oracle")

            # 标记：dirty 和 oracle 是独立数据集
            self._dirty_oracle_independent = True

            # 所有 dirty 样本都标记为可能含有噪声（让检测器来判断）
            self._oracle_feat_dirty = np.ones((n_dirty, len(feat)), dtype=bool)
            self._oracle_sample_dirty = np.ones(n_dirty, dtype=float)
        else:
            # 长度一致时，使用原有的 oracle 对比逻辑
            # 直接对比 dirty 和 clean，不需要阈值（clean 就是标准答案）
            nan_mask = np.isnan(dirty_feats)

            # 特征噪声：直接比较是否相等（NaN 也算噪声）
            feat_diff = np.isnan(dirty_feats) | (dirty_feats != clean_feats)
            self._oracle_feat_dirty = feat_diff

            # 标签噪声：直接比较
            label_noise = (self.dirty_raw[lbl].values != self.clean_oracle[lbl].values)
            self._oracle_sample_dirty = (self._oracle_feat_dirty.any(axis=1) | label_noise).astype(float)

    # ──────────────────────────────────────────────────────────────────────
    # Episode management
    # ──────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset to dirty dataset and return initial state."""
        self.current_data = self.dirty_raw.copy().reset_index(drop=True)
        self.aug_pool = self.aug_pool_init.copy().reset_index(drop=True)

        # sample_ids[i] = row index in original dirty_raw (-1 for added rows)
        self.sample_ids = list(range(len(self.dirty_raw)))
        # Per-current-sample dirty flag (mirrors oracle_sample_dirty initially)
        self.sample_dirty = self._oracle_sample_dirty.copy()
        
        # 记录初始数据大小（用于 reward 计算中的保留率奖励）
        self.initial_data_size = len(self.dirty_raw)
        
        # 记录初始噪声比例（用于 reward 计算中的清理进度奖励）
        self._initial_dirty_ratio = self.sample_dirty.mean()
        
        self.step_count = 0
        self.action_counts = np.zeros(self.cfg.n_actions, dtype=float)
        self.recent_rewards = []
        self._add_candidates_from_smote = False
        self._current_smote_pool = None
        # 标记：每个 joint 结束后才训练一次分类器（lazy 训练）
        self._needs_clf_train = True

        # 初始化 IDE 标签噪声检测器
        use_simple_detector = getattr(self.cfg, 'use_simple_label_detector', False)
        if use_simple_detector:
            self._label_detector = SimpleLabelNoiseDetector(
                noise_ratio=getattr(self.cfg, 'label_noise_ratio', 0.3),
                random_state=self.cfg.seed,
            )
        else:
            self._label_detector = IDELabelDetector(
                clf_type=getattr(self.cfg, 'ide_clf_type', 'mlp'),
                hidden_size=getattr(self.cfg, 'ide_hidden', (64, 32)),
                n_iterations=getattr(self.cfg, 'ide_n_iterations', 3),
                confidence_threshold=getattr(self.cfg, 'ide_confidence_threshold', 0.7),
                noise_ratio_estimate=getattr(self.cfg, 'ide_noise_ratio_estimate', 0.3),
                random_state=self.cfg.seed,
            )

        # 初始化 ED2-RPT 特征噪声检测器（只在第一次 reset 时初始化）
        if not hasattr(self, '_ed2_rpt_initialized') or not self._ed2_rpt_initialized:
            self._ed2_rpt_detector = None
            self._ed2_rpt_step_counter = 0  # 步数计数器
            use_ed2_rpt = getattr(self.cfg, 'use_ed2_rpt', True)
            if use_ed2_rpt:
                self._ed2_rpt_detector = ED2RPTDetector(
                    n_features=self.cfg.n_features,
                    hidden_dims=getattr(self.cfg, 'ed2_rpt_hidden_dims', [128, 64]),
                    device=getattr(self.cfg, 'device', 'cpu'),
                    noise_threshold=getattr(self.cfg, 'ed2_rpt_noise_threshold', 0.5),
                    correction_scale=getattr(self.cfg, 'ed2_rpt_correction_scale', 0.3),
                    random_state=self.cfg.seed,
                )
                # 用干净数据预训练 ED2-RPT（不传标签，保持与检测时输入维度一致）
                feat = self.cfg.feature_cols
                lbl = self.cfg.label_col
                X_clean = self._float_matrix(self.clean_oracle)
                pretrain_epochs = getattr(self.cfg, 'ed2_rpt_pretrain_epochs', 50)
                try:
                    self._ed2_rpt_detector.pretrain(
                        X_clean,  # 不传 y_clean，保持输入维度一致
                        epochs=pretrain_epochs,
                        lr=getattr(self.cfg, 'ed2_rpt_lr', 1e-3),
                        batch_size=getattr(self.cfg, 'ed2_rpt_batch_size', 64),
                    )
                    print(f"  [ED2-RPT] 预训练完成 ({pretrain_epochs} epochs)")
                except Exception as e:
                    print(f"  [ED2-RPT] 预训练失败: {e}, 将使用 oracle 方法")
                    self._ed2_rpt_detector = None
            self._ed2_rpt_initialized = True

        # 初始化数据生成器（支持单生成器或多生成器组合）
        # 使用 get_data_generator_from_config 自动根据配置选择
        from env.data_generators import get_data_generator_from_config
        self._data_generator = get_data_generator_from_config(self.cfg, np.random.default_rng(self.cfg.seed))

        # Train initial classifier and record baseline accuracy
        acc, clf, scaler, imputer = self._train_and_eval()
        self.current_acc = acc
        self.clf = clf
        self.scaler = scaler
        self.imputer = imputer

        # 用已训练的分类器计算标签噪声比例（基于 IDE 迭代检测）
        # IDE 通过迭代翻转标签 + 监控损失变化来识别噪声
        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col
        X_dirty = self._float_matrix(self.dirty_raw)
        y_dirty = self.dirty_raw[lbl].values
        label_noise_ratio = 0.0
        try:
            if self._label_detector is not None:
                _, label_noise_mask, _ = self._label_detector.fit_predict(
                    X_dirty, y_dirty, return_noise_scores=True
                )
                label_noise_ratio = float(label_noise_mask.mean())
            else:
                # Fallback: 用置信度排序估计
                X_imp = self.imputer.transform(X_dirty)
                X_sc = self.scaler.transform(X_imp)
                probs = self.clf.predict_proba(X_sc)
                max_probs = probs.max(axis=1)
                # 取置信度最低的 20% 作为噪声估计
                threshold = np.percentile(max_probs, 20)
                label_noise_ratio = float((max_probs <= threshold).mean())
        except Exception:
            pass
        self._initial_label_noise_ratio = label_noise_ratio

        # 用已预训练的 ED2-RPT 检测器计算特征噪声比例
        feature_noise_ratio = 0.0
        if self._ed2_rpt_detector is not None:
            try:
                _, feat_noise_mask, feat_scores = self._ed2_rpt_detector.detect_and_correct(
                    X_dirty, y_labels=None, return_noise_scores=True
                )
                feature_noise_ratio = float(feat_noise_mask.mean())
            except Exception:
                pass
        self._initial_feat_noise_ratio = feature_noise_ratio

        # 计算缺失率
        dirty_feat = self._float_matrix(self.dirty_raw)
        self._initial_missing_rate = float(np.sum(np.isnan(dirty_feat)) / dirty_feat.size)

        # Rollback state bookkeeping
        self.best_acc = acc
        self._best_snap = self._snapshot()
        self.consecutive_neg = 0

        return self._state()

    # ──────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action_idx, selected_indices, user_feedback=None, selector_pred=None):
        """
        Apply high-level action to the selected candidate indices.

        Parameters
        ----------
        action_idx : int
            动作索引 (0=modify_features, 1=modify_labels, 2=delete, 3=add, 4=no_op)
        selected_indices : list
            选择的样本索引
        user_feedback : str or None
            用户反馈：
            - None: 自动判断（模拟模式，用 ground truth）
            - "accept": 用户接受
            - "reject": 用户拒绝
            - "auto": 自动判断
        selector_pred : np.ndarray or None
            Selector 预测的值：
            - action_idx == 0: feat_pred (n_selected, n_features) 预测的干净特征
            - action_idx == 1: label_pred (n_selected,) 预测的正确标签
            - 其他: None

        Returns
        -------
        next_state : np.ndarray  (state_dim,)
        reward     : float
        done       : bool
        info       : dict
        """
        self.step_count += 1
        self.action_counts[action_idx] += 1

        # 记录修改前的快照（用于用户拒绝时恢复）
        pre_snapshot = self._snapshot()
        pre_acc = self.current_acc

        # 执行动作（传入 Selector 预测值）
        if action_idx == 0:
            self._act_modify_features(selected_indices, selector_pred)
        elif action_idx == 1:
            self._act_modify_labels(selected_indices, selector_pred)
        elif action_idx == 2:
            self._act_delete_samples(selected_indices)
        elif action_idx == 3:
            self._act_add_samples(selected_indices)
        # 4 → no_op: nothing to do

        # 检查用户反馈
        feedback_mode = user_feedback if user_feedback is not None else getattr(self.cfg, 'user_feedback_mode', 'auto')

        if feedback_mode != "auto":
            # 显式用户反馈模式
            user_accepted = (feedback_mode == "accept")
        else:
            # 自动判断模式：用 ground truth 模拟用户决策
            user_accepted = self._simulate_user_feedback(action_idx, selected_indices)

        # 如果用户拒绝，恢复修改并给予惩罚
        if not user_accepted:
            self._restore(pre_snapshot)
            # 惩罚性 reward：基于错误程度
            reward = getattr(self.cfg, 'reject_penalty', -0.1)
            # 记录 reward breakdown（用于调试和分析）
            self._last_reward_breakdown = {
                'base_reward': reward, 'delete_penalty': 0,
                'preserve_reward': 0, 'action_bonus': 0, 'exploration_bonus': 0, 'diversity_bonus': 0, 'total': reward
            }
            info = {
                "accuracy": self.current_acc,
                "best_accuracy": self.best_acc,
                "n_samples": len(self.current_data),
                "action_name": self.cfg.action_names[action_idx],
                "n_selected": len(selected_indices),
                "user_feedback": "reject",
                "rejected": True,
                # 新增：噪声检测指标（即使是拒绝也要记录）
                "selected_noise_count": int(sum(self.sample_dirty[idx] for idx in selected_indices if idx < len(self.sample_dirty))) if len(selected_indices) > 0 else 0,
                "selected_noise_ratio": 0.0,  # 拒绝时修改未生效
            }
            return self._state(), reward, False, info

        # 用户接受：继续正常流程
        # 标记：在 joint 结束后统一训练分类器，这里不再每步训练
        self._needs_clf_train = True

        # 在线微调 ED2-RPT（每隔一定步数）
        if (getattr(self.cfg, 'ed2_rpt_online_finetune', True)
            and self._ed2_rpt_detector is not None):
            self._ed2_rpt_step_counter += 1
            finetune_interval = getattr(self.cfg, 'ed2_rpt_finetune_interval', 5)
            if self._ed2_rpt_step_counter % finetune_interval == 0:
                self._finetune_ed2_rpt()

        # 先用当前 acc/reward 计算，后续在 done 确定后再决定是否训练分类器
        new_acc = self.current_acc
        new_clf = self.clf
        new_scaler = self.scaler
        new_imp = self.imputer
        
        # 基础 reward：准确率变化
        acc_delta = new_acc - self.current_acc
        base_reward = acc_delta * self.cfg.reward_scale
        
        # ── 改进的 reward 设计 ─────────────────────────────────────────────
        # 1. 删除/修改惩罚与奖励：根据检测精度给予差异化 reward
        n_selected = len(selected_indices)
        current_size = len(self.current_data) + n_selected  # 修改前的大小

        # 新增：基于选中样本中实际噪声比例的奖励/惩罚
        correct_selection_reward = 0.0
        if action_idx in (0, 1, 2) and n_selected > 0:  # modify or delete
            # 计算选中的样本中有多少是真正的噪声
            selected_noise_count = int(sum(self.sample_dirty[idx] for idx in selected_indices if idx < len(self.sample_dirty)))
            noise_precision = selected_noise_count / n_selected if n_selected > 0 else 0
            
            # 如果选中大多是噪声，给予正奖励；否则给予惩罚
            # 大幅增加奖励/惩罚的幅度，让信号更明显
            if noise_precision >= 0.7:  # 高精度：选中的 mostly 是噪声
                correct_selection_reward = 0.30 * noise_precision
            elif noise_precision >= 0.5:  # 中等精度
                correct_selection_reward = 0.15 * (noise_precision - 0.5)
            else:  # 低精度：删除了很多好样本
                correct_selection_reward = -0.35 * (0.5 - noise_precision)

        if action_idx == 2 and n_selected > 0:  # delete action
            delete_ratio = n_selected / current_size if current_size > 0 else 0
            # 改进：基于删除比例的惩罚，但如果有高噪声精度则减免
            delete_penalty = -delete_ratio * self.cfg.delete_penalty_scale
            if correct_selection_reward > 0:  # 如果删除的是噪声，减轻惩罚
                delete_penalty *= 0.5
        else:
            delete_penalty = 0.0

        # 2. 保留率奖励：保留更多数据获得额外奖励（相对于初始数据）
        preserve_ratio = len(self.current_data) / self.initial_data_size if self.initial_data_size > 0 else 0
        preserve_reward = preserve_ratio * self.cfg.preserve_reward_scale

        # 3. 动作平衡奖励：鼓励使用 modify，适度 delete，减少 add
        action_names = getattr(self.cfg, 'action_names',
                               ['modify_features', 'modify_labels', 'delete_samples', 'add_samples', 'no_op'])
        action_name = action_names[action_idx] if action_idx < len(action_names) else 'no_op'
        action_bonus_dict = getattr(self.cfg, 'action_bonus', {
            "modify_features": 0.05,
            "modify_labels": 0.05,
            "delete_samples": 0.03,
            "add_samples": 0.05,
            "no_op": -0.50,  # 大幅增加不操作惩罚
        })
        action_bonus = action_bonus_dict.get(action_name, 0.0)

        # 4. 动作多样性奖励：鼓励选择不同的动作（与历史动作不同）
        diversity_bonus = 0.0
        if hasattr(self, 'action_history') and len(self.action_history) > 0:
            # 如果当前动作与最近的动作不同，给奖励
            if action_idx != self.action_history[-1]:
                diversity_bonus = getattr(self.cfg, 'diversity_bonus', 0.01)
        # 记录动作历史
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action_idx)
        # 只保留最近10个动作
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]

        # 5. 探索奖励：只要采取了有效动作（不是 no_op 且有选中样本），给一个小的正奖励
        exploration_bonus = 0.0
        if action_idx != 4 and n_selected > 0:  # 不是 no_op 且有选中样本
            exploration_bonus = getattr(self.cfg, 'exploration_bonus', 0.01)
        
        # 6. 噪声清理进度奖励：鼓励降低 dirty ratio
        dirty_progress_reward = 0.0
        if hasattr(self, '_initial_dirty_ratio') and action_idx != 4:  # 只有有效动作才计算
            current_dirty = self.sample_dirty.mean()
            dirty_delta = self._initial_dirty_ratio - current_dirty  # 清理了多少噪声
            # 如果噪声比例下降，给予奖励
            if dirty_delta > 0.01:  # 至少有 1% 的噪声被清理
                dirty_progress_reward = 0.20 * dirty_delta
        
        # 7. 修正 no_op 惩罚：增大惩罚并加入冷却机制
        no_op_penalty = 0.0
        if action_idx == 4:
            # 检查是否连续 no_op
            recent_actions = self.action_history[-5:] if hasattr(self, 'action_history') and len(self.action_history) >= 5 else []
            no_op_count = recent_actions.count(4) if recent_actions else 0
            # 连续 no_op 越多，惩罚越大
            base_noop_penalty = getattr(self.cfg, 'no_op_penalty', -0.20)
            no_op_penalty = base_noop_penalty * (1 + no_op_count * 0.8)
        
        # 8. 新增：基于分类器置信度变化的奖励
        confidence_reward = 0.0
        if hasattr(self, '_recent_avg_confidence'):
            if self.current_acc > self.best_acc * 0.99:  # 准确率接近最优
                confidence_reward = 0.05
        # 更新历史置信度
        if not hasattr(self, '_recent_avg_confidence'):
            self._recent_avg_confidence = []
        if len(self.recent_rewards) > 0:
            self._recent_avg_confidence.append(self.current_acc)
            if len(self._recent_avg_confidence) > 10:
                self._recent_avg_confidence = self._recent_avg_confidence[-10:]
        
        # 综合 reward（加入正确选择奖励和清理进度奖励）
        reward = (base_reward + delete_penalty + preserve_reward + 
                  action_bonus + exploration_bonus + diversity_bonus + 
                  correct_selection_reward + dirty_progress_reward + no_op_penalty + confidence_reward)
        
        # 记录各 reward 分量（用于调试和分析）
        self._last_reward_breakdown = {
            'base_reward': base_reward,
            'delete_penalty': delete_penalty,
            'preserve_reward': preserve_reward,
            'action_bonus': action_bonus,
            'exploration_bonus': exploration_bonus,
            'diversity_bonus': diversity_bonus,
            'correct_selection_reward': correct_selection_reward,
            'dirty_progress_reward': dirty_progress_reward,
            'no_op_penalty': no_op_penalty,
            'confidence_reward': confidence_reward,
            'total': reward,
        }

        # Update model references
        self.current_acc = new_acc
        if new_clf is not None:
            self.clf = new_clf
            self.scaler = new_scaler
            self.imputer = new_imp

        self.recent_rewards.append(reward)

        # Rollback safety
        if new_acc > self.best_acc:
            self.best_acc = new_acc
            self._best_snap = self._snapshot()
            self.consecutive_neg = 0
            # 在达到 best 的当步保存数据，避免保存到“本 episode 末尾”导致 Best state 偏低
            best_path = getattr(self.cfg, "best_data_path", None)
            if best_path:
                self.current_data.to_csv(best_path, index=False)
        else:
            self.consecutive_neg = self.consecutive_neg + 1 if reward < 0 else 0

        if self.consecutive_neg >= self.cfg.rollback_patience:
            self._restore(self._best_snap)
            self.consecutive_neg = 0
            reward = -0.05  # small rollback penalty

        done = (
            self.step_count >= self.cfg.max_steps_per_episode
            or len(self.current_data) < self.cfg.min_dataset_size
        )

        # ── 在 joint 结束时（done=True）或每个 action_idx==4 (no_op) 后训练一次 ──
        if done or action_idx == 4:
            new_acc, new_clf, new_scaler, new_imp = self._train_and_eval(force=True)

        # 计算选中样本中的噪声比例（用于观察智能体是否真的找到了噪声）
        selected_noise_count = 0
        selected_total = len(selected_indices)
        if selected_total > 0 and action_idx in (0, 1, 2):
            # 对于 modify/delete 操作，用 oracle 判断选中样本中有多少是真正的噪声
            selected_noise_count = int(sum(self.sample_dirty[idx] for idx in selected_indices if idx < len(self.sample_dirty)))

        # 计算特征噪声比例、标签噪声比例、缺失率（基于初始 dirty_raw 恒定追踪）
        label_noise_ratio = self._initial_label_noise_ratio
        feature_noise_ratio = self._initial_feat_noise_ratio
        dirty_feat = self._float_matrix(self.dirty_raw)
        missing_rate = self._initial_missing_rate
        
        # 计算类别比例（格式: "30%/70%"）
        lbl = self.cfg.label_col
        class_counts = self.current_data[lbl].value_counts().sort_index()
        class_pcts = (class_counts / class_counts.sum() * 100).round(1)
        class_ratio_str = "/".join([f"{p:.1f}%" for p in class_pcts.values])
        
        info = {
            "accuracy": self.current_acc,
            "best_accuracy": self.best_acc,
            "n_samples": len(self.current_data),
            "action_name": self.cfg.action_names[action_idx],
            "n_selected": len(selected_indices),
            "class_ratio": class_ratio_str,
            "label_noise_ratio": round(label_noise_ratio, 4),
            "feature_noise_ratio": round(feature_noise_ratio, 4),
            "missing_rate": round(missing_rate, 4),
            "user_feedback": "accept",
            "rejected": False,
            # 新增：噪声检测指标
            "selected_noise_count": selected_noise_count,
            "selected_noise_ratio": selected_noise_count / selected_total if selected_total > 0 else 0.0,
            # 新增：reward breakdown（用于调试和分析）
            "reward_breakdown": getattr(self, '_last_reward_breakdown', {
                'base_reward': reward, 'delete_penalty': 0,
                'preserve_reward': 0, 'action_bonus': 0, 'exploration_bonus': 0, 'diversity_bonus': 0, 'total': reward
            }),
        }
        return self._state(), reward, done, info

    # ──────────────────────────────────────────────────────────────────────
    # Selector inputs
    # ──────────────────────────────────────────────────────────────────────

    def get_candidates(self, action_idx):
        """
        Return candidate data for the low-level selector.

        Returns None for no_op (action 4).

        Returns dict with keys:
          X           – (n, n_features) float array (NaN possible)
          y           – (n,) float labels
          u           – (n, sample_feature_dim) per-sample features
          oracle_dirty– (n,) float [0,1]  oracle dirty signal
          clean_feats – (n, n_features) oracle clean features
          is_aug_pool – bool  True when candidates come from aug pool
        """
        if action_idx == 4:
            return None

        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col

        if action_idx == 3:  # add_samples – candidates from SMOTE or aug pool
            use_smote = getattr(self.cfg, "use_smote_for_add", False) and _HAS_IMBLEARN
            if use_smote:
                cand = self._generate_smote_candidates()
                if cand is None:
                    if len(self.aug_pool) == 0:
                        return None
                    use_smote = False
                else:
                    self._add_candidates_from_smote = True
                    return cand
            if not use_smote and len(self.aug_pool) == 0:
                return None
            self._add_candidates_from_smote = False
            X = self._float_matrix(self.aug_pool)
            y = self.aug_pool[lbl].values.astype(float)
            oracle_dirty = np.zeros(len(X), dtype=float)
            clean_feats = X.copy()
            is_aug = True
        else:  # actions 0, 1, 2 – candidates from current dataset
            X = self._float_matrix(self.current_data)
            y = self.current_data[lbl].values.astype(float)
            oracle_dirty = self.sample_dirty.copy()
            clean_feats = self._get_current_clean_feats()
            is_aug = False

        u = self._per_sample_features(X, y, oracle_dirty)
        return dict(X=X, y=y, u=u, oracle_dirty=oracle_dirty,
                    clean_feats=clean_feats, is_aug_pool=is_aug)

    def _get_current_clean_feats(self):
        """Oracle clean features for every row in current_data (vectorised)."""
        feat = self.cfg.feature_cols
        clean_all = self._clean_feats_arr   # pre-extracted numpy array (set in _load_data)
        current_X = self._float_matrix(self.current_data)

        n = len(self.current_data)
        n_clean = len(clean_all)
        result = np.empty((n, self.cfg.n_features), dtype=float)

        ids = np.array(self.sample_ids, dtype=int)

        # 边界检查：sample_ids 必须在 [0, n_clean) 范围内
        valid = (ids >= 0) & (ids < n_clean)
        if valid.any():
            result[valid] = clean_all[ids[valid]]
        if (~valid).any():
            # 对于无效索引（新增样本或超出范围的索引），使用当前数据
            result[~valid] = current_X[~valid]
        return result

    def _generate_smote_candidates(self):
        """
        用数据生成器（方案一）从当前数据生成合成样本，作为 add_samples 的候选池。

        支持多种生成方法：
        - random: 随机插值
        - smote: SMOTE
        - adasyn: ADASYN (自适应合成采样)
        - borderline: BorderlineSMOTE
        - mixed: 混合方法

        返回与 get_candidates 相同结构的 dict，失败时返回 None。
        """
        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col
        X = self._float_matrix(self.current_data)
        y = self.current_data[lbl].values.astype(float)
        n_orig = len(X)

        if n_orig < 10 or len(np.unique(y)) < 2:
            return None

        # 对输入数据进行填充处理
        if self.imputer is not None:
            try:
                X_imp = self.imputer.transform(X)
            except Exception:
                X_imp = np.nan_to_num(X, nan=0.0)
        else:
            X_imp = np.nan_to_num(X, nan=0.0)

        # 设置当前步数到生成器（用于随机种子）
        if hasattr(self._data_generator, 'step_count'):
            self._data_generator.step_count = self.step_count

        # 计算需要生成的样本数量
        n_want = min(
            getattr(self.cfg, "smote_candidates_mult", 4) * self.cfg.max_add_samples,
            getattr(self.cfg, "smote_max_candidates", 120),
            n_orig * 2,  # 限制最大生成数量
        )

        # 使用数据生成器生成合成样本
        try:
            X_syn, y_syn = self._data_generator.generate(X_imp, y, n_want)
        except Exception:
            return None

        if len(X_syn) == 0:
            return None

        self._current_smote_pool = self._synthetic_float_matrix_to_df(X_syn, y_syn)
        oracle_dirty = np.zeros(len(X_syn), dtype=float)
        clean_feats = X_syn.copy()
        u = self._per_sample_features(X_syn, y_syn, oracle_dirty)
        return dict(
            X=X_syn,
            y=y_syn,
            u=u,
            oracle_dirty=oracle_dirty,
            clean_feats=clean_feats,
            is_aug_pool=True,
        )

    def _per_sample_features(self, X, y, oracle_dirty):
        """
        Build u_i = [entropy, loss, margin, norm_features×6, oracle_dirty]
        Shape: (n, 10) or (n, 9) if use_oracle_in_u=False

        根据 cfg.use_oracle_in_u 决定是否包含 oracle_dirty
        """
        n = len(X)
        feat = self.cfg.feature_cols

        # Lazy training: ensure classifier is trained before using it
        # (the first call in each joint will trigger _train_and_eval)
        self._ensure_classifier_trained()
        
        # 根据配置决定是否包含oracle_dirty
        use_oracle_in_u = getattr(self.cfg, 'use_oracle_in_u', False)
        
        # Impute for model inference
        if self.imputer is not None:
            try:
                X_imp = self.imputer.transform(X)
            except Exception:
                X_imp = np.nan_to_num(X, nan=0.0)
        else:
            X_imp = np.nan_to_num(X, nan=0.0)

        # Model-based uncertainty signals  (fully vectorised)
        if self.clf is not None and self.scaler is not None:
            try:
                X_sc = self.scaler.transform(X_imp)
                probs = self.clf.predict_proba(X_sc)   # (n, 2)
                entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
                margin = np.abs(probs[:, 1] - 0.5)
                # Vectorised per-sample cross-entropy loss
                classes = np.array(self.clf.classes_)
                y_int = y.astype(int)
                # Map each label to its column index in probs
                label_to_col = {int(c): j for j, c in enumerate(classes)}
                col_idx = np.array(
                    [label_to_col.get(int(yi), 0) for yi in y_int], dtype=int
                )
                prob_true = probs[np.arange(n), col_idx]
                loss = -np.log(np.clip(prob_true, 1e-8, 1.0))
            except Exception:
                entropy = np.zeros(n)
                margin = np.zeros(n)
                loss = np.zeros(n)
        else:
            entropy = np.zeros(n)
            margin = np.zeros(n)
            loss = np.zeros(n)

        # Normalised raw features: NaN → –5 (strong out-of-range signal)
        feat_mean = np.nanmean(X, axis=0)
        feat_std = np.nanstd(X, axis=0) + 1e-8
        X_norm = (X - feat_mean) / feat_std
        X_norm = np.where(np.isnan(X_norm), -5.0, X_norm)
        X_norm = np.clip(X_norm, -10, 10)

        # 根据配置决定u向量的组成
        if use_oracle_in_u:
            # 包含 oracle_dirty: [entropy, loss, margin, norm_features, oracle_dirty]
            return np.column_stack([
                entropy, loss, margin, X_norm, oracle_dirty
            ]).astype(np.float32)
        else:
            # 不包含 oracle_dirty: [entropy, loss, margin, norm_features]
            # 更符合实际应用场景（没有Oracle答案）
            return np.column_stack([
                entropy, loss, margin, X_norm
            ]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # State vector
    # ──────────────────────────────────────────────────────────────────────

    def _state(self):
        """Compute global state vector S_t (dim = 22)."""
        feat = self.cfg.feature_cols
        X = self._float_matrix(self.current_data)
        y_raw = self.current_data[self.cfg.label_col].values
        
        # 处理标签：如果是字符串，转换为数值
        if y_raw.dtype.kind in ['U', 'S', 'O'] or (hasattr(y_raw, 'dtype') and y_raw.dtype == object):
            label_map = {'<=50K': 0, '>50K': 1}
            if str(y_raw[0]) in label_map:
                y = np.array([label_map.get(str(v), v) for v in y_raw], dtype=float)
            else:
                try:
                    y = np.array([float(v) for v in y_raw], dtype=float)
                except:
                    y = y_raw.astype(float)
        else:
            y = y_raw.astype(float)

        n_samples_ratio = len(X) / len(self.dirty_raw)
        class_ratio = np.nanmean(y)  # 使用 nanmean 处理可能的 NaN
        accuracy = self.current_acc
        recent_rwd = float(np.mean(self.recent_rewards[-5:])) if self.recent_rewards else 0.0
        dirty_ratio = float(self.sample_dirty.mean()) if len(self.sample_dirty) else 0.0

        denom = np.nanmax(np.abs(X), axis=0) + 1e-8
        feat_means = np.nanmean(X, axis=0) / denom
        feat_stds = np.nanstd(X, axis=0) / denom

        action_norm = self.action_counts / (self.step_count + 1)

        state = np.concatenate([
            [n_samples_ratio, class_ratio, accuracy, recent_rwd, dirty_ratio],
            feat_means, feat_stds, action_norm,
        ]).astype(np.float32)

        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Classifier
    # ──────────────────────────────────────────────────────────────────────

    def _build_classifier(self):
        """Build classifier instance from config (MLP or LogisticRegression)."""
        cfg = self.cfg
        # 训练时优先用 env_classifier_type（可设为 logistic 加速），否则用 classifier_type
        which = getattr(cfg, "env_classifier_type", None) or getattr(cfg, "classifier_type", "logistic")
        if which == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=getattr(cfg, "mlp_hidden", (128, 64)),
                max_iter=cfg.classifier_max_iter,
                random_state=cfg.seed,
                early_stopping=getattr(cfg, "mlp_early_stopping", True),
                validation_fraction=getattr(cfg, "mlp_val_fraction", 0.1),
            )
        elif which == "image":
            return ImageClassifier(num_classes=10)
        
        else:
            return LogisticRegression(
                max_iter=cfg.classifier_max_iter,
                random_state=cfg.seed,
                solver="lbfgs",
            )

    def _ensure_classifier_trained(self):
        """Lazy training: only train when _needs_clf_train is set."""
        if self._needs_clf_train:
            self.current_acc, self.clf, self.scaler, self.imputer = self._train_and_eval(force=True)

    def _train_and_eval(self, force=False):
        """Train classifier on current_data, evaluate on reference set.
        
        Parameters
        ----------
        force : bool
            If True, always retrain. If False (default), only retrain when
            _needs_clf_train is set (i.e. at the end of each joint).
        """
        # Skip if not forced and not flagged as needing training
        if not force and not getattr(self, '_needs_clf_train', True):
            return self.current_acc, self.clf, self.scaler, self.imputer

        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col
        X = self._float_matrix(self.current_data)
        y_raw = self.current_data[lbl].values
        
        # 处理标签：如果是字符串，转换为数值
        if y_raw.dtype.kind in ['U', 'S', 'O'] or (hasattr(y_raw, 'dtype') and y_raw.dtype == object):
            label_map = {'<=50K': 0, '>50K': 1}
            if str(y_raw[0]) in label_map:
                y = np.array([label_map.get(str(v), np.nan) for v in y_raw], dtype=float)
            else:
                try:
                    y = np.array([float(v) if not pd.isna(v) else np.nan for v in y_raw], dtype=float)
                except:
                    y = y_raw.astype(float)
        else:
            y = y_raw.astype(float)
        
        # 过滤掉标签为 NaN 的样本
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 20 or len(np.unique(y[valid_mask])) < 2:
            return self.current_acc, self.clf, self.scaler, self.imputer
        X = X[valid_mask]
        y = y[valid_mask]

        # Impute NaN with column mean
        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_imp)

        clf = self._build_classifier()
        clf.fit(X_sc, y)

        # Evaluate on reference set (already clean, no NaN)
        X_ref_sc = scaler.transform(imputer.transform(self.X_ref))
        acc = clf.score(X_ref_sc, self.y_ref)
        # Clear the flag after training
        self._needs_clf_train = False
        return acc, clf, scaler, imputer

    # ──────────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────────

    def _act_modify_features(self, indices, selector_pred=None):
        """
        使用 Selector 预测值修复特征噪声。

        优先级：
        1. Selector 预测值 (selector_pred) - 优先使用训练好的 Selector
        2. ED2-RPT 检测器 - 如果 Selector 不可用或失败
        3. Oracle 修复 - 最后回退选项

        Parameters
        ----------
        indices : list
            选中的样本索引
        selector_pred : np.ndarray or None
            Selector 预测的干净特征 (n_selected, n_features)
        """
        max_k = max(1, int(len(self.current_data) * self.cfg.max_modify_ratio))
        indices = list(indices)[:max_k]

        if len(indices) == 0:
            return

        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col

        # 优先使用 Selector 预测值
        if selector_pred is not None:
            pred = np.asarray(selector_pred)
            n_selected = len(indices)
            n_pred = len(pred)

            # 确保 pred 与 indices 长度匹配
            if n_pred >= n_selected:
                for i, idx in enumerate(indices):
                    if idx >= len(self.current_data):
                        continue
                    for j, col in enumerate(feat):
                        if j < pred.shape[1]:
                            raw_p = float(pred[i, j])
                            raw_p = np.clip(raw_p, -1e6, 1e6)
                            if col in getattr(self, "_cat_cols", []):
                                self.current_data.at[idx, col] = self.decode_selector_feat_value(
                                    col, raw_p
                                )
                                continue
                            val = raw_p
                            # 根据列类型进行转换
                            col_dtype = self.current_data[col].dtype
                            if np.issubdtype(col_dtype, np.integer):
                                val = int(round(val))
                                # 处理无符号整数类型（如 uint8 像素值 0-255）
                                if np.issubdtype(col_dtype, np.unsignedinteger):
                                    info = np.iinfo(col_dtype)
                                    val = np.clip(val, info.min, info.max)
                                elif val < 0:
                                    val = max(0, val)
                            elif np.issubdtype(col_dtype, np.floating):
                                pass  # 浮点类型直接使用
                            self.current_data.at[idx, col] = val

                # 更新 dirty 标记（假设 Selector 正确修复了）
                self._update_dirty_after_modify(indices, feat)
                return
            else:
                print(f"  [Selector] pred 长度 {n_pred} < indices 长度 {n_selected}，使用备用方法")

        # 回退：使用 ED2-RPT
        if self._ed2_rpt_detector is not None:
            X_selected = self._float_matrix(self.current_data.iloc[indices])

            try:
                X_corrected, noise_mask, noise_prob = self._ed2_rpt_detector.detect_and_correct(
                    X_selected, y_labels=None, return_noise_scores=True
                )

                n_fixed = 0
                for i, idx in enumerate(indices):
                    if idx >= len(self.current_data):
                        continue
                    for j, col in enumerate(feat):
                        if noise_mask[i, j]:
                            val = X_corrected[i, j]
                            if col in getattr(self, "_cat_cols", []):
                                self.current_data.at[idx, col] = self.decode_selector_feat_value(
                                    col, float(val)
                                )
                                n_fixed += 1
                                continue
                            # 根据列类型进行转换
                            col_dtype = self.current_data[col].dtype
                            if np.issubdtype(col_dtype, np.integer):
                                val = int(round(val))
                                if np.issubdtype(col_dtype, np.unsignedinteger):
                                    info = np.iinfo(col_dtype)
                                    val = np.clip(val, info.min, info.max)
                                elif val < 0:
                                    val = max(0, val)
                            self.current_data.at[idx, col] = val
                            n_fixed += 1

                self._update_dirty_after_modify(indices, feat)
                return

            except Exception as e:
                print(f"  [ED2-RPT] 修复失败，回退到 oracle: {e}")

        # 最后回退：使用 oracle 修复
        self._oracle_fix_features(indices, feat)

    def _update_dirty_after_modify(self, indices, feat):
        """修改特征后更新 dirty 标记"""
        max_orig_id = len(self.clean_oracle) - 1
        for idx in indices:
            if idx >= len(self.current_data):
                continue
            if getattr(self, '_dirty_oracle_independent', False):
                if self._ed2_rpt_detector is not None:
                    try:
                        X_after = self._float_matrix(self.current_data.iloc[[idx]])
                        _, _, noise_probs_after = self._ed2_rpt_detector.detect_and_correct(
                            X_after, y_labels=None, return_noise_scores=True
                        )
                        score_after = noise_probs_after.mean()
                        if score_after < 0.6:
                            self.sample_dirty[idx] = 0.0
                    except:
                        pass
            else:
                orig_id = self.sample_ids[idx]
                if 0 <= orig_id <= max_orig_id:
                    try:
                        current_val = self._float_matrix(self.current_data.iloc[[idx]])[0]
                        oracle_val = self._clean_feats_arr[orig_id]
                        remaining_noise = np.isnan(current_val) | (current_val != oracle_val)
                        if not remaining_noise.any():
                            self.sample_dirty[idx] = 0.0
                    except (IndexError, KeyError):
                        pass

    def _oracle_fix_features(self, indices, feat):
        """使用 oracle 修复特征（最后回退选项）"""
        max_orig_id = len(self.clean_oracle) - 1
        for idx in indices:
            if idx >= len(self.current_data):
                continue
            orig_id = self.sample_ids[idx]
            if orig_id < 0 or orig_id >= len(self.clean_oracle):
                continue
            try:
                clean_row = self.clean_oracle.iloc[orig_id]
            except (IndexError, KeyError):
                continue
            for col in feat:
                val = clean_row[col]
                # 根据列类型进行转换（分类列保持字符串，与 oracle 一致）
                col_dtype = self.current_data[col].dtype
                if col not in getattr(self, "_cat_cols", []):
                    if np.issubdtype(col_dtype, np.integer):
                        val = int(round(float(val)))
                        if np.issubdtype(col_dtype, np.unsignedinteger):
                            info = np.iinfo(col_dtype)
                            val = np.clip(val, info.min, info.max)
                        elif val < 0:
                            val = max(0, val)
                    elif np.issubdtype(col_dtype, np.floating):
                        val = float(val)
                self.current_data.at[idx, col] = val
            self.sample_dirty[idx] = 0.0

    def _finetune_ed2_rpt(self):
        """
        使用当前数据中已被修复的样本（oracle 标记为干净的）对 ED2-RPT 进行在线微调。
        这让模型能够持续适应实际的噪声模式。
        """
        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col
        
        # 找出当前被标记为"干净"的样本（orig_id >= 0 且 sample_dirty == 0）
        clean_mask = (np.array(self.sample_ids) >= 0) & (self.sample_dirty == 0)
        if clean_mask.sum() < 10:
            return  # 样本太少，不进行微调
        
        try:
            X_clean = self._float_matrix(self.current_data.iloc[clean_mask])
            # y_clean 不再传给 pretrain，保持与预训练和检测时输入维度一致
            
            finetune_epochs = getattr(self.cfg, 'ed2_rpt_finetune_epochs', 10)
            finetune_lr = getattr(self.cfg, 'ed2_rpt_lr', 1e-4)  # 用更小的学习率
            batch_size = getattr(self.cfg, 'ed2_rpt_batch_size', 64)
            
            self._ed2_rpt_detector.pretrain(
                X_clean,  # 不传 y_clean，保持输入维度一致
                epochs=finetune_epochs,
                lr=finetune_lr,
                batch_size=batch_size,
            )
        except Exception as e:
            pass  # 微调失败不影响主流程

    def _act_modify_labels(self, indices, selector_pred=None):
        """
        使用 Selector 预测值修复标签噪声。

        优先级：
        1. Selector 预测值 (selector_pred) - 优先使用训练好的 Selector
        2. IDE 检测器 - 如果 Selector 不可用或失败
        3. Oracle 修复 - 最后回退选项

        Parameters
        ----------
        indices : list
            选中的样本索引
        selector_pred : np.ndarray or None
            Selector 预测的正确标签 (n_selected,)
        """
        max_k = max(1, int(len(self.current_data) * self.cfg.max_modify_ratio))
        indices = list(indices)[:max_k]

        if len(indices) == 0:
            return

        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col

        # 优先使用 Selector 预测值
        if selector_pred is not None:
            pred = np.asarray(selector_pred)
            n_selected = len(indices)
            n_pred = len(pred)

            if n_pred >= n_selected:
                for i, idx in enumerate(indices):
                    if idx >= len(self.current_data):
                        continue
                    # Selector 预测值是 0-1 之间的概率，阈值 0.5 决定是否翻转
                    pred_label = 1.0 if float(pred[i]) > 0.5 else 0.0
                    self.current_data.at[idx, lbl] = pred_label

                # 更新 dirty 标记
                self._update_label_dirty_after_modify(indices, lbl)
                return
            else:
                print(f"  [Selector] pred 长度 {n_pred} < indices 长度 {n_selected}，使用备用方法")

        # 回退：使用 IDE 检测器
        X_selected = self._float_matrix(self.current_data.iloc[indices])
        y_current = self.current_data.iloc[indices][lbl].values.astype(float)

        if self._label_detector is not None:
            try:
                y_corrected, noise_mask, noise_scores = self._label_detector.fit_predict(
                    X_selected, y_current
                )

                for i, idx in enumerate(indices):
                    if idx >= len(self.current_data):
                        continue
                    if noise_mask[i]:
                        val = float(y_corrected[i])
                        # 标签列转换为整数
                        lbl_dtype = self.current_data[lbl].dtype
                        if np.issubdtype(lbl_dtype, np.integer):
                            val = int(round(val))
                        self.current_data.at[idx, lbl] = val

                self._update_label_dirty_after_modify(indices, lbl)
                return
            except Exception as e:
                print(f"  [IDE] 标签修复失败: {e}")

        # 最后回退：使用 oracle 修复
        self._oracle_fix_labels(indices, lbl)

    def _update_label_dirty_after_modify(self, indices, lbl):
        """修改标签后更新 dirty 标记"""
        max_orig_id = len(self.clean_oracle) - 1
        for idx in indices:
            if idx >= len(self.current_data):
                continue
            orig_id = self.sample_ids[idx]
            if orig_id >= 0 and orig_id <= max_orig_id:
                try:
                    current_label = float(self.current_data.iloc[idx][lbl])
                    oracle_label = float(self.clean_oracle.iloc[orig_id][lbl])
                    if current_label == oracle_label:
                        self.sample_dirty[idx] = 0.0
                except (IndexError, KeyError):
                    pass

    def _oracle_fix_labels(self, indices, lbl):
        """使用 oracle 修复标签（最后回退选项）"""
        max_orig_id = len(self.clean_oracle) - 1
        for idx in indices:
            if idx >= len(self.current_data):
                continue
            orig_id = self.sample_ids[idx]
            if orig_id >= 0 and orig_id <= max_orig_id:
                try:
                    oracle_label = float(self.clean_oracle.iloc[orig_id][lbl])
                    self.current_data.at[idx, lbl] = oracle_label
                    self.sample_dirty[idx] = 0.0
                except (IndexError, KeyError):
                    pass

    def _act_delete_samples(self, indices):
        """Remove selected samples (conservative: ≤ max_delete_ratio per step)."""
        ratio = getattr(self.cfg, "max_delete_ratio", self.cfg.max_modify_ratio)
        max_k = max(1, int(len(self.current_data) * ratio))
        indices = sorted(set(list(indices)[:max_k]))

        if not indices:
            return

        mask = np.ones(len(self.current_data), dtype=bool)
        for idx in indices:
            if idx < len(mask):
                mask[idx] = False

        self.current_data = (
            self.current_data[mask].reset_index(drop=True)
        )
        self.sample_ids = [s for i, s in enumerate(self.sample_ids) if mask[i]]
        self.sample_dirty = self.sample_dirty[mask]

    def _act_add_samples(self, indices):
        """Append selected samples from SMOTE pool or augmentation pool to current dataset."""
        indices = list(indices)[: self.cfg.max_add_samples]

        if getattr(self, "_add_candidates_from_smote", False) and getattr(
            self, "_current_smote_pool", None
        ) is not None:
            pool = self._current_smote_pool
            indices = [i for i in indices if i < len(pool)]
            if not indices:
                self._add_candidates_from_smote = False
                self._current_smote_pool = None
                return
            new_rows = pool.iloc[indices].copy()
            self._add_candidates_from_smote = False
            self._current_smote_pool = None
        else:
            indices = [i for i in indices if i < len(self.aug_pool)]
            if not indices:
                return
            new_rows = self.aug_pool.iloc[indices].copy()
            pool_mask = np.ones(len(self.aug_pool), dtype=bool)
            for i in indices:
                pool_mask[i] = False
            self.aug_pool = self.aug_pool[pool_mask].reset_index(drop=True)

        self.current_data = pd.concat(
            [self.current_data, new_rows], ignore_index=True
        )
        self.sample_ids.extend([-1] * len(new_rows))
        self.sample_dirty = np.concatenate(
            [self.sample_dirty, np.zeros(len(new_rows))]
        )

    # ──────────────────────────────────────────────────────────────────────
    # Snapshot / rollback
    # ──────────────────────────────────────────────────────────────────────

    def _snapshot(self):
        return dict(
            data=self.current_data.copy(),
            sample_ids=self.sample_ids.copy(),
            sample_dirty=self.sample_dirty.copy(),
            aug_pool=self.aug_pool.copy(),
            acc=self.current_acc,
        )

    def _restore(self, snap):
        self.current_data = snap["data"].copy()
        self.sample_ids = snap["sample_ids"].copy()
        self.sample_dirty = snap["sample_dirty"].copy()
        self.aug_pool = snap["aug_pool"].copy()
        self.current_acc = snap["acc"]
        # Mark classifier as needing retrain after rollback (data changed)
        self._needs_clf_train = True

    def _simulate_user_feedback(self, action_idx, selected_indices):
        """
        模拟用户反馈：用 ground truth 判断智能体的决策是否正确

        Returns
        -------
        bool
            True = 用户接受（决策正确）
            False = 用户拒绝（决策错误）
        """
        if len(selected_indices) == 0:
            return True  # 空操作视为正确

        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col

        # 脏数据中有 oracle 对应的干净数据
        if self.clean_oracle is None:
            # 没有 ground truth，默认接受
            return True
        
        # 如果 dirty 和 oracle 是独立数据集，无法准确判断 ground truth
        # 此时应该信任检测器的结果，让 RL 自己探索
        if getattr(self, '_dirty_oracle_independent', False):
            return True
        
        # 对于 action 0 (modify_features)，只有在修改后才知道是否正确
        # 所以先接受动作，后面再验证修复效果
        if action_idx == 0:
            return True  # modify_features 总是接受，让检测器验证
        
        # 逐个检查选中的样本
        correct_count = 0
        total_count = 0

        for idx in selected_indices:
            if idx >= len(self.current_data):
                continue
            
            total_count += 1
            row = self.current_data.iloc[idx]
            clean_row = self.clean_oracle.iloc[idx]

            if action_idx == 1:
                # 修改标签：检查标签是否与 ground truth 一致
                if row[lbl] == clean_row[lbl]:
                    correct_count += 1

            elif action_idx == 2:
                # 删除样本：检查删除的是否是噪声样本
                if self.sample_dirty[idx] == 1:  # 脏样本应该删除
                    correct_count += 1
                # 注意：删除干净样本是错误的

            # action_idx == 3 (add) 不涉及用户反馈判断

        # 正确率超过阈值则接受
        accuracy_threshold = getattr(self.cfg, 'user_feedback_accuracy_threshold', 0.5)
        accuracy = correct_count / total_count if total_count > 0 else 1.0

        return accuracy >= accuracy_threshold

    def _check_feature_modification_correct(self, row, clean_row, feat):
        """
        检查特征修改是否正确

        如果修改后的值等于 ground truth，则为正确
        """
        for f in feat:
            dirty_val = row[f]
            clean_val = clean_row[f]
            if pd.isna(dirty_val) and pd.isna(clean_val):
                continue
            if dirty_val == clean_val:
                continue
            try:
                if abs(float(dirty_val) - float(clean_val)) <= 1e-6:
                    continue
            except (TypeError, ValueError):
                pass
            return False
        return True
