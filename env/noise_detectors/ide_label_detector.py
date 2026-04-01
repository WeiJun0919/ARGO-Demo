"""
IDE: A System for Iterative Mislabel Detection

实现论文: IDE: A System for Iterative Mislabel Detection (SIGMOD 2024)

核心思想:
1. Early Loss Observation - 训练早期高损失的样本可能是噪声标签
2. Influence-based Verification - 用影响函数验证噪声样本
3. 迭代检测直到收敛
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class IDELabelDetector:
    """
    IDE 标签噪声检测器
    
    用于在不使用 ground truth 的情况下检测和修复噪声标签。
    """
    
    def __init__(self, clf_type='mlp', hidden_size=(64, 32), 
                 n_iterations=3, early_stop_patience=2,
                 confidence_threshold=0.7, noise_ratio_estimate=0.3,
                 random_state=42):
        """
        Parameters:
        -----------
        clf_type : str
            分类器类型, 'mlp' 或 'logistic'
        hidden_size : tuple
            MLP 隐藏层大小
        n_iterations : int
            最大迭代次数
        early_stop_patience : int
            早停耐心值
        confidence_threshold : float
            噪声修复的置信度阈值
        noise_ratio_estimate : float
            估计的噪声比例上限
        random_state : int
            随机种子
        """
        self.clf_type = clf_type
        self.hidden_size = hidden_size
        self.n_iterations = n_iterations
        self.patience = early_stop_patience
        self.confidence_threshold = confidence_threshold
        self.noise_ratio_estimate = noise_ratio_estimate
        self.random_state = random_state
        
    def fit_predict(self, X, y):
        """
        检测噪声标签并返回修复后的标签
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        y : array-like, shape (n_samples,)
            当前标签（可能含噪声）
        
        Returns:
        --------
        y_corrected : ndarray
            修复后的标签
        noise_mask : ndarray (bool)
            噪声样本的布尔掩码
        noise_scores : ndarray
            每个样本的噪声概率 [0, 1]
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        n = len(X)
        
        # 数据预处理
        self.imputer_ = SimpleImputer(strategy='mean')
        X_imp = self.imputer_.fit_transform(X)
        
        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X_imp)
        
        # 迭代检测
        y_current = y.copy()
        best_noise_mask = None
        best_noise_scores = None
        best_flip_count = 0
        
        for iteration in range(self.n_iterations):
            # Step 1: 训练分类器
            clf = self._build_classifier()
            clf.fit(X_sc, y_current)
            
            # Step 2: 计算每个样本的损失
            losses = self._compute_losses(clf, X_sc, y_current)
            
            # Step 3: 计算噪声分数
            noise_scores = self._compute_noise_scores(losses, y_current, clf)
            
            # Step 4: 确定噪声阈值
            threshold = self._find_threshold(noise_scores)
            
            # Step 5: 生成噪声掩码
            current_noise_mask = noise_scores > threshold
            
            # Step 6: 标签翻转
            flip_indices = np.where(current_noise_mask)[0]
            flipped_y = y_current.copy()
            flipped_y[flip_indices] = 1 - flipped_y[flip_indices]
            
            # Step 7: 评估翻转后的标签质量
            clf_eval = self._build_classifier()
            clf_eval.fit(X_sc, flipped_y)
            flipped_losses = self._compute_losses(clf_eval, X_sc, y_current)
            
            # 如果翻转后整体损失下降，说明翻转是对的
            if np.mean(flipped_losses) < np.mean(losses):
                y_current = flipped_y
                best_noise_mask = current_noise_mask.copy()
                best_noise_scores = noise_scores.copy()
                best_flip_count = len(flip_indices)
            else:
                # 早停检查
                if best_noise_mask is not None and len(flip_indices) == 0:
                    break
                    
        # 最终：用最佳结果计算最终噪声分数
        if best_noise_scores is None:
            best_noise_scores = noise_scores
            best_noise_mask = current_noise_mask
            
        # 计算最终的噪声分数（基于翻转后的标签）
        final_clf = self._build_classifier()
        final_clf.fit(X_sc, y_current)
        
        # 返回修复后的标签
        y_corrected = y_current.copy()
        
        return y_corrected, best_noise_mask, best_noise_scores
    
    def _build_classifier(self):
        """构建分类器"""
        if self.clf_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=self.hidden_size,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_state,
            )
        else:
            return LogisticRegression(
                max_iter=500,
                random_state=self.random_state,
            )
    
    def _compute_losses(self, clf, X, y):
        """计算每个样本的交叉熵损失"""
        probs = clf.predict_proba(X)
        classes = clf.classes_
        
        # 找到每个标签对应的概率列
        label_to_col = {int(c): j for j, c in enumerate(classes)}
        col_idx = np.array([label_to_col.get(int(yi), 0) for yi in y])
        prob_true = probs[np.arange(len(y)), col_idx]
        
        # 交叉熵损失
        losses = -np.log(np.clip(prob_true, 1e-10, 1.0))
        return losses
    
    def _compute_noise_scores(self, losses, y, clf):
        """
        计算噪声分数
        
        核心思想：
        - 噪声样本的损失通常比同类干净样本高
        - 使用类别级别的 z-score 归一化
        """
        n = len(losses)
        noise_scores = np.zeros(n)
        
        for label in np.unique(y):
            mask = (y == label)
            class_losses = losses[mask]
            
            if len(class_losses) < 2:
                noise_scores[mask] = 0.0
                continue
                
            # 计算该类别的统计量
            mean_loss = np.mean(class_losses)
            std_loss = np.std(class_losses) + 1e-8
            
            # z-score
            z_scores = (class_losses - mean_loss) / std_loss
            
            # 用 sigmoid 转换为概率
            noise_scores[mask] = 1 / (1 + np.exp(-z_scores))
            
        return noise_scores
    
    def _find_threshold(self, noise_scores):
        """
        找到最佳阈值
        策略：假设最多 noise_ratio_estimate 的样本是噪声
        """
        k = self.noise_ratio_estimate
        threshold = np.percentile(noise_scores, (1 - k) * 100)
        return threshold
    
    def predict(self, X, y):
        """
        预测噪声标签（不修改输入）
        
        Parameters:
        -----------
        X : array-like
            特征矩阵
        y : array-like
            当前标签
        
        Returns:
        --------
        noise_mask : ndarray (bool)
            噪声样本的布尔掩码
        noise_scores : ndarray
            噪声分数
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        # 预处理
        X_imp = self.imputer_.transform(X)
        X_sc = self.scaler_.transform(X_imp)
        
        # 训练分类器
        clf = self._build_classifier()
        clf.fit(X_sc, y)
        
        # 计算损失和噪声分数
        losses = self._compute_losses(clf, X_sc, y)
        noise_scores = self._compute_noise_scores(losses, y, clf)
        
        # 阈值化
        threshold = self._find_threshold(noise_scores)
        noise_mask = noise_scores > threshold
        
        return noise_mask, noise_scores


# ═══════════════════════════════════════════════════════════════════════════════
# GPU 加速版本
# ═══════════════════════════════════════════════════════════════════════════════

class TorchMLPClassifier:
    """PyTorch GPU 加速的 MLP 分类器"""
    
    def __init__(self, hidden_layers=(256, 128), max_epochs=50, batch_size=512, lr=0.001, random_state=42):
        self.hidden_layers = hidden_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.imputer = None
        
    def fit(self, X, y):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 预处理
        self.imputer = SimpleImputer(strategy='mean')
        X_imp = self.imputer.fit_transform(X)
        
        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X_imp)
        
        # 转换为 PyTorch tensor
        X_tensor = torch.FloatTensor(X_sc).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # 构建 MLP
        layers = []
        in_features = X_sc.shape[1]
        for hidden_size in self.hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, len(np.unique(y))))
        
        self.model = nn.Sequential(*layers).to(device)
        
        # 训练
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.max_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict_proba(self, X):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_imp = self.imputer.transform(X)
        X_sc = self.scaler.transform(X_imp)
        X_tensor = torch.FloatTensor(X_sc).to(device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    @property
    def classes_(self):
        return self._classes
    
    @classes_.setter
    def classes_(self, value):
        self._classes = value


class TorchLabelNoiseDetector:
    """
    PyTorch GPU 加速的标签噪声检测器
    
    使用 GPU 训练 MLP，大幅加速高维数据处理
    """
    
    def __init__(self, noise_ratio=0.3, hidden_layers=(256, 128), 
                 max_epochs=50, batch_size=512, random_state=42):
        self.noise_ratio = noise_ratio
        self.hidden_layers = hidden_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
    def fit_predict(self, X, y):
        import numpy as np
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        # 训练分类器（使用 GPU）
        print("    [TorchDetector] 使用 PyTorch GPU 训练 MLP...")
        clf = TorchMLPClassifier(
            hidden_layers=self.hidden_layers,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            random_state=self.random_state
        )
        clf.fit(X, y)
        clf.classes_ = np.unique(y)
        
        # 计算损失
        print("    [TorchDetector] 计算噪声分数...")
        probs = clf.predict_proba(X)
        classes = clf.classes_
        label_to_col = {int(c): j for j, c in enumerate(classes)}
        col_idx = np.array([label_to_col.get(int(yi), 0) for yi in y])
        prob_true = probs[np.arange(len(y)), col_idx]
        losses = -np.log(np.clip(prob_true, 1e-10, 1.0))
        
        # 噪声分数
        noise_scores = np.zeros(len(losses))
        for label in np.unique(y):
            mask = (y == label)
            class_losses = losses[mask]
            mean_loss = np.mean(class_losses)
            std_loss = np.std(class_losses) + 1e-8
            z_scores = (class_losses - mean_loss) / std_loss
            noise_scores[mask] = 1 / (1 + np.exp(-z_scores))
        
        # 阈值
        threshold = np.percentile(noise_scores, (1 - self.noise_ratio) * 100)
        noise_mask = noise_scores > threshold
        
        # 翻转噪声标签
        y_corrected = y.copy()
        y_corrected[noise_mask] = 1 - y_corrected[noise_mask]
        
        print(f"    [TorchDetector] 检测完成，噪声样本: {noise_mask.sum()}/{len(y)}")
        return y_corrected, noise_mask, noise_scores


class SimpleLabelNoiseDetector:
    """
    简化版标签噪声检测器（基于损失排序）
    
    适用于快速检测，不需要迭代
    """
    
    def __init__(self, noise_ratio=0.3, random_state=42):
        self.noise_ratio = noise_ratio
        self.random_state = random_state
        
    def fit_predict(self, X, y):
        """检测噪声并返回修复后的标签"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        # 预处理
        imputer = SimpleImputer(strategy='mean')
        X_imp = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_imp)
        
        # 训练 MLP - 使用更快的配置处理高维图像数据
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),  # 更大的网络处理高维数据
            max_iter=100,  # 减少迭代
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,  # 早点停止
            batch_size=256,  # 批量训练
            random_state=self.random_state,
            solver='adam',
        )
        clf.fit(X_sc, y)
        
        # 计算损失
        probs = clf.predict_proba(X_sc)
        classes = clf.classes_
        label_to_col = {int(c): j for j, c in enumerate(classes)}
        col_idx = np.array([label_to_col.get(int(yi), 0) for yi in y])
        prob_true = probs[np.arange(len(y)), col_idx]
        losses = -np.log(np.clip(prob_true, 1e-10, 1.0))
        
        # 噪声分数
        noise_scores = np.zeros(len(losses))
        for label in np.unique(y):
            mask = (y == label)
            class_losses = losses[mask]
            mean_loss = np.mean(class_losses)
            std_loss = np.std(class_losses) + 1e-8
            z_scores = (class_losses - mean_loss) / std_loss
            noise_scores[mask] = 1 / (1 + np.exp(-z_scores))
        
        # 阈值
        threshold = np.percentile(noise_scores, (1 - self.noise_ratio) * 100)
        noise_mask = noise_scores > threshold
        
        # 翻转噪声标签
        y_corrected = y.copy()
        y_corrected[noise_mask] = 1 - y_corrected[noise_mask]
        
        return y_corrected, noise_mask, noise_scores
