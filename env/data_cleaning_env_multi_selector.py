"""
使用 MultiSelectorAgent 的数据清洗环境。

特点：
- 4个独立 Selector，分别对应4个动作
- 每个 Selector 有自己的噪声检测头和预测头
- 先检测噪声，再选择样本，最后修改
"""

import numpy as np
import torch
import pandas as pd


class DataCleaningEnvMultiSelector:
    """
    使用 MultiSelectorAgent 的数据清洗环境。
    """
    
    def __init__(self, config):
        """初始化环境"""
        from env.data_cleaning_env import DataCleaningEnv
        
        self._env = DataCleaningEnv(config)
        self.cfg = config
        self.sel_agent = None
        
        # 当前状态和动作
        self._current_state = None
        self._current_action_idx = None
        
        # 缓存检测结果
        self._detection_results = {}
    
    def set_sel_agent(self, sel_agent):
        """设置 MultiSelector 智能体"""
        self.sel_agent = sel_agent
    
    def __getattr__(self, name):
        # 如果属性不存在，从原始环境获取
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._env, name)
    
    def _sync_attrs(self):
        """同步原始环境的属性"""
        self.__dict__.update(self._env.__dict__)
    
    def reset(self):
        """重置环境"""
        result = self._env.reset()
        self._sync_attrs()
        return result
    
    def get_candidates(self, action_idx):
        """
        获取候选样本（两阶段噪声检测版本）。

        第一阶段：使用传统方法（ED2/IDE）进行噪声预筛选
        第二阶段：RL Selector 从噪声候选中选择

        Returns:
            dict: 包含预筛选后的候选样本，添加了以下字段：
              - is_noisy: 噪声检测结果（第一阶段）
              - detector_scores: 传统检测器的分数
        """
        self._sync_attrs()

        # 获取底层环境的候选
        cand = self._env.get_candidates(action_idx)

        if cand is None:
            return None

        # 第一阶段：使用传统方法进行噪声预筛选
        cand = self._apply_first_stage_detection(action_idx, cand)

        return cand

    def _apply_first_stage_detection(self, action_idx, cand):
        """
        第一阶段：使用传统方法进行噪声预筛选。

        - action_idx=0 (modify_features): 使用 ED2 检测特征噪声
        - action_idx=1 (modify_labels): 使用 IDE 检测标签噪声
        - action_idx=2 (delete_samples): 使用损失值筛选低质量样本
        - action_idx=3 (add_samples): 不需要预筛选，返回所有候选
        """
        X = cand["X"]
        y = cand["y"]
        n = len(X)

        # 初始化检测结果
        is_noisy = np.zeros(n, dtype=bool)
        detector_scores = np.zeros(n, dtype=float)

        if action_idx == 0:  # modify_features - 使用 ED2 检测特征噪声
            if self._env._ed2_rpt_detector is not None:
                try:
                    # ED2 检测噪声特征（使用概率分数）
                    _, noise_mask, noise_probs = self._env._ed2_rpt_detector.detect_and_correct(
                        X, y_labels=None, return_noise_scores=True
                    )
                    # 如果任意特征被标记为噪声，则样本是噪声样本
                    is_noisy = noise_mask.any(axis=1) if noise_mask.ndim > 1 else noise_mask.astype(bool)
                    # 使用噪声概率作为分数（不是硬判决）
                    detector_scores = noise_probs.mean(axis=1) if noise_probs.ndim > 1 else noise_probs
                except Exception as e:
                    print(f"  [Warning] ED2 检测失败: {e}")
                    # 回退到损失值筛选
                    is_noisy, detector_scores = self._loss_based_detection(X, y)
            else:
                # 如果没有 ED2，使用损失值筛选
                is_noisy, detector_scores = self._loss_based_detection(X, y)

        elif action_idx == 1:  # modify_labels - 使用 IDE 检测标签噪声
            if hasattr(self._env, '_label_detector') and self._env._label_detector is not None:
                try:
                    # IDE 检测噪声标签（返回修复后的标签、噪声掩码、噪声分数）
                    predicted_labels, noise_mask, noise_scores = self._env._label_detector.fit_predict(X, y)
                    # 使用噪声概率作为分数
                    is_noisy = noise_scores > 0.5 if noise_scores is not None else noise_mask.astype(bool)
                    detector_scores = noise_scores if noise_scores is not None else is_noisy.astype(float)
                except Exception as e:
                    print(f"  [Warning] IDE 检测失败: {e}")
                    is_noisy, detector_scores = self._loss_based_detection(X, y)
            else:
                # 如果没有 IDE，使用损失值筛选
                is_noisy, detector_scores = self._loss_based_detection(X, y)

        elif action_idx == 2:  # delete_samples - 使用损失值筛选低质量样本
            is_noisy, detector_scores = self._loss_based_detection(X, y)
            # 对于 delete，分数高表示低质量
            is_noisy = detector_scores > np.percentile(detector_scores, 50)

        elif action_idx == 3:  # add_samples - 不需要预筛选
            # 返回所有候选
            is_noisy = np.ones(n, dtype=bool)  # 假设所有候选都可以添加
            detector_scores = np.ones(n, dtype=float)

        # 将第一阶段检测结果添加到候选字典
        cand["is_noisy"] = is_noisy
        cand["detector_scores"] = detector_scores
        cand["first_stage_detected"] = is_noisy.sum()  # 噪声样本数量

        return cand

    def _loss_based_detection(self, X, y):
        """
        基于损失值的噪声检测（备用方法）。

        Returns:
            is_noisy: 布尔数组，表示样本是否为噪声
            scores: 损失值分数
        """
        n = len(X)

        # 使用环境中的分类器计算损失
        if self._env.clf is not None and self._env.scaler is not None:
            try:
                # 标准化输入
                if self._env.imputer is not None:
                    X_imp = self._env.imputer.transform(X)
                else:
                    X_imp = np.nan_to_num(X, nan=0.0)

                X_sc = self._env.scaler.transform(X_imp)
                probs = self._env.clf.predict_proba(X_sc)

                # 计算交叉熵损失
                classes = np.array(self._env.clf.classes_)
                y_int = y.astype(int)
                label_to_col = {int(c): j for j, c in enumerate(classes)}
                col_idx = np.array([label_to_col.get(int(yi), 0) for yi in y_int], dtype=int)
                prob_true = probs[np.arange(n), col_idx]
                loss = -np.log(np.clip(prob_true, 1e-8, 1.0))

                # 高损失样本可能是噪声
                threshold = np.percentile(loss, 75)
                is_noisy = loss > threshold
                scores = loss

            except Exception:
                # 如果计算失败，返回默认值
                is_noisy = np.zeros(n, dtype=bool)
                scores = np.zeros(n, dtype=float)
        else:
            is_noisy = np.zeros(n, dtype=bool)
            scores = np.zeros(n, dtype=float)

        return is_noisy, scores
    
    def step(self, action_idx, selected_indices, user_feedback=None, selector_pred=None):
        """执行一步操作

        Parameters
        ----------
        action_idx : int
            动作索引
        selected_indices : list
            选中的样本索引
        user_feedback : str or None
            用户反馈：
            - None: 自动判断（模拟模式，用 ground truth）
            - "accept": 用户接受
            - "reject": 用户拒绝
        selector_pred : np.ndarray or None
            Selector 预测的值，用于实际修改
        """
        # 同步最新属性
        self._sync_attrs()

        # 获取当前 state
        self._current_state = self._env._state()
        self._current_state_idx = action_idx

        # 对所有候选运行对应动作的 Selector，获取噪声检测分数
        cand = self._env.get_candidates(action_idx)

        if cand is not None and self.sel_agent is not None:
            u = cand["u"]
            X = cand["X"]
            y = cand["y"]
            z = self.sel_agent.build_input(
                self._current_state.astype(np.float32),
                u.astype(np.float32),
                action_idx,
                X=X,
                y=y
            )

            with torch.no_grad():
                scores, pred, hidden = self.sel_agent.forward(z, action_idx)

            self._detection_results = {
                'scores': scores.cpu().numpy(),
                'pred': pred.cpu().numpy() if pred is not None else None,
                'hidden': hidden.cpu().numpy()
            }

        # 调用原始环境的 step（传入用户反馈和 Selector 预测值）
        result = self._env.step(
            action_idx, selected_indices, 
            user_feedback=user_feedback,
            selector_pred=selector_pred
        )

        return result
    
    def detect_noise(self, action_idx, indices=None):
        """
        噪声检测：对候选样本进行噪声检测。
        
        返回每个样本的噪声分数和预测值。
        
        Returns
        -------
        scores : np.ndarray
            噪声分数
        pred : np.ndarray 或 None
            预测值（特征或标签）
        """
        if self.sel_agent is None:
            return None, None
        
        # 确保属性同步
        self._sync_attrs()
        
        # 获取当前 state
        if self._current_state is None:
            self._current_state = self._env._state()
        
        # 获取候选
        cand = self._env.get_candidates(action_idx)
        if cand is None:
            return None, None

        u = cand["u"]
        X = cand["X"]
        y = cand["y"]

        # 可只对指定 indices 检测
        if indices is not None:
            u = u[indices]
            X = X[indices]
            y = y[indices]

        z = self.sel_agent.build_input(
            self._current_state.astype(np.float32),
            u.astype(np.float32),
            action_idx,
            X=X,
            y=y
        )
        
        with torch.no_grad():
            scores, pred, hidden = self.sel_agent.forward(z, action_idx)
        
        scores_np = scores.cpu().numpy()
        pred_np = pred.cpu().numpy() if pred is not None else None
        
        return scores_np, pred_np
    
    def select_with_detection(self, action_idx, n_select, strategy="top", use_prescreening=True, use_two_stage=True):
        """
        基于噪声检测结果选择样本（支持两阶段检测）。

        Parameters
        ----------
        action_idx : int
            动作索引
        n_select : int
            选择数量
        strategy : str
            选择策略：
            - "top": 选分数最高的（用于特征噪声、标签错误）
            - "bottom": 选分数最低的（用于删除无用样本）
            - "threshold": 选分数超过阈值的
            - "adaptive": 自适应阈值筛选
        use_prescreening : bool
            是否使用预筛选机制（先筛选高置信噪声，再从中选择）
        use_two_stage : bool
            是否使用两阶段检测（第一阶段ED2/IDE + 第二阶段Selector）

        Returns
        -------
        selected : list[int]
            选中的样本索引
        pred_selected : np.ndarray 或 None
            选中样本的预测值
        """
        scores, pred = self.detect_noise(action_idx)

        if scores is None:
            # 如果没有 Selector，返回随机选择
            n_cand = len(self._env.get_candidates(action_idx)["u"])
            selected = list(range(min(n_select, n_cand)))
            return selected, None

        n_cand = len(scores)
        n_select = min(n_select, n_cand)

        # ===== 两阶段检测：使用第一阶段的噪声预筛选结果 =====
        if use_two_stage:
            # 获取第一阶段检测结果（从 get_candidates 获取）
            cand = self._env.get_candidates(action_idx)
            if cand is not None and "is_noisy" in cand:
                first_stage_detected = cand["first_stage_detected"]
                detector_scores = cand["detector_scores"]

                if first_stage_detected > 0:
                    # 第一阶段检测到噪声样本
                    # 获取第一阶段认为是噪声的样本索引
                    noise_mask = cand["is_noisy"]
                    noise_indices = np.where(noise_mask)[0]

                    if len(noise_indices) >= n_select:
                        # 噪声候选足够多，从噪声候选中选择
                        noise_scores = scores[noise_indices]
                        top_local_indices = np.argsort(noise_scores)[-n_select:]
                        selected = noise_indices[top_local_indices].tolist()
                    else:
                        # 噪声候选不够，优先选噪声样本，不够的用 Selector 分数补充
                        selected = noise_indices.tolist()
                        remaining = n_select - len(selected)
                        if remaining > 0:
                            # 从非噪声样本中选择分数最高的
                            clean_indices = np.where(~noise_mask)[0]
                            if len(clean_indices) > 0:
                                clean_scores = scores[clean_indices]
                                top_clean = np.argsort(clean_scores)[-remaining:]
                                selected.extend(clean_indices[top_clean].tolist())

                    # 获取选中样本的预测值
                    pred_selected = pred[selected] if pred is not None else None
                    return selected, pred_selected

        # ===== 原始的单阶段 Selector 选择逻辑 =====
        # 预筛选机制：根据动作类型确定噪声分数方向
        if use_prescreening:
            # 计算自适应阈值（使用分数分布的分位数）
            threshold_percentile = 0.5  # 默认阈值

            if action_idx == 0 or action_idx == 1:
                # modify_features / modify_labels: 分数越高越可能是噪声
                threshold = np.percentile(scores, 75)  # 75分位数
                noise_candidates = np.where(scores >= threshold)[0]

                if len(noise_candidates) >= n_select:
                    # 只从高置信噪声中选择
                    noise_scores = scores[noise_candidates]
                    top_indices = np.argsort(noise_scores)[-n_select:]
                    selected = noise_candidates[top_indices].tolist()
                else:
                    # 如果高置信噪声不够，从所有候选中选
                    _, selected = torch.topk(torch.FloatTensor(scores), n_select)
                    selected = selected.tolist()

            elif action_idx == 2:
                # delete_samples: 分数越高越可能是低质量样本
                threshold = np.percentile(scores, 75)
                low_quality_candidates = np.where(scores >= threshold)[0]

                if len(low_quality_candidates) >= n_select:
                    lq_scores = scores[low_quality_candidates]
                    top_indices = np.argsort(lq_scores)[-n_select:]
                    selected = low_quality_candidates[top_indices].tolist()
                else:
                    _, selected = torch.topk(torch.FloatTensor(scores), n_select)
                    selected = selected.tolist()

            elif action_idx == 3:
                # add_samples: 分数越高越可能是高质量样本
                threshold = np.percentile(scores, 25)  # 25分位数，选分数高的
                high_quality_candidates = np.where(scores >= threshold)[0]

                if len(high_quality_candidates) >= n_select:
                    hq_scores = scores[high_quality_candidates]
                    top_indices = np.argsort(hq_scores)[-n_select:]
                    selected = high_quality_candidates[top_indices].tolist()
                else:
                    _, selected = torch.topk(torch.FloatTensor(scores), n_select)
                    selected = selected.tolist()

            else:
                # 默认使用 top 策略
                _, selected = torch.topk(torch.FloatTensor(scores), n_select)
                selected = selected.tolist()

        else:
            # 不使用预筛选，使用原有的策略
            if strategy == "top":
                _, selected = torch.topk(torch.FloatTensor(scores), n_select)
                selected = selected.tolist()
            elif strategy == "bottom":
                _, selected = torch.topk(torch.FloatTensor(-scores), n_select)
                selected = selected.tolist()
            elif strategy == "threshold":
                threshold = 0.5
                above_threshold = np.where(scores > threshold)[0]
                selected = above_threshold[:n_select].tolist()
            elif strategy == "adaptive":
                threshold = np.percentile(scores, 75)
                above_threshold = np.where(scores > threshold)[0]
                selected = above_threshold[:n_select].tolist()
            else:
                selected = list(range(n_select))

        # 返回选中样本的预测值
        pred_selected = pred[selected] if pred is not None else None

        return selected, pred_selected

    # ──────────────────────────────────────────────────────────────────────
    # 便捷方法：一步完成检测+选择+修改
    # ──────────────────────────────────────────────────────────────────────
    
    def detect_select_act(self, action_idx, n_select=None):
        """
        一步完成：噪声检测 → 选择样本 → 执行动作
        
        Parameters
        ----------
        action_idx : int
            动作索引
        n_select : int, optional
            选择数量，默认使用配置中的比例
        
        Returns
        -------
        state : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        self._sync_attrs()
        self._current_state = self._env._state()
        self._current_action_idx = action_idx
        
        # 确定选择数量
        if n_select is None:
            ratio = getattr(self.cfg, "max_modify_ratio", 0.1)
            n_cand = len(self._env.current_data)
            n_select = max(1, int(n_cand * ratio))

        # 选择策略 - 使用两阶段检测
        if action_idx == 2:  # delete_samples - 选分数低的
            strategy = "bottom"
        else:  # 其他动作 - 选分数高的
            strategy = "top"

        # 检测 + 选择（启用两阶段检测）
        selected, pred = self.select_with_detection(
            action_idx, n_select, strategy,
            use_prescreening=True,
            use_two_stage=True  # 启用两阶段检测
        )
        
        # 执行动作（应用预测值）
        self._apply_prediction(action_idx, selected, pred)
        
        # 计算 reward
        state, reward, done, info = self._env._compute_reward(action_idx, selected)
        
        return state, reward, done, info
    
    def _apply_prediction(self, action_idx, selected, pred):
        """
        根据预测值修改样本。
        
        Parameters
        ----------
        action_idx : int
            动作索引
        selected : list[int]
            选中的样本索引
        pred : np.ndarray 或 None
            预测值
        """
        if len(selected) == 0 or pred is None:
            return
        
        # 转换为 numpy（如果是 tensor）
        if hasattr(pred, 'cpu'):
            pred = pred.cpu().numpy()
        
        feat = self.cfg.feature_cols
        lbl = self.cfg.label_col
        
        if action_idx == 0:  # modify_features - 用 feat_pred 替换特征
            for i, idx in enumerate(selected):
                if idx >= len(self.current_data):
                    continue
                for j, col in enumerate(feat):
                    self.current_data.at[idx, col] = float(pred[i, j])
        
        elif action_idx == 1:  # modify_labels - 用 label_pred 翻转/保持标签
            for i, idx in enumerate(selected):
                if idx >= len(self.current_data):
                    continue
                # 阈值 0.5 决定是否翻转
                new_label = 1 if float(pred[i]) > 0.5 else 0
                self.current_data.at[idx, lbl] = new_label
        
        # action_idx == 2 (delete) 和 3 (add) 由原始环境处理
        
        # 同步属性
        self._sync_attrs()
