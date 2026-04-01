"""
Multi-Selector Agent: 4个独立的 Selector，分别用于不同的动作。

每个 Selector 针对特定任务优化：
- Selector 0 (modify_features): 检测特征噪声 + 预测干净特征
- Selector 1 (modify_labels): 检测标签噪声 + 预测正确标签  
- Selector 2 (delete_samples): 检测无用样本
- Selector 3 (add_samples): 从候选池选优质样本 + 预测增强特征
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks_multi_selector import (
    SELECTOR_NETS, 
    ACTION_NAMES,
    SelectorNetFeatures,
    SelectorNetLabels,
    SelectorNetDelete,
    SelectorNetAdd,
)


class MultiSelectorAgent:
    """
    管理4个独立 Selector 的智能体。
    """
    
    def __init__(self, config):
        self.cfg = config
        self.n_actions = 4
        
        # 获取设备
        self._device = torch.device(config.device)
        
        # 为每个动作创建独立的 Selector
        self.nets = {}
        self.optimizers = {}
        
        for action_idx in range(self.n_actions):
            net_cls = SELECTOR_NETS[action_idx]
            
            # 不同动作可以使用不同的输入维度和隐藏层配置
            input_dim = self._get_input_dim(action_idx)
            hidden = self._get_hidden(action_idx)
            
            net = net_cls(
                input_dim=input_dim,
                n_features=config.n_features,
                hidden=hidden,
            )
            # 将网络移到 GPU
            net = net.to(self.device)
            self.nets[action_idx] = net
            
            # 独立的优化器
            self.optimizers[action_idx] = optim.Adam(
                net.parameters(), 
                lr=config.lr_selector
            )
        
        print(f"[MultiSelector] 初始化了 {self.n_actions} 个 Selector，设备: {self.device}:")
        for idx in range(self.n_actions):
            n_params = sum(p.numel() for p in self.nets[idx].parameters())
            print(f"  - Action {idx} ({ACTION_NAMES[idx]}): {n_params:,} 参数")
    
    def _get_input_dim(self, action_idx):
        """获取每个动作的输入维度"""
        # 根据 use_oracle_in_u 调整 sample_feature_dim
        sample_dim = self.cfg.sample_feature_dim
        if not getattr(self.cfg, 'use_oracle_in_u', False):
            sample_dim = sample_dim - 1  # 移除 oracle_dirty 后减少1维
        
        base_dim = self.cfg.state_dim + sample_dim
        base_dim = base_dim + self.n_actions  # +n_actions for one-hot

        # 根据动作类型添加额外的特征维度
        action_extra_dims = {
            0: 3,  # modify_features: anomaly_per_sample, extreme_count, nan_ratio
            1: 5,  # modify_labels: entropy, loss, margin, class_balance_0, class_balance_1
            2: 3,  # delete_samples: loss, -margin, combined_quality
            3: 2,  # add_samples: margin, uncertainty
        }

        return base_dim + action_extra_dims.get(action_idx, 0)
    
    def _get_hidden(self, action_idx):
        """获取每个动作的隐藏层配置"""
        # 可以为不同动作设置不同的隐藏层大小
        hidden_cfg = getattr(self.cfg, 'selector_hidden', [256, 128])
        
        if action_idx == 0:  # modify_features - 需要预测特征，隐藏层可以大一些
            return hidden_cfg
        elif action_idx == 1:  # modify_labels - 标签预测
            return hidden_cfg
        elif action_idx == 2:  # delete_samples - 只需打分
            return hidden_cfg
        elif action_idx == 3:  # add_samples - 需要预测特征
            return hidden_cfg
        return hidden_cfg
    
    # ──────────────────────────────────────────────────────────────────────
    # Build selector input tensor
    # ──────────────────────────────────────────────────────────────────────
    
    def build_input(
        self, state: np.ndarray, u: np.ndarray, action_idx: int,
        X: np.ndarray = None, y: np.ndarray = None
    ) -> torch.Tensor:
        """
        构建 Selector 输入 z_i = [S_t || u_i || a_onehot || action_specific_features]

        针对不同动作添加特定的特征：
        - action_idx=0 (modify_features): 特征级别的噪声信息
        - action_idx=1 (modify_labels): 标签级别的噪声信息
        - action_idx=2 (delete_samples): 样本质量信息
        - action_idx=3 (add_samples): 合成样本质量信息
        """
        n = len(u)
        state_rep = np.tile(state, (n, 1)).astype(np.float32)

        a_onehot = np.zeros((n, self.n_actions), dtype=np.float32)
        a_onehot[:, action_idx] = 1.0

        # 基础输入
        z = np.concatenate([state_rep, u.astype(np.float32), a_onehot], axis=1)

        # 添加动作特定的额外特征
        if X is not None and y is not None:
            action_features = self._build_action_specific_features(
                action_idx, X, y, u
            )
            if action_features is not None:
                z = np.concatenate([z, action_features], axis=1)

        # 创建张量并移到正确设备
        z_tensor = torch.from_numpy(z)
        return z_tensor.to(self.device)

    def _build_action_specific_features(
        self, action_idx: int, X: np.ndarray, y: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """
        根据动作类型构建特定的特征。

        Parameters
        ----------
        action_idx : int
            动作索引
        X : np.ndarray
            样本特征 (n, n_features)
        y : np.ndarray
            样本标签 (n,)
        u : np.ndarray
            基础的 per-sample 特征

        Returns
        -------
        np.ndarray or None
            动作特定的额外特征
        """
        n = len(X)

        if action_idx == 0:  # modify_features - 特征级别噪声信息
            # 计算每个特征列的统计异常
            feat_mean = np.nanmean(X, axis=0)
            feat_std = np.nanstd(X, axis=0) + 1e-8

            # 每个样本与均值的偏离程度
            X_centered = (X - feat_mean) / feat_std
            X_centered = np.nan_to_num(X_centered, nan=0.0)

            # 每个样本的异常程度：所有特征偏离的均值
            anomaly_per_sample = np.abs(X_centered).mean(axis=1)

            # 超过2个标准差的特征数量
            extreme_count = (np.abs(X_centered) > 2.0).sum(axis=1)

            # 缺失值比例
            nan_ratio = np.isnan(X).sum(axis=1) / X.shape[1]

            return np.column_stack([
                anomaly_per_sample,
                extreme_count,
                nan_ratio,
            ]).astype(np.float32)

        elif action_idx == 1:  # modify_labels - 标签级别噪声信息
            # 基于预测概率的标签不确定性
            if u.shape[1] >= 1:
                entropy = u[:, 0]  # 假设 u 的第一列是 entropy
            else:
                entropy = np.zeros(n)

            # 高损失样本可能标签错误
            if u.shape[1] >= 2:
                loss = u[:, 1]
            else:
                loss = np.zeros(n)

            # 预测概率接近0.5的边界样本
            if u.shape[1] >= 3:
                margin = u[:, 2]
            else:
                margin = np.zeros(n)

            # 类别平衡信息
            class_counts = np.bincount(y.astype(int), minlength=2)
            class_balance = class_counts / (class_counts.sum() + 1e-8)

            return np.column_stack([
                entropy,
                loss,
                margin,
                np.full(n, class_balance[0]),
                np.full(n, class_balance[1]),
            ]).astype(np.float32)

        elif action_idx == 2:  # delete_samples - 样本质量信息
            # 高损失 + 低置信度 = 低质量样本
            if u.shape[1] >= 2:
                loss = u[:, 1]
            else:
                loss = np.zeros(n)

            if u.shape[1] >= 3:
                margin = u[:, 3] if u.shape[1] > 3 else u[:, 2]
            else:
                margin = np.zeros(n)

            # 样本与多数类的距离
            class_mode = np.argmax(np.bincount(y.astype(int)))

            return np.column_stack([
                loss,
                -margin,  # margin 越小越可能是边界样本
                loss * (1 - margin),  # 组合质量分数
            ]).astype(np.float32)

        elif action_idx == 3:  # add_samples - 合成样本质量信息
            # SMOTE/增强样本的可靠性评估
            # 基于与原始数据的距离
            if u.shape[1] >= 3:
                margin = u[:, 2]
            else:
                margin = np.zeros(n)

            # 原始增强池的样本应该有较低的风险
            # 这里可以添加更多的分布一致性特征
            return np.column_stack([
                margin,  # 预测置信度
                1 - margin,  # 不确定性
            ]).astype(np.float32)

        return None
    
    def build_input_for_action(
        self, state: np.ndarray, u: np.ndarray, action_idx: int
    ) -> torch.Tensor:
        """build_input 的别名"""
        return self.build_input(state, u, action_idx)
    
    # ──────────────────────────────────────────────────────────────────────
    # Select samples (使用对应的 Selector)
    # ──────────────────────────────────────────────────────────────────────
    
    def select(
        self, z: torch.Tensor, action_idx: int, n_select: int
    ):
        """
        使用对应动作的 Selector 选择样本。
        
        Parameters
        ----------
        z : torch.Tensor
            输入张量
        action_idx : int
            动作索引 (0-3)
        n_select : int
            选择数量
        
        Returns
        -------
        selected_indices : list[int]
        scores : torch.Tensor
            所有候选的分数
        pred : torch.Tensor 或 None
            预测值（feat_pred 或 label_pred）
        hidden : torch.Tensor
            隐藏层表示
        """
        n = z.shape[0]
        n_select = min(max(1, n_select), n)
        net = self.nets[action_idx]
        
        device = next(net.parameters()).device
        z = z.to(device)
         
        
        with torch.no_grad():
            scores, pred, hidden = net(z)
        
        # 根据动作选择策略：
        # - modify_features/delete/add: 选分数高的
        # - modify_labels: 分数高表示标签可能错误
        _, top_idx = torch.topk(scores, n_select)
        
        return top_idx.tolist(), scores, pred, hidden
    
    def forward(self, z: torch.Tensor, action_idx: int):
        """
        前向传播，返回指定动作的 Selector 输出。
        """
        net = self.nets[action_idx]
        device = next(net.parameters()).device
        z = z.to(device)
        return net(z)
    
    # ──────────────────────────────────────────────────────────────────────
    # Loss computation (针对特定动作)
    # ──────────────────────────────────────────────────────────────────────
    
    def compute_loss(
        self,
        z: torch.Tensor,
        action_idx: int,
        selected_indices: list,
        reward: float,
        oracle_dirty: np.ndarray = None,
        clean_feats: np.ndarray = None,
        clean_labels: np.ndarray = None,
        train_mode: str = "joint",
    ):
        """
        计算指定动作的 Selector 损失。
        """
        net = self.nets[action_idx]
        scores, pred, hidden = net(z)
        
        loss = torch.tensor(0.0, device=z.device)
        details = {"action": ACTION_NAMES[action_idx]}
        
        # 1. RL loss (REINFORCE) - 基于奖励的直接策略梯度
        # 核心思想：如果动作带来了正奖励，说明选中的样本是正确的，应该增加选中它们的概率
        #         如果动作带来了负奖励，说明选中的样本是错误的，应该降低选中它们的概率
        if train_mode in ("rl", "joint") and len(selected_indices) > 0:
            sel_scores = scores[selected_indices]
            
            # 改进的策略梯度：
            # - 使用 log 概率来增加/减少选择高/低分样本的概率
            # - 奖励越高，越应该选择高分样本；奖励越低，越应该避免选择
            # 
            # 对于 modify/delete：高分 = 更可能是噪声 = 应该被操作
            # 对于 add：高分 = 更可能是高质量样本 = 应该被添加
            
            # 归一化奖励到 [-1, 1] 范围，使其更稳定
            normalized_reward = torch.clamp(
                torch.tensor(reward, device=z.device), 
                min=-1.0, max=1.0
            )
            
            # 关键改进：使用正确的梯度方向
            # - 正奖励：增加选中高分样本的概率 -> 最大化 log(sel_scores)
            # - 负奖励：减少选中样本的概率 -> 最小化 log(sel_scores)
            # 
            # 原代码的问题：当 reward < 0 时使用 log(1-sel_scores)，这会让智能体选择低分样本
            # 但实际上应该根据 action 类型来决定：
            # - modify/delete: 即使 reward < 0，仍然应该选择噪声样本（高分），只是选择得不够准确
            # - add: 即使 reward < 0，仍然应该选择高质量样本
            
            # 解决方案：使用 sign(reward) * log(sel_scores)
            # - 正奖励：最大化选中样本的分数
            # - 负奖励：最小化选中样本的分数（避免选择错误的样本）
            rl_loss = -normalized_reward * torch.log(sel_scores.clamp(1e-7, 1 - 1e-7)).mean()
            loss = loss + self.cfg.lambda_rl * rl_loss
            details["rl_loss"] = rl_loss.item()
            details["reward"] = reward
            details["normalized_reward"] = normalized_reward.item()
        
        # 2. Auxiliary losses (根据动作类型)
        if train_mode in ("aux", "rl", "joint"):
            
            # 检查是否允许使用Oracle进行辅助损失训练
            use_oracle_in_aux = getattr(self.cfg, 'use_oracle_in_aux_loss', False)
            
            # 课程学习：自适应调整 lambda_aux
            # 随着训练进行，逐渐增加 aux loss 的重要性
            # 这是通过外部传入的课程权重来实现的，默认值为 1.0
            curriculum_weight = getattr(self, '_curriculum_weight', 1.0)
            
            # Action 0: modify_features - BCE + MSE(feat_pred)
            if action_idx == 0 and oracle_dirty is not None and use_oracle_in_aux:
                oracle_t = torch.FloatTensor(oracle_dirty).to(z.device)
                # 确保 scores 在 [0,1] 范围内（处理 NaN/Inf）
                scores_safe = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
                scores_clamped = torch.clamp(scores_safe, min=0.0, max=1.0)
                aux_bce = nn.BCELoss()(scores_clamped, oracle_t)
                loss = loss + self.cfg.lambda_aux * curriculum_weight * aux_bce
                details["aux_bce"] = aux_bce.item()
                details["curriculum_weight"] = curriculum_weight
                
                # 只对 selected 的样本计算 feat_mse
                if clean_feats is not None and pred is not None and len(selected_indices) > 0:
                    cf_selected = torch.FloatTensor(clean_feats[selected_indices]).to(z.device)
                    pred_selected = pred[selected_indices]
                    feat_mean = np.nanmean(clean_feats[selected_indices], axis=0)
                    feat_std = np.nanstd(clean_feats[selected_indices], axis=0)
                    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
                    feat_mean = torch.FloatTensor(feat_mean).to(z.device)
                    feat_std = torch.FloatTensor(feat_std).to(z.device)
                    cf_norm = (cf_selected - feat_mean) / feat_std
                    fp_norm = (pred_selected - feat_mean) / feat_std
                    feat_mse = nn.MSELoss()(fp_norm, cf_norm.detach())
                    loss = loss + self.cfg.lambda_aux * feat_mse
                    details["feat_mse"] = feat_mse.item()
            
            # Action 1: modify_labels - BCE + BCE(label_pred)
            elif action_idx == 1 and oracle_dirty is not None and use_oracle_in_aux:
                oracle_t = torch.FloatTensor(oracle_dirty).to(z.device)
                scores_safe = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
                scores_clamped = torch.clamp(scores_safe, min=0.0, max=1.0)
                aux_bce = nn.BCELoss()(scores_clamped, oracle_t)
                loss = loss + self.cfg.lambda_aux * aux_bce
                details["aux_bce"] = aux_bce.item()
                
                # 只对 selected 的样本计算 label_loss
                # clean_labels 已经是按 selected 筛选后的结果，长度等于 len(selected)
                # selected_indices 用于从 z/scores/pred 中选取，所以 clean_labels 直接对应
                if clean_labels is not None and pred is not None and len(selected_indices) > 0:
                    # clean_labels 长度 = len(selected)，pred 也是按 selected 筛选后的
                    # 不需要再用 selected_indices 索引，直接使用
                    clean_lbl_selected = torch.FloatTensor(clean_labels).to(z.device)
                    pred_selected = pred[selected_indices]
                    # 确保维度一致
                    if clean_lbl_selected.shape == pred_selected.shape:
                        label_loss = nn.BCELoss()(pred_selected, clean_lbl_selected)
                        loss = loss + self.cfg.lambda_aux * label_loss
                        details["label_loss"] = label_loss.item()
            
            # Action 2: delete_samples - BCE
            elif action_idx == 2 and oracle_dirty is not None and use_oracle_in_aux:
                oracle_t = torch.FloatTensor(oracle_dirty).to(z.device)
                scores_safe = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
                scores_clamped = torch.clamp(scores_safe, min=0.0, max=1.0)
                aux_bce = nn.BCELoss()(scores_clamped, oracle_t)
                loss = loss + self.cfg.lambda_aux * aux_bce
                details["aux_bce"] = aux_bce.item()
            
            # Action 3: add_samples - BCE + MSE(feat_pred)
            elif action_idx == 3 and oracle_dirty is not None and use_oracle_in_aux:
                # 对于 add_samples，oracle_dirty=0 表示优质样本应该被添加
                oracle_t = torch.FloatTensor(oracle_dirty).to(z.device)
                scores_safe = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
                scores_clamped = torch.clamp(scores_safe, min=0.0, max=1.0)
                aux_bce = nn.BCELoss()(scores_clamped, oracle_t)
                loss = loss + self.cfg.lambda_aux * aux_bce
                details["aux_bce"] = aux_bce.item()
                
                # 只对 selected 的样本计算 feat_mse
                if clean_feats is not None and pred is not None and len(selected_indices) > 0:
                    cf_selected = torch.FloatTensor(clean_feats[selected_indices]).to(z.device)
                    pred_selected = pred[selected_indices]
                    feat_mean = np.nanmean(clean_feats[selected_indices], axis=0)
                    feat_std = np.nanstd(clean_feats[selected_indices], axis=0)
                    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
                    feat_mean = torch.FloatTensor(feat_mean).to(z.device)
                    feat_std = torch.FloatTensor(feat_std).to(z.device)
                    cf_norm = (cf_selected - feat_mean) / feat_std
                    fp_norm = (pred_selected - feat_mean) / feat_std
                    feat_mse = nn.MSELoss()(fp_norm, cf_norm.detach())
                    loss = loss + self.cfg.lambda_aux * feat_mse
                    details["feat_mse"] = feat_mse.item()
        
        # 3. Diversity loss
        if self.cfg.lambda_div > 0 and len(selected_indices) > 1:
            sel_h = hidden[selected_indices]
            # Cosine similarity
            sel_h_norm = sel_h / (sel_h.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = torch.mm(sel_h_norm, sel_h_norm.t())
            # 只计算上三角（避免重复）
            n = len(selected_indices)
            mask = torch.triu(torch.ones(n, n), diagonal=1).to(z.device)
            div_loss = (sim_matrix * mask).sum() / (n * (n - 1) / 2)
            loss = loss + self.cfg.lambda_div * div_loss
            details["div_loss"] = div_loss.item()
        
        # 4. Contrastive loss: 选中的样本应该与未选中的样本在特征空间中分离
        n_all = z.shape[0]
        if self.cfg.lambda_contrastive > 0 and len(selected_indices) > 0 and len(selected_indices) < n_all - 1:
            # 安全检查：确保 batch 大小足够
            if n_all < 10:
                pass  # skip contrastive loss for small batches
            else:
                selected_set = set(selected_indices)
                unselected_indices = [i for i in range(n_all) if i not in selected_set]
                
                if len(unselected_indices) > 0 and len(selected_indices) > 0:
                    # 获取选中和未选中的隐藏表示
                    sel_h = hidden[selected_indices]
                    unsel_h = hidden[unselected_indices]
                    
                    # 归一化
                    sel_h_norm = sel_h / (sel_h.norm(dim=1, keepdim=True) + 1e-8)
                    unsel_h_norm = unsel_h / (unsel_h.norm(dim=1, keepdim=True) + 1e-8)
                    
                    # 计算选中与未选中之间的相似度矩阵
                    sim_matrix = torch.mm(sel_h_norm, unsel_h_norm.t())
                    
                    # 使用温度系数
                    temperature = getattr(self.cfg, 'contrastive_temperature', 0.5)
                    sim_matrix = sim_matrix / temperature
                    
                    exp_sim = torch.exp(sim_matrix)
                    denominator = exp_sim.sum(dim=1) + 1e-8
                    numerator = torch.exp(torch.zeros_like(sim_matrix))
                    
                    contrastive_loss = -(numerator / denominator).log().mean()
                    
                    loss = loss + self.cfg.lambda_contrastive * contrastive_loss
                    details["contrastive_loss"] = contrastive_loss.item()
        
        details["total_loss"] = loss.item()
        return loss, details
    
    # ──────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────
    
    def update(
        self,
        z: torch.Tensor,
        action_idx: int,
        selected_indices: list,
        reward: float,
        oracle_dirty: np.ndarray = None,
        clean_feats: np.ndarray = None,
        clean_labels: np.ndarray = None,
        train_mode: str = "joint",
    ):
        """
        更新指定动作的 Selector。
        """
        optimizer = self.optimizers[action_idx]
        
        # 确保输入张量在正确的设备上
        z = z.to(self._device)
        
        optimizer.zero_grad()
        loss, details = self.compute_loss(
            z=z,
            action_idx=action_idx,
            selected_indices=selected_indices,
            reward=reward,
            oracle_dirty=oracle_dirty,
            clean_feats=clean_feats,
            clean_labels=clean_labels,
            train_mode=train_mode,
        )
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.cfg, 'max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.nets[action_idx].parameters(), 
                self.cfg.max_grad_norm
            )
        
        optimizer.step()
        
        return details
    
    # ──────────────────────────────────────────────────────────────────────
    # Save/Load
    # ──────────────────────────────────────────────────────────────────────
    
    def save(self, path: str):
        """保存所有 Selector 的状态"""
        checkpoint = {
            'nets': {idx: net.state_dict() for idx, net in self.nets.items()},
            'optimizers': {idx: opt.state_dict() for idx, opt in self.optimizers.items()},
        }
        torch.save(checkpoint, path)
        print(f"[MultiSelector] 保存到 {path}")
    
    def load(self, path: str):
        """加载所有 Selector 的状态"""
        checkpoint = torch.load(path)
        for idx, state_dict in checkpoint['nets'].items():
            self.nets[idx].load_state_dict(state_dict)
        for idx, state_dict in checkpoint['optimizers'].items():
            self.optimizers[idx].load_state_dict(state_dict)
        print(f"[MultiSelector] 从 {path} 加载")
    
    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────
    
    def get_net(self, action_idx: int):
        """获取指定动作的 Selector 网络"""
        return self.nets[action_idx]
    
    def train_mode(self, mode: str = "train"):
        """设置所有 Selector 的训练模式"""
        for net in self.nets.values():
            if mode == "train":
                net.train()
            else:
                net.eval()
    
    def eval_mode(self):
        """设置评估模式"""
        for net in self.nets.values():
            net.eval()
    
    @property
    def device(self):
        """获取设备"""
        return self._device
