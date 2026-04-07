"""
PPO-HRL 统一训练脚本 - 支持表格、文本、图像数据集

通过 --dataset 参数自动选择对应的数据处理策略：
- adult / smartfactory: 表格数据
- imdb: 文本数据
- cifar10_resnet50: 图像数据

使用方式：
    # 表格数据
    python train_multi_selector_v2.py --dataset adult

    # 文本数据
    python train_multi_selector_v2.py --dataset imdb

    # 图像数据
    python train_multi_selector_v2.py --dataset cifar10_resnet50
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from config import Config
from env.data_cleaning_env_multi_selector import DataCleaningEnvMultiSelector
from agents.ppo_agent import PPOAgent
from agents.multi_selector_agent import MultiSelectorAgent
from utils.common import get_data_type, detection_threshold_to_fraction, FourDecimalEncoder


def create_env_for_data_type(config, data_type):
    """
    根据数据类型创建环境

    参数：
        config: 配置对象
        data_type: 'tabular', 'text', 'image'

    返回：
        环境实例
    """
    if data_type == 'image':
        # 图像数据环境
        from env.image_cleaning_env import ImageDataCleaningEnv
        return ImageDataCleaningEnv(config)
    else:
        # 表格/文本数据环境
        return DataCleaningEnvMultiSelector(config)


def _detection_threshold_to_fraction(threshold, default=0.6):
    """将 Config 中的检测阈值转为 [0, 1] 比例。"""
    return detection_threshold_to_fraction(threshold, default)


def create_selector_for_data_type(config, data_type):
    """
    根据数据类型创建 Selector Agent

    参数：
        config: 配置对象
        data_type: 'tabular', 'text', 'image'

    返回：
        Selector Agent 实例
    """
    if data_type == 'image':
        from agents.image_selector_agent import ImageSelectorAgent
        return ImageSelectorAgent(
            config=config,
            feature_dim=config.image_feature_dim,
            num_classes=config.num_classes,
        )
    elif data_type == 'text':
        # 文本数据使用 3 个 selector（modify_labels, delete, add），与图像一致
        from agents.image_selector_agent import ImageSelectorAgent
        return ImageSelectorAgent(
            config=config,
            feature_dim=config.n_features,  # TF-IDF 特征维度
            num_classes=2,  # IMDB 二分类
            n_actions=3,    # 3 个动作
        )
    else:
        return MultiSelectorAgent(config)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: ED2/IDE 预检测 + 用检测结果做 Warmup
# ══════════════════════════════════════════════════════════════════════════════

def run_detection_phase(env, cfg, confidence_threshold=0.2, clean_threshold=None, min_gap=0.05):
    """
    Step 1: 用 ED2/IDE 对脏数据做噪声检测，得到伪干净/伪噪声标签。

    对于图像/文本数据，使用分类器预测一致性检测标签噪声。
    对于表格数据，使用 ED2/IDE 检测。
    """
    data_type = get_data_type(cfg.dataset_name)

    if data_type in ('image', 'text'):
        env.train_classifier_for_detection(epochs=20, batch_size=128)
        return run_detection_phase_image(env, cfg, confidence_threshold)
    else:
        return run_detection_phase_tabular(env, cfg, confidence_threshold, clean_threshold, min_gap)


def run_detection_phase_image(env, cfg, confidence_threshold=0.2):
    """
    图像/文本数据：使用分类器预测一致性检测标签噪声
    """
    base_env = env._env if hasattr(env, '_env') else env

    features = base_env.features
    labels = base_env.noisy_labels
    true_labels = base_env.true_labels

    # 调试：检查标签是否正确
    print(f"  [DEBUG] labels 前5: {labels[:5]}")
    print(f"  [DEBUG] true_labels 前5: {true_labels[:5]}")
    print(f"  [DEBUG] 一致数: {(labels == true_labels).sum()}/{len(labels)}")

    # 获取分类器预测
    device = torch.device(cfg.device)

    with torch.no_grad():
        if hasattr(base_env, 'classifier') and base_env.classifier is not None:
            # 图像数据
            feats_tensor = torch.FloatTensor(features).to(device)
            probs = base_env.classifier.predict_proba(feats_tensor)
            pred_labels = probs.argmax(axis=1)
            confidence = probs.max(axis=1)
        elif hasattr(base_env, '_predict_with_text_clf'):
            # 文本数据
            probs = base_env._predict_with_text_clf(features, return_proba=True)
            pred_labels = probs.argmax(axis=1)
            confidence = probs.max(axis=1)
        else:
            pred_labels = labels.copy()
            confidence = np.ones(len(labels)) * 0.5

    # 标签噪声分数
    label_agreement = (pred_labels == labels).astype(float)
    noise_scores = 1.0 - label_agreement

    # 低置信度 + 标签不一致 = 高噪声分数
    noise_scores = noise_scores * (1.0 - confidence)

    # 高置信度噪声（与 detection_noise_threshold：0~1 或 0~100 百分位一致）
    pct = _detection_threshold_to_fraction(confidence_threshold, default=0.20) * 100.0
    threshold = np.percentile(noise_scores, pct)
    high_conf_noise = noise_scores > threshold

    print(f"\n  [Detection] 总样本: {len(labels)}")
    print(f"  [Detection] 检测到噪声样本: {high_conf_noise.sum()} ({high_conf_noise.mean()*100:.1f}%)")
    print(f"  [Detection] 实际噪声样本: {(labels != true_labels).sum()} ({(labels != true_labels).mean()*100:.1f}%)")

    return ~high_conf_noise, high_conf_noise, noise_scores


def run_detection_phase_tabular(env, cfg, confidence_threshold=0.2, clean_threshold=None, min_gap=0.05):
    """
    表格数据：使用 ED2/IDE 检测
    """
    base_env = env._env
    feat = cfg.feature_cols
    lbl = cfg.label_col

    X_dirty = base_env.encode_features_to_float(base_env.current_data)
    y_dirty_raw = base_env.current_data[lbl].values

    # 处理标签
    if y_dirty_raw.dtype.kind in ['U', 'S', 'O'] or (hasattr(y_dirty_raw, 'dtype') and y_dirty_raw.dtype == object):
        label_map = {'<=50K': 0, '>50K': 1}
        if str(y_dirty_raw[0]) in label_map:
            y_dirty = np.array([label_map.get(str(v), v) for v in y_dirty_raw], dtype=np.float32)
        else:
            y_dirty = np.array([float(v) for v in y_dirty_raw], dtype=np.float32)
    else:
        y_dirty = y_dirty_raw.astype(np.float32)

    # 处理缺失值
    has_nan = np.isnan(X_dirty).any()
    if has_nan:
        imputer = SimpleImputer(strategy="mean")
        X_dirty = imputer.fit_transform(X_dirty)

    n_samples = len(X_dirty)
    detection_scores = np.zeros(n_samples, dtype=float)

    # ED2 检测特征噪声 (action_idx=0)
    if hasattr(base_env, '_ed2_rpt_detector') and base_env._ed2_rpt_detector is not None:
        try:
            _, feat_noise_mask, feat_probs = base_env._ed2_rpt_detector.detect_and_correct(
                X_dirty, y_labels=None, return_noise_scores=True
            )
            feat_noise_ratio = feat_noise_mask.mean(axis=1) if feat_noise_mask.ndim > 1 else feat_noise_mask
            feat_scores = feat_probs.mean(axis=1) if feat_probs.ndim > 1 else feat_probs
        except Exception as e:
            print(f"  [Warning] ED2 检测失败: {e}")
            feat_scores = np.zeros(n_samples)
    else:
        feat_scores = np.zeros(n_samples)

    # IDE 检测标签噪声 (action_idx=1)
    if hasattr(base_env, '_label_detector') and base_env._label_detector is not None:
        try:
            _, label_noise_mask, label_scores = base_env._label_detector.fit_predict(X_dirty, y_dirty)
            label_scores = label_scores if label_scores is not None else label_noise_mask.astype(float)
        except Exception as e:
            print(f"  [Warning] IDE 检测失败: {e}")
            label_scores = np.zeros(n_samples)
    else:
        label_scores = np.zeros(n_samples)

    # 综合分数
    combined_scores = np.maximum(feat_scores, label_scores)
    detection_scores = combined_scores

    # 使用自适应阈值（与 Config 中 detection_* 一致：0~1 或 1~100 百分位）
    sorted_scores = np.sort(combined_scores)
    # 噪声比例应该接近实际噪声比例（约 20%），提高精确率
    noise_frac = _detection_threshold_to_fraction(confidence_threshold, default=0.15)
    # 干净阈值设为更低，只标记确定是噪声的样本
    clean_t = clean_threshold if clean_threshold is not None else 0.1
    clean_frac = _detection_threshold_to_fraction(clean_t, default=0.1)
    noise_quantile_idx = int(n_samples * noise_frac)
    clean_quantile_idx = int(n_samples * clean_frac)

    noise_threshold = sorted_scores[noise_quantile_idx] if noise_quantile_idx < n_samples else 0.8
    clean_threshold_val = sorted_scores[clean_quantile_idx] if clean_quantile_idx < n_samples else 0.3

    if noise_threshold - clean_threshold_val < min_gap:
        mid = (noise_threshold + clean_threshold_val) / 2
        noise_threshold = mid + min_gap / 2
        clean_threshold_val = mid - min_gap / 2

    high_conf_noise = combined_scores > noise_threshold
    high_conf_clean = combined_scores <= clean_threshold_val

    print(f"\n  [Detection] 总样本: {n_samples}")
    print(f"  [Detection] 分数分布: min={combined_scores.min():.3f}, max={combined_scores.max():.3f}, "
          f"mean={combined_scores.mean():.3f}, median={np.median(combined_scores):.3f}")
    print(f"  [Detection] 阈值设置: noise>{noise_threshold:.3f}, clean<={clean_threshold_val:.3f}")
    print(f"  [Detection] 高置信度噪声: {high_conf_noise.sum()} ({high_conf_noise.mean()*100:.1f}%)")
    print(f"  [Detection] 高置信度干净: {high_conf_clean.sum()} ({high_conf_clean.mean()*100:.1f}%)")

    return high_conf_clean, high_conf_noise, detection_scores


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 续: Supervised Warmup
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1_supervised_with_detection(
    env, sel_agent, cfg,
    clean_mask, noisy_mask,
    n_epochs=20, batch_size=64
):
    """
    Step 1 (续): 用检测结果得到的伪标签做 supervised warmup。
    """
    data_type = get_data_type(cfg.dataset_name)

    if data_type in ('image', 'text'):
        run_phase1_supervised_image(env, sel_agent, cfg, clean_mask, noisy_mask, n_epochs, batch_size)
    else:
        run_phase1_supervised_tabular(env, sel_agent, cfg, clean_mask, noisy_mask, n_epochs, batch_size)


def run_phase1_supervised_image(env, sel_agent, cfg, clean_mask, noisy_mask, n_epochs, batch_size):
    """图像/文本数据的 Phase1 Supervised"""
    base_env = env._env if hasattr(env, '_env') else env

    # 图像数据使用 env.features，文本数据也使用
    features = base_env.features
    labels = base_env.noisy_labels
    true_labels = base_env.true_labels

    n_clean = clean_mask.sum()
    n_noisy = noisy_mask.sum()

    print(f"\n  [Phase1-Supervised] 干净样本: {n_clean}, 噪声样本: {n_noisy}")

    # 使用分类器预测正确标签
    if hasattr(base_env, 'classifier') and base_env.classifier is not None:
        # 图像数据使用已有的分类器
        device = torch.device(cfg.device)
        with torch.no_grad():
            feats_tensor = torch.FloatTensor(features).to(device)
            pred_labels = base_env.classifier.predict(feats_tensor)
    elif hasattr(base_env, '_predict_with_text_clf'):
        # 文本数据使用文本分类器
        pred_labels = base_env._predict_with_text_clf(features)
    else:
        pred_labels = labels.copy()

    # Oracle
    oracle_dirty = noisy_mask.astype(np.float32)
    clean_labels = pred_labels

    rng = np.random.default_rng(cfg.seed)

    for epoch in range(n_epochs):
        for action_idx in [1, 2, 3]:  # 三个动作
            clean_indices = np.where(clean_mask)[0]
            noisy_indices = np.where(noisy_mask)[0]

            half = batch_size // 2
            idx_clean = rng.choice(len(clean_indices), size=min(half, len(clean_indices)), replace=False)
            idx_noisy = rng.choice(len(noisy_indices), size=min(half, len(noisy_indices)), replace=False)

            if len(idx_clean) == 0 or len(idx_noisy) == 0:
                continue

            batch_idx = np.concatenate([clean_indices[idx_clean], noisy_indices[idx_noisy]])
            X_batch = features[batch_idx]
            oracle_batch = oracle_dirty[batch_idx]

            if action_idx == 3:  # add: 选干净样本
                selected_in_batch = np.where(oracle_batch < 0.5)[0].tolist()
            else:  # modify/delete: 选噪声样本
                selected_in_batch = np.where(oracle_batch >= 0.5)[0].tolist()

            if len(selected_in_batch) == 0:
                continue

            z = sel_agent.build_input(X_batch, action_idx)
            clean_labels_batch = clean_labels[batch_idx]

            if action_idx == 1:  # modify_labels
                sel_agent.update(
                    z, action_idx, selected_in_batch, 0.0,
                    oracle_dirty=oracle_batch,
                    clean_labels=clean_labels_batch,
                    train_mode="aux",
                )
            else:
                sel_agent.update(
                    z, action_idx, selected_in_batch, 0.0,
                    oracle_dirty=oracle_batch,
                    train_mode="aux",
                )

        if (epoch + 1) % 5 == 0:
            print(f"  [Phase1-Supervised] epoch {epoch+1}/{n_epochs}")


def run_phase1_supervised_tabular(env, sel_agent, cfg, clean_mask, noisy_mask, n_epochs, batch_size):
    """表格数据的 Phase1 Supervised"""
    base_env = env._env
    feat = cfg.feature_cols
    lbl = cfg.label_col

    X_all = base_env.encode_features_to_float(base_env.current_data)
    y_all_raw = base_env.current_data[lbl].values

    # 处理标签
    if y_all_raw.dtype.kind in ['U', 'S', 'O'] or (hasattr(y_all_raw, 'dtype') and y_all_raw.dtype == object):
        label_map = {'<=50K': 0, '>50K': 1}
        if str(y_all_raw[0]) in label_map:
            y_all = np.array([label_map.get(str(v), v) for v in y_all_raw], dtype=np.float32)
        else:
            y_all = np.array([float(v) for v in y_all_raw], dtype=np.float32)
    else:
        y_all = y_all_raw.astype(np.float32)

    # 处理缺失值
    has_nan = np.isnan(X_all).any()
    if has_nan:
        imputer = SimpleImputer(strategy="mean")
        X_all = imputer.fit_transform(X_all)

    X_clean_detected = X_all[clean_mask]
    y_clean_detected = y_all[clean_mask]
    X_noisy_detected = X_all[noisy_mask]
    y_noisy_detected = y_all[noisy_mask]

    n_clean = len(X_clean_detected)
    n_noisy = len(X_noisy_detected)

    print(f"\n  [Phase1-Supervised] 干净样本: {n_clean}, 噪声样本: {n_noisy}")

    # 目标1: 预测干净特征
    clean_mean = X_clean_detected.mean(axis=0) if n_clean > 0 else np.zeros(X_all.shape[1])
    clean_feats_noisy = np.tile(clean_mean, (n_noisy, 1))
    clean_feats_all = np.vstack([X_clean_detected, X_noisy_detected])

    # 目标2: 预测干净标签
    unique_labels = np.unique(y_clean_detected) if n_clean > 0 else np.array([])
    has_valid_labels = n_clean >= 10 and len(unique_labels) >= 2

    if has_valid_labels:
        clf_label = LogisticRegression(max_iter=500, random_state=cfg.seed)
        clf_label.fit(X_clean_detected, y_clean_detected)
        if len(unique_labels) == 2:
            pred_labels_noisy = clf_label.predict_proba(X_noisy_detected)[:, 1]
        else:
            pred_labels_noisy = y_noisy_detected.copy()
        print(f"  [Phase1] 使用干净样本训练分类器 ({n_clean}个, {len(unique_labels)}类)")
    else:
        pred_labels_noisy = y_noisy_detected.copy()

    pred_labels_all = np.concatenate([y_clean_detected, pred_labels_noisy])

    X_combined = np.vstack([X_clean_detected, X_noisy_detected])
    y_combined = np.concatenate([y_clean_detected, y_noisy_detected])

    oracle_dirty = np.concatenate([
        np.zeros(n_clean),
        np.ones(n_noisy),
    ]).astype(np.float32)

    state = base_env._state()
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(n_epochs):
        for action_idx in range(4):  # 4个 selector
            half = batch_size // 2
            idx_clean = rng.choice(n_clean, size=min(half, n_clean), replace=n_clean < half)
            idx_noisy = rng.choice(n_noisy, size=min(half, n_noisy), replace=n_noisy < half)

            if len(idx_clean) == 0 or len(idx_noisy) == 0:
                continue

            X_batch = np.vstack([X_clean_detected[idx_clean], X_noisy_detected[idx_noisy]])
            y_batch = np.concatenate([y_clean_detected[idx_clean], y_noisy_detected[idx_noisy]])
            oracle_batch = np.concatenate([oracle_dirty[:n_clean][idx_clean], oracle_dirty[n_clean:][idx_noisy]])

            if action_idx == 3:  # add: 选干净样本
                selected_in_batch = np.where(oracle_batch < 0.5)[0].tolist()
            else:  # modify/delete: 选噪声样本
                selected_in_batch = np.where(oracle_batch >= 0.5)[0].tolist()

            if len(selected_in_batch) == 0:
                continue

            u_batch = base_env._per_sample_features(X_batch, y_batch, oracle_batch)
            z = sel_agent.build_input(state, u_batch, action_idx, X=X_batch, y=y_batch)

            clean_feats_batch = np.vstack([clean_feats_all[idx_clean], clean_feats_noisy[idx_noisy]])
            pred_labels_batch = np.concatenate([y_clean_detected[idx_clean], pred_labels_noisy[idx_noisy]])

            if action_idx == 0:  # modify_features
                sel_agent.update(
                    z, action_idx, selected_in_batch, 0.0,
                    oracle_dirty=oracle_batch,
                    clean_feats=clean_feats_batch,
                    train_mode="aux",
                )
            elif action_idx == 1:  # modify_labels
                sel_agent.update(
                    z, action_idx, selected_in_batch, 0.0,
                    oracle_dirty=oracle_batch,
                    clean_labels=pred_labels_batch,
                    train_mode="aux",
                )
            elif action_idx == 2:  # delete
                sel_agent.update(
                    z, action_idx, selected_in_batch, 0.0,
                    oracle_dirty=oracle_batch,
                    train_mode="aux",
                )
            else:  # add
                sel_agent.update(
                    z, action_idx, selected_in_batch, 0.0,
                    oracle_dirty=oracle_batch,
                    clean_feats=clean_feats_batch,
                    train_mode="aux",
                )

        if (epoch + 1) % 5 == 0:
            print(f"  [Phase1-Supervised] epoch {epoch+1}/{n_epochs}")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def heuristic_action(action_counts, n_actions, step, total_steps):
    """Phase 2 固定启发式动作"""
    progress = step / max(total_steps, 1)
    if progress < 0.3:
        w = np.array([0.65, 0.02, 0.05, 0.25, 0.03])
    elif progress < 0.7:
        w = np.array([0.50, 0.05, 0.10, 0.28, 0.07])
    else:
        w = np.array([0.40, 0.07, 0.13, 0.30, 0.10])
    return int(np.random.choice(n_actions, p=w))


def _eval_data_on_ref(cfg, env, data_df):
    """在 data_df 上训练分类器，在独立参考集上评估准确率"""
    inner = getattr(env, "_env", env)
    feat = cfg.feature_cols
    lbl = cfg.label_col
    X = inner.encode_features_to_float(data_df)
    y = data_df[lbl].values.astype(float)
    if len(np.unique(y)) < 2 or len(X) < 20:
        return 0.0
    has_nan = np.isnan(X).any()
    if has_nan:
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
        X_ref = imputer.transform(env._env.X_ref)
    else:
        X_ref = env._env.X_ref.copy()

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=cfg.seed, n_jobs=-1)
    clf.fit(X, y)
    return clf.score(X_ref, env._env.y_ref)


# ══════════════════════════════════════════════════════════════════════════════
# Run Episode
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(env, ppo_agent, sel_agent, cfg, phase, ep, data_type, interactive_mode=False, auto_accept_ref=None):
    """
    运行一个 episode

    参数：
        data_type: 'tabular', 'text', 'image'
    """
    if auto_accept_ref is None:
        auto_accept_ref = {"auto_accept": False}

    state = env.reset()
    done = False
    ep_reward = 0.0
    ep_log = []
    env.action_counts = np.zeros(cfg.n_actions)
    prev_accuracy = None

    global_step = ep * cfg.max_steps_per_episode

    # ── 图像/文本数据：Episode 开始时进行完整评估 ───────────────────
    # 初始准确率已经在 reset() 后由 Phase1-End 设置，这里只在 joint 阶段需要
    # 注意：我们不在每个 episode 开始时训练完整模型，因为太慢
    # 模型复用由 _evaluate_accuracy 的缓存机制处理
    if data_type in ('image', 'text') and hasattr(env, '_evaluate_accuracy'):
        # 确保缓存有初始值
        if not hasattr(env, '_acc_is_fresh') or not env._acc_is_fresh:
            # 使用快速评估初始化缓存
            env._cached_acc = env.current_acc if hasattr(env, 'current_acc') else 0.0

    # 根据数据类型确定有效的动作数量
    # 图像和文本：使用 4 个动作 (1=modify_labels, 2=delete, 3=add, 4=no_op)
    # 表格数据：使用 5 个动作 (0=modify_features, 1=modify_labels, 2=delete, 3=add, 4=no_op)
    if data_type in ('image', 'text'):
        effective_actions = 4  # 1,2,3,4
    else:
        effective_actions = cfg.n_actions

    while not done:
        # 1) 选择高层动作
        raw_action_idx = 0  # 默认值，joint 阶段会被重新赋值

        if phase == "warmup":
            if data_type in ('image', 'text'):
                w = np.array([0.7, 0.05, 0.2, 0.05])  # labels, delete, add, no_op
                action_idx = int(np.random.choice([1, 2, 3, 4], p=w))
                ac_idx = action_idx - 1  # 映射到 0-3
                raw_action_idx = action_idx - 1  # 存储用 0-3 索引
            else:
                w = np.array([0.65, 0.0, 0.0, 0.30, 0.05])
                action_idx = int(np.random.choice(cfg.n_actions, p=w))
                ac_idx = action_idx
                raw_action_idx = action_idx
            log_prob, value = 0.0, 0.0

        elif phase == "freeze_hl":
            if data_type in ('image', 'text'):
                action_idx = heuristic_action_image(env.action_counts, 4, env.step_count, cfg.n_freeze_hl * cfg.max_steps_per_episode)
                ac_idx = action_idx - 1  # 映射到 0-3
                raw_action_idx = action_idx - 1  # 存储用 0-3 索引
            else:
                action_idx = heuristic_action(
                    env.action_counts, cfg.n_actions,
                    env.step_count, cfg.n_freeze_hl * cfg.max_steps_per_episode,
                )
                ac_idx = action_idx
                raw_action_idx = action_idx
            log_prob, value = 0.0, 0.0

        elif phase == "joint":
            raw_action_idx, log_prob, value = ppo_agent.select_action(state)

            # 图像/文本数据：PPO 输出 0-3，需要映射到 1-4
            # 映射：0->1, 1->2, 2->3, 3->4
            if data_type in ('image', 'text'):
                action_idx = raw_action_idx + 1  # 映射到 1-4
                ac_idx = raw_action_idx  # action_counts 使用 0-3 索引
            else:
                action_idx = raw_action_idx
                ac_idx = raw_action_idx

            epsilon = max(0.05, 1.0 - ep / cfg.n_joint * 0.5)

            if np.random.random() < epsilon:
                if data_type in ('image', 'text'):
                    valid_actions = [1, 2, 3]  # 无 modify_features (text/图像用 1,2,3)
                    action_idx = int(np.random.choice(valid_actions))
                    ac_idx = action_idx - 1  # 映射回 0-2
                    raw_action_idx = ac_idx  # 更新存储用索引
                else:
                    valid_actions = [0, 1, 2, 3]  # 包含 modify_features (表格用 0,1,2,3)
                    action_idx = int(np.random.choice(valid_actions))
                    ac_idx = action_idx
                log_prob, value = 0.0, 0.0
        else:
            break

        env.action_counts[ac_idx] += 1

        # 图像/文本：仅 1~3 有 Selector 子网；0=modify_features、4=no_op 直接走环境，不调用 sel_agent
        if data_type in ('image', 'text') and action_idx not in (1, 2, 3):
            next_state, reward, done, info = env.step(action_idx, [])
            ep_reward += reward
            if phase == "joint":
                ppo_agent.store(state, raw_action_idx, log_prob, reward, value, done)
            step_log = {
                "step": env.step_count,
                "action": cfg.action_names[action_idx] if action_idx < len(cfg.action_names) else f"action_{action_idx}",
                "action_id": int(action_idx),
                "reward": round(float(reward), 4),
                "n_selected": 0,
                "selected_noise_count": 0,
                "selected_noise_ratio": 0.0,
                "accuracy": round(float(info.get("accuracy", 0)), 4),
                "best_accuracy": round(float(info.get("best_accuracy", 0)), 4),
                "n_samples": info.get("n_samples", 0),
                "class_ratio": info.get("class_ratio", ""),
                "label_noise_ratio": round(float(info.get("label_noise_ratio", 0)), 4),
                "data_type": data_type,
            }
            ep_log.append(step_log)
            state = next_state
            continue

        # 2) 获取候选样本
        if data_type in ('image', 'text'):
            cand = env.get_candidates(action_idx)
        else:
            cand = env.get_candidates(action_idx)

        if cand is None or len(cand.get("u", [])) == 0:
            next_state, reward, done, info = env.step(action_idx, [])
            state = next_state
            ep_reward += reward
            continue

        u = cand["u"]
        X = cand["X"] if "X" in cand else cand.get("features")
        y = cand["y"] if "y" in cand else cand.get("labels")
        n_cand = len(u)

        is_noisy = cand.get("is_noisy", np.zeros(n_cand, dtype=bool))

        if action_idx == 3:
            n_sel = min(cfg.max_add_samples, n_cand)
        else:
            n_sel = max(1, int(n_cand * cfg.max_modify_ratio))

        # 3) 构建输入并选择
        if data_type in ('image', 'text'):
            z = sel_agent.build_input(X, action_idx)
        else:
            z = sel_agent.build_input(state, u, action_idx, X=X, y=y)

        selected, scores, pred, hidden = sel_agent.select(z, action_idx, n_sel)

        # 获取预测值 (action_idx 1=modify_labels)
        selector_pred = None
        if action_idx == 1 and pred is not None:
            pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
            selector_pred = pred_np[selected]

        # 4) 执行动作
        current_step = env.step_count
        print(f"    [Ep{ep} step{current_step}] action={action_idx}, n_sel={len(selected)}, calling env.step...", flush=True)
        next_state, reward, done, info = env.step(
            action_idx, selected,
            user_feedback=None,
            selector_pred=selector_pred
        )
        print(f"    [Ep{ep} step{current_step}] done, acc={info.get('accuracy', 0):.4f}, reward={reward:.4f}", flush=True)

        # 5) 训练 Selector
        if phase == "warmup":
            sel_mode = "aux"
        else:
            sel_mode = "joint"

        if phase in ["freeze_hl", "joint"]:
            rl_reward = reward
        else:
            rl_reward = 0.0

        if phase == "joint":
            progress = ep / cfg.n_joint
            curriculum_weight = 0.3 + progress * 0.7
            sel_agent._curriculum_weight = curriculum_weight
        else:
            sel_agent._curriculum_weight = 0.5

        sel_agent.update(z, action_idx, selected, rl_reward, train_mode=sel_mode)

        # 6) 存储 transition
        if phase == "joint":
            ppo_agent.store(state, raw_action_idx, log_prob, reward, value, done)

        # 7) 记录日志
        selected_noise = is_noisy[selected].sum() if len(selected) > 0 else 0

        scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
        pred_np = pred.cpu().numpy() if pred is not None and hasattr(pred, 'cpu') else pred

        step_log = {
            "step": env.step_count,
            "action": cfg.action_names[action_idx] if action_idx < len(cfg.action_names) else f"action_{action_idx}",
            "action_id": int(action_idx),
            "reward": round(float(reward), 4),
            "n_selected": len(selected),
            "selected_noise_count": int(selected_noise),
            "selected_noise_ratio": round(int(selected_noise) / len(selected), 4) if len(selected) > 0 else 0.0,
            "accuracy": round(float(info.get("accuracy", 0)), 4),
            "best_accuracy": round(float(info.get("best_accuracy", 0)), 4),
            "n_samples": info.get("n_samples", 0),
            "class_ratio": info.get("class_ratio", ""),
            "label_noise_ratio": round(float(info.get("label_noise_ratio", 0)), 4),
            "data_type": data_type,
        }
        ep_log.append(step_log)

        state = next_state
        ep_reward += reward

    # PPO update
    ppo_metrics = {}
    if phase == "joint":
        if ep_reward < -20:
            ppo_metrics = {"policy_loss": 0.0, "skipped": True}
        else:
            ppo_metrics = ppo_agent.update()

    # ── 图像/文本数据：episode 结束时进行完整评估 ──────────────────────
    # 只有当缓存不新鲜时才需要评估
    final_acc = env.current_acc if hasattr(env, '_acc_is_fresh') and env._acc_is_fresh else None
    
    if data_type in ('image', 'text') and not (hasattr(env, '_acc_is_fresh') and env._acc_is_fresh):
        # 缓存不新鲜，需要完整评估
        if hasattr(env, '_evaluate_accuracy'):
            final_acc = env._evaluate_accuracy(quick=False)
            env.current_acc = final_acc
            if final_acc > env.best_acc:
                env.best_acc = final_acc
    
    # 如果 final_acc 仍为 None（表格数据或缓存新鲜），使用 current_acc
    if final_acc is None:
        final_acc = env.current_acc if hasattr(env, 'current_acc') else info.get("accuracy", 0)

    # 统一 final_info 格式（与 adult 数据集一致）
    final_info = {
        "accuracy": round(final_acc, 4),
        "best_accuracy": round(env.best_acc if hasattr(env, 'best_acc') else info.get("best_accuracy", 0), 4),
        "n_samples": len(env.current_indices) if hasattr(env, 'current_indices') else info.get("n_samples", 0),
        "data_type": data_type,
        "label_noise_ratio": round(info.get("label_noise_ratio", 0), 4),
        "class_ratio": info.get("class_ratio", ""),
    }

    return ep_reward, final_info, ep_log, ppo_metrics


def heuristic_action_image(action_counts, n_actions, step, total_steps):
    """图像/文本数据的启发式动作
    
    n_actions: 实际动作数（4：modify_labels, delete, add, no_op）
    返回: 动作索引 (1, 2, 3, 4)
    """
    progress = step / max(total_steps, 1)

    if progress < 0.3:
        weights = [0.7, 0.05, 0.2, 0.05]  # labels, delete, add, no_op
    elif progress < 0.7:
        weights = [0.5, 0.15, 0.25, 0.1]
    else:
        weights = [0.4, 0.2, 0.3, 0.1]

    # 返回动作索引 1-4
    return int(np.random.choice([1, 2, 3, 4], p=weights))


# ══════════════════════════════════════════════════════════════════════════════
# Main Training
# ══════════════════════════════════════════════════════════════════════════════

def train(dataset=None, interactive_mode=False, auto_accept_mode=False):
    """主训练函数"""
    cfg = Config(dataset=dataset)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    cfg.best_data_path = os.path.join(cfg.checkpoint_dir, "best_data.csv")

    # 检测数据类型
    data_type = get_data_type(dataset or cfg.dataset_name)
    print("\n" + "=" * 70)
    print(f"PPO-HRL 训练 - 数据类型: {data_type.upper()}")
    print("=" * 70)
    print(f"  数据集: {cfg.dataset_name}")
    print(f"  设备: {cfg.device}")
    print("=" * 70)

    # 根据数据类型选择环境
    env = create_env_for_data_type(cfg, data_type)

    # 根据数据类型选择 Selector
    sel_agent = create_selector_for_data_type(cfg, data_type)

    # 对于图像/文本数据，使用 4 个动作（1-4：modify_labels, delete, add, no_op）
    # 对于表格数据，使用 5 个动作（0-4：包含 modify_features）
    if data_type in ('image', 'text'):
        original_n_actions = cfg.n_actions
        cfg.n_actions = 4  # 使用动作 1,2,3,4

    ppo_agent = PPOAgent(cfg)

    if data_type in ('image', 'text'):
        cfg.n_actions = original_n_actions  # 恢复原始值

    env.set_sel_agent(sel_agent)
    env.reset()

    training_log = []
    best_acc = 0.0
    start_time = time.time()

    auto_accept_ref = {"auto_accept": auto_accept_mode}

    def log_episode(phase, ep, ep_reward, final_info, ep_log=None, ppo_metrics=None):
        acc = final_info.get("accuracy", 0)
        best = final_info.get("best_accuracy", acc)
        n = final_info.get("n_samples", 0)
        record = dict(phase=phase, episode=ep, reward=ep_reward,
                      accuracy=acc, best_accuracy=best, n_samples=n, data_type=data_type)

        # 添加噪声相关指标
        if 'label_noise_ratio' in final_info:
            record['label_noise_ratio'] = final_info['label_noise_ratio']
        if 'class_ratio' in final_info:
            record['class_ratio'] = final_info['class_ratio']

        if ep_log:
            total_selected = sum(step.get("n_selected", 0) for step in ep_log)
            total_noise = sum(step.get("selected_noise_count", 0) for step in ep_log)
            record["total_selected"] = total_selected
            record["total_noise_found"] = total_noise

        if ppo_metrics:
            record.update(ppo_metrics)
        training_log.append(record)
        return acc, best_acc

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: 噪声检测 + Selector Warmup
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 1: 噪声检测 + Selector Warmup")
    print("=" * 70)

    print(f"\n[Step 1a] 运行噪声检测 ({data_type} 数据)...")

    noise_threshold = getattr(cfg, 'detection_noise_threshold', 0.5)
    clean_threshold = getattr(cfg, 'detection_clean_threshold', 0.3)
    min_gap = getattr(cfg, 'detection_min_gap', 0.1)

    clean_mask, noisy_mask, detection_scores = run_detection_phase(
        env, cfg, confidence_threshold=noise_threshold,
        clean_threshold=clean_threshold, min_gap=min_gap
    )

    print(f"\n[Step 1b] 用检测结果训练 Selector...")

    if data_type in ('image', 'text'):
        init_dirty = noisy_mask.mean()
        init_n = len(noisy_mask)
    else:
        init_dirty = env.sample_dirty.mean()
        init_n = len(env.sample_dirty)

    run_phase1_supervised_with_detection(
        env, sel_agent, cfg,
        clean_mask, noisy_mask,
        n_epochs=cfg.n_warmup,
        batch_size=64
    )

    print(f"\n  [Initial] Dirty={init_dirty:.3f} ({int(init_dirty*init_n)}/{init_n})")

    # ── 图像/文本数据：Phase 1 结束后初始化评估缓存 ──────────────────
    if data_type in ('image', 'text'):
        if hasattr(env, '_evaluate_accuracy') and hasattr(env, '_eval_every_n'):
            print(f"\n  [Phase1-End] 初始化评估缓存...")
            init_acc = env._evaluate_accuracy(quick=True)
            env._cached_acc = init_acc
            env._last_eval_step = 0
            env.current_acc = init_acc
            print(f"  [Phase1-End] 初始准确率: {init_acc:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: 固定高层动作 + 轮流训练 Selector
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 2: 固定高层动作 + 训练 Selector")
    print("=" * 70)

    for ep in range(cfg.n_freeze_hl):
        ep_rew, f_info, ep_log, _ = run_episode(
            env, ppo_agent, sel_agent, cfg, "freeze_hl", ep, data_type
        )
        acc, best = log_episode("freeze_hl", ep, ep_rew, f_info, ep_log)

        # 更新全局 best_acc
        if acc > best_acc:
            best_acc = acc

        if (ep + 1) % cfg.log_interval == 0:
            total_selected = sum(step.get("n_selected", 0) for step in ep_log)
            total_noise = sum(step.get("selected_noise_count", 0) for step in ep_log)
            avg_noise_ratio = total_noise / total_selected if total_selected > 0 else 0.0
            best = f_info.get("best_accuracy", acc)

            print(
                f"  [FreezeHL {ep+1:2d}/{cfg.n_freeze_hl}] "
                f"Rew={ep_rew:+.3f}  Acc={acc:.4f}  Best={best:.4f}  "
                f"SelNoise={avg_noise_ratio:.2%}({total_noise}/{total_selected})"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: 高低层联合训练
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 3: 高低层联合训练 (Joint)")
    print("=" * 70)

    # Step 3 内部最佳准确率（独立于 Step 2）
    joint_best_acc = 0.0

    for ep in range(cfg.n_joint):
        ep_rew, f_info, ep_log, ppo_m = run_episode(
            env, ppo_agent, sel_agent, cfg, "joint", ep, data_type,
            interactive_mode=interactive_mode,
            auto_accept_ref=auto_accept_ref
        )
        acc, best = log_episode("joint", ep, ep_rew, f_info, ep_log, ppo_m)

        # Step 3 内部最佳数据保存（独立比较）
        if acc > joint_best_acc:
            joint_best_acc = acc
            sel_agent.save(os.path.join(cfg.checkpoint_dir, "multiselector_best.pt"))
            ppo_agent.save(os.path.join(cfg.checkpoint_dir, "ppo_best.pt"))

            if data_type not in ('image', 'text') and hasattr(env, 'current_data'):
                # 表格数据：保存为 CSV
                best_data = env.current_data.copy()
                best_data.to_csv(os.path.join(cfg.checkpoint_dir, "best_data.csv"), index=False)
            elif data_type in ('image', 'text'):
                # 图像/文本数据：保存清洗后的数据为 npz 格式
                base_env = env._env if hasattr(env, '_env') else env

                if data_type == 'image':
                    # 图像数据
                    cleaned_images = base_env.images[base_env.current_indices]
                    cleaned_labels = base_env.noisy_labels[base_env.current_indices]
                    cleaned_true_labels = base_env.true_labels[base_env.current_indices]
                    original_n = len(base_env.images)
                    data_key = 'data'
                    data_desc = 'Images'
                else:
                    # 文本数据：TF-IDF 特征
                    if hasattr(base_env, 'current_indices'):
                        cleaned_features = base_env.features[base_env.current_indices]
                        cleaned_labels = base_env.noisy_labels[base_env.current_indices]
                        cleaned_true_labels = base_env.true_labels[base_env.current_indices]
                    else:
                        # 使用 current_data 中的索引
                        cleaned_features = base_env.features
                        cleaned_labels = base_env.noisy_labels
                        cleaned_true_labels = base_env.true_labels
                    original_n = len(base_env.features)
                    data_key = 'features'
                    data_desc = 'Features'

                # 保存为 npz 格式
                npz_path = os.path.join(cfg.checkpoint_dir, "best_cleaned_data.npz")
                save_dict = {
                    data_key: cleaned_features if data_type == 'text' else cleaned_images,
                    'labels': cleaned_labels,
                    'true_labels': cleaned_true_labels,
                }
                np.savez_compressed(npz_path, **save_dict)

                # 计算样本数量
                n_samples = len(base_env.current_indices) if hasattr(base_env, 'current_indices') else len(base_env.features)

                # 同时保存元信息为 JSON
                meta = {
                    'accuracy': float(acc),
                    'n_samples': int(n_samples),
                    'original_n_samples': int(original_n),
                    'indices': base_env.current_indices.tolist() if hasattr(base_env, 'current_indices') else list(range(original_n)),
                    'label_noise_ratio': float((cleaned_labels != cleaned_true_labels).mean()),
                    'data_type': data_type,
                }
                meta_path = os.path.join(cfg.checkpoint_dir, "best_cleaned_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)

                print(f"\n  [Best Joint Data] Saved! Acc={acc:.4f}, Samples={n_samples}")
                print(f"    - {data_desc}+Labels: {npz_path}")
                print(f"    - Metadata: {meta_path}")

        a_dist = env.action_counts / (env.action_counts.sum() + 1e-8)
        p_loss = ppo_m.get("policy_loss", 0)

        total_selected = sum(step.get("n_selected", 0) for step in ep_log)
        total_noise = sum(step.get("selected_noise_count", 0) for step in ep_log)
        avg_noise_ratio = total_noise / total_selected if total_selected > 0 else 0.0

        if (ep + 1) % cfg.log_interval == 0:
            print(
                f"  [Joint    {ep+1:3d}/{cfg.n_joint}] "
                f"Rew={ep_rew:+.3f}  Acc={acc:.4f}  "
                f"Best={best_acc:.4f}  "
                f"SelNoise={avg_noise_ratio:.2%}({total_noise}/{total_selected})  "
                f"PLoss={p_loss:.4f}"
            )

        # 保存日志
        iter_detail = {
            "iteration": ep + 1,
            "data_type": data_type,
            "reward": round(float(ep_rew), 4),
            "accuracy": round(float(acc), 4),
            "best_accuracy": round(float(best_acc), 4),
            "label_noise_ratio": round(float(f_info.get("label_noise_ratio", 0)), 4),
            "class_ratio": f_info.get("class_ratio", ""),
            "n_selected": int(total_selected),
            "total_noise_found": int(total_noise),
            "noise_precision": round(avg_noise_ratio, 4),
            "policy_loss": round(float(p_loss), 4),
            "steps": ep_log,
        }

        latest_path = os.path.join(cfg.checkpoint_dir, "joint_latest.json")
        with open(latest_path, "w", encoding="utf-8") as fh:
            json.dump(iter_detail, fh, indent=2, ensure_ascii=False, cls=FourDecimalEncoder)

        training_log.append(iter_detail)

        if (ep + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, "training_log.json")
            with open(ckpt_path, "w") as fh:
                json.dump(training_log, fh, indent=2, cls=FourDecimalEncoder)

    # ══════════════════════════════════════════════════════════════════════════
    # Final
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min  |  Best acc = {best_acc:.4f}")

    # 表格数据额外评估
    if data_type != 'image' and os.path.exists(os.path.join(cfg.checkpoint_dir, "best_data.csv")):
        best_df = pd.read_csv(os.path.join(cfg.checkpoint_dir, "best_data.csv"))
        acc_best_state = _eval_data_on_ref(cfg, env, best_df)
        print(f"\n  Best state accuracy: {acc_best_state:.4f}")
        print(f"  Best state dataset size: {len(best_df)}")
    elif data_type in ('image', 'text') and os.path.exists(os.path.join(cfg.checkpoint_dir, "best_cleaned_meta.json")):
        with open(os.path.join(cfg.checkpoint_dir, "best_cleaned_meta.json"), "r") as f:
            best_meta = json.load(f)
        print(f"\n  Best cleaned data:")
        print(f"    - Accuracy: {best_meta.get('accuracy', 0):.4f}")
        print(f"    - Samples: {best_meta.get('n_samples', 0)} (原始: {best_meta.get('original_n_samples', 0)})")
        print(f"    - Label noise ratio: {best_meta.get('label_noise_ratio', 0):.4f}")
        print(f"    - Saved to: best_cleaned_data.npz + best_cleaned_meta.json")

    log_path = os.path.join(cfg.checkpoint_dir, "training_log.json")
    with open(log_path, "w") as fh:
        json.dump(training_log, fh, indent=2)
    print(f"\nLog saved to {log_path}")

    return best_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='adult')
    parser.add_argument("--use-split-data", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--auto-accept", action="store_true", default=False)
    args = parser.parse_args()

    if '--use-split-data' in sys.argv:
        Config._args.use_split_data = True

    train(dataset=args.dataset, interactive_mode=args.interactive, auto_accept_mode=args.auto_accept)
