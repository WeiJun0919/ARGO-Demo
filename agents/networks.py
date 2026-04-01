"""
Neural network definitions for PPO-HRL data cleaning.

ActorCritic  – high-level PPO policy (shared backbone)
SelectorNet  – low-level conditional sample selector (two-head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# High-level Actor-Critic  (PPO)
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic for the high-level PPO agent.

    Architecture (from spec):
        S_t (state_dim)
          → Linear(state_dim → hidden[0]) + LayerNorm + ReLU
          → Linear(hidden[0]  → hidden[1]) + ReLU
          → Actor: Linear(hidden[1] → n_actions) → Softmax
          → Critic: Linear(hidden[1] → 1)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: list):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden[1], n_actions)
        self.critic_head = nn.Linear(hidden[1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))  # 标准初始化
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor):
        """Returns (logits, value)."""
        state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.backbone(state)
        logits = self.actor_head(feat)
        value = self.critic_head(feat)
        return torch.nan_to_num(logits, nan=0.0), value

    def select_action(self, state: torch.Tensor):
        """Sample action; return (action, log_prob, value)."""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate log-prob, value and entropy for a batch."""
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, values.squeeze(-1), entropy

    def action_probs(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(state)
        return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level Selector Network  (Two-Head)
# ─────────────────────────────────────────────────────────────────────────────

class SelectorNet(nn.Module):
    """
    Conditional sample selector: SelectorNet = Shared Encoder + Two Heads

    Input  z_i = [S_t || u_i || action_onehot]   shape (N, selector_input_dim)

    Head 1 – Relabel/Select Score : p_i ∈ (0, 1)   (Sigmoid)
    Head 2 – Feature Prediction   : ĥ_i ∈ R^n_features  (for aux regression loss)

    The intermediate hidden representation h_i is exposed for diversity loss.
    """

    def __init__(self, input_dim: int, n_features: int, hidden: list):
        super().__init__()

        dims = [input_dim] + hidden
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # Head 1: selection probability
        self.score_head = nn.Sequential(
            nn.Linear(hidden[-1], 1),
            nn.Sigmoid(),
        )

        # Head 2: feature prediction (auxiliary regression)
        self.feat_head = nn.Linear(hidden[-1], n_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor):
        """
        z : (N, input_dim)

        Returns
        -------
        scores   : (N,)         selection probabilities
        feat_pred: (N, n_feat)  predicted clean feature values
        hidden   : (N, hidden[-1]) intermediate representation (for diversity)
        """
        h = self.encoder(z)
        scores = self.score_head(h).squeeze(-1)
        feat_pred = self.feat_head(h)
        return scores, feat_pred, h
