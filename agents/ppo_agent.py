"""
High-level PPO Agent.

Selects one of the n_actions discrete actions each step.
Uses Generalised Advantage Estimation (GAE) and a clipped surrogate loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import ActorCritic


class PPOAgent:
    def __init__(self, config):
        self.cfg = config
        self.net = ActorCritic(
            config.state_dim,
            config.n_actions,
            config.actor_hidden,
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr_ppo)

        # Per-episode trajectory storage
        self._reset_buffer()

    # ──────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False):
        """
        Parameters
        ----------
        state   : (state_dim,) numpy array
        greedy  : if True use argmax (for evaluation)

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        """
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if greedy:
                logits, value = self.net(s)
                action = logits.argmax(dim=-1)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
            else:
                action, log_prob, value = self.net.select_action(s)
        return action.item(), log_prob.item(), value.item()

    # ──────────────────────────────────────────────────────────────────────
    # Trajectory buffer
    # ──────────────────────────────────────────────────────────────────────

    def _reset_buffer(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_rewards = []
        self.buf_values = []
        self.buf_dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_log_probs.append(log_prob)
        self.buf_rewards.append(reward)
        self.buf_values.append(value)
        self.buf_dones.append(done)

    # ──────────────────────────────────────────────────────────────────────
    # PPO update (called once per episode)
    # ──────────────────────────────────────────────────────────────────────

    def update(self):
        """Run PPO update on the stored episode trajectory."""
        if len(self.buf_states) == 0:
            self._reset_buffer()
            return {}
        
        # 过滤异常 reward
        rewards = np.array(self.buf_rewards)
        values = np.array(self.buf_values)
        
        # 如果 reward 标准差异常大，跳过更新
        if len(rewards) > 0 and rewards.std() > 10.0:
            self._reset_buffer()
            return {"policy_loss": 0.0, "skipped": True, "reason": "high_variance"}
        
        states = torch.FloatTensor(np.array(self.buf_states))
        actions = torch.LongTensor(self.buf_actions)
        old_log_probs = torch.FloatTensor(self.buf_log_probs)
        
        advantages, returns = self._gae(
            self.buf_rewards, self.buf_values, self.buf_dones
        )
        
        # 过滤异常 advantage
        advantages = torch.clamp(torch.FloatTensor(advantages), -10.0, 10.0)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.cfg.ppo_epochs):
            log_probs, values, entropy = self.net.evaluate(states, actions)

            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                + self.cfg.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.mean().item())

        self._reset_buffer()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ──────────────────────────────────────────────────────────────────────
    # GAE computation
    # ──────────────────────────────────────────────────────────────────────

    def _gae(self, rewards, values, dones):
        """Compute GAE advantages and discounted returns."""
        advantages = []
        gae = 0.0
        next_val = 0.0

        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.cfg.gamma * next_val * (1 - float(d)) - v
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - float(d)) * gae
            advantages.insert(0, gae)
            next_val = v

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        return advantages, returns

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
