"""PPO agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ptd
from utils.state_encoder import StateEncoder


class PPOAgent(nn.Module):
    def __init__(self, state_dim=256, hidden_dim=256, max_elements=64,
                 num_action_types=2, element_feature_dim=24):
        super().__init__()
        self.state_encoder = StateEncoder(
            element_feature_dim=element_feature_dim,
            max_elements=max_elements, state_dim=state_dim,
        )
        embed_dim = self.state_encoder.embed_dim

        self.action_type_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )
        self.element_score = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.max_elements = max_elements

    def forward(self, element_features, element_mask):
        state, element_embeds = self.state_encoder(element_features, element_mask)
        at_logits = self.action_type_head(state)
        B, N, D = element_embeds.shape
        state_exp = state.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([state_exp, element_embeds], dim=-1)
        el_logits = self.element_score(combined).squeeze(-1)
        el_logits = el_logits.masked_fill(element_mask == 0, float('-inf'))
        value = self.value_head(state).squeeze(-1)
        return at_logits, el_logits, value

    def action_distribution(self, element_features, element_mask):
        at_logits, el_logits, _ = self.forward(element_features, element_mask)
        return ptd.Categorical(logits=at_logits), ptd.Categorical(logits=el_logits)

    def act(self, element_features, element_mask, return_log_prob=False):
        at_dist, el_dist = self.action_distribution(element_features, element_mask)
        action_type = at_dist.sample()
        element_idx = el_dist.sample()
        if return_log_prob:
            log_prob = at_dist.log_prob(action_type) + el_dist.log_prob(element_idx)
            return action_type.item(), element_idx.item(), log_prob.item()
        return action_type.item(), element_idx.item()

    def get_value(self, element_features, element_mask):
        state, _ = self.state_encoder(element_features, element_mask)
        return self.value_head(state).squeeze(-1)

    def get_action(self, element_features, element_mask):
        with torch.no_grad():
            at_logits, el_logits, _ = self.forward(
                element_features.unsqueeze(0), element_mask.unsqueeze(0))
        return {
            "action_type_idx": at_logits.argmax(dim=-1).item(),
            "element_idx": el_logits.argmax(dim=-1).item(),
        }


def get_returns(paths, gamma):
    all_returns = []
    for path in paths:
        rewards = path["reward"]
        returns = [0.0] * len(rewards)
        future = 0.0
        for i in reversed(range(len(rewards))):
            future = rewards[i] + gamma * future
            returns[i] = future
        all_returns.append(returns)
    return np.concatenate(all_returns)


def calculate_advantage(returns, features, masks, agent, device, normalize=True):
    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = agent.get_value(features_t, masks_t).cpu().numpy()
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def update_policy(agent, optimizer, features, masks, action_types, element_idxs,
                  advantages, old_logprobs, eps_clip, device, entropy_coeff=0.01):
    n = min(len(features), len(advantages))
    if n == 0:
        return
    features_t = torch.tensor(features[:n], dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks[:n], dtype=torch.float32, device=device)
    at_t = torch.tensor(action_types[:n], dtype=torch.long, device=device)
    ei_t = torch.tensor(element_idxs[:n], dtype=torch.long, device=device)
    adv_t = torch.tensor(advantages[:n], dtype=torch.float32, device=device)
    old_lp_t = torch.tensor(old_logprobs[:n], dtype=torch.float32, device=device)

    at_dist, el_dist = agent.action_distribution(features_t, masks_t)
    log_probs = at_dist.log_prob(at_t) + el_dist.log_prob(ei_t)

    ratio = torch.exp(log_probs - old_lp_t)
    clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_t
    objective = torch.min(ratio * adv_t, clipped)
    entropy = at_dist.entropy().mean() + el_dist.entropy().mean()

    optimizer.zero_grad()
    loss = -(objective.mean()) - entropy_coeff * entropy
    loss.backward()
    optimizer.step()


def update_value(agent, optimizer, features, masks, returns, device):
    n = min(len(features), len(returns))
    if n == 0:
        return
    features_t = torch.tensor(features[:n], dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks[:n], dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns[:n], dtype=torch.float32, device=device)

    optimizer.zero_grad()
    values = agent.get_value(features_t, masks_t)
    loss = F.mse_loss(values, returns_t)
    loss.backward()
    optimizer.step()