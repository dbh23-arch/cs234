import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ptd
from utils.state_encoder import StateEncoder

class PPOAgent(nn.Module):
    """PPO with factored action space (action_type x element)."""

    def __init__(self, state_dim=256, hidden_dim=256, max_elements=64,
                 num_action_types=2, element_feature_dim=24):
        super().__init__()

        self.state_encoder = StateEncoder(
            element_feature_dim=element_feature_dim,
            max_elements=max_elements,
            state_dim=state_dim,
        )
        embed_dim = self.state_encoder.embed_dim

        self.action_type_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )

        self.element_score = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.max_elements = max_elements

    def _encode(self, element_features, element_mask):
        return self.state_encoder(element_features, element_mask)

    def forward(self, element_features, element_mask):
        state, element_embeds = self._encode(element_features, element_mask)

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
            log_prob = (at_dist.log_prob(action_type) +
                        el_dist.log_prob(element_idx))
            return action_type.item(), element_idx.item(), log_prob.item()
        return action_type.item(), element_idx.item()

    def log_prob(self, element_features, element_mask, action_types, element_idxs):
        at_dist, el_dist = self.action_distribution(element_features, element_mask)
        return at_dist.log_prob(action_types) + el_dist.log_prob(element_idxs)

    def get_value(self, element_features, element_mask):
        state, _ = self._encode(element_features, element_mask)
        return self.value_head(state).squeeze(-1)

    def get_action(self, element_features, element_mask):
        with torch.no_grad():
            ef = element_features.unsqueeze(0)
            em = element_mask.unsqueeze(0)
            at_logits, el_logits, _ = self.forward(ef, em)
        return {
            "action_type_idx": at_logits.argmax(dim=-1).item(),
            "element_idx": el_logits.argmax(dim=-1).item(),
        }
    

def get_returns(paths, gamma):
    all_returns = []
    for path in paths:
        rewards = path["reward"]
        returns = [0] * len(rewards)
        future = 0
        for i in range(len(rewards)):
            current = rewards[len(rewards) - 1 - i]
            future = current + gamma * future
            returns[len(rewards) - 1 - i] = future
        all_returns.append(returns)
    return np.concatenate(all_returns)

def normalize_advantage(advantages):
    return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

def calculate_advantage(returns, features, masks, agent, device, normalize=True):
    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = agent.get_value(features_t, masks_t).cpu().numpy()
    advantages = returns - values
    if normalize:
        advantages = normalize_advantage(advantages)
    return advantages

def update_policy(agent, optimizer, features, masks, action_types, element_idxs,
                  advantages, old_logprobs, eps_clip, device, entropy_coeff=0.01):
    n = min(
        len(features), len(masks), len(action_types), len(element_idxs),
        len(advantages), len(old_logprobs),
    )
    if n == 0:
        return
    features = features[:n]
    masks = masks[:n]
    action_types = action_types[:n]
    element_idxs = element_idxs[:n]
    advantages = advantages[:n]
    old_logprobs = old_logprobs[:n]

    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    action_types_t = torch.tensor(action_types, dtype=torch.long, device=device)
    element_idxs_t = torch.tensor(element_idxs, dtype=torch.long, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)

    at_dist, el_dist = agent.action_distribution(features_t, masks_t)
    log_prob_actions = at_dist.log_prob(action_types_t) + el_dist.log_prob(element_idxs_t)

    # PPO clipped surrogate objective
    z_theta = torch.exp(log_prob_actions - old_logprobs_t)
    unclipped = z_theta * advantages_t
    clipped = torch.clamp(z_theta, 1 - eps_clip, 1 + eps_clip) * advantages_t
    objective = torch.min(unclipped, clipped)

    entropy = at_dist.entropy().mean() + el_dist.entropy().mean()

    optimizer.zero_grad()
    loss = -torch.sum(objective) / action_types_t.shape[0] - entropy_coeff * entropy
    loss.backward()
    optimizer.step()

def update_value(agent, optimizer, features, masks, returns, device):
    n = min(len(features), len(masks), len(returns))
    if n == 0:
        return
    features = features[:n]
    masks = masks[:n]
    returns = returns[:n]

    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    values = agent.get_value(features_t, masks_t)
    optimizer.zero_grad()
    loss = F.mse_loss(values, returns_t)
    loss.backward()
    optimizer.step()
