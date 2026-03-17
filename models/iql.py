"""IQL agent (Kostrikov et al. 2021)."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.state_encoder import StateEncoder


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_action_types=2, embed_dim=64):
        super().__init__()
        self.action_type_q = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )
        self.element_q = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, element_embeds, element_mask):
        at_q = self.action_type_q(state)
        B, N, D = element_embeds.shape
        state_exp = state.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([state_exp, element_embeds], dim=-1)
        el_q = self.element_q(combined).squeeze(-1)
        el_q = el_q.masked_fill(element_mask == 0, float('-inf'))
        return at_q, el_q


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class IQLAgent(nn.Module):
    def __init__(self, state_dim=256, hidden_dim=256, max_elements=64,
                 num_action_types=2, element_feature_dim=24,
                 discount=0.99, tau=0.7, beta=3.0, target_update_rate=0.005):
        super().__init__()
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.target_update_rate = target_update_rate
        self.max_elements = max_elements

        self.state_encoder = StateEncoder(
            element_feature_dim=element_feature_dim,
            max_elements=max_elements, state_dim=state_dim,
        )
        embed_dim = self.state_encoder.embed_dim

        self.q1 = QNetwork(state_dim, hidden_dim, num_action_types, embed_dim)
        self.q2 = QNetwork(state_dim, hidden_dim, num_action_types, embed_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.value = ValueNetwork(state_dim, hidden_dim)

        self.policy_action_type = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )
        self.policy_element = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action(self, element_features, element_mask):
        with torch.no_grad():
            ef = element_features.unsqueeze(0)
            em = element_mask.unsqueeze(0)
            state, element_embeds = self.state_encoder(ef, em)

            at_logits = self.policy_action_type(state)
            action_type = at_logits.argmax(dim=-1).item()

            B, N, D = element_embeds.shape
            state_exp = state.unsqueeze(1).expand(-1, N, -1)
            combined = torch.cat([state_exp, element_embeds], dim=-1)
            el_logits = self.policy_element(combined).squeeze(-1)
            el_logits = el_logits.masked_fill(em == 0, float('-inf'))
            element_idx = el_logits.argmax(dim=-1).item()
        return {"action_type_idx": action_type, "element_idx": element_idx}

    def compute_loss(self, batch):
        state, elem_embeds = self.state_encoder(
            batch["state_features"], batch["state_mask"])
        with torch.no_grad():
            next_state, next_elem_embeds = self.state_encoder(
                batch["next_state_features"], batch["next_state_mask"])

        elem_mask = batch["state_mask"]
        action_types = batch["action_type"]
        element_idxs = batch["element_idx"]
        rewards = batch["reward"]
        dones = batch["done"]

        # value loss (expectile)
        with torch.no_grad():
            at_q1, el_q1 = self.q1_target(state, elem_embeds, elem_mask)
            at_q2, el_q2 = self.q2_target(state, elem_embeds, elem_mask)
            q1_val = at_q1.gather(1, action_types.unsqueeze(1)).squeeze(1) + \
                     el_q1.gather(1, element_idxs.unsqueeze(1)).squeeze(1)
            q2_val = at_q2.gather(1, action_types.unsqueeze(1)).squeeze(1) + \
                     el_q2.gather(1, element_idxs.unsqueeze(1)).squeeze(1)
            q_target = torch.min(q1_val, q2_val)

        v = self.value(state)
        diff = q_target - v
        weight = torch.where(diff > 0, self.tau, 1 - self.tau)
        value_loss = (weight * diff.pow(2)).mean()

        # q loss
        with torch.no_grad():
            next_v = self.value(next_state)
            q_backup = rewards + (1 - dones) * self.discount * next_v

        at_q1, el_q1 = self.q1(state.detach(), elem_embeds.detach(), elem_mask)
        at_q2, el_q2 = self.q2(state.detach(), elem_embeds.detach(), elem_mask)
        q1_pred = at_q1.gather(1, action_types.unsqueeze(1)).squeeze(1) + \
                  el_q1.gather(1, element_idxs.unsqueeze(1)).squeeze(1)
        q2_pred = at_q2.gather(1, action_types.unsqueeze(1)).squeeze(1) + \
                  el_q2.gather(1, element_idxs.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_pred, q_backup) + F.mse_loss(q2_pred, q_backup)

        # policy loss (AWR)
        with torch.no_grad():
            advantage = q_target - v
            exp_advantage = torch.exp(self.beta * advantage).clamp(max=100.0)

        at_logits = self.policy_action_type(state.detach())
        at_log_prob = F.log_softmax(at_logits, dim=-1).gather(
            1, action_types.unsqueeze(1)).squeeze(1)

        B, N, D = elem_embeds.shape
        state_exp = state.detach().unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([state_exp, elem_embeds.detach()], dim=-1)
        el_logits = self.policy_element(combined).squeeze(-1)
        el_logits = el_logits.masked_fill(elem_mask == 0, float('-inf'))
        el_log_prob = F.log_softmax(el_logits, dim=-1).gather(
            1, element_idxs.unsqueeze(1)).squeeze(1)

        policy_loss = -(exp_advantage * (at_log_prob + el_log_prob)).mean()

        return {
            "loss": value_loss + q_loss + policy_loss,
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def update_targets(self):
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(self.target_update_rate * p.data +
                         (1 - self.target_update_rate) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(self.target_update_rate * p.data +
                         (1 - self.target_update_rate) * tp.data)
