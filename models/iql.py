import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.state_encoder import StateEncoder


class QNetwork(nn.Module):
    """Q-network that outputs separate Q-values for action type and element."""

    def __init__(self, state_dim, hidden_dim, max_elements, num_action_types=2,
                 embed_dim=64):
        super().__init__()

        self.action_type_q = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )

        self.element_q = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
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
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

class IQLAgent(nn.Module):
    """Implicit Q-Learning with optional CQL regularization.

    Based on Kostrikov et al. 2021 (IQL) + Kumar et al. 2020 (CQL).
    We decompose Q into action_type Q + element Q since our action space
    is factored (pick type, then pick element).
    """

    def __init__(self, state_dim=256, hidden_dim=256, max_elements=64,
                 num_action_types=2, element_feature_dim=24,
                 discount=0.99, tau=0.7, beta=3.0, target_update_rate=0.005,
                 cql_alpha=0.0, cql_temperature=1.0,
                 encoder_type="dom", freeze_lm=True):
        super().__init__()

        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.target_update_rate = target_update_rate
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature
        self.max_elements = max_elements
        self.encoder_type = encoder_type

        if encoder_type == "text":
            from utils.text_state_encoder import TextStateEncoder
            self.state_encoder = TextStateEncoder(
                state_dim=state_dim,
                embed_dim=64,
                max_elements=max_elements,
                freeze_lm=freeze_lm,
            )
        else:
            self.state_encoder = StateEncoder(
                element_feature_dim=element_feature_dim,
                max_elements=max_elements,
                state_dim=state_dim,
            )

        embed_dim = self.state_encoder.embed_dim

        self.q1 = QNetwork(state_dim, hidden_dim, max_elements, num_action_types,
                           embed_dim)
        self.q2 = QNetwork(state_dim, hidden_dim, max_elements, num_action_types,
                           embed_dim)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.value = ValueNetwork(state_dim, hidden_dim)

        self.policy_action_type = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )

        self.policy_element = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_state_from_batch(self, batch, prefix=""):
        if self.encoder_type == "text":
            p = prefix
            return self.state_encoder(
                batch[f"{p}input_ids"], batch[f"{p}attention_mask"],
                batch[f"{p}element_token_spans"], batch[f"{p}element_mask"],
            )
        else:
            return self.state_encoder(
                batch[f"{p}state_features"], batch[f"{p}state_mask"],
            ) if prefix == "" else self.state_encoder(
                batch[f"next_state_features"], batch[f"next_state_mask"],
            )

    def encode_state(self, element_features, element_mask):
        return self.state_encoder(element_features, element_mask)

    def get_action(self, element_features, element_mask):
        with torch.no_grad():
            ef = element_features.unsqueeze(0)
            em = element_mask.unsqueeze(0)
            state, element_embeds = self.encode_state(ef, em)

            at_logits = self.policy_action_type(state)
            action_type = at_logits.argmax(dim=-1).item()

            B, N, D = element_embeds.shape
            state_exp = state.unsqueeze(1).expand(-1, N, -1)
            combined = torch.cat([state_exp, element_embeds], dim=-1)
            el_logits = self.policy_element(combined).squeeze(-1)
            el_logits = el_logits.masked_fill(em == 0, float('-inf'))
            element_idx = el_logits.argmax(dim=-1).item()

        return {"action_type_idx": action_type, "element_idx": element_idx}

    def get_action_text(self, input_ids, attention_mask, element_token_spans,
                        element_mask):
        with torch.no_grad():
            state, element_embeds = self.state_encoder(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0),
                element_token_spans.unsqueeze(0), element_mask.unsqueeze(0),
            )
            at_logits = self.policy_action_type(state)
            action_type = at_logits.argmax(dim=-1).item()

            B, N, D = element_embeds.shape
            state_exp = state.unsqueeze(1).expand(-1, N, -1)
            combined = torch.cat([state_exp, element_embeds], dim=-1)
            el_logits = self.policy_element(combined).squeeze(-1)
            el_logits = el_logits.masked_fill(element_mask.unsqueeze(0) == 0, float('-inf'))
            element_idx = el_logits.argmax(dim=-1).item()

        return {"action_type_idx": action_type, "element_idx": element_idx}

    def _get_element_mask(self, batch, prefix=""):
        if self.encoder_type == "text":
            return batch[f"{prefix}element_mask"]
        else:
            return batch[f"{prefix}state_mask"] if prefix == "" else batch["next_state_mask"]

    def compute_loss(self, batch):
        elem_mask = self._get_element_mask(batch)
        next_elem_mask = self._get_element_mask(batch, "next_")

        if self.encoder_type == "text":
            state, elem_embeds = self.state_encoder(
                batch["input_ids"], batch["attention_mask"],
                batch["element_token_spans"], batch["element_mask"],
            )
            with torch.no_grad():
                next_state, next_elem_embeds = self.state_encoder(
                    batch["next_input_ids"], batch["next_attention_mask"],
                    batch["next_element_token_spans"], batch["next_element_mask"],
                )
        else:
            state, elem_embeds = self.encode_state(
                batch["state_features"], batch["state_mask"]
            )
            with torch.no_grad():
                next_state, next_elem_embeds = self.encode_state(
                    batch["next_state_features"], batch["next_state_mask"]
                )

        action_types = batch["action_type"]
        element_idxs = batch["element_idx"]
        rewards = batch["reward"]
        dones = batch["done"]

        with torch.no_grad():
            at_q1, el_q1 = self.q1_target(state, elem_embeds, elem_mask)
            at_q2, el_q2 = self.q2_target(state, elem_embeds, elem_mask)

            q1_at = at_q1.gather(1, action_types.unsqueeze(1)).squeeze(1)
            q1_el = el_q1.gather(1, element_idxs.unsqueeze(1)).squeeze(1)
            q2_at = at_q2.gather(1, action_types.unsqueeze(1)).squeeze(1)
            q2_el = el_q2.gather(1, element_idxs.unsqueeze(1)).squeeze(1)

            q_target = torch.min(q1_at + q1_el, q2_at + q2_el)

        v = self.value(state)
        diff = q_target - v
        weight = torch.where(diff > 0, self.tau, 1 - self.tau)
        value_loss = (weight * diff.pow(2)).mean()

        with torch.no_grad():
            next_v = self.value(next_state)
            q_backup = rewards + (1 - dones) * self.discount * next_v

        at_q1, el_q1 = self.q1(state.detach(), elem_embeds.detach(), elem_mask)
        at_q2, el_q2 = self.q2(state.detach(), elem_embeds.detach(), elem_mask)

        q1_pred = at_q1.gather(1, action_types.unsqueeze(1)).squeeze(1) + \
                  el_q1.gather(1, element_idxs.unsqueeze(1)).squeeze(1)
        q2_pred = at_q2.gather(1, action_types.unsqueeze(1)).squeeze(1) + \
                  el_q2.gather(1, element_idxs.unsqueeze(1)).squeeze(1)

        q_bellman_loss = F.mse_loss(q1_pred, q_backup) + F.mse_loss(q2_pred, q_backup)

        # CQL penalty: logsumexp(Q) - Q(data) pushes down Q for unseen actions
        temp = max(self.cql_temperature, 1e-6)
        # can't use -inf here because logsumexp would give nan
        finite_neg = torch.tensor(-1e9, device=state.device, dtype=state.dtype)
        safe_el_q1 = torch.where(elem_mask > 0, el_q1, finite_neg)
        safe_el_q2 = torch.where(elem_mask > 0, el_q2, finite_neg)
        cql1 = (
            temp * torch.logsumexp(at_q1 / temp, dim=-1) +
            temp * torch.logsumexp(safe_el_q1 / temp, dim=-1) -
            q1_pred
        ).mean()
        cql2 = (
            temp * torch.logsumexp(at_q2 / temp, dim=-1) +
            temp * torch.logsumexp(safe_el_q2 / temp, dim=-1) -
            q2_pred
        ).mean()
        cql_loss = cql1 + cql2
        q_loss = q_bellman_loss + self.cql_alpha * cql_loss

        with torch.no_grad():
            advantage = q_target - v
            exp_advantage = torch.exp(self.beta * advantage).clamp(max=100.0)

        at_logits = self.policy_action_type(state.detach())
        at_log_probs = F.log_softmax(at_logits, dim=-1)
        at_log_prob = at_log_probs.gather(1, action_types.unsqueeze(1)).squeeze(1)

        B, N, D = elem_embeds.shape
        state_exp = state.detach().unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([state_exp, elem_embeds.detach()], dim=-1)
        el_logits = self.policy_element(combined).squeeze(-1)
        el_logits = el_logits.masked_fill(elem_mask == 0, float('-inf'))
        el_log_probs = F.log_softmax(el_logits, dim=-1)
        el_log_prob = el_log_probs.gather(1, element_idxs.unsqueeze(1)).squeeze(1)

        policy_loss = -(exp_advantage * (at_log_prob + el_log_prob)).mean()

        total_loss = value_loss + q_loss + policy_loss

        return {
            "loss": total_loss,
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "q_bellman_loss": q_bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "policy_loss": policy_loss.item(),
            "v_mean": v.mean().item(),
            "q_mean": q1_pred.mean().item(),
            "advantage_mean": advantage.mean().item(),
        }

    def update_targets(self):
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(self.target_update_rate * p.data +
                         (1 - self.target_update_rate) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(self.target_update_rate * p.data +
                         (1 - self.target_update_rate) * tp.data)
