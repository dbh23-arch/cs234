"""BC agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.state_encoder import StateEncoder


class BCAgent(nn.Module):
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
        self.max_elements = max_elements

    def forward(self, element_features, element_mask):
        state, element_embeds = self.state_encoder(element_features, element_mask)
        at_logits = self.action_type_head(state)

        B, N, D = element_embeds.shape
        state_exp = state.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([state_exp, element_embeds], dim=-1)
        el_logits = self.element_score(combined).squeeze(-1)
        el_logits = el_logits.masked_fill(element_mask == 0, float('-inf'))
        return at_logits, el_logits

    def get_action(self, element_features, element_mask):
        with torch.no_grad():
            at_logits, el_logits = self.forward(
                element_features.unsqueeze(0), element_mask.unsqueeze(0))
        return {
            "action_type_idx": at_logits.argmax(dim=-1).item(),
            "element_idx": el_logits.argmax(dim=-1).item(),
        }

    def compute_loss(self, batch):
        at_logits, el_logits = self.forward(
            batch["state_features"], batch["state_mask"])
        at_loss = F.cross_entropy(at_logits, batch["action_type"])
        el_loss = F.cross_entropy(el_logits, batch["element_idx"])
        return {
            "loss": at_loss + el_loss,
            "action_type_acc": (at_logits.argmax(-1) == batch["action_type"]).float().mean().item(),
            "element_acc": (el_logits.argmax(-1) == batch["element_idx"]).float().mean().item(),
        }
