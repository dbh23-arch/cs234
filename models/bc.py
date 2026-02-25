"""
Behavioral Cloning (BC) baseline.
Learns a policy by supervised learning on demonstration actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.state_encoder import StateEncoder
from utils.text_state_encoder import TextStateEncoder


class BCAgent(nn.Module):
    """
    Behavioral Cloning agent.
    Predicts action_type and element_idx from state via supervised learning.
    Supports both DOM (hand-crafted) and text (DistilBERT) encoders.
    """

    def __init__(self, state_dim=256, hidden_dim=256, max_elements=64,
                 num_action_types=2, element_feature_dim=24,
                 encoder_type="dom", freeze_lm=True):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "text":
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

        # Action type head (click vs type)
        self.action_type_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_types),
        )

        # Element selection head - uses element embeddings
        embed_dim = self.state_encoder.embed_dim
        self.element_score = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.max_elements = max_elements
        self.num_action_types = num_action_types

    def encode_state(self, batch):
        """Encode state using the appropriate encoder."""
        if self.encoder_type == "text":
            return self.state_encoder(
                batch["input_ids"], batch["attention_mask"],
                batch["element_token_spans"], batch["element_mask"],
            )
        else:
            return self.state_encoder(
                batch["state_features"], batch["state_mask"],
            )

    def forward_from_encoded(self, state, element_embeds, element_mask):
        """Forward pass from pre-encoded state."""
        action_type_logits = self.action_type_head(state)

        B, N, D = element_embeds.shape
        state_expanded = state.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([state_expanded, element_embeds], dim=-1)
        element_logits = self.element_score(combined).squeeze(-1)
        element_logits = element_logits.masked_fill(element_mask == 0, float('-inf'))

        return action_type_logits, element_logits

    def forward(self, element_features, element_mask):
        """
        DOM encoder forward (backward compatible).

        Args:
            element_features: (B, max_elements, feature_dim)
            element_mask: (B, max_elements)
        """
        state, element_embeds = self.state_encoder(element_features, element_mask)
        return self.forward_from_encoded(state, element_embeds, element_mask)

    def get_action(self, element_features, element_mask):
        """Get action for a single observation (no batch dim). DOM encoder only."""
        with torch.no_grad():
            ef = element_features.unsqueeze(0)
            em = element_mask.unsqueeze(0)
            at_logits, el_logits = self.forward(ef, em)

            action_type = at_logits.argmax(dim=-1).item()
            element_idx = el_logits.argmax(dim=-1).item()

        return {"action_type_idx": action_type, "element_idx": element_idx}

    def get_action_text(self, input_ids, attention_mask, element_token_spans,
                        element_mask):
        """Get action for a single observation. Text encoder."""
        with torch.no_grad():
            batch = {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "element_token_spans": element_token_spans.unsqueeze(0),
                "element_mask": element_mask.unsqueeze(0),
            }
            state, element_embeds = self.encode_state(batch)
            at_logits, el_logits = self.forward_from_encoded(
                state, element_embeds, batch["element_mask"]
            )
            action_type = at_logits.argmax(dim=-1).item()
            element_idx = el_logits.argmax(dim=-1).item()
        return {"action_type_idx": action_type, "element_idx": element_idx}

    def compute_loss(self, batch):
        """Compute BC loss on a batch of transitions."""
        if self.encoder_type == "text":
            state, element_embeds = self.encode_state(batch)
            at_logits, el_logits = self.forward_from_encoded(
                state, element_embeds, batch["element_mask"]
            )
        else:
            at_logits, el_logits = self.forward(
                batch["state_features"], batch["state_mask"]
            )

        at_loss = F.cross_entropy(at_logits, batch["action_type"])
        el_loss = F.cross_entropy(el_logits, batch["element_idx"])
        total_loss = at_loss + el_loss

        return {
            "loss": total_loss,
            "action_type_loss": at_loss.item(),
            "element_loss": el_loss.item(),
            "action_type_acc": (at_logits.argmax(-1) == batch["action_type"]).float().mean().item(),
            "element_acc": (el_logits.argmax(-1) == batch["element_idx"]).float().mean().item(),
        }
