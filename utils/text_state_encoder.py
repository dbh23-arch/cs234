"""
DistilBERT-based state encoder for MiniWoB++ web navigation.
Replaces hand-crafted 24-dim DOM features with contextual LM embeddings.

Key design: outputs SAME signature as StateEncoder:
    state: (B, state_dim=256)
    element_embeds: (B, max_elements=64, embed_dim=64)
So all model action heads (BC, IQL, DT) work without changes.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextStateEncoder(nn.Module):
    """
    Encodes DOM page text via DistilBERT.

    Architecture:
        1. Frozen DistilBERT encodes full page text (768-dim hidden states)
        2. CLS token -> project to state_dim (global state)
        3. Mean-pool each element's token span -> project to embed_dim (per-element)

    Same output contract as StateEncoder:
        state: (B, state_dim)
        element_embeds: (B, max_elements, embed_dim)
    """

    def __init__(self, state_dim=256, embed_dim=64, max_elements=64,
                 model_name="distilbert-base-uncased", freeze_lm=True):
        super().__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.max_elements = max_elements
        self.lm_hidden = 768  # DistilBERT hidden size

        # Load pretrained DistilBERT
        self.lm = DistilBertModel.from_pretrained(model_name)
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        # Project CLS token (768) -> state_dim (256)
        self.state_proj = nn.Sequential(
            nn.Linear(self.lm_hidden, state_dim),
            nn.ReLU(),
        )

        # Project per-element embeddings (768) -> embed_dim (64)
        self.element_proj = nn.Sequential(
            nn.Linear(self.lm_hidden, embed_dim),
            nn.ReLU(),
        )

    def forward(self, input_ids, attention_mask, element_token_spans,
                element_mask):
        """
        Args:
            input_ids: (B, seq_len) tokenized page text
            attention_mask: (B, seq_len)
            element_token_spans: (B, max_elements, 2) start/end token indices
            element_mask: (B, max_elements) 1 for valid elements

        Returns:
            state: (B, state_dim)
            element_embeds: (B, max_elements, embed_dim)
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Run DistilBERT
        with torch.set_grad_enabled(any(p.requires_grad for p in self.lm.parameters())):
            lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = lm_out.last_hidden_state  # (B, seq_len, 768)

        # Global state from CLS token
        cls_hidden = hidden[:, 0, :]  # (B, 768)
        state = self.state_proj(cls_hidden)  # (B, state_dim)

        # Per-element embeddings via mean pooling of token spans
        element_embeds = torch.zeros(B, self.max_elements, self.embed_dim,
                                     device=device)

        for b in range(B):
            for i in range(self.max_elements):
                if element_mask[b, i] == 0:
                    continue
                start = element_token_spans[b, i, 0].item()
                end = element_token_spans[b, i, 1].item()
                if start >= end:
                    continue
                # Mean pool the element's token hidden states
                elem_hidden = hidden[b, start:end, :].mean(dim=0)  # (768,)
                element_embeds[b, i] = self.element_proj(elem_hidden)

        return state, element_embeds


_tokenizer_cache = {}

def get_tokenizer(model_name="distilbert-base-uncased"):
    """Get cached tokenizer instance."""
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = DistilBertTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]
