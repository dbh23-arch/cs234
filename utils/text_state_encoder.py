"""State encoder that uses DistilBERT to encode DOM text instead of hand-crafted features."""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextStateEncoder(nn.Module):

    def __init__(self, state_dim=256, embed_dim=64, max_elements=64,
                 model_name="distilbert-base-uncased", freeze_lm=True):
        super().__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.max_elements = max_elements
        self.lm_hidden = 768  # distilbert hidden size

        self.lm = DistilBertModel.from_pretrained(model_name)
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        # project CLS token to global state
        self.state_proj = nn.Sequential(
            nn.Linear(self.lm_hidden, state_dim),
            nn.ReLU(),
        )

        # project each element's token span to element embedding
        self.element_proj = nn.Sequential(
            nn.Linear(self.lm_hidden, embed_dim),
            nn.ReLU(),
        )

    def forward(self, input_ids, attention_mask, element_token_spans,
                element_mask):
        B = input_ids.shape[0]
        device = input_ids.device

        with torch.set_grad_enabled(any(p.requires_grad for p in self.lm.parameters())):
            lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = lm_out.last_hidden_state  # (B, seq_len, 768)

        cls_hidden = hidden[:, 0, :]
        state = self.state_proj(cls_hidden)  # (B, state_dim)

        # for each element, average its token span to get an embedding
        # TODO: this loop is slow, could batch with scatter but it works fine for now
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
                elem_hidden = hidden[b, start:end, :].mean(dim=0)
                element_embeds[b, i] = self.element_proj(elem_hidden)

        return state, element_embeds

_tokenizer_cache = {}

def get_tokenizer(model_name="distilbert-base-uncased"):
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = DistilBertTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]
