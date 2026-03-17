"""Decision Transformer (Chen et al. 2021)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.state_encoder import StateEncoder


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformerAgent(nn.Module):
    def __init__(self, state_dim=256, embed_dim=128, n_layers=4, n_heads=4,
                 context_length=20, max_elements=64, num_action_types=2,
                 element_feature_dim=24, max_timestep=100, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.max_elements = max_elements

        self.state_encoder = StateEncoder(
            element_feature_dim=element_feature_dim,
            max_elements=max_elements, state_dim=state_dim,
        )
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.return_embed = nn.Linear(1, embed_dim)
        self.action_type_embed = nn.Embedding(num_action_types, embed_dim)
        self.element_embed = nn.Embedding(max_elements, embed_dim)
        self.action_embed = nn.Linear(2 * embed_dim, embed_dim)
        self.timestep_embed = nn.Embedding(max_timestep, embed_dim)
        self.pos_embed = nn.Embedding(3 * context_length, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, 3 * context_length, dropout)
            for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.action_type_head = nn.Linear(embed_dim, num_action_types)
        self.element_head = nn.Linear(embed_dim, max_elements)

    def forward(self, state_features, state_masks, action_types, element_idxs,
                returns_to_go, timesteps, attention_mask=None):
        B, K = returns_to_go.shape

        flat_f = state_features.view(B * K, self.max_elements, -1)
        flat_m = state_masks.view(B * K, self.max_elements)
        encoded, _ = self.state_encoder(flat_f, flat_m)
        state_encoded = encoded.view(B, K, -1)

        state_tokens = self.state_embed(state_encoded)
        return_tokens = self.return_embed(returns_to_go.unsqueeze(-1))
        at_emb = self.action_type_embed(action_types)
        el_emb = self.element_embed(element_idxs)
        action_tokens = self.action_embed(torch.cat([at_emb, el_emb], dim=-1))

        time_emb = self.timestep_embed(timesteps)
        state_tokens = state_tokens + time_emb
        return_tokens = return_tokens + time_emb
        action_tokens = action_tokens + time_emb

        # interleave (R, s, a)
        tokens = torch.zeros(B, 3 * K, self.embed_dim, device=state_tokens.device)
        tokens[:, 0::3] = return_tokens
        tokens[:, 1::3] = state_tokens
        tokens[:, 2::3] = action_tokens

        positions = torch.arange(3 * K, device=tokens.device)
        tokens = self.dropout(tokens + self.pos_embed(positions))

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.ln_final(tokens)

        state_output = tokens[:, 1::3]
        return self.action_type_head(state_output), self.element_head(state_output)

    def get_action(self, state_features, state_mask,
                   past_states=None, past_state_masks=None, past_actions=None,
                   past_returns=None, past_timesteps=None,
                   target_return=1.0, current_timestep=0):
        with torch.no_grad():
            device = next(self.parameters()).device
            if past_states is None:
                sf = state_features.unsqueeze(0).unsqueeze(0).to(device)
                sm = state_mask.unsqueeze(0).unsqueeze(0).to(device)
                at = torch.zeros(1, 1, dtype=torch.long, device=device)
                ei = torch.zeros(1, 1, dtype=torch.long, device=device)
                rtg = torch.tensor([[target_return]], dtype=torch.float32, device=device)
                ts = torch.tensor([[current_timestep]], dtype=torch.long, device=device)
            else:
                K = min(len(past_states) + 1, self.context_length)
                start = max(0, len(past_states) + 1 - K)
                sf_list = past_states[start:] + [state_features]
                sm_list = (past_state_masks or [torch.ones_like(state_mask)] * len(past_states))[start:] + [state_mask]
                sf = torch.stack(sf_list).unsqueeze(0).to(device)
                sm = torch.stack(sm_list).unsqueeze(0).to(device)
                at_list = past_actions[start:] + [torch.tensor([0, 0])]
                at = torch.stack([a[0] for a in at_list]).unsqueeze(0).to(device)
                ei = torch.stack([a[1] for a in at_list]).unsqueeze(0).to(device)
                rtg = torch.tensor([past_returns[start:] + [target_return]], dtype=torch.float32, device=device)
                ts = torch.tensor([past_timesteps[start:] + [current_timestep]], dtype=torch.long, device=device)

            at_preds, el_preds = self.forward(sf, sm, at, ei, rtg, ts)
        return {
            "action_type_idx": at_preds[0, -1].argmax().item(),
            "element_idx": el_preds[0, -1].argmax().item(),
        }

    def compute_loss(self, batch):
        at_preds, el_preds = self.forward(
            batch["state_features"], batch["state_masks"],
            batch["action_types"], batch["element_idxs"],
            batch["returns_to_go"], batch["timesteps"], batch["attention_mask"])

        mask = batch["attention_mask"]
        at_loss = F.cross_entropy(
            at_preds.reshape(-1, at_preds.size(-1)),
            batch["action_types"].reshape(-1), reduction='none'
        ).reshape(mask.shape)
        at_loss = (at_loss * mask).sum() / mask.sum()

        el_loss = F.cross_entropy(
            el_preds.reshape(-1, el_preds.size(-1)),
            batch["element_idxs"].reshape(-1), reduction='none'
        ).reshape(mask.shape)
        el_loss = (el_loss * mask).sum() / mask.sum()

        at_acc = ((at_preds.argmax(-1) == batch["action_types"]).float() * mask).sum() / mask.sum()
        el_acc = ((el_preds.argmax(-1) == batch["element_idxs"]).float() * mask).sum() / mask.sum()

        return {
            "loss": at_loss + el_loss,
            "action_type_acc": at_acc.item(),
            "element_acc": el_acc.item(),
        }
