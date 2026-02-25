"""
Dataset utilities for storing and loading demonstration trajectories.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.text_features import dom_to_text, tokenize_page


class TrajectoryDataset(Dataset):
    """
    Dataset of demonstration trajectories for offline RL.

    Each trajectory is a sequence of (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, data_dir, task_name, max_demos=None, seed=42):
        self.data_dir = data_dir
        self.task_name = task_name

        # Load trajectories
        self.trajectories = self._load_trajectories(task_name)

        # Subsample if requested
        if max_demos is not None and max_demos < len(self.trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.trajectories), max_demos, replace=False)
            self.trajectories = [self.trajectories[i] for i in sorted(indices)]

        # Flatten into transitions for BC/IQL training
        self.transitions = []
        for traj in self.trajectories:
            for t in range(len(traj["states"]) - 1):
                self.transitions.append({
                    "state_features": np.array(traj["states"][t]["features"], dtype=np.float32),
                    "state_mask": np.array(traj["states"][t]["mask"], dtype=np.float32),
                    "action_type": traj["actions"][t]["action_type_idx"],
                    "element_idx": traj["actions"][t]["element_idx"],
                    "reward": traj["rewards"][t],
                    "next_state_features": np.array(traj["states"][t + 1]["features"], dtype=np.float32),
                    "next_state_mask": np.array(traj["states"][t + 1]["mask"], dtype=np.float32),
                    "done": traj["dones"][t],
                })

    def _load_trajectories(self, task_name):
        """Load trajectories from disk."""
        path = os.path.join(self.data_dir, f"{task_name}.json")
        if not os.path.exists(path):
            print(f"Warning: No data found at {path}")
            return []

        with open(path, "r") as f:
            data = json.load(f)

        return data["trajectories"]

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            "state_features": torch.tensor(t["state_features"]),
            "state_mask": torch.tensor(t["state_mask"]),
            "action_type": torch.tensor(t["action_type"], dtype=torch.long),
            "element_idx": torch.tensor(t["element_idx"], dtype=torch.long),
            "reward": torch.tensor(t["reward"], dtype=torch.float32),
            "next_state_features": torch.tensor(t["next_state_features"]),
            "next_state_mask": torch.tensor(t["next_state_mask"]),
            "done": torch.tensor(t["done"], dtype=torch.float32),
        }


class SequenceDataset(Dataset):
    """
    Dataset that returns full trajectory sequences for Decision Transformer.
    Pads/truncates to a fixed context length.
    """

    def __init__(self, data_dir, task_name, context_length=20,
                 max_demos=None, seed=42):
        self.context_length = context_length
        self.data_dir = data_dir

        # Load trajectories
        trajectories = self._load_trajectories(task_name)

        if max_demos is not None and max_demos < len(trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        # Process into sequences
        self.sequences = []
        for traj in trajectories:
            T = len(traj["states"])
            if T < 2:
                continue

            # Compute returns-to-go
            rewards = traj["rewards"]
            rtg = np.zeros(T, dtype=np.float32)
            rtg[-1] = rewards[-1] if len(rewards) >= T else 0
            for t in range(T - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]

            # Create subsequences
            for start in range(T - 1):
                end = min(start + context_length, T)
                self.sequences.append({
                    "states": traj["states"][start:end],
                    "actions": traj["actions"][start:end],
                    "rewards": rewards[start:end],
                    "returns_to_go": rtg[start:end].tolist(),
                    "timesteps": list(range(start, end)),
                    "length": end - start,
                })

    def _load_trajectories(self, task_name):
        """Load trajectories from disk."""
        path = os.path.join(self.data_dir, f"{task_name}.json")
        if not os.path.exists(path):
            print(f"Warning: No data found at {path}")
            return []
        with open(path, "r") as f:
            return json.load(f)["trajectories"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = seq["length"]
        K = self.context_length

        # Pad sequences to context_length
        state_features = np.zeros((K, 64, 24), dtype=np.float32)  # max_elements=64, feature_dim=24
        state_masks = np.zeros((K, 64), dtype=np.float32)
        action_types = np.zeros(K, dtype=np.int64)
        element_idxs = np.zeros(K, dtype=np.int64)
        returns_to_go = np.zeros(K, dtype=np.float32)
        timesteps = np.zeros(K, dtype=np.int64)
        attention_mask = np.zeros(K, dtype=np.float32)

        for t in range(L):
            state_features[t] = np.array(seq["states"][t]["features"])
            state_masks[t] = np.array(seq["states"][t]["mask"])
            if t < len(seq["actions"]):
                action_types[t] = seq["actions"][t]["action_type_idx"]
                element_idxs[t] = seq["actions"][t]["element_idx"]
            returns_to_go[t] = seq["returns_to_go"][t]
            timesteps[t] = seq["timesteps"][t]
            attention_mask[t] = 1.0

        return {
            "state_features": torch.tensor(state_features),
            "state_masks": torch.tensor(state_masks),
            "action_types": torch.tensor(action_types),
            "element_idxs": torch.tensor(element_idxs),
            "returns_to_go": torch.tensor(returns_to_go),
            "timesteps": torch.tensor(timesteps),
            "attention_mask": torch.tensor(attention_mask),
        }


class TextTrajectoryDataset(Dataset):
    """
    Dataset of transitions for BC/IQL with LM (text) encoder.
    Tokenizes DOM text on-the-fly from stored raw DOM elements.
    """

    def __init__(self, data_dir, task_name, tokenizer, max_demos=None,
                 seed=42, max_elements=64, max_length=512):
        self.tokenizer = tokenizer
        self.max_elements = max_elements
        self.max_length = max_length
        self.data_dir = data_dir

        trajectories = self._load_trajectories(task_name)

        if max_demos is not None and max_demos < len(trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        # Flatten into transitions
        self.transitions = []
        for traj in trajectories:
            for t in range(len(traj["states"]) - 1):
                s = traj["states"][t]
                s_next = traj["states"][t + 1]
                # Need raw DOM elements and utterance
                if "dom_elements_raw" not in s:
                    continue
                self.transitions.append({
                    "dom_elements_raw": s["dom_elements_raw"],
                    "utterance": s.get("utterance", ""),
                    "next_dom_elements_raw": s_next["dom_elements_raw"],
                    "next_utterance": s_next.get("utterance", ""),
                    "action_type": traj["actions"][t]["action_type_idx"],
                    "element_idx": traj["actions"][t]["element_idx"],
                    "reward": traj["rewards"][t],
                    "done": traj["dones"][t],
                })

    def _load_trajectories(self, task_name):
        path = os.path.join(self.data_dir, f"{task_name}.json")
        if not os.path.exists(path):
            print(f"Warning: No data found at {path}")
            return []
        with open(path, "r") as f:
            return json.load(f)["trajectories"]

    def __len__(self):
        return len(self.transitions)

    def _tokenize_state(self, dom_elements_raw, utterance):
        text, char_spans = dom_to_text(
            dom_elements_raw, utterance, self.max_elements
        )
        input_ids, attn_mask, elem_tok_spans, elem_mask = tokenize_page(
            text, char_spans, self.tokenizer, self.max_elements, self.max_length
        )
        return input_ids, attn_mask, elem_tok_spans, elem_mask

    def __getitem__(self, idx):
        t = self.transitions[idx]

        ids, amask, spans, emask = self._tokenize_state(
            t["dom_elements_raw"], t["utterance"]
        )
        n_ids, n_amask, n_spans, n_emask = self._tokenize_state(
            t["next_dom_elements_raw"], t["next_utterance"]
        )

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(amask, dtype=torch.float32),
            "element_token_spans": torch.tensor(spans, dtype=torch.long),
            "element_mask": torch.tensor(emask, dtype=torch.float32),
            "next_input_ids": torch.tensor(n_ids, dtype=torch.long),
            "next_attention_mask": torch.tensor(n_amask, dtype=torch.float32),
            "next_element_token_spans": torch.tensor(n_spans, dtype=torch.long),
            "next_element_mask": torch.tensor(n_emask, dtype=torch.float32),
            "action_type": torch.tensor(t["action_type"], dtype=torch.long),
            "element_idx": torch.tensor(t["element_idx"], dtype=torch.long),
            "reward": torch.tensor(t["reward"], dtype=torch.float32),
            "done": torch.tensor(t["done"], dtype=torch.float32),
        }


class TextSequenceDataset(Dataset):
    """
    Sequence dataset for Decision Transformer with LM (text) encoder.
    Tokenizes DOM text on-the-fly from stored raw DOM elements.
    """

    def __init__(self, data_dir, task_name, tokenizer, context_length=20,
                 max_demos=None, seed=42, max_elements=64, max_length=512):
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.max_elements = max_elements
        self.max_length = max_length
        self.data_dir = data_dir

        trajectories = self._load_trajectories(task_name)

        if max_demos is not None and max_demos < len(trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        self.sequences = []
        for traj in trajectories:
            T = len(traj["states"])
            if T < 2:
                continue
            # Check that raw DOM data exists
            if "dom_elements_raw" not in traj["states"][0]:
                continue

            rewards = traj["rewards"]
            rtg = np.zeros(T, dtype=np.float32)
            rtg[-1] = rewards[-1] if len(rewards) >= T else 0
            for t_idx in range(T - 2, -1, -1):
                rtg[t_idx] = rewards[t_idx] + rtg[t_idx + 1]

            for start in range(T - 1):
                end = min(start + context_length, T)
                self.sequences.append({
                    "states": traj["states"][start:end],
                    "actions": traj["actions"][start:end],
                    "rewards": rewards[start:end],
                    "returns_to_go": rtg[start:end].tolist(),
                    "timesteps": list(range(start, end)),
                    "length": end - start,
                })

    def _load_trajectories(self, task_name):
        path = os.path.join(self.data_dir, f"{task_name}.json")
        if not os.path.exists(path):
            print(f"Warning: No data found at {path}")
            return []
        with open(path, "r") as f:
            return json.load(f)["trajectories"]

    def __len__(self):
        return len(self.sequences)

    def _tokenize_state(self, state):
        text, char_spans = dom_to_text(
            state["dom_elements_raw"], state.get("utterance", ""),
            self.max_elements
        )
        input_ids, attn_mask, elem_tok_spans, elem_mask = tokenize_page(
            text, char_spans, self.tokenizer, self.max_elements, self.max_length
        )
        return input_ids, attn_mask, elem_tok_spans, elem_mask

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = seq["length"]
        K = self.context_length
        ME = self.max_elements
        ML = self.max_length

        # Pre-allocate padded arrays
        all_input_ids = np.zeros((K, ML), dtype=np.int64)
        all_attn_masks = np.zeros((K, ML), dtype=np.float32)
        all_elem_spans = np.zeros((K, ME, 2), dtype=np.int64)
        all_elem_masks = np.zeros((K, ME), dtype=np.float32)
        action_types = np.zeros(K, dtype=np.int64)
        element_idxs = np.zeros(K, dtype=np.int64)
        returns_to_go = np.zeros(K, dtype=np.float32)
        timesteps = np.zeros(K, dtype=np.int64)
        seq_attention_mask = np.zeros(K, dtype=np.float32)

        for t in range(L):
            ids, amask, spans, emask = self._tokenize_state(seq["states"][t])
            all_input_ids[t] = ids
            all_attn_masks[t] = amask
            all_elem_spans[t] = spans
            all_elem_masks[t] = emask
            if t < len(seq["actions"]):
                action_types[t] = seq["actions"][t]["action_type_idx"]
                element_idxs[t] = seq["actions"][t]["element_idx"]
            returns_to_go[t] = seq["returns_to_go"][t]
            timesteps[t] = seq["timesteps"][t]
            seq_attention_mask[t] = 1.0

        return {
            "input_ids": torch.tensor(all_input_ids),
            "text_attention_mask": torch.tensor(all_attn_masks),
            "element_token_spans": torch.tensor(all_elem_spans),
            "element_masks": torch.tensor(all_elem_masks),
            "action_types": torch.tensor(action_types),
            "element_idxs": torch.tensor(element_idxs),
            "returns_to_go": torch.tensor(returns_to_go),
            "timesteps": torch.tensor(timesteps),
            "attention_mask": torch.tensor(seq_attention_mask),
        }


def save_trajectories(trajectories, data_dir, task_name):
    """Save collected trajectories to disk."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{task_name}.json")

    data = {
        "task_name": task_name,
        "num_trajectories": len(trajectories),
        "trajectories": trajectories,
    }

    with open(path, "w") as f:
        json.dump(data, f)

    print(f"Saved {len(trajectories)} trajectories to {path}")
