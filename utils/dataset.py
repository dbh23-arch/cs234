"""Dataset classes for offline RL training (trajectory and sequence formats)."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.text_features import dom_to_text, tokenize_page

def _demo_path(data_dir, task_name, source="human"):
    if source == "human":
        return os.path.join(data_dir, f"human_{task_name}.json")
    return os.path.join(data_dir, f"{task_name}.json")

def _load_trajectories(data_dir, task_name, source="human"):
    path = _demo_path(data_dir, task_name, source)
    if not os.path.exists(path):
        print(f"Warning: No data found at {path}")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data["trajectories"]

def _safe_step_rewards(traj, horizon):
    rewards = np.zeros(horizon, dtype=np.float32)
    raw = traj.get("rewards", [])
    n = min(len(raw), horizon)
    if n > 0:
        rewards[:n] = np.array(raw[:n], dtype=np.float32)
    return rewards

def _trajectory_return(traj):
    return float(np.sum(np.array(traj.get("rewards", []), dtype=np.float32)))

def _to_tensor(value, dtype=None):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if dtype is not None:
        return torch.tensor(value, dtype=dtype)
    return torch.tensor(value)

def _build_stitched_trajectories(trajectories, rng, stitch_prob=0.0,
                                 topk_frac=0.3):
    if stitch_prob <= 0.0 or len(trajectories) < 2:
        return []

    topk = max(2, int(np.ceil(len(trajectories) * topk_frac)))
    ranked = sorted(
        [t for t in trajectories if len(t.get("states", [])) >= 2],
        key=_trajectory_return,
        reverse=True,
    )[:topk]

    if len(ranked) < 2:
        return []

    n_synth = max(1, int(round(stitch_prob * len(trajectories))))
    stitched = []
    for _ in range(n_synth):
        i, j = rng.choice(len(ranked), size=2, replace=False)
        t1 = ranked[i]
        t2 = ranked[j]
        s1 = t1.get("states", [])
        s2 = t2.get("states", [])
        if len(s1) < 2 or len(s2) < 2:
            continue

        split1 = rng.randint(1, len(s1))
        split2 = rng.randint(0, len(s2) - 1)

        states = s1[:split1] + s2[split2:]
        actions = t1.get("actions", [])[:split1] + t2.get("actions", [])[split2:]
        rewards = t1.get("rewards", [])[:split1] + t2.get("rewards", [])[split2:]
        dones = t1.get("dones", [])[:split1] + t2.get("dones", [])[split2:]

        if len(states) < 2:
            continue

        stitched.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "_is_stitched": True,
        })

    return stitched

class TrajectoryDataset(Dataset):

    def __init__(self, data_dir, task_name, max_demos=None, seed=42,
                 source="human"):
        self.data_dir = data_dir
        self.task_name = task_name
        self.source = source

        self.trajectories = _load_trajectories(data_dir, task_name, source)

        # subsample if we only want a limited number of demos
        if max_demos is not None and max_demos < len(self.trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.trajectories), max_demos, replace=False)
            self.trajectories = [self.trajectories[i] for i in sorted(indices)]

        # flatten trajectories into (s, a, r, s') transitions
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

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            "state_features": _to_tensor(t["state_features"], dtype=torch.float32),
            "state_mask": _to_tensor(t["state_mask"], dtype=torch.float32),
            "action_type": _to_tensor(t["action_type"], dtype=torch.long),
            "element_idx": _to_tensor(t["element_idx"], dtype=torch.long),
            "reward": _to_tensor(t["reward"], dtype=torch.float32),
            "next_state_features": _to_tensor(t["next_state_features"], dtype=torch.float32),
            "next_state_mask": _to_tensor(t["next_state_mask"], dtype=torch.float32),
            "done": _to_tensor(t["done"], dtype=torch.float32),
        }

class SequenceDataset(Dataset):

    def __init__(self, data_dir, task_name, context_length=20,
                 max_demos=None, seed=42, source="human",
                 stitch_prob=0.0, stitch_topk_frac=0.3):
        self.context_length = context_length
        self.data_dir = data_dir
        self.source = source

        trajectories = _load_trajectories(data_dir, task_name, source)
        rng = np.random.RandomState(seed)

        if max_demos is not None and max_demos < len(trajectories):
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        stitched = _build_stitched_trajectories(
            trajectories, rng, stitch_prob=stitch_prob, topk_frac=stitch_topk_frac
        )
        if stitched:
            trajectories = trajectories + stitched

        # build all context windows for the decision transformer
        self.sequences = []
        for traj in trajectories:
            T = len(traj["states"])
            if T < 2:
                continue

            # compute returns-to-go (sum of future rewards from each timestep)
            rewards = _safe_step_rewards(traj, T)
            rtg = np.zeros(T, dtype=np.float32)
            rtg[-1] = rewards[-1]
            for t in range(T - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]

            for start in range(T - 1):
                end = min(start + context_length, T)
                self.sequences.append({
                    "states": traj["states"][start:end],
                    "actions": traj["actions"][start:end],
                    "rewards": rewards[start:end].tolist(),
                    "returns_to_go": rtg[start:end].tolist(),
                    "timesteps": list(range(start, end)),
                    "length": end - start,
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = seq["length"]
        K = self.context_length

        # pad to context_length (zero-padded, masked out in attention)
        state_features = np.zeros((K, 64, 24), dtype=np.float32)
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
            "state_features": _to_tensor(state_features, dtype=torch.float32),
            "state_masks": _to_tensor(state_masks, dtype=torch.float32),
            "action_types": _to_tensor(action_types, dtype=torch.long),
            "element_idxs": _to_tensor(element_idxs, dtype=torch.long),
            "returns_to_go": _to_tensor(returns_to_go, dtype=torch.float32),
            "timesteps": _to_tensor(timesteps, dtype=torch.long),
            "attention_mask": _to_tensor(attention_mask, dtype=torch.float32),
        }

class TextTrajectoryDataset(Dataset):

    def __init__(self, data_dir, task_name, tokenizer, max_demos=None,
                 seed=42, max_elements=64, max_length=512, source="human"):
        self.tokenizer = tokenizer
        self.max_elements = max_elements
        self.max_length = max_length
        self.data_dir = data_dir
        self.source = source

        trajectories = _load_trajectories(data_dir, task_name, source)

        if max_demos is not None and max_demos < len(trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        self.transitions = []
        for traj in trajectories:
            for t in range(len(traj["states"]) - 1):
                s = traj["states"][t]
                s_next = traj["states"][t + 1]
                
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
            "input_ids": _to_tensor(ids, dtype=torch.long),
            "attention_mask": _to_tensor(amask, dtype=torch.float32),
            "element_token_spans": _to_tensor(spans, dtype=torch.long),
            "element_mask": _to_tensor(emask, dtype=torch.float32),
            "next_input_ids": _to_tensor(n_ids, dtype=torch.long),
            "next_attention_mask": _to_tensor(n_amask, dtype=torch.float32),
            "next_element_token_spans": _to_tensor(n_spans, dtype=torch.long),
            "next_element_mask": _to_tensor(n_emask, dtype=torch.float32),
            "action_type": _to_tensor(t["action_type"], dtype=torch.long),
            "element_idx": _to_tensor(t["element_idx"], dtype=torch.long),
            "reward": _to_tensor(t["reward"], dtype=torch.float32),
            "done": _to_tensor(t["done"], dtype=torch.float32),
        }

class TextSequenceDataset(Dataset):

    def __init__(self, data_dir, task_name, tokenizer, context_length=20,
                 max_demos=None, seed=42, max_elements=64, max_length=512,
                 source="human", stitch_prob=0.0, stitch_topk_frac=0.3):
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.max_elements = max_elements
        self.max_length = max_length
        self.data_dir = data_dir
        self.source = source

        trajectories = _load_trajectories(data_dir, task_name, source)
        rng = np.random.RandomState(seed)

        if max_demos is not None and max_demos < len(trajectories):
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        stitched = _build_stitched_trajectories(
            trajectories, rng, stitch_prob=stitch_prob, topk_frac=stitch_topk_frac
        )
        if stitched:
            trajectories = trajectories + stitched

        self.sequences = []
        for traj in trajectories:
            T = len(traj["states"])
            if T < 2:
                continue
            
            if "dom_elements_raw" not in traj["states"][0]:
                continue

            rewards = _safe_step_rewards(traj, T)
            rtg = np.zeros(T, dtype=np.float32)
            rtg[-1] = rewards[-1]
            for t_idx in range(T - 2, -1, -1):
                rtg[t_idx] = rewards[t_idx] + rtg[t_idx + 1]

            for start in range(T - 1):
                end = min(start + context_length, T)
                self.sequences.append({
                    "states": traj["states"][start:end],
                    "actions": traj["actions"][start:end],
                    "rewards": rewards[start:end].tolist(),
                    "returns_to_go": rtg[start:end].tolist(),
                    "timesteps": list(range(start, end)),
                    "length": end - start,
                })

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
            "input_ids": _to_tensor(all_input_ids, dtype=torch.long),
            "text_attention_mask": _to_tensor(all_attn_masks, dtype=torch.float32),
            "element_token_spans": _to_tensor(all_elem_spans, dtype=torch.long),
            "element_masks": _to_tensor(all_elem_masks, dtype=torch.float32),
            "action_types": _to_tensor(action_types, dtype=torch.long),
            "element_idxs": _to_tensor(element_idxs, dtype=torch.long),
            "returns_to_go": _to_tensor(returns_to_go, dtype=torch.float32),
            "timesteps": _to_tensor(timesteps, dtype=torch.long),
            "attention_mask": _to_tensor(seq_attention_mask, dtype=torch.float32),
        }

def save_trajectories(trajectories, data_dir, task_name):
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
