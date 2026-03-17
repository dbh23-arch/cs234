"""Datasets for offline RL."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_trajectories(data_dir, task_name):
    path = os.path.join(data_dir, f"{task_name}.json")
    if not os.path.exists(path):
        print(f"Warning: No data found at {path}")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data["trajectories"]


def _trajectory_return(traj):
    return float(np.sum(np.array(traj.get("rewards", []), dtype=np.float32)))


class TrajectoryDataset(Dataset):

    def __init__(self, data_dir, task_name, max_demos=None, seed=42):
        trajectories = _load_trajectories(data_dir, task_name)

        if max_demos is not None and max_demos < len(trajectories):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        self.transitions = []
        for traj in trajectories:
            for t in range(len(traj["states"]) - 1):
                self.transitions.append({
                    "state_features": np.array(traj["states"][t]["features"], dtype=np.float32),
                    "state_mask": np.array(traj["states"][t]["mask"], dtype=np.float32),
                    "action_type": traj["actions"][t]["action_type_idx"],
                    "element_idx": min(traj["actions"][t]["element_idx"], 63),
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
            "state_features": torch.tensor(t["state_features"], dtype=torch.float32),
            "state_mask": torch.tensor(t["state_mask"], dtype=torch.float32),
            "action_type": torch.tensor(t["action_type"], dtype=torch.long),
            "element_idx": torch.tensor(t["element_idx"], dtype=torch.long),
            "reward": torch.tensor(t["reward"], dtype=torch.float32),
            "next_state_features": torch.tensor(t["next_state_features"], dtype=torch.float32),
            "next_state_mask": torch.tensor(t["next_state_mask"], dtype=torch.float32),
            "done": torch.tensor(t["done"], dtype=torch.float32),
        }


class SequenceDataset(Dataset):

    def __init__(self, data_dir, task_name, context_length=20,
                 max_demos=None, seed=42):
        self.context_length = context_length
        trajectories = _load_trajectories(data_dir, task_name)
        rng = np.random.RandomState(seed)

        if max_demos is not None and max_demos < len(trajectories):
            indices = rng.choice(len(trajectories), max_demos, replace=False)
            trajectories = [trajectories[i] for i in sorted(indices)]

        self.sequences = []
        for traj in trajectories:
            T = len(traj["states"])
            if T < 2:
                continue

            rewards = np.zeros(T, dtype=np.float32)
            raw = traj.get("rewards", [])
            n = min(len(raw), T)
            if n > 0:
                rewards[:n] = np.array(raw[:n], dtype=np.float32)

            rtg = np.zeros(T, dtype=np.float32)
            rtg[-1] = rewards[-1]
            for t in range(T - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]

            for start in range(T - 1):
                end = min(start + context_length, T)
                self.sequences.append({
                    "states": traj["states"][start:end],
                    "actions": traj["actions"][start:end],
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
                element_idxs[t] = min(seq["actions"][t]["element_idx"], 63)
            returns_to_go[t] = seq["returns_to_go"][t]
            timesteps[t] = seq["timesteps"][t]
            attention_mask[t] = 1.0

        return {
            "state_features": torch.tensor(state_features, dtype=torch.float32),
            "state_masks": torch.tensor(state_masks, dtype=torch.float32),
            "action_types": torch.tensor(action_types, dtype=torch.long),
            "element_idxs": torch.tensor(element_idxs, dtype=torch.long),
            "returns_to_go": torch.tensor(returns_to_go, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }
