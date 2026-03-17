"""Training script for all methods."""

import os
import re
import json
import argparse
import yaml
import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
from models.ppo import (PPOAgent, get_returns, calculate_advantage,
                        update_policy, update_value)
from utils.dataset import TrajectoryDataset, SequenceDataset
from utils.state_encoder import extract_dom_features


def get_device(config_device="auto"):
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


# --- BC ---

def train_bc(config, task_name, num_demos, seed, device):
    dataset = TrajectoryDataset(
        config["data"]["save_dir"], task_name, max_demos=num_demos, seed=seed)
    if len(dataset) == 0:
        print(f"  No data for {task_name}, skipping")
        return None

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=False)
    agent = BCAgent(
        state_dim=config["state"]["state_dim"],
        hidden_dim=config["bc"]["hidden_dim"],
        max_elements=config["state"]["max_dom_elements"],
    ).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        agent.train()
        total_loss, total_at, total_el, n = 0, 0, 0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)
            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            bs = len(batch["reward"])
            total_loss += metrics["loss"].item() * bs
            total_at += metrics["action_type_acc"] * bs
            total_el += metrics["element_acc"] * bs
            n += bs

        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n:.4f}  "
                  f"at_acc={total_at/n:.3f}  el_acc={total_el/n:.3f}")
    return agent


# --- IQL ---

def train_iql(config, task_name, num_demos, seed, device):
    dataset = TrajectoryDataset(
        config["data"]["save_dir"], task_name, max_demos=num_demos, seed=seed)
    if len(dataset) == 0:
        print(f"  No data for {task_name}, skipping")
        return None

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=False)
    agent = IQLAgent(
        state_dim=config["state"]["state_dim"],
        hidden_dim=config["iql"]["hidden_dim"],
        max_elements=config["state"]["max_dom_elements"],
        discount=config["iql"]["discount"],
        tau=config["iql"]["tau"],
        beta=config["iql"]["beta"],
        target_update_rate=config["iql"]["target_update_rate"],
    ).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        agent.train()
        total_loss, total_v, total_q, total_p, n = 0, 0, 0, 0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)
            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            agent.update_targets()
            bs = len(batch["reward"])
            total_loss += metrics["loss"].item() * bs
            total_v += metrics["value_loss"] * bs
            total_q += metrics["q_loss"] * bs
            total_p += metrics["policy_loss"] * bs
            n += bs

        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n:.4f}  "
                  f"v={total_v/n:.4f}  q={total_q/n:.4f}  pi={total_p/n:.4f}")
    return agent


# --- DT ---

def train_dt(config, task_name, num_demos, seed, device):
    dataset = SequenceDataset(
        config["data"]["save_dir"], task_name,
        context_length=config["dt"]["context_length"],
        max_demos=num_demos, seed=seed)
    if len(dataset) == 0:
        print(f"  No data for {task_name}, skipping")
        return None

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=False)
    agent = DecisionTransformerAgent(
        state_dim=config["state"]["state_dim"],
        embed_dim=config["dt"]["embed_dim"],
        n_layers=config["dt"]["n_layers"],
        n_heads=config["dt"]["n_heads"],
        context_length=config["dt"]["context_length"],
        max_elements=config["state"]["max_dom_elements"],
        dropout=config["dt"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        agent.train()
        total_loss, total_at, total_el, n = 0, 0, 0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)
            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            bs = batch["attention_mask"].sum().item()
            total_loss += metrics["loss"].item() * bs
            total_at += metrics["action_type_acc"] * bs
            total_el += metrics["element_acc"] * bs
            n += bs

        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n:.4f}  "
                  f"at_acc={total_at/n:.3f}  el_acc={total_el/n:.3f}")
    return agent


# --- PPO ---

def train_ppo(config, task_name, seed, device):
    import gymnasium as gym
    import miniwob
    from miniwob.action import ActionTypes

    ppo_cfg = config["ppo"]
    gamma = ppo_cfg["gamma"]
    eps_clip = ppo_cfg["eps_clip"]
    update_freq = ppo_cfg["update_freq"]
    batch_size = ppo_cfg["batch_size"]
    max_ep_len = ppo_cfg["max_ep_len"]
    num_batches = ppo_cfg["num_batches"]
    entropy_coeff = ppo_cfg.get("entropy_coeff", 0.02)
    step_penalty = ppo_cfg.get("step_penalty", 0.01)
    lr = config["training"]["lr"]

    def parse_dom_elements(obs):
        elements = []
        for elem in obs.get("dom_elements", []):
            def to_float(v):
                return v.item() if hasattr(v, 'item') else float(v or 0)
            elements.append({
                "tag": str(elem.get("tag", "div")),
                "type": str(elem.get("type", "")),
                "text": str(elem.get("text", "")),
                "value": str(elem.get("value", "")),
                "left": to_float(elem.get("left", 0)),
                "top": to_float(elem.get("top", 0)),
                "width": to_float(elem.get("width", 0)),
                "height": to_float(elem.get("height", 0)),
                "visible": bool(elem.get("visible", True)),
                "focused": bool(elem.get("focused", False)),
                "ref": int(elem.get("ref", 0)) if elem.get("ref") is not None else 0,
            })
        return elements

    def element_center(elem):
        cx = max(0.0, min(elem["left"] + elem["width"] / 2, 160.0))
        cy = max(0.0, min(elem["top"] + elem["height"] / 2, 210.0))
        return cx, cy

    def is_input_like(elem):
        tag = str(elem.get("tag", "")).lower()
        typ = str(elem.get("type", "")).lower()
        if tag in ("input_text", "input_password", "input_search", "textarea"):
            return True
        if tag == "input":
            return typ in ("", "text", "password", "search", "email", "url", "tel")
        return False

    def is_clickable(elem):
        tag = str(elem.get("tag", "")).lower()
        if int(elem.get("ref", 0)) <= 0:
            return False
        return tag in {"button", "a", "span", "label", "option", "select",
                       "input", "li"} or tag.startswith("input_")

    def resolve_target(elements, idx, predicate):
        if 0 <= idx < len(elements) and predicate(elements[idx]):
            return idx
        if not elements:
            return idx
        sx, sy = element_center(elements[idx]) if 0 <= idx < len(elements) else (80.0, 105.0)
        candidates = [i for i, e in enumerate(elements) if predicate(e)]
        if not candidates:
            return idx
        return min(candidates,
                   key=lambda i: (element_center(elements[i])[0] - sx)**2 +
                                 (element_center(elements[i])[1] - sy)**2)

    def infer_type_text(utterance, elements, idx):
        elem = elements[idx] if 0 <= idx < len(elements) else {}
        is_password = str(elem.get("type", "")).lower() == "password" or \
                      str(elem.get("tag", "")).lower() == "input_password"
        quoted = re.findall(r'"([^"]*)"', utterance)
        user_q = re.search(r'username[^"]*"([^"]+)"', utterance, re.IGNORECASE)
        pass_q = re.search(r'password[^"]*"([^"]+)"', utterance, re.IGNORECASE)
        if is_password and pass_q:
            return pass_q.group(1).strip()
        if not is_password and user_q:
            return user_q.group(1).strip()
        if quoted:
            return quoted[1] if is_password and len(quoted) >= 2 else quoted[0]
        m = re.search(r'(?:enter|type|input|search for)\s+(.+)', utterance, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('"').rstrip('.')
        return utterance.strip()

    def make_env_action(env, elements, action_type_idx, element_idx, utterance):
        action_types = env.unwrapped.action_space_config.action_types
        if action_type_idx == 0:
            element_idx = resolve_target(elements, element_idx, is_clickable)
        else:
            element_idx = resolve_target(elements, element_idx, is_input_like)

        elem = elements[element_idx] if 0 <= element_idx < len(elements) else {}
        cx, cy = element_center(elem) if elem else (80.0, 105.0)
        ref = int(elem.get("ref", 0))

        act = OrderedDict()
        act["ref"] = np.int64(ref if ref > 0 else 0)
        act["coords"] = np.array([cx, cy], dtype=np.float32)
        act["text"] = ""
        act["field"] = np.int64(0)
        act["key"] = np.int64(0)
        if action_type_idx == 0:
            if ref > 0 and ActionTypes.CLICK_ELEMENT in action_types:
                act["action_type"] = np.int64(action_types.index(ActionTypes.CLICK_ELEMENT))
            else:
                act["action_type"] = np.int64(action_types.index(ActionTypes.CLICK_COORDS))
        else:
            act["text"] = infer_type_text(utterance, elements, element_idx)
            if ref > 0 and ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT in action_types:
                act["action_type"] = np.int64(action_types.index(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT))
            else:
                act["action_type"] = np.int64(action_types.index(ActionTypes.TYPE_TEXT))
        return act

    # training loop

    agent = PPOAgent(
        state_dim=config["state"]["state_dim"],
        hidden_dim=ppo_cfg["hidden_dim"],
        max_elements=config["state"]["max_dom_elements"],
    ).to(device)

    if pretrain_checkpoint is not None and pretrain_method is not None:
        load_pretrain_weights(agent, pretrain_checkpoint, pretrain_method)

    policy_params = (list(agent.state_encoder.parameters()) +
                     list(agent.action_type_head.parameters()) +
                     list(agent.element_score.parameters()))
    value_params = list(agent.value_head.parameters())
    optimizer_pi = torch.optim.Adam(policy_params, lr=lr)
    optimizer_v = torch.optim.Adam(value_params, lr=lr)

    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)

    for t in range(num_batches):
        paths = []
        episode_rewards = []
        episode = 0
        steps = 0

        while steps < batch_size:
            try:
                obs, info = env.reset(seed=seed + episode)
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass
                env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
                obs, info = env.reset(seed=seed + episode)

            utterance = obs.get("utterance", "")
            elements = parse_dom_elements(obs)
            features, mask = extract_dom_features(elements, utterance)

            ep_features, ep_masks = [], []
            ep_at, ep_ei, ep_lp, ep_rewards = [], [], [], []
            ep_reward = 0

            for step in range(max_ep_len):
                ep_features.append(features)
                ep_masks.append(mask)
                f_t = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    at, ei, lp = agent.act(f_t, m_t, return_log_prob=True)

                env_action = make_env_action(env, elements, at, ei, utterance)
                try:
                    obs, reward, done, truncated, info = env.step(env_action)
                except Exception:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
                    break

                reward -= step_penalty
                ep_at.append(at)
                ep_ei.append(ei)
                ep_lp.append(lp)
                ep_rewards.append(reward)
                ep_reward += reward
                steps += 1

                if done or truncated:
                    break
                if steps >= batch_size:
                    break

                utterance = obs.get("utterance", utterance)
                elements = parse_dom_elements(obs)
                features, mask = extract_dom_features(elements, utterance)

            vl = min(len(ep_features), len(ep_at), len(ep_rewards))
            if vl > 0:
                paths.append({
                    "features": np.array(ep_features[:vl]),
                    "masks": np.array(ep_masks[:vl]),
                    "action_types": np.array(ep_at[:vl]),
                    "element_idxs": np.array(ep_ei[:vl]),
                    "old_logprobs": np.array(ep_lp[:vl]),
                    "reward": np.array(ep_rewards[:vl]),
                })
                episode_rewards.append(ep_reward)
            episode += 1

        if not paths:
            continue

        all_features = np.concatenate([p["features"] for p in paths])
        all_masks = np.concatenate([p["masks"] for p in paths])
        all_at = np.concatenate([p["action_types"] for p in paths])
        all_ei = np.concatenate([p["element_idxs"] for p in paths])
        old_lp = np.concatenate([p["old_logprobs"] for p in paths])

        returns = get_returns(paths, gamma)
        advantages = calculate_advantage(returns, all_features, all_masks,
                                         agent, device)

        for _ in range(update_freq):
            update_value(agent, optimizer_v, all_features, all_masks, returns, device)
            update_policy(agent, optimizer_pi, all_features, all_masks,
                          all_at, all_ei, advantages, old_lp, eps_clip, device,
                          entropy_coeff=entropy_coeff)

        avg_r = np.mean(episode_rewards)
        std_r = np.std(episode_rewards) / max(len(episode_rewards)**0.5, 1)
        print(f"    Batch {t+1}/{num_batches}: reward={avg_r:.3f} +/- {std_r:.3f}  "
              f"({len(episode_rewards)} eps)", flush=True)

    env.close()
    return agent


# --- Main ---

TRAIN_FNS = {"bc": train_bc, "iql": train_iql, "dt": train_dt}
def load_pretrain_weights(ppo_agent, checkpoint_path, pretrain_method):
    sd = torch.load(checkpoint_path, map_location="cpu")
    ppo_sd = ppo_agent.state_dict()
    for k in sd:
        if k.startswith("state_encoder."):
            ppo_sd[k] = sd[k]
    if pretrain_method == "bc":
        for k in ("action_type_head.0.weight", "action_type_head.0.bias",
                  "action_type_head.2.weight", "action_type_head.2.bias",
                  "element_score.0.weight", "element_score.0.bias",
                  "element_score.2.weight", "element_score.2.bias"):
            if k in sd:
                ppo_sd[k] = sd[k]
    elif pretrain_method == "iql":
        key_map = {
            "policy_action_type.0.weight": "action_type_head.0.weight",
            "policy_action_type.0.bias": "action_type_head.0.bias",
            "policy_action_type.2.weight": "action_type_head.2.weight",
            "policy_action_type.2.bias": "action_type_head.2.bias",
            "policy_element.0.weight": "element_score.0.weight",
            "policy_element.0.bias": "element_score.0.bias",
            "policy_element.2.weight": "element_score.2.weight",
            "policy_element.2.bias": "element_score.2.bias",
        }
        for iql_k, ppo_k in key_map.items():
            if iql_k in sd:
                ppo_sd[ppo_k] = sd[iql_k]
    ppo_agent.load_state_dict(ppo_sd)

def run_experiment(config, method, task_name, num_demos, seed, device,
                   encoder_type="dom", source="human",
                   pretrain_checkpoint=None, pretrain_method=None):
    enc_label = f" [{encoder_type.upper()}]" if encoder_type != "dom" else ""
    src_label = f" [{source}]"
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}{enc_label}{src_label} | Task: {task_name} | "
          f"Demos: {num_demos} | Seed: {seed}")
    print(f"{'='*60}")

    train_fn = TRAIN_FNS[method]
    if method == "ppo":
        agent = train_fn(config, task_name, num_demos, seed, device, encoder_type,
                         pretrain_checkpoint=pretrain_checkpoint,
                         pretrain_method=pretrain_method)
    else:
        agent = train_fn(config, task_name, num_demos, seed, device,
                         encoder_type, source=source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default="bc",
                        choices=["bc", "iql", "dt", "ppo", "all"])
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to train on (default: all)")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Single demo count (default: sweep all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: sweep all)")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--encoder_type", type=str, default="dom",
                        choices=["dom", "text"],
                        help="State encoder type (dom=hand-crafted, text=DistilBERT)")
    parser.add_argument("--source", type=str, default="human",
                        choices=["human", "heuristic"],
                        help="Demo source: human (Stanford) or heuristic (scripted)")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                        help="Path to BC or IQL checkpoint to warm-start PPO policy")
    parser.add_argument("--pretrain_method", type=str, default=None,
                        choices=["bc", "iql"],
                        help="Method that produced the pretrain checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    print(f"Device: {device}")

    methods = ["bc", "iql", "dt", "ppo"] if args.method == "all" else [args.method]
    if args.task:
        tasks = [args.task]
    else:
        tasks = (config["env"]["tasks"]["simple"] +
                 config["env"]["tasks"]["medium"] +
                 config["env"]["tasks"]["hard"])
    demo_sizes = [args.num_demos] if args.num_demos else config["data"]["data_sizes"]
    seeds = [args.seed] if args.seed else config["training"]["seeds"]

    results = {}
    for method in methods:
        for task in tasks:
            sizes = [0] if method == "ppo" else demo_sizes
            for n_demos in sizes:
                for seed in seeds:
                    print(f"\n{'='*60}")
                    print(f"{method.upper()} | {task} | demos={n_demos} | seed={seed}")
                    print(f"{'='*60}")

                    if method == "ppo":
                        agent = train_ppo(config, task, seed, device)
                    else:
                        agent = TRAIN_FNS[method](config, task, n_demos, seed, device)

                    if agent is not None:
                        path = os.path.join(
                            args.save_dir, "models",
                            f"{method}_{task}_n{n_demos}_s{seed}.pt")
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        torch.save(agent.state_dict(), path)
                        results[f"{method}_{task}_n{n_demos}_s{seed}"] = {
                            "method": method, "task": task,
                            "num_demos": n_demos, "seed": seed,
                        }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "experiment_index.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
