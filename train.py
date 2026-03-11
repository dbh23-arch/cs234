"""
Main training script for BC, IQL, and Decision Transformer.
Runs experiments across methods, tasks, and dataset sizes.
Supports both DOM (hand-crafted) and text (DistilBERT) state encoders.
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
import gymnasium as gym
import miniwob
from collections import OrderedDict
from miniwob.action import ActionTypes
from models.ppo import (PPOAgent, get_returns, calculate_advantage,
                            update_policy, update_value)
from utils.dataset import TrajectoryDataset, SequenceDataset
from utils.dataset import TextTrajectoryDataset, TextSequenceDataset
from utils.state_encoder import extract_dom_features


def get_device(config_device="auto"):
    """Select compute device."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def get_tokenizer(encoder_type):
    """Get tokenizer if using text encoder."""
    if encoder_type == "text":
        from utils.text_state_encoder import get_tokenizer
        return get_tokenizer()
    return None


def train_bc(config, task_name, num_demos, seed, device, encoder_type="dom"):
    """Train BC agent on a task."""
    if encoder_type == "text":
        tokenizer = get_tokenizer(encoder_type)
        dataset = TextTrajectoryDataset(
            config["data"]["save_dir"], task_name,
            tokenizer=tokenizer, max_demos=num_demos, seed=seed,
        )
    else:
        dataset = TrajectoryDataset(
            config["data"]["save_dir"], task_name,
            max_demos=num_demos, seed=seed,
        )

    if len(dataset) == 0:
        print(f"  No data for {task_name}, skipping")
        return None

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=False)

    agent = BCAgent(
        state_dim=config["state"]["state_dim"],
        hidden_dim=config["bc"]["hidden_dim"],
        max_elements=config["state"]["max_dom_elements"],
        encoder_type=encoder_type,
        freeze_lm=config.get("encoder", {}).get("freeze_lm", True),
    ).to(device)

    # Only optimize non-frozen params
    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"])

    best_loss = float('inf')
    for epoch in range(config["training"]["epochs"]):
        agent.train()
        epoch_metrics = {"loss": 0, "action_type_acc": 0, "element_acc": 0, "n": 0}

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)

            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_metrics["loss"] += metrics["loss"].item() * len(batch["reward"])
            epoch_metrics["action_type_acc"] += metrics["action_type_acc"] * len(batch["reward"])
            epoch_metrics["element_acc"] += metrics["element_acc"] * len(batch["reward"])
            epoch_metrics["n"] += len(batch["reward"])

        n = epoch_metrics["n"]
        avg_loss = epoch_metrics["loss"] / n

        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, "
                  f"at_acc={epoch_metrics['action_type_acc']/n:.3f}, "
                  f"el_acc={epoch_metrics['element_acc']/n:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    return agent


def train_iql(config, task_name, num_demos, seed, device, encoder_type="dom"):
    """Train IQL agent on a task."""
    if encoder_type == "text":
        tokenizer = get_tokenizer(encoder_type)
        dataset = TextTrajectoryDataset(
            config["data"]["save_dir"], task_name,
            tokenizer=tokenizer, max_demos=num_demos, seed=seed,
        )
    else:
        dataset = TrajectoryDataset(
            config["data"]["save_dir"], task_name,
            max_demos=num_demos, seed=seed,
        )

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
        encoder_type=encoder_type,
        freeze_lm=config.get("encoder", {}).get("freeze_lm", True),
    ).to(device)

    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        agent.train()
        epoch_metrics = {"loss": 0, "v_loss": 0, "q_loss": 0, "p_loss": 0, "n": 0}

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)

            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            agent.update_targets()

            epoch_metrics["loss"] += metrics["loss"].item() * len(batch["reward"])
            epoch_metrics["v_loss"] += metrics["value_loss"] * len(batch["reward"])
            epoch_metrics["q_loss"] += metrics["q_loss"] * len(batch["reward"])
            epoch_metrics["p_loss"] += metrics["policy_loss"] * len(batch["reward"])
            epoch_metrics["n"] += len(batch["reward"])

        n = epoch_metrics["n"]
        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={epoch_metrics['loss']/n:.4f}, "
                  f"v={epoch_metrics['v_loss']/n:.4f}, "
                  f"q={epoch_metrics['q_loss']/n:.4f}, "
                  f"pi={epoch_metrics['p_loss']/n:.4f}")

    return agent


def train_dt(config, task_name, num_demos, seed, device, encoder_type="dom"):
    """Train Decision Transformer on a task."""
    if encoder_type == "text":
        tokenizer = get_tokenizer(encoder_type)
        dataset = TextSequenceDataset(
            config["data"]["save_dir"], task_name,
            tokenizer=tokenizer,
            context_length=config["dt"]["context_length"],
            max_demos=num_demos, seed=seed,
        )
    else:
        dataset = SequenceDataset(
            config["data"]["save_dir"], task_name,
            context_length=config["dt"]["context_length"],
            max_demos=num_demos, seed=seed,
        )

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
        encoder_type=encoder_type,
        freeze_lm=config.get("encoder", {}).get("freeze_lm", True),
    ).to(device)

    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        agent.train()
        epoch_metrics = {"loss": 0, "at_acc": 0, "el_acc": 0, "n": 0}

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)

            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            bs = batch["attention_mask"].sum().item()
            epoch_metrics["loss"] += metrics["loss"].item() * bs
            epoch_metrics["at_acc"] += metrics["action_type_acc"] * bs
            epoch_metrics["el_acc"] += metrics["element_acc"] * bs
            epoch_metrics["n"] += bs

        n = epoch_metrics["n"]
        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={epoch_metrics['loss']/n:.4f}, "
                  f"at_acc={epoch_metrics['at_acc']/n:.3f}, "
                  f"el_acc={epoch_metrics['el_acc']/n:.3f}")

    return agent


def train_ppo(config, task_name, num_demos, seed, device, encoder_type="dom"):
    ppo_cfg = config["ppo"]
    gamma      = ppo_cfg["gamma"]
    eps_clip   = ppo_cfg["eps_clip"]
    update_freq = ppo_cfg["update_freq"]
    batch_size  = ppo_cfg["batch_size"]
    max_ep_len  = ppo_cfg["max_ep_len"]
    num_batches = ppo_cfg["num_batches"]
    lr          = config["training"]["lr"]

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
            })
        return elements

    def make_env_action(env, elements, action_type_idx, element_idx):
        action_types = env.unwrapped.action_space_config.action_types
        if element_idx < len(elements):
            elem = elements[element_idx]
            cx = elem["left"] + elem["width"] / 2
            cy = elem["top"] + elem["height"] / 2
            cx = max(0, min(cx, 160.0))
            cy = max(0, min(cy, 210.0))

        else:
            cx, cy = 80.0, 105.0
        act = OrderedDict()
        act["ref"] = np.int64(0)
        act["coords"] = np.array([cx, cy], dtype=np.float32)
        act["text"] = ""
        act["field"] = np.int64(0)
        act["key"] = np.int64(0)
        if action_type_idx == 0:
            act["action_type"] = np.int64(action_types.index(ActionTypes.CLICK_COORDS))
        else:
            act["action_type"] = np.int64(action_types.index(ActionTypes.TYPE_TEXT))
        return act

    agent = PPOAgent(
        state_dim=config["state"]["state_dim"],
        hidden_dim=ppo_cfg["hidden_dim"],
        max_elements=config["state"]["max_dom_elements"],
    ).to(device)

    policy_params = (list(agent.state_encoder.parameters()) +
                     list(agent.action_type_head.parameters()) +
                     list(agent.element_score.parameters()))
    value_params  = list(agent.value_head.parameters())
    optimizer_pi = torch.optim.Adam(policy_params, lr=lr)
    optimizer_v  = torch.optim.Adam(value_params,  lr=lr)

    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)

    all_total_rewards = []
    averaged_total_rewards = []

    for t in range(num_batches):
        print(f"  Batch {t+1}/{num_batches}: collecting {batch_size} steps...", flush=True)

        paths = []
        episode_rewards = []
        episode = 0
        steps = 0

        while steps < batch_size:
            obs, info = env.reset(seed=seed + episode)
            utterance = obs.get("utterance", "")
            elements = parse_dom_elements(obs)
            features, mask = extract_dom_features(elements, utterance)

            ep_features, ep_masks = [], []
            ep_action_types, ep_element_idxs, ep_old_logprobs, ep_rewards = [], [], [], []
            episode_reward = 0

            for step in range(max_ep_len):
                ep_features.append(features)
                ep_masks.append(mask)

                features_t = torch.tensor(features, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                mask_t = torch.tensor(mask, dtype=torch.float32,
                                      device=device).unsqueeze(0)

                with torch.no_grad():
                    action_type, element_idx, old_logprob = agent.act(
                        features_t, mask_t, return_log_prob=True
                    )

                env_action = make_env_action(env, elements, action_type, element_idx)
                obs, reward, done, truncated, info = env.step(env_action)

                ep_action_types.append(action_type)
                ep_element_idxs.append(element_idx)
                ep_old_logprobs.append(old_logprob)
                ep_rewards.append(reward)
                episode_reward += reward
                steps += 1

                if done or truncated or step == max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if steps == batch_size:
                    break

                elements = parse_dom_elements(obs)
                features, mask = extract_dom_features(elements, utterance)

            paths.append({
                "features":      np.array(ep_features),
                "masks":         np.array(ep_masks),
                "action_types":  np.array(ep_action_types),
                "element_idxs":  np.array(ep_element_idxs),
                "old_logprobs":  np.array(ep_old_logprobs),
                "reward":        np.array(ep_rewards),
            })
            episode += 1

        all_total_rewards.extend(episode_rewards)
        all_features     = np.concatenate([p["features"]     for p in paths])
        all_masks        = np.concatenate([p["masks"]        for p in paths])
        all_action_types = np.concatenate([p["action_types"] for p in paths])
        all_element_idxs = np.concatenate([p["element_idxs"] for p in paths])
        old_logprobs     = np.concatenate([p["old_logprobs"] for p in paths])

        returns    = get_returns(paths, gamma)
        advantages = calculate_advantage(returns, all_features, all_masks,
                                         agent, device)

        for k in range(update_freq):
            update_value(agent, optimizer_v, all_features, all_masks,
                         returns, device)
            update_policy(agent, optimizer_pi, all_features, all_masks,
                          all_action_types, all_element_idxs,
                          advantages, old_logprobs, eps_clip, device)

        avg_reward = np.mean(episode_rewards)
        sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
        averaged_total_rewards.append(avg_reward)
        avg_reward = np.mean(episode_rewards)
        sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
        averaged_total_rewards.append(avg_reward)
        print(f"  Batch {t+1}/{num_batches}: avg_reward={avg_reward:.3f} +/- {sigma_reward:.3f}  ({len(episode_rewards)} eps)", flush=True)
    env.close()
    return agent



TRAIN_FNS = {"bc": train_bc, "iql": train_iql, "dt": train_dt, "ppo": train_ppo}


def run_experiment(config, method, task_name, num_demos, seed, device,
                   encoder_type="dom"):
    """Run a single experiment: train + return trained agent."""
    enc_label = f" [{encoder_type.upper()}]" if encoder_type != "dom" else ""
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}{enc_label} | Task: {task_name} | "
          f"Demos: {num_demos} | Seed: {seed}")
    print(f"{'='*60}")

    train_fn = TRAIN_FNS[method]
    agent = train_fn(config, task_name, num_demos, seed, device, encoder_type)

    return agent


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
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    encoder_type = args.encoder_type
    print(f"Using device: {device}")
    print(f"Encoder type: {encoder_type}")

    # Determine what to sweep
    methods = ["bc", "iql", "dt"] if args.method == "all" else [args.method]

    if args.task:
        tasks = [args.task]
    else:
        tasks = (config["env"]["tasks"]["simple"] +
                 config["env"]["tasks"]["medium"] +
                 config["env"]["tasks"]["hard"])

    demo_sizes = [args.num_demos] if args.num_demos is not None else config["data"]["data_sizes"]
    seeds = [args.seed] if args.seed else config["training"]["seeds"]

    # Encoder prefix for model file names
    enc_prefix = f"{encoder_type}_" if encoder_type != "dom" else ""

    # Run experiments
    results = {}
    for method in methods:
        for task in tasks:
            for n_demos in demo_sizes:
                for seed in seeds:
                    agent = run_experiment(
                        config, method, task, n_demos, seed, device,
                        encoder_type,
                    )

                    if agent is not None:
                        save_path = os.path.join(
                            args.save_dir, "models",
                            f"{enc_prefix}{method}_{task}_n{n_demos}_s{seed}.pt"
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(agent.state_dict(), save_path)

                        key = f"{enc_prefix}{method}_{task}_n{n_demos}_s{seed}"
                        results[key] = {
                            "method": method, "task": task,
                            "num_demos": n_demos, "seed": seed,
                            "encoder_type": encoder_type,
                        }

    # Save results index
    os.makedirs(args.save_dir, exist_ok=True)
    index_name = f"{enc_prefix}experiment_index.json"
    with open(os.path.join(args.save_dir, index_name), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll experiments complete. Results in {args.save_dir}/")


if __name__ == "__main__":
    main()
