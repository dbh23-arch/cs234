"""Training script for BC, IQL, DT, and PPO on MiniWoB++ tasks."""

import os
import json
import argparse
import re
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
from collections import OrderedDict
from models.ppo import (PPOAgent, get_returns, calculate_advantage,
                            update_policy, update_value)
from utils.dataset import TrajectoryDataset, SequenceDataset
from utils.dataset import TextTrajectoryDataset, TextSequenceDataset
from utils.state_encoder import extract_dom_features

def get_device(config_device="auto"):
    """Pick GPU if available, otherwise MPS (Apple Silicon), otherwise CPU."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)

def get_tokenizer(encoder_type):
    if encoder_type == "text":
        from utils.text_state_encoder import get_tokenizer
        return get_tokenizer()
    return None

def train_bc(config, task_name, num_demos, seed, device, encoder_type="dom",
             source="human"):
    if encoder_type == "text":
        tokenizer = get_tokenizer(encoder_type)
        dataset = TextTrajectoryDataset(
            config["data"]["save_dir"], task_name,
            tokenizer=tokenizer, max_demos=num_demos, seed=seed,
            source=source,
        )
    else:
        dataset = TrajectoryDataset(
            config["data"]["save_dir"], task_name,
            max_demos=num_demos, seed=seed, source=source,
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

    # only optimize unfrozen params (matters for text encoder)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"])

    best_loss = float('inf')
    for epoch in range(config["training"]["epochs"]):
        agent.train()
        total_loss, total_at_acc, total_el_acc, n_samples = 0, 0, 0, 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = agent.compute_loss(batch)

            optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            bs = len(batch["reward"])
            total_loss += metrics["loss"].item() * bs
            total_at_acc += metrics["action_type_acc"] * bs
            total_el_acc += metrics["element_acc"] * bs
            n_samples += bs

        avg_loss = total_loss / n_samples

        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, "
                  f"at_acc={total_at_acc/n_samples:.3f}, "
                  f"el_acc={total_el_acc/n_samples:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    return agent

def train_iql(config, task_name, num_demos, seed, device, encoder_type="dom",
              source="human"):
    if encoder_type == "text":
        tokenizer = get_tokenizer(encoder_type)
        dataset = TextTrajectoryDataset(
            config["data"]["save_dir"], task_name,
            tokenizer=tokenizer, max_demos=num_demos, seed=seed,
            source=source,
        )
    else:
        dataset = TrajectoryDataset(
            config["data"]["save_dir"], task_name,
            max_demos=num_demos, seed=seed, source=source,
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
        cql_alpha=config["iql"].get("cql_alpha", 0.0),
        cql_temperature=config["iql"].get("cql_temperature", 1.0),
        encoder_type=encoder_type,
        freeze_lm=config.get("encoder", {}).get("freeze_lm", True),
    ).to(device)

    trainable = [p for p in agent.parameters() if p.requires_grad]  # skip frozen LM
    optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        agent.train()
        epoch_metrics = {
            "loss": 0, "v_loss": 0, "q_loss": 0, "q_bellman": 0,
            "cql_loss": 0, "p_loss": 0, "n": 0,
        }

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
            epoch_metrics["q_bellman"] += metrics["q_bellman_loss"] * len(batch["reward"])
            epoch_metrics["cql_loss"] += metrics["cql_loss"] * len(batch["reward"])
            epoch_metrics["p_loss"] += metrics["policy_loss"] * len(batch["reward"])
            epoch_metrics["n"] += len(batch["reward"])

        n = epoch_metrics["n"]
        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            print(f"    Epoch {epoch+1}: loss={epoch_metrics['loss']/n:.4f}, "
                  f"v={epoch_metrics['v_loss']/n:.4f}, "
                  f"q={epoch_metrics['q_loss']/n:.4f}, "
                  f"q_td={epoch_metrics['q_bellman']/n:.4f}, "
                  f"cql={epoch_metrics['cql_loss']/n:.4f}, "
                  f"pi={epoch_metrics['p_loss']/n:.4f}")

    return agent

def train_dt(config, task_name, num_demos, seed, device, encoder_type="dom",
             source="human"):
    # NOTE: stitch_prob > 0 splices top-k trajectories together for data augmentation
    if encoder_type == "text":
        tokenizer = get_tokenizer(encoder_type)
        dataset = TextSequenceDataset(
            config["data"]["save_dir"], task_name,
            tokenizer=tokenizer,
            context_length=config["dt"]["context_length"],
            max_demos=num_demos, seed=seed, source=source,
            stitch_prob=config["dt"].get("stitch_prob", 0.0),
            stitch_topk_frac=config["dt"].get("stitch_topk_frac", 0.3),
        )
    else:
        dataset = SequenceDataset(
            config["data"]["save_dir"], task_name,
            context_length=config["dt"]["context_length"],
            max_demos=num_demos, seed=seed, source=source,
            stitch_prob=config["dt"].get("stitch_prob", 0.0),
            stitch_topk_frac=config["dt"].get("stitch_topk_frac", 0.3),
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
    import gymnasium as gym
    import miniwob
    from miniwob.action import ActionTypes

    ppo_cfg = config["ppo"]
    gamma      = ppo_cfg["gamma"]
    eps_clip   = ppo_cfg["eps_clip"]
    update_freq = ppo_cfg["update_freq"]
    batch_size  = ppo_cfg["batch_size"]
    max_ep_len  = ppo_cfg["max_ep_len"]
    num_batches = ppo_cfg["num_batches"]
    entropy_coeff = ppo_cfg.get("entropy_coeff", 0.02)
    lr          = config["training"]["lr"]

    def parse_dom_elements(obs):
        elements = []
        for elem in obs.get("dom_elements", []):
            def to_float(v):
                return v.item() if hasattr(v, 'item') else float(v or 0)
            def to_int(v):
                if hasattr(v, "item"):
                    v = v.item()
                try:
                    return int(v)
                except Exception:
                    return 0
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
                "ref": to_int(elem.get("ref", 0)),
            })
        return elements

    def element_center(elem):
        cx = elem["left"] + elem["width"] / 2
        cy = elem["top"] + elem["height"] / 2
        cx = max(0.0, min(cx, 160.0))
        cy = max(0.0, min(cy, 210.0))
        return cx, cy

    def is_input_like(elem):
        tag = str(elem.get("tag", "")).lower()
        typ = str(elem.get("type", "")).lower()
        if tag in ("input_text", "input_password", "input_search", "textarea"):
            return True
        if tag == "input":
            return typ in ("", "text", "password", "search", "email", "url", "tel")
        return tag.startswith("input_") and tag not in (
            "input_checkbox", "input_radio", "input_button", "input_submit",
            "input_reset", "input_file", "input_image", "input_hidden",
        )

    def is_clickable(elem):
        tag = str(elem.get("tag", "")).lower()
        ref = int(elem.get("ref", 0))
        if ref <= 0:
            return False
        if tag.startswith("input_"):
            return True
        return tag in {
            "button", "a", "span", "label", "option", "select",
            "input", "li",
        }

    def resolve_target_idx(elements, element_idx, predicate):
        if 0 <= element_idx < len(elements) and predicate(elements[element_idx]):
            return element_idx
        if not elements:
            return element_idx
        if 0 <= element_idx < len(elements):
            sx, sy = element_center(elements[element_idx])
        else:
            sx, sy = 80.0, 105.0
        candidates = [i for i, e in enumerate(elements) if predicate(e)]
        if not candidates:
            return element_idx
        return min(
            candidates,
            key=lambda i: (element_center(elements[i])[0] - sx) ** 2 +
                          (element_center(elements[i])[1] - sy) ** 2,
        )

    def infer_type_text(utterance, elements, element_idx):
        elem = elements[element_idx] if 0 <= element_idx < len(elements) else {}
        elem_type = str(elem.get("type", "")).lower()
        elem_tag = str(elem.get("tag", "")).lower()
        is_password = elem_type == "password" or elem_tag == "input_password"

        quoted = re.findall(r'"([^"]*)"', utterance)

        user_q = re.search(r'username[^"]*"([^"]+)"', utterance, re.IGNORECASE)
        pass_q = re.search(r'password[^"]*"([^"]+)"', utterance, re.IGNORECASE)
        if is_password and pass_q:
            return pass_q.group(1).strip()
        if (not is_password) and user_q:
            return user_q.group(1).strip()

        starts_m = re.search(r'starts with\s+"([^"]+)"', utterance, re.IGNORECASE)
        ends_m = re.search(r'ends with\s+"([^"]+)"', utterance, re.IGNORECASE)
        if starts_m or ends_m:
            prefix = starts_m.group(1).strip() if starts_m else ""
            suffix = ends_m.group(1).strip() if ends_m else ""
            if prefix and suffix:
                overlap = 0
                for k in range(min(len(prefix), len(suffix)), 0, -1):
                    if prefix[-k:].lower() == suffix[:k].lower():
                        overlap = k
                        break
                return prefix + suffix[overlap:]
            if prefix:
                return prefix
            if suffix:
                return suffix

        if quoted:
            if is_password and len(quoted) >= 2:
                return quoted[1]
            return quoted[0]

        m = re.search(r'(?:enter|type|input|search for)\s+(.+)', utterance, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('"').rstrip('.')
        return utterance.strip()

    def make_env_action(env, elements, action_type_idx, element_idx, utterance):
        action_types = env.unwrapped.action_space_config.action_types
        if action_type_idx == 0:
            element_idx = resolve_target_idx(elements, element_idx, is_clickable)
        else:
            element_idx = resolve_target_idx(elements, element_idx, is_input_like)

        if 0 <= element_idx < len(elements):
            elem = elements[element_idx]
            cx, cy = element_center(elem)
            ref = int(elem.get("ref", 0))
        else:
            cx, cy = 80.0, 105.0
            ref = 0

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
                act["action_type"] = np.int64(
                    action_types.index(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT)
                )
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

                env_action = make_env_action(
                    env, elements, action_type, element_idx, utterance
                )
                try:
                    obs, reward, done, truncated, info = env.step(env_action)
                except Exception:
                    # env crashed, remake it and bail on this episode
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
                    ep_rewards.append(-1.0)
                    episode_reward += -1.0
                    steps += 1
                    episode_rewards.append(episode_reward)
                    break

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

                utterance = obs.get("utterance", utterance)
                elements = parse_dom_elements(obs)
                features, mask = extract_dom_features(elements, utterance)

            valid_len = min(
                len(ep_features), len(ep_masks), len(ep_action_types),
                len(ep_element_idxs), len(ep_old_logprobs), len(ep_rewards),
            )
            if valid_len == 0:
                episode += 1
                continue

            paths.append({
                "features":      np.array(ep_features[:valid_len]),
                "masks":         np.array(ep_masks[:valid_len]),
                "action_types":  np.array(ep_action_types[:valid_len]),
                "element_idxs":  np.array(ep_element_idxs[:valid_len]),
                "old_logprobs":  np.array(ep_old_logprobs[:valid_len]),
                "reward":        np.array(ep_rewards[:valid_len]),
            })
            episode += 1

        if not paths:
            continue

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
                          advantages, old_logprobs, eps_clip, device,
                          entropy_coeff=entropy_coeff)

        avg_reward = np.mean(episode_rewards)
        sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
        averaged_total_rewards.append(avg_reward)
        print(f"  Batch {t+1}/{num_batches}: avg_reward={avg_reward:.3f} +/- {sigma_reward:.3f}  ({len(episode_rewards)} eps)", flush=True)
    env.close()
    return agent

TRAIN_FNS = {"bc": train_bc, "iql": train_iql, "dt": train_dt, "ppo": train_ppo}

def run_experiment(config, method, task_name, num_demos, seed, device,
                   encoder_type="dom", source="human"):
    enc_label = f" [{encoder_type.upper()}]" if encoder_type != "dom" else ""
    src_label = f" [{source}]"
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}{enc_label}{src_label} | Task: {task_name} | "
          f"Demos: {num_demos} | Seed: {seed}")
    print(f"{'='*60}")

    train_fn = TRAIN_FNS[method]
    if method == "ppo":
        agent = train_fn(config, task_name, num_demos, seed, device, encoder_type)
    else:
        agent = train_fn(config, task_name, num_demos, seed, device,
                         encoder_type, source=source)

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
    parser.add_argument("--source", type=str, default="human",
                        choices=["human", "heuristic"],
                        help="Demo source: human (Stanford) or heuristic (scripted)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    encoder_type = args.encoder_type
    source = args.source
    print(f"Using device: {device}")
    print(f"Encoder type: {encoder_type}")
    print(f"Data source: {source}")

    methods = ["bc", "iql", "dt", "ppo"] if args.method == "all" else [args.method]

    if args.task:
        tasks = [args.task]
    else:
        tasks = (config["env"]["tasks"]["simple"] +
                 config["env"]["tasks"]["medium"] +
                 config["env"]["tasks"]["hard"])

    demo_sizes = [args.num_demos] if args.num_demos is not None else config["data"]["data_sizes"]
    seeds = [args.seed] if args.seed else config["training"]["seeds"]

    enc_prefix = f"{encoder_type}_" if encoder_type != "dom" else ""
    src_prefix = f"{source}_"

    # run all combos of method x task x demos x seed
    results = {}
    for method in methods:
        method_demo_sizes = [0] if method == "ppo" else demo_sizes
        for task in tasks:
            for n_demos in method_demo_sizes:
                for seed in seeds:
                    agent = run_experiment(
                        config, method, task, n_demos, seed, device,
                        encoder_type, source=source,
                    )

                    if agent is not None:
                        save_path = os.path.join(
                            args.save_dir, "models",
                            f"{src_prefix}{enc_prefix}{method}_{task}_n{n_demos}_s{seed}.pt"
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(agent.state_dict(), save_path)

                        key = f"{src_prefix}{enc_prefix}{method}_{task}_n{n_demos}_s{seed}"
                        results[key] = {
                            "method": method, "task": task,
                            "num_demos": n_demos, "seed": seed,
                            "encoder_type": encoder_type,
                            "source": source,
                        }

    os.makedirs(args.save_dir, exist_ok=True)
    index_name = f"{src_prefix}{enc_prefix}experiment_index.json"
    with open(os.path.join(args.save_dir, index_name), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll experiments complete. Results in {args.save_dir}/")

if __name__ == "__main__":
    main()
