"""
Evaluate trained agents on MiniWoB++ tasks.
Runs agents in the environment and reports success rates.
Supports both DOM (hand-crafted) and text (DistilBERT) state encoders.
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
from collections import OrderedDict

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
from models.ppo import PPOAgent
from utils.state_encoder import extract_dom_features
from utils.text_features import dom_to_text, tokenize_page


def get_device(config_device="auto"):
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def load_agent(method, config, model_path, device, encoder_type="dom"):
    """Load a trained agent from checkpoint."""
    freeze_lm = config.get("encoder", {}).get("freeze_lm", True)

    if method == "bc":
        agent = BCAgent(
            state_dim=config["state"]["state_dim"],
            hidden_dim=config["bc"]["hidden_dim"],
            max_elements=config["state"]["max_dom_elements"],
            encoder_type=encoder_type,
            freeze_lm=freeze_lm,
        )
    elif method == "iql":
        agent = IQLAgent(
            state_dim=config["state"]["state_dim"],
            hidden_dim=config["iql"]["hidden_dim"],
            max_elements=config["state"]["max_dom_elements"],
            discount=config["iql"]["discount"],
            tau=config["iql"]["tau"],
            beta=config["iql"]["beta"],
            encoder_type=encoder_type,
            freeze_lm=freeze_lm,
        )
    elif method == "dt":
        agent = DecisionTransformerAgent(
            state_dim=config["state"]["state_dim"],
            embed_dim=config["dt"]["embed_dim"],
            n_layers=config["dt"]["n_layers"],
            n_heads=config["dt"]["n_heads"],
            context_length=config["dt"]["context_length"],
            max_elements=config["state"]["max_dom_elements"],
            dropout=config["dt"]["dropout"],
            encoder_type=encoder_type,
            freeze_lm=freeze_lm,
        )

    elif method == "ppo":
        agent = PPOAgent(
            state_dim=config["state"]["state_dim"],
            hidden_dim=config["ppo"]["hidden_dim"],
            max_elements=config["state"]["max_dom_elements"]

        )

    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.to(device)
    agent.set_grad_enabled = False
    return agent


def parse_dom_elements(obs):
    """Parse DOM elements from MiniWoB observation."""
    dom_elements = obs.get("dom_elements", [])
    elements = []
    for elem in dom_elements:
        def to_float(val):
            if hasattr(val, 'item'):
                return val.item()
            return float(val) if val is not None else 0.0

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


def make_click_action(env, coords):
    """Create a click action at (x, y) coordinates."""
    action_types = env.unwrapped.action_space_config.action_types
    click_idx = action_types.index(ActionTypes.CLICK_COORDS)
    action = OrderedDict()
    action["action_type"] = np.int64(click_idx)
    action["coords"] = np.array(coords, dtype=np.float32)
    action["ref"] = np.int64(0)
    action["text"] = ""
    action["field"] = np.int64(0)
    action["key"] = np.int64(0)
    return action


def make_type_action(env, coords, text):
    """Create a type action at (x, y) with given text."""
    action_types = env.unwrapped.action_space_config.action_types
    type_idx = action_types.index(ActionTypes.TYPE_TEXT)
    action = OrderedDict()
    action["action_type"] = np.int64(type_idx)
    action["coords"] = np.array(coords, dtype=np.float32)
    action["ref"] = np.int64(0)
    action["text"] = str(text)
    action["field"] = np.int64(0)
    action["key"] = np.int64(0)
    return action


def run_episode_dom(agent, method, env, device, seed=None, max_steps=20):
    """Run a single episode with DOM encoder."""
    obs, info = env.reset(seed=seed)
    utterance = obs.get("utterance", "")

    elements = parse_dom_elements(obs)
    features, mask = extract_dom_features(elements, utterance)
    features_t = torch.tensor(features, device=device)
    mask_t = torch.tensor(mask, device=device)

    ep_reward = 0
    ep_steps = 0

    for step in range(max_steps):
        with torch.no_grad():
            if method == "dt":
                action = agent.get_action(
                    features_t, mask_t,
                    target_return=1.0,
                    current_timestep=step,
                )
            else:
                action = agent.get_action(features_t, mask_t)

        elem_idx = action["element_idx"]
        if elem_idx < len(elements):
            elem = elements[elem_idx]
            cx = elem["left"] + elem["width"] / 2
            cy = elem["top"] + elem["height"] / 2
        else:
            cx, cy = 80.0, 105.0

        if action["action_type_idx"] == 0:
            env_action = make_click_action(env, [cx, cy])
        else:
            env_action = make_type_action(env, [cx, cy], "")

        obs, reward, done, truncated, info = env.step(env_action)
        ep_reward += reward
        ep_steps += 1

        if done or truncated:
            break

        elements = parse_dom_elements(obs)
        features, mask = extract_dom_features(elements, utterance)
        features_t = torch.tensor(features, device=device)
        mask_t = torch.tensor(mask, device=device)

    return ep_reward, ep_steps


def run_episode_text(agent, method, env, device, tokenizer, seed=None,
                     max_steps=20):
    """Run a single episode with text (LM) encoder."""
    obs, info = env.reset(seed=seed)
    utterance = obs.get("utterance", "")

    elements = parse_dom_elements(obs)
    text, char_spans = dom_to_text(elements, utterance)
    input_ids, attn_mask, elem_spans, elem_mask = tokenize_page(
        text, char_spans, tokenizer
    )
    ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
    amask_t = torch.tensor(attn_mask, dtype=torch.float32, device=device)
    spans_t = torch.tensor(elem_spans, dtype=torch.long, device=device)
    emask_t = torch.tensor(elem_mask, dtype=torch.float32, device=device)

    ep_reward = 0
    ep_steps = 0

    for step in range(max_steps):
        with torch.no_grad():
            if method == "dt":
                # DT text: single-step prediction (no rolling context yet)
                batch = {
                    "input_ids": ids_t.unsqueeze(0).unsqueeze(0),
                    "text_attention_mask": amask_t.unsqueeze(0).unsqueeze(0),
                    "element_token_spans": spans_t.unsqueeze(0).unsqueeze(0),
                    "element_masks": emask_t.unsqueeze(0).unsqueeze(0),
                    "action_types": torch.zeros(1, 1, dtype=torch.long, device=device),
                    "element_idxs": torch.zeros(1, 1, dtype=torch.long, device=device),
                    "returns_to_go": torch.tensor([[1.0]], device=device),
                    "timesteps": torch.tensor([[step]], dtype=torch.long, device=device),
                    "attention_mask": torch.ones(1, 1, device=device),
                }
                at_preds, el_preds = agent.forward_text(
                    batch["input_ids"], batch["text_attention_mask"],
                    batch["element_token_spans"], batch["element_masks"],
                    batch["action_types"], batch["element_idxs"],
                    batch["returns_to_go"], batch["timesteps"],
                    batch["attention_mask"],
                )
                action = {
                    "action_type_idx": at_preds[0, -1].argmax().item(),
                    "element_idx": el_preds[0, -1].argmax().item(),
                }
            else:
                action = agent.get_action_text(ids_t, amask_t, spans_t, emask_t)

        elem_idx = action["element_idx"]
        if elem_idx < len(elements):
            elem = elements[elem_idx]
            cx = elem["left"] + elem["width"] / 2
            cy = elem["top"] + elem["height"] / 2
        else:
            cx, cy = 80.0, 105.0

        if action["action_type_idx"] == 0:
            env_action = make_click_action(env, [cx, cy])
        else:
            env_action = make_type_action(env, [cx, cy], "")

        obs, reward, done, truncated, info = env.step(env_action)
        ep_reward += reward
        ep_steps += 1

        if done or truncated:
            break

        elements = parse_dom_elements(obs)
        text, char_spans = dom_to_text(elements, utterance)
        input_ids, attn_mask, elem_spans, elem_mask = tokenize_page(
            text, char_spans, tokenizer
        )
        ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        amask_t = torch.tensor(attn_mask, dtype=torch.float32, device=device)
        spans_t = torch.tensor(elem_spans, dtype=torch.long, device=device)
        emask_t = torch.tensor(elem_mask, dtype=torch.float32, device=device)

    return ep_reward, ep_steps


def evaluate_agent(agent, method, task_name, config, device,
                   num_episodes=100, seed=0, encoder_type="dom",
                   tokenizer=None, render_mode=None):
    """Evaluate an agent on a MiniWoB++ task."""
    env = gym.make(f"miniwob/{task_name}-v1", render_mode=render_mode,
                   wait_ms=0)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        if encoder_type == "text":
            ep_reward, ep_steps = run_episode_text(
                agent, method, env, device, tokenizer,
                seed=seed + ep, max_steps=20,
            )
        else:
            ep_reward, ep_steps = run_episode_dom(
                agent, method, env, device,
                seed=seed + ep, max_steps=20,
            )

        if ep_reward > 0:
            successes += 1
        total_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

    env.close()

    return {
        "success_rate": successes / num_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "num_episodes": num_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default="bc",
                        choices=["bc", "iql", "dt", "ppo", "all"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--encoder_type", type=str, default="dom",
                        choices=["dom", "text"])
    parser.add_argument("--render", action="store_true",
                        help="Render environment (opens browser)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    encoder_type = args.encoder_type
    render_mode = "human" if args.render else None
    print(f"Using device: {device}")
    print(f"Encoder type: {encoder_type}")

    # Load tokenizer for text encoder
    tokenizer = None
    if encoder_type == "text":
        from utils.text_state_encoder import get_tokenizer
        tokenizer = get_tokenizer()

    methods = ["bc", "iql", "dt"] if args.method == "all" else [args.method]

    if args.task:
        tasks = [args.task]
    else:
        tasks = (config["env"]["tasks"]["simple"] +
                 config["env"]["tasks"]["medium"] +
                 config["env"]["tasks"]["hard"])

    demo_sizes = [args.num_demos] if args.num_demos is not None else config["data"]["data_sizes"]
    seeds = [args.seed] if args.seed else config["training"]["seeds"]

    enc_prefix = f"{encoder_type}_" if encoder_type != "dom" else ""

    all_results = []

    for method in methods:
        for task in tasks:
            for n_demos in demo_sizes:
                for seed in seeds:
                    model_path = os.path.join(
                        args.results_dir, "models",
                        f"{enc_prefix}{method}_{task}_n{n_demos}_s{seed}.pt"
                    )

                    if not os.path.exists(model_path):
                        print(f"Skipping {method}/{task}/n{n_demos}/s{seed} "
                              f"(no model found)")
                        continue

                    print(f"\nEvaluating {method.upper()} [{encoder_type}] on "
                          f"{task} (n={n_demos}, seed={seed})...")

                    agent = load_agent(method, config, model_path, device,
                                       encoder_type)
                    metrics = evaluate_agent(
                        agent, method, task, config, device,
                        num_episodes=args.num_episodes, seed=seed * 1000,
                        encoder_type=encoder_type, tokenizer=tokenizer,
                        render_mode=render_mode,
                    )

                    result = {
                        "method": method,
                        "task": task,
                        "num_demos": n_demos,
                        "seed": seed,
                        "encoder_type": encoder_type,
                        **metrics,
                    }
                    all_results.append(result)

                    print(f"  Success rate: {metrics['success_rate']:.1%} "
                          f"({int(metrics['success_rate']*args.num_episodes)}"
                          f"/{args.num_episodes})")
                    print(f"  Mean reward: {metrics['mean_reward']:.3f} "
                          f"+/- {metrics['std_reward']:.3f}")

    # Save all results
    os.makedirs(args.results_dir, exist_ok=True)
    results_name = f"{enc_prefix}results.json"
    results_path = os.path.join(args.results_dir, results_name)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Method':<8} {'Enc':<6} {'Task':<20} {'N':<6} {'Success Rate':<15}")
        print(f"{'='*70}")
        for r in all_results:
            print(f"{r['method']:<8} {r.get('encoder_type','dom'):<6} "
                  f"{r['task']:<20} {r['num_demos']:<6} "
                  f"{r['success_rate']:.1%}")


if __name__ == "__main__":
    main()
