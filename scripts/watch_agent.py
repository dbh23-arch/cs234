"""Watch a trained agent play MiniWoB++ tasks in a browser window."""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
from utils.state_encoder import extract_dom_features
from utils.text_features import dom_to_text, tokenize_page
from evaluate import extract_text_from_utterance

def get_device(config_device="auto"):
    if config_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)

def load_agent(method, config, model_path, device, encoder_type="dom"):
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

    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.to(device)
    return agent

def parse_dom_elements(obs):
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

def describe_element(elem, idx):
    tag = elem["tag"]
    text = elem["text"].strip()[:30]
    x = elem["left"] + elem["width"] / 2
    y = elem["top"] + elem["height"] / 2
    if text:
        return f"[{idx}] {tag}: '{text}' at ({x:.0f}, {y:.0f})"
    else:
        return f"[{idx}] {tag} at ({x:.0f}, {y:.0f})"

def watch_episodes(agent, method, task_name, config, device, num_episodes=3,
                   encoder_type="dom", tokenizer=None, seed=42):
    env = gym.make(f"miniwob/{task_name}-v1", render_mode="human", wait_ms=800)

    successes = 0

    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{num_episodes}")
        print(f"{'='*60}")

        obs, info = env.reset(seed=seed + ep)
        utterance = obs.get("utterance", "")
        print(f"Task: {utterance}")

        elements = parse_dom_elements(obs)
        print(f"Page has {len(elements)} DOM elements")

        ep_reward = 0
        max_steps = 20
        dt_running_return = 1.0
        dt_past_ids = []
        dt_past_amasks = []
        dt_past_spans = []
        dt_past_emasks = []
        dt_past_action_types = []
        dt_past_element_idxs = []
        dt_past_returns = []
        dt_past_timesteps = []

        for step in range(max_steps):
            with torch.no_grad():
                if encoder_type == "text":
                    text, char_spans = dom_to_text(elements, utterance)
                    input_ids, attn_mask, elem_spans, elem_mask = tokenize_page(
                        text, char_spans, tokenizer
                    )
                    ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
                    amask_t = torch.tensor(attn_mask, dtype=torch.float32, device=device)
                    spans_t = torch.tensor(elem_spans, dtype=torch.long, device=device)
                    emask_t = torch.tensor(elem_mask, dtype=torch.float32, device=device)

                    if method == "dt":
                        K = min(step + 1, agent.context_length)
                        start = max(0, step + 1 - K)

                        all_ids = dt_past_ids[start:] + [ids_t]
                        all_amasks = dt_past_amasks[start:] + [amask_t]
                        all_spans = dt_past_spans[start:] + [spans_t]
                        all_emasks = dt_past_emasks[start:] + [emask_t]
                        all_at = dt_past_action_types[start:] + [0]
                        all_ei = dt_past_element_idxs[start:] + [0]
                        all_rtg = dt_past_returns[start:] + [dt_running_return]
                        all_ts = dt_past_timesteps[start:] + [step]
                        ctx_len = len(all_ids)

                        at_preds, el_preds = agent.forward_text(
                            torch.stack(all_ids).unsqueeze(0),
                            torch.stack(all_amasks).unsqueeze(0),
                            torch.stack(all_spans).unsqueeze(0),
                            torch.stack(all_emasks).unsqueeze(0),
                            torch.tensor([all_at], dtype=torch.long, device=device),
                            torch.tensor([all_ei], dtype=torch.long, device=device),
                            torch.tensor([all_rtg], dtype=torch.float32, device=device),
                            torch.tensor([all_ts], dtype=torch.long, device=device),
                            torch.ones(1, ctx_len, device=device),
                        )
                        action = {
                            "action_type_idx": at_preds[0, -1].argmax().item(),
                            "element_idx": el_preds[0, -1].argmax().item(),
                        }
                    else:
                        action = agent.get_action_text(ids_t, amask_t, spans_t, emask_t)
                else:
                    features, mask = extract_dom_features(elements, utterance)
                    features_t = torch.tensor(features, device=device)
                    mask_t = torch.tensor(mask, device=device)

                    if method == "dt":
                        action = agent.get_action(
                            features_t, mask_t,
                            target_return=1.0,
                            current_timestep=step,
                        )
                    else:
                        action = agent.get_action(features_t, mask_t)

            elem_idx = action["element_idx"]
            action_type = "CLICK" if action["action_type_idx"] == 0 else "TYPE"

            if elem_idx < len(elements):
                elem = elements[elem_idx]
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                elem_desc = describe_element(elem, elem_idx)
                print(f"  Step {step+1}: {action_type} -> {elem_desc}")
            else:
                cx, cy = 80.0, 105.0
                print(f"  Step {step+1}: {action_type} -> [invalid elem {elem_idx}] "
                      f"at (80, 105)")

            if action["action_type_idx"] == 0:
                env_action = make_click_action(env, [cx, cy])
            else:
                type_text = extract_text_from_utterance(
                    utterance, elements, elem_idx, task_name
                )
                env_action = make_type_action(env, [cx, cy], type_text)

            obs, reward, done, truncated, info = env.step(env_action)
            ep_reward += reward
            if method == "dt" and encoder_type == "text":
                dt_past_ids.append(ids_t)
                dt_past_amasks.append(amask_t)
                dt_past_spans.append(spans_t)
                dt_past_emasks.append(emask_t)
                dt_past_action_types.append(action["action_type_idx"])
                dt_past_element_idxs.append(action["element_idx"])
                dt_past_returns.append(dt_running_return)
                dt_past_timesteps.append(step)
                dt_running_return = max(0.0, dt_running_return - reward)

            if reward != 0:
                print(f"         -> reward: {reward:.2f}")

            if done or truncated:
                break

            elements = parse_dom_elements(obs)

        status = "SUCCESS" if ep_reward > 0 else "FAIL"
        print(f"\n  Result: {status} (total reward: {ep_reward:.2f})")
        if ep_reward > 0:
            successes += 1

    env.close()

    print(f"\n{'='*60}")
    print(f"Summary: {successes}/{num_episodes} episodes succeeded "
          f"({successes/num_episodes:.0%})")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained agent play MiniWoB++ tasks"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default="bc",
                        choices=["bc", "iql", "dt"])
    parser.add_argument("--task", type=str, default="click-button")
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--encoder_type", type=str, default="dom",
                        choices=["dom", "text"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")
    print(f"Encoder: {args.encoder_type}")

    tokenizer = None
    if args.encoder_type == "text":
        from utils.text_state_encoder import get_tokenizer
        tokenizer = get_tokenizer()

    enc_prefix = f"{args.encoder_type}_" if args.encoder_type != "dom" else ""
    model_path = os.path.join(
        args.results_dir, "models",
        f"{enc_prefix}{args.method}_{args.task}_n{args.num_demos}_s{args.seed}.pt"
    )

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Available models:")
        models_dir = os.path.join(args.results_dir, "models")
        if os.path.exists(models_dir):
            models = sorted(os.listdir(models_dir))
            for m in models[:20]:
                print(f"  {m}")
            if len(models) > 20:
                print(f"  ... and {len(models)-20} more")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    agent = load_agent(args.method, config, model_path, device, args.encoder_type)

    watch_episodes(
        agent, args.method, args.task, config, device,
        num_episodes=args.num_episodes,
        encoder_type=args.encoder_type,
        tokenizer=tokenizer,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
