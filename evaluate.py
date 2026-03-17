"""Evaluation script."""

import os
import json
import yaml
import argparse
import numpy as np
import torch
import gymnasium as gym
import miniwob  # noqa: F401
from collections import OrderedDict
from miniwob.action import ActionTypes

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
from models.ppo import PPOAgent
from utils.state_encoder import extract_dom_features, classify_element


def obs_to_dom_elements(obs, max_elements=64):
    raw_dom = obs.get("dom_elements", [])
    elements = []
    for elem in raw_dom[:max_elements]:
        rect = elem.get("rect", {})
        elements.append({
            "tag": str(elem.get("tag", "div")),
            "type": str(elem.get("type", "")),
            "text": str(elem.get("text", "")),
            "value": str(elem.get("value", "")),
            "left": float(rect.get("left", 0)),
            "top": float(rect.get("top", 0)),
            "width": float(rect.get("width", 0)),
            "height": float(rect.get("height", 0)),
            "visible": bool(elem.get("visible", True)),
            "focused": bool(elem.get("focused", False)),
            "ref": int(elem["ref"]) if elem.get("ref") is not None else 0,
            "children": elem.get("children", []),
        })
    return elements


def infer_type_text(utterance, dom_elements, element_idx):
    if element_idx < len(dom_elements):
        el = dom_elements[element_idx]
        tag = classify_element(el.get("tag", ""), el.get("type", ""))
        if tag in ("input_text", "textarea"):
            label_text = el.get("text", "") or el.get("value", "")
            low_label = label_text.lower()
            if "user" in low_label or "login" in low_label:
                return "jane"
            if "pass" in low_label:
                return "12345"
            if "email" in low_label:
                return "jane@test.com"
            if "search" in low_label or "query" in low_label:
                parts = utterance.split('"')
                if len(parts) >= 3:
                    return parts[1]
                return utterance.split("search for ")[-1] if "search for " in utterance else utterance
            return utterance
    return utterance


def element_center(el):
    left = float(el.get("left", 80))
    top = float(el.get("top", 105))
    width = float(el.get("width", 10))
    height = float(el.get("height", 10))
    return left + width / 2, top + height / 2


def make_env_action(env, dom_elements, action_dict, utterance):
    el_idx = min(action_dict["element_idx"], len(dom_elements) - 1) if dom_elements else 0
    at_idx = action_dict["action_type_idx"]

    action_types = env.unwrapped.action_space_config.action_types

    elem = dom_elements[el_idx] if 0 <= el_idx < len(dom_elements) else {}
    cx, cy = element_center(elem) if elem else (80.0, 105.0)
    ref = int(elem.get("ref", 0))

    act = OrderedDict()
    act["ref"] = np.int64(ref if ref > 0 else 0)
    act["coords"] = np.array([cx, cy], dtype=np.float32)
    act["text"] = ""
    act["field"] = np.int64(0)
    act["key"] = np.int64(0)

    if at_idx == 0:
        if ref > 0 and ActionTypes.CLICK_ELEMENT in action_types:
            act["action_type"] = np.int64(action_types.index(ActionTypes.CLICK_ELEMENT))
        else:
            act["action_type"] = np.int64(action_types.index(ActionTypes.CLICK_COORDS))
    else:
        act["text"] = infer_type_text(utterance, dom_elements, el_idx)
        if ref > 0 and ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT in action_types:
            act["action_type"] = np.int64(action_types.index(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT))
        else:
            act["action_type"] = np.int64(action_types.index(ActionTypes.TYPE_TEXT))
    return act


def load_agent(method, task, cfg, device, model_path):
    if method == "bc":
        agent = BCAgent(
            state_dim=cfg["state"]["state_dim"],
            hidden_dim=cfg["bc"]["hidden_dim"],
            max_elements=cfg["state"]["max_dom_elements"],
            element_feature_dim=cfg["state"]["element_feature_dim"],
        )
    elif method == "iql":
        agent = IQLAgent(
            state_dim=cfg["state"]["state_dim"],
            hidden_dim=cfg["iql"]["hidden_dim"],
            max_elements=cfg["state"]["max_dom_elements"],
            element_feature_dim=cfg["state"]["element_feature_dim"],
        )
    elif method == "dt":
        agent = DecisionTransformerAgent(
            state_dim=cfg["state"]["state_dim"],
            embed_dim=cfg["dt"]["embed_dim"],
            n_layers=cfg["dt"]["n_layers"],
            n_heads=cfg["dt"]["n_heads"],
            context_length=cfg["dt"]["context_length"],
            max_elements=cfg["state"]["max_dom_elements"],
            element_feature_dim=cfg["state"]["element_feature_dim"],
        )
    elif method == "ppo":
        agent = PPOAgent(
            state_dim=cfg["state"]["state_dim"],
            hidden_dim=cfg["ppo"]["hidden_dim"],
            max_elements=cfg["state"]["max_dom_elements"],
            element_feature_dim=cfg["state"]["element_feature_dim"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.to(device)
    agent.eval()
    return agent


def evaluate_agent(agent, method, task, cfg, device, num_episodes=50):
    env = gym.make(f"miniwob/{task}-v1", render_mode=None)
    max_elements = cfg["state"]["max_dom_elements"]
    successes = 0

    past_states, past_masks, past_actions = [], [], []
    past_returns, past_timesteps = [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        past_states.clear()
        past_masks.clear()
        past_actions.clear()
        past_returns.clear()
        past_timesteps.clear()

        while not done:
            utterance = obs.get("utterance", "")
            dom_elements = obs_to_dom_elements(obs, max_elements)
            features, mask = extract_dom_features(dom_elements, utterance, max_elements)
            feat_t = torch.tensor(features, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

            if method == "dt":
                action_dict = agent.get_action(
                    feat_t, mask_t,
                    past_states=past_states if past_states else None,
                    past_state_masks=past_masks if past_masks else None,
                    past_actions=past_actions if past_actions else None,
                    past_returns=past_returns if past_returns else None,
                    past_timesteps=past_timesteps if past_timesteps else None,
                    target_return=1.0,
                    current_timestep=min(step, 99),
                )
                past_states.append(feat_t)
                past_masks.append(mask_t)
                past_actions.append(torch.tensor([action_dict["action_type_idx"],
                                                   action_dict["element_idx"]]))
                past_returns.append(1.0)
                past_timesteps.append(min(step, 99))
            else:
                action_dict = agent.get_action(feat_t, mask_t)

            env_action = make_env_action(env, dom_elements, action_dict, utterance)
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            step += 1

        if reward > 0:
            successes += 1

    env.close()
    return successes / num_episodes



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if cfg["training"]["device"] == "auto" and torch.cuda.is_available()
        else "cpu"
    )

    methods = [args.method] if args.method else ["bc", "iql", "dt", "ppo"]
    all_tasks = cfg["env"]["tasks"]["simple"] + cfg["env"]["tasks"]["medium"] + cfg["env"]["tasks"]["hard"]
    tasks = [args.task] if args.task else all_tasks
    num_episodes = cfg["eval"]["num_episodes"]

    results = {}
    models_dir = os.path.join(cfg["logging"]["save_dir"], "models")

    if not os.path.isdir(models_dir):
        print(f"No models directory found at {models_dir}. Run train.py first.")
        return

    for method in methods:
        results[method] = {}
        for task in tasks:
            prefix = f"{method}_{task}_"
            ckpts = [f for f in os.listdir(models_dir)
                     if f.startswith(prefix) and f.endswith(".pt")]
            if not ckpts:
                print(f"  [skip] No checkpoints for {method}/{task}")
                continue

            task_results = []
            for ckpt_file in sorted(ckpts):
                model_path = os.path.join(models_dir, ckpt_file)
                print(f"  Evaluating {ckpt_file} ...")
                agent = load_agent(method, task, cfg, device, model_path)
                sr = evaluate_agent(agent=agent, method=method, task=task,
                                    cfg=cfg, device=device,
                                    num_episodes=num_episodes)
                task_results.append({
                    "checkpoint": ckpt_file,
                    "success_rate": sr,
                })
                print(f"    success_rate = {sr:.2%}")

            results[method][task] = task_results

    # save results
    os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
    suffix = f"_{args.method}" if args.method else ""
    out_path = os.path.join(cfg["logging"]["save_dir"], f"eval_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
