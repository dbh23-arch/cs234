
import argparse
import json
import os
import re
from collections import OrderedDict

import gymnasium as gym
import miniwob  
import numpy as np
import torch
import yaml
from miniwob.action import ActionTypes

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ppo import PPOAgent, get_returns, calculate_advantage, update_policy, update_value
from utils.state_encoder import extract_dom_features

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_batches", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_ep_len", type=int, default=20)
    p.add_argument("--update_freq", type=int, default=3)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--results_path", type=str, default="results/ppo_smoke_results.json")
    return p.parse_args()

def parse_dom(obs):
    elements = []
    for elem in obs.get("dom_elements", []):
        def to_float(v):
            return v.item() if hasattr(v, "item") else float(v or 0)
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
        return m.group(1).strip().strip('"').rstrip(".")
    return utterance.strip()

def make_action(env, elements, at_idx, el_idx, utterance):
    action_types = env.unwrapped.action_space_config.action_types
    if at_idx == 0:
        el_idx = resolve_target_idx(elements, el_idx, is_clickable)
    else:
        el_idx = resolve_target_idx(elements, el_idx, is_input_like)

    if 0 <= el_idx < len(elements):
        e = elements[el_idx]
        cx, cy = element_center(e)
        ref = int(e.get("ref", 0))
    else:
        cx, cy = 80.0, 105.0
        ref = 0

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
        act["text"] = infer_type_text(utterance, elements, el_idx)
        if ref > 0 and ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT in action_types:
            act["action_type"] = np.int64(
                action_types.index(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT)
            )
        else:
            act["action_type"] = np.int64(action_types.index(ActionTypes.TYPE_TEXT))
    return act

def evaluate(agent, task_name, device, max_ep_len, eval_episodes, seed):
    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
    successes = 0
    rewards = []
    for ep in range(eval_episodes):
        try:
            obs, _ = env.reset(seed=seed * 1000 + ep)
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
            obs, _ = env.reset(seed=seed * 1000 + ep)
        utt = obs.get("utterance", "")
        elems = parse_dom(obs)
        feat, mask = extract_dom_features(elems, utt)
        ep_rew = 0.0
        for _ in range(max_ep_len):
            ft = torch.tensor(feat, dtype=torch.float32, device=device)
            mt = torch.tensor(mask, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = agent.get_action(ft, mt)
            ea = make_action(env, elems, action["action_type_idx"], action["element_idx"], utt)
            try:
                obs, reward, done, trunc, _ = env.step(ea)
            except Exception:
                reward, done, trunc = -1.0, True, False
            ep_rew += reward
            if done or trunc:
                break
            elems = parse_dom(obs)
            utt = obs.get("utterance", utt)
            feat, mask = extract_dom_features(elems, utt)
        rewards.append(ep_rew)
        if ep_rew > 0:
            successes += 1
    env.close()
    return {
        "success_rate": successes / eval_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }

def train_smoke_task(task_name, cfg, device, args):
    lr = cfg["training"]["lr"]
    gamma = cfg["ppo"]["gamma"]
    eps_clip = cfg["ppo"]["eps_clip"]
    entropy_coeff = cfg["ppo"].get("entropy_coeff", 0.02)

    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
    agent = PPOAgent(
        state_dim=cfg["state"]["state_dim"],
        hidden_dim=cfg["ppo"]["hidden_dim"],
        max_elements=cfg["state"]["max_dom_elements"],
    ).to(device)

    policy_params = (
        list(agent.state_encoder.parameters()) +
        list(agent.action_type_head.parameters()) +
        list(agent.element_score.parameters())
    )
    value_params = list(agent.value_head.parameters())
    opt_pi = torch.optim.Adam(policy_params, lr=lr)
    opt_v = torch.optim.Adam(value_params, lr=lr)

    for t in range(args.num_batches):
        paths = []
        episode = 0
        steps = 0
        episode_rewards = []

        while steps < args.batch_size:
            try:
                obs, _ = env.reset(seed=args.seed + episode)
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass
                env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
                obs, _ = env.reset(seed=args.seed + episode)

            utt = obs.get("utterance", "")
            elems = parse_dom(obs)
            feat, mask = extract_dom_features(elems, utt)

            ep_f, ep_m, ep_at, ep_ei, ep_lp, ep_r = [], [], [], [], [], []
            ep_rew = 0.0

            for _ in range(args.max_ep_len):
                ep_f.append(feat)
                ep_m.append(mask)

                ft = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                mt = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    at, ei, lp = agent.act(ft, mt, return_log_prob=True)

                ea = make_action(env, elems, at, ei, utt)
                try:
                    obs, reward, done, trunc, _ = env.step(ea)
                    ep_at.append(at)
                    ep_ei.append(ei)
                    ep_lp.append(lp)
                    ep_r.append(reward)
                    ep_rew += reward
                    steps += 1
                except Exception:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=0)
                    ep_r.append(-1.0)
                    ep_rew += -1.0
                    steps += 1
                    episode_rewards.append(ep_rew)
                    break

                if done or trunc or steps >= args.batch_size:
                    episode_rewards.append(ep_rew)
                    break

                elems = parse_dom(obs)
                utt = obs.get("utterance", utt)
                feat, mask = extract_dom_features(elems, utt)
            else:
                episode_rewards.append(ep_rew)

            valid_len = min(len(ep_f), len(ep_m), len(ep_at), len(ep_ei), len(ep_lp), len(ep_r))
            if valid_len > 0:
                paths.append({
                    "features": np.array(ep_f[:valid_len]),
                    "masks": np.array(ep_m[:valid_len]),
                    "action_types": np.array(ep_at[:valid_len]),
                    "element_idxs": np.array(ep_ei[:valid_len]),
                    "old_logprobs": np.array(ep_lp[:valid_len]),
                    "reward": np.array(ep_r[:valid_len]),
                })
            episode += 1

        if not paths:
            continue

        af = np.concatenate([p["features"] for p in paths])
        am = np.concatenate([p["masks"] for p in paths])
        aat = np.concatenate([p["action_types"] for p in paths])
        aei = np.concatenate([p["element_idxs"] for p in paths])
        olp = np.concatenate([p["old_logprobs"] for p in paths])
        rets = get_returns(paths, gamma)
        adv = calculate_advantage(rets, af, am, agent, device)

        for _ in range(args.update_freq):
            update_value(agent, opt_v, af, am, rets, device)
            update_policy(
                agent, opt_pi, af, am, aat, aei, adv, olp, eps_clip, device,
                entropy_coeff=entropy_coeff,
            )

        avg = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        print(f"{task_name:<22} train batch {t+1}/{args.num_batches} avg_reward={avg:.3f}", flush=True)

    env.close()
    return agent

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tasks = (
        cfg["env"]["tasks"]["simple"] +
        cfg["env"]["tasks"]["medium"] +
        cfg["env"]["tasks"]["hard"]
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Device:", device)
    print("Tasks:", len(tasks))
    print("Smoke params:", {
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "max_ep_len": args.max_ep_len,
        "update_freq": args.update_freq,
        "eval_episodes": args.eval_episodes,
    })
    print("-" * 72)

    summary = []
    for task in tasks:
        try:
            agent = train_smoke_task(task, cfg, device, args)
            metrics = evaluate(
                agent, task, device,
                max_ep_len=args.max_ep_len,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
            )
            row = {"task": task, **metrics}
            summary.append(row)
            print(
                f"{task:<22} eval success={metrics['success_rate']*100:5.1f}% "
                f"mean={metrics['mean_reward']:.3f} std={metrics['std_reward']:.3f}",
                flush=True,
            )
        except Exception as e:
            row = {"task": task, "error": str(e)}
            summary.append(row)
            print(f"{task:<22} ERROR {e}", flush=True)

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("-" * 72)
    print("Saved:", args.results_path)

if __name__ == "__main__":
    main()
