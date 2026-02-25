"""
Collect demonstration trajectories from MiniWoB++ environments.
Uses scripted heuristic policies for simple tasks.
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes
from utils.state_encoder import extract_dom_features
from utils.text_features import dom_elements_to_raw


def get_dom_elements_from_obs(obs):
    """Extract DOM element list from MiniWoB observation."""
    elements = []
    dom_info = obs.get("dom_elements", [])
    for elem in dom_info:
        # Handle numpy arrays from MiniWoB (e.g., left/top can be arrays)
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
            "ref": int(elem["ref"]) if elem.get("ref") is not None else None,
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


def make_click_element_action(env, ref):
    """Create a click-element action by ref."""
    action_types = env.unwrapped.action_space_config.action_types
    click_idx = action_types.index(ActionTypes.CLICK_ELEMENT)
    action = OrderedDict()
    action["action_type"] = np.int64(click_idx)
    action["coords"] = np.array([0.0, 0.0], dtype=np.float32)
    action["ref"] = np.int64(ref)
    action["text"] = ""
    action["field"] = np.int64(0)
    action["key"] = np.int64(0)
    return action


def heuristic_policy(env, obs, dom_elements, utterance, task_name):
    """
    Simple heuristic policy for MiniWoB++ tasks.
    Returns (env_action, action_record) or (None, None).
    """
    utt_lower = utterance.lower()

    # For click tasks: find the element matching the utterance
    if "click" in task_name:
        # First pass: exact text match in utterance
        for i, elem in enumerate(dom_elements):
            text = elem.get("text", "").lower().strip()
            if not text:
                continue
            # Check if the element's text appears in the utterance
            if text in utt_lower and elem["tag"] in ("button", "a", "span", "div", "option", "label"):
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i}
                return env_action, action_record

        # Second pass: check quoted text in utterance
        import re
        quoted = re.findall(r'"([^"]*)"', utterance)
        for q in quoted:
            q_lower = q.lower()
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if text == q_lower:
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i}
                    return env_action, action_record

        # Third pass: click first button
        for i, elem in enumerate(dom_elements):
            if elem["tag"] == "button" and elem.get("text", "").strip():
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i}
                return env_action, action_record

    # For login-user: find username/password fields and submit
    if "login" in task_name:
        import re
        # Extract username and password from utterance
        utt = utterance
        username_match = re.search(r'username\s*[:\s]\s*(\S+)', utt, re.IGNORECASE)
        password_match = re.search(r'password\s*[:\s]\s*(\S+)', utt, re.IGNORECASE)

        if username_match and password_match:
            username = username_match.group(1).strip().rstrip('.')
            password = password_match.group(1).strip().rstrip('.')

            # Find input fields
            for i, elem in enumerate(dom_elements):
                if elem["tag"] == "input" and elem.get("type", "") in ("text", ""):
                    # Check if this is a username field
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_type_action(env, [cx, cy], username)
                    action_record = {"action_type_idx": 1, "element_idx": i}
                    return env_action, action_record

            for i, elem in enumerate(dom_elements):
                if elem["tag"] == "input" and elem.get("type", "") == "password":
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_type_action(env, [cx, cy], password)
                    action_record = {"action_type_idx": 1, "element_idx": i}
                    return env_action, action_record

            # Look for submit button
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if elem["tag"] == "button" and text in ("login", "submit", "log in", "sign in", "ok"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i}
                    return env_action, action_record

    # For enter-text: find text input and type the specified text
    if "enter-text" in task_name:
        import re
        # Extract quoted text from utterance
        quoted = re.findall(r'"([^"]*)"', utterance)
        if quoted:
            target_text = quoted[0]
            for i, elem in enumerate(dom_elements):
                if elem["tag"] in ("input", "textarea") and elem.get("type", "") in ("text", "", "search"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_type_action(env, [cx, cy], target_text)
                    action_record = {"action_type_idx": 1, "element_idx": i}
                    return env_action, action_record

        # Look for submit button after typing
        for i, elem in enumerate(dom_elements):
            text = elem.get("text", "").lower().strip()
            if elem["tag"] == "button" and text in ("submit", "ok", "go"):
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i}
                return env_action, action_record

    return None, None


def collect_mixed_demonstrations(task_name, num_demos=500, success_rate=1.0,
                                  max_steps=20, save_dir="data/demos", seed=42):
    """
    Collect mixed-quality demonstrations (successful + failed).
    success_rate: fraction of successful demos (1.0 = all expert, 0.5 = half expert).
    """
    print(f"\nCollecting {num_demos} mixed demos for {task_name} "
          f"(target quality: {success_rate:.0%})...")

    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=500)

    num_success_target = int(num_demos * success_rate)
    num_fail_target = num_demos - num_success_target

    success_trajs = []
    fail_trajs = []
    attempts = 0
    max_attempts = num_demos * 15

    rng = np.random.RandomState(seed)

    while (len(success_trajs) < num_success_target or
           len(fail_trajs) < num_fail_target) and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset(seed=int(rng.randint(0, 100000)))

        utterance = obs.get("utterance", "")
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_dones = []

        dom_elements = get_dom_elements_from_obs(obs)
        features, mask = extract_dom_features(dom_elements, utterance)
        trajectory_states.append({
            "features": features.tolist(),
            "mask": mask.tolist(),
            "dom_elements_raw": dom_elements_to_raw(dom_elements),
            "utterance": utterance,
        })

        total_reward = 0

        for step in range(max_steps):
            env_action, action_record = heuristic_policy(
                env, obs, dom_elements, utterance, task_name
            )

            if env_action is None:
                env_action = env.action_space.sample()
                action_record = {"action_type_idx": 0, "element_idx": 0}

            obs, reward, done, truncated, info = env.step(env_action)
            total_reward += reward

            dom_elements = get_dom_elements_from_obs(obs)
            features, mask = extract_dom_features(dom_elements, utterance)

            trajectory_actions.append(action_record)
            trajectory_rewards.append(float(reward))
            trajectory_dones.append(bool(done or truncated))
            trajectory_states.append({
                "features": features.tolist(),
                "mask": mask.tolist(),
                "dom_elements_raw": dom_elements_to_raw(dom_elements),
                "utterance": utterance,
            })

            if done or truncated:
                break

        traj = {
            "states": trajectory_states,
            "actions": trajectory_actions,
            "rewards": trajectory_rewards,
            "dones": trajectory_dones,
            "total_reward": float(total_reward),
            "length": len(trajectory_actions),
        }

        if total_reward > 0 and len(success_trajs) < num_success_target:
            success_trajs.append(traj)
        elif total_reward <= 0 and len(fail_trajs) < num_fail_target:
            fail_trajs.append(traj)

        collected = len(success_trajs) + len(fail_trajs)
        if collected % 50 == 0 and collected > 0:
            print(f"  Collected {collected}/{num_demos} "
                  f"({len(success_trajs)} success, {len(fail_trajs)} fail, "
                  f"{attempts} attempts)")

    env.close()

    trajectories = success_trajs + fail_trajs
    rng.shuffle(trajectories)

    actual_sr = len(success_trajs) / max(len(trajectories), 1)

    os.makedirs(save_dir, exist_ok=True)
    quality_tag = f"_q{int(success_rate*100)}"
    save_path = os.path.join(save_dir, f"{task_name}{quality_tag}.json")
    data = {
        "task_name": task_name,
        "num_trajectories": len(trajectories),
        "target_success_rate": success_rate,
        "actual_success_rate": actual_sr,
        "num_successful": len(success_trajs),
        "num_failed": len(fail_trajs),
        "trajectories": trajectories,
    }
    with open(save_path, "w") as f:
        json.dump(data, f)

    print(f"  Saved {len(trajectories)} trajectories to {save_path}")
    print(f"  Quality: {len(success_trajs)} success + {len(fail_trajs)} fail "
          f"= {actual_sr:.1%} actual success rate")

    return trajectories


def collect_demonstrations(task_name, num_demos=500, max_steps=20,
                           save_dir="data/demos", seed=42):
    """Collect demonstration trajectories for a task."""
    print(f"\nCollecting {num_demos} demos for {task_name}...")

    env = gym.make(f"miniwob/{task_name}-v1", render_mode=None, wait_ms=500)

    trajectories = []
    successes = 0
    attempts = 0
    max_attempts = num_demos * 10

    rng = np.random.RandomState(seed)

    while successes < num_demos and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset(seed=int(rng.randint(0, 100000)))

        utterance = obs.get("utterance", "")
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_dones = []

        dom_elements = get_dom_elements_from_obs(obs)
        features, mask = extract_dom_features(dom_elements, utterance)
        trajectory_states.append({
            "features": features.tolist(),
            "mask": mask.tolist(),
            "dom_elements_raw": dom_elements_to_raw(dom_elements),
            "utterance": utterance,
        })

        total_reward = 0

        for step in range(max_steps):
            env_action, action_record = heuristic_policy(
                env, obs, dom_elements, utterance, task_name
            )

            if env_action is None:
                env_action = env.action_space.sample()
                action_record = {"action_type_idx": 0, "element_idx": 0}

            obs, reward, done, truncated, info = env.step(env_action)
            total_reward += reward

            dom_elements = get_dom_elements_from_obs(obs)
            features, mask = extract_dom_features(dom_elements, utterance)

            trajectory_actions.append(action_record)
            trajectory_rewards.append(float(reward))
            trajectory_dones.append(bool(done or truncated))
            trajectory_states.append({
                "features": features.tolist(),
                "mask": mask.tolist(),
                "dom_elements_raw": dom_elements_to_raw(dom_elements),
                "utterance": utterance,
            })

            if done or truncated:
                break

        if total_reward > 0:
            trajectories.append({
                "states": trajectory_states,
                "actions": trajectory_actions,
                "rewards": trajectory_rewards,
                "dones": trajectory_dones,
                "total_reward": float(total_reward),
                "length": len(trajectory_actions),
            })
            successes += 1

            if successes % 50 == 0:
                print(f"  Collected {successes}/{num_demos} successful demos "
                      f"({attempts} attempts)")

    env.close()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_name}.json")
    data = {
        "task_name": task_name,
        "num_trajectories": len(trajectories),
        "success_rate": successes / max(attempts, 1),
        "trajectories": trajectories,
    }
    with open(save_path, "w") as f:
        json.dump(data, f)

    print(f"  Saved {len(trajectories)} trajectories to {save_path}")
    print(f"  Success rate: {successes}/{attempts} = "
          f"{successes/max(attempts,1):.1%}")

    return trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="click-button",
                        help="MiniWoB task name")
    parser.add_argument("--num_demos", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="data/demos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=float, default=None,
                        help="Data quality (0.0-1.0). If set, collects mixed "
                             "success/fail trajectories. E.g., 0.5 = 50%% expert.")
    args = parser.parse_args()

    if args.quality is not None:
        collect_mixed_demonstrations(
            args.task, args.num_demos, success_rate=args.quality,
            save_dir=args.save_dir, seed=args.seed,
        )
    else:
        collect_demonstrations(args.task, args.num_demos, save_dir=args.save_dir,
                              seed=args.seed)
