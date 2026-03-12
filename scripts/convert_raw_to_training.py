
import os
import sys
import gzip
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state_encoder import extract_dom_features
from utils.text_features import dom_elements_to_raw
from scripts.convert_human_demos import flatten_dom_tree, find_clicked_element, extract_actions

def convert_demo_to_training_format(demo, task_name):
    utterance = demo.get("utterance", "")
    reward = demo.get("rawReward", 0)
    if reward <= 0:
        return None

    raw_states = demo.get("states", [])
    if not raw_states:
        return None

    
    actions = extract_actions(demo)
    if not actions:
        return None

    
    initial_dom = raw_states[0].get("dom", {})
    elements = flatten_dom_tree(initial_dom)
    if not elements:
        return None

    
    
    
    action_state_indices = _find_action_state_indices(raw_states)

    trajectory_states = []
    trajectory_actions = []
    trajectory_rewards = []
    trajectory_dones = []

    
    current_elements = elements
    for i, action in enumerate(actions):
        
        if i < len(action_state_indices) and action_state_indices[i] < len(raw_states):
            state_idx = action_state_indices[i]
            dom_tree = raw_states[state_idx].get("dom")
            if dom_tree:
                new_elements = flatten_dom_tree(dom_tree)
                if new_elements:
                    current_elements = new_elements

        
        features, mask = extract_dom_features(current_elements, utterance)

        
        raw_elems = []
        for elem in current_elements:
            raw_elems.append({
                "tag": elem["tag"], "type": elem.get("type", ""),
                "text": elem.get("text", ""), "value": elem.get("value", ""),
                "left": elem["left"], "top": elem["top"],
                "width": elem["width"], "height": elem["height"],
                "visible": elem.get("visible", True),
                "focused": elem.get("focused", False),
            })

        trajectory_states.append({
            "features": features.tolist(),
            "mask": mask.tolist(),
            "dom_elements_raw": raw_elems,
            "utterance": utterance,
        })

        
        if action["type"] == "click":
            elem_idx = find_clicked_element(current_elements, action["x"], action["y"])
            if elem_idx < 0:
                elem_idx = 0
            trajectory_actions.append({"action_type_idx": 0, "element_idx": int(elem_idx)})
        elif action["type"] == "type":
            elem_idx = find_clicked_element(current_elements, action["x"], action["y"])
            if elem_idx < 0:
                elem_idx = 0
            trajectory_actions.append({"action_type_idx": 1, "element_idx": int(elem_idx)})

        
        is_last = (i == len(actions) - 1)
        trajectory_rewards.append(float(reward) if is_last else 0.0)
        trajectory_dones.append(is_last)

    
    features, mask = extract_dom_features(current_elements, utterance)
    raw_elems = []
    for elem in current_elements:
        raw_elems.append({
            "tag": elem["tag"], "type": elem.get("type", ""),
            "text": elem.get("text", ""), "value": elem.get("value", ""),
            "left": elem["left"], "top": elem["top"],
            "width": elem["width"], "height": elem["height"],
            "visible": elem.get("visible", True),
            "focused": elem.get("focused", False),
        })
    trajectory_states.append({
        "features": features.tolist(),
        "mask": mask.tolist(),
        "dom_elements_raw": raw_elems,
        "utterance": utterance,
    })

    if not trajectory_actions:
        return None

    return {
        "states": trajectory_states,
        "actions": trajectory_actions,
        "rewards": trajectory_rewards,
        "dones": trajectory_dones,
        "total_reward": float(reward),
        "length": len(trajectory_actions),
    }

def _find_action_state_indices(raw_states):
    indices = []
    i = 0
    while i < len(raw_states):
        state = raw_states[i]
        action = state.get("action")
        if action and action.get("timing") == 1:
            ev_type = action.get("type", "")
            if ev_type == "mousedown":
                indices.append(i)
                
                i += 1
                while i < len(raw_states):
                    a = raw_states[i].get("action")
                    if a and a.get("timing") == 1 and a.get("type") in ("mouseup", "click"):
                        i += 1
                    else:
                        break
                
                typed = False
                while i < len(raw_states):
                    a = raw_states[i].get("action")
                    if a and a.get("type") in ("keydown", "keypress", "keyup"):
                        if a.get("type") == "keypress" and not typed:
                            typed = True
                        i += 1
                    else:
                        break
                if typed:
                    
                    indices.append(indices[-1])
                continue
            elif ev_type == "keypress":
                indices.append(i)
                while i < len(raw_states):
                    a = raw_states[i].get("action")
                    if a and a.get("type") in ("keydown", "keypress", "keyup"):
                        i += 1
                    else:
                        break
                continue
        i += 1
    return indices

def convert_task(task_name, human_raw_dir, output_dir, target_demos=500):
    trajectories = []
    files_found = 0
    files_converted = 0

    for batch_folder in sorted(os.listdir(human_raw_dir)):
        batch_path = os.path.join(human_raw_dir, batch_folder)
        if not os.path.isdir(batch_path):
            continue
        task_path = os.path.join(batch_path, task_name)
        if not os.path.isdir(task_path):
            continue

        demo_files = sorted([f for f in os.listdir(task_path) if f.endswith(".json.gz")])
        for demo_file in demo_files:
            files_found += 1
            filepath = os.path.join(task_path, demo_file)
            try:
                with gzip.open(filepath, "rt") as f:
                    demo = json.load(f)
                traj = convert_demo_to_training_format(demo, task_name)
                if traj is not None:
                    trajectories.append(traj)
                    files_converted += 1
            except Exception as e:
                print(f"  Warning: Failed to convert {filepath}: {e}")

    if not trajectories:
        print(f"  No valid trajectories for {task_name} ({files_found} files scanned)")
        return 0

    
    if len(trajectories) < target_demos:
        print(f"  Only {len(trajectories)} demos available (target: {target_demos})")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}.json")

    data = {
        "task_name": task_name,
        "num_trajectories": len(trajectories),
        "success_rate": files_converted / max(files_found, 1),
        "trajectories": trajectories,
    }
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"  Saved {len(trajectories)} trajectories -> {output_path}")
    return len(trajectories)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--human_raw_dir", type=str, default=None,
                        help="Path to raw miniwob-plusplus demos "
                             "(default: auto-detect miniwob-plusplus-demos)")
    parser.add_argument("--output_dir", type=str, default="data/demos")
    parser.add_argument("--target_demos", type=int, default=500)
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to convert")
    args = parser.parse_args()

    human_raw_dir = args.human_raw_dir
    if human_raw_dir is None:
        candidates = ["miniwob-plusplus-demos", "data/human_raw"]
        human_raw_dir = next((p for p in candidates if os.path.isdir(p)), None)

    if not human_raw_dir or not os.path.isdir(human_raw_dir):
        print(f"Error: {human_raw_dir} not found")
        print("Clone: git clone https://github.com/stanfordnlp/miniwob-plusplus-demos data/human_raw")
        sys.exit(1)
    print(f"Using human demo source: {human_raw_dir}")

    if args.tasks:
        tasks = args.tasks
    elif args.task == "all":
        
        tasks = [
            "click-button", "click-link", "click-option", "click-dialog", "click-dialog-2",
            "login-user", "enter-text", "search-engine", "navigate-tree", "click-checkboxes",
            "email-inbox", "social-media", "use-autocomplete",
        ]
    else:
        tasks = [args.task]

    total = 0
    for task in tasks:
        print(f"\nConverting: {task}")
        count = convert_task(task, human_raw_dir, args.output_dir, args.target_demos)
        total += count

    print(f"\nDone! Total trajectories: {total}")

if __name__ == "__main__":
    main()
