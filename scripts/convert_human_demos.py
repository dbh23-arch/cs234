"""
Convert Stanford MiniWoB++ human demonstrations into our training format.

This version:
1. Handles nested DOM trees.
2. Recomputes features from the correct DOM snapshot for each action.
3. Preserves text, value, focused, and geometry information.
4. Reads multiple files from an input folder like the original script.

Usage:
    python scripts/convert_human_demos.py --task click-button
    python scripts/convert_human_demos.py --task all

Expected input layout:
    <input_dir>/<task>/*.json
    <input_dir>/<task>/*.json.gz

Output:
    <output_dir>/<task>.json
"""

import os
import sys
import gzip
import json
import argparse
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state_encoder import extract_dom_features, classify_element
LEAF_TAGS = {"span", "div", "li", "td", "th", "p", "h1", "h2", "h3", "h4", "h5", "h6", "t"}
INTERACTIVE_TAGS = {"button", "input", "a", "select", "option", "textarea", "label"}

MAX_ELEMENTS = 64
def parse_tag_and_type(tag_raw):
    tag_raw = (tag_raw or "").upper()
    if "_" in tag_raw and tag_raw.startswith("INPUT"):
        base, subtype = tag_raw.split("_", 1)
        return base.lower(), subtype.lower()
    return tag_raw.lower(), ""


def collect_text_recursive(node):
    pieces = []

    def _walk(n):
        txt = str(n.get("text", "") or "").strip()
        if txt:
            pieces.append(txt)
        for child in n.get("children", []):
            _walk(child)

    _walk(node)
    return " ".join(pieces).strip()


def flatten_dom_tree(node):
    elements = []
    _flatten_recursive(node, elements)
    return elements


def _flatten_recursive(node, elements):
    if not isinstance(node, dict):
        return

    tag, type_attr = parse_tag_and_type(node.get("tag", ""))
    children = node.get("children", [])
    text = str(node.get("text", "") or "")
    value = str(node.get("value", "") or "")
    left = float(node.get("left", 0) or 0)
    top = float(node.get("top", 0) or 0)
    width = float(node.get("width", 0) or 0)
    height = float(node.get("height", 0) or 0)
    focused = bool(node.get("focused", False))
    visible = width > 0 and height > 0

    # Special case: LABEL that wraps checkbox/radio + visible text
    if tag == "label" and visible:
        label_text = collect_text_recursive(node)

        interactive_children = []
        for child in children:
            child_tag, child_type = parse_tag_and_type(child.get("tag", ""))
            if child_tag in {"input", "button", "select", "textarea", "a"}:
                interactive_children.append((child, child_tag, child_type))

        if interactive_children:
            for child, child_tag, child_type in interactive_children:
                c_left = float(child.get("left", 0) or 0)
                c_top = float(child.get("top", 0) or 0)
                c_width = float(child.get("width", 0) or 0)
                c_height = float(child.get("height", 0) or 0)

                elements.append({
                    "tag": child_tag,
                    "type": child_type,
                    "text": label_text,
                    "value": str(child.get("value", "") or ""),
                    "left": c_left,
                    "top": c_top,
                    "width": c_width,
                    "height": c_height,
                    "visible": c_width > 0 and c_height > 0,
                    "focused": bool(child.get("focused", False)),
                })
            return

    has_payload = bool(text.strip()) or bool(value.strip())
    is_leaf = len(children) == 0
    is_interactive = tag in {"button", "input", "a", "select", "option", "textarea"}
    is_leaf_textual = is_leaf and (tag in LEAF_TAGS) and has_payload

    if visible and (is_interactive or is_leaf_textual):
        elements.append({
            "tag": tag,
            "type": type_attr,
            "text": text.strip(),
            "value": value,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "visible": visible,
            "focused": focused,
        })

    for child in children:
        _flatten_recursive(child, elements)


def dedupe_elements(elements):
    deduped = []
    seen = set()

    for e in elements:
        key = (
            e["tag"],
            e.get("type", ""),
            e.get("text", ""),
            e.get("value", ""),
            e["left"],
            e["top"],
            e["width"],
            e["height"],
            e.get("focused", False),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    return deduped


def trim_dom_elements(elements, max_elements=MAX_ELEMENTS):
    return elements[:max_elements]


def pad_features(features, mask, max_elements=MAX_ELEMENTS):
    features = np.asarray(features, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    feat_dim = features.shape[1] if features.size > 0 else 24

    out_feats = np.zeros((max_elements, feat_dim), dtype=np.float32)
    out_mask = np.zeros((max_elements,), dtype=np.float32)

    n = min(len(features), max_elements)
    if n > 0:
        out_feats[:n] = features[:n]
        out_mask[:n] = mask[:n]

    return out_feats, out_mask


def build_state_from_dom(dom_tree, utterance):
    dom_elements = flatten_dom_tree(dom_tree)
    dom_elements = dedupe_elements(dom_elements)
    dom_elements = trim_dom_elements(dom_elements, MAX_ELEMENTS)

    features, valid_mask = extract_dom_features(dom_elements, utterance)
    features, valid_mask = pad_features(features, valid_mask, MAX_ELEMENTS)

    return {
        "features": features.tolist(),
        "mask": valid_mask.tolist(),
        "dom_elements_raw": dom_elements,
        "utterance": utterance,
    }


def point_in_box(x, y, elem, tolerance=5):
    left = elem["left"] - tolerance
    top = elem["top"] - tolerance
    right = elem["left"] + elem["width"] + tolerance
    bottom = elem["top"] + elem["height"] + tolerance
    return left <= x <= right and top <= y <= bottom


def find_clicked_element(elements, x, y, tolerance=5):
    best_idx = -1
    best_area = float("inf")

    for i, elem in enumerate(elements):
        if point_in_box(x, y, elem, tolerance=tolerance):
            area = elem["width"] * elem["height"]
            if area < best_area:
                best_area = area
                best_idx = i

    if best_idx != -1:
        return best_idx

    best_dist = float("inf")
    for i, elem in enumerate(elements):
        cx = elem["left"] + elem["width"] / 2
        cy = elem["top"] + elem["height"] / 2
        dist = (cx - x) ** 2 + (cy - y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def collapse_events_with_state_indices(demo):
    """
    Convert raw browser events into high-level actions while retaining the
    source state index whose DOM snapshot should be used.
    """
    states = demo.get("states", [])
    timed_events = []

    for idx, state in enumerate(states):
        action = state.get("action")
        if action is not None and action.get("timing") == 1:
            timed_events.append((idx, action))

    actions = []
    i = 0

    while i < len(timed_events):
        state_idx, ev = timed_events[i]
        ev_type = ev.get("type", "")

        if ev_type == "mousedown":
            x = float(ev.get("x", 0))
            y = float(ev.get("y", 0))

            j = i + 1
            while j < len(timed_events) and timed_events[j][1].get("type") in {"mouseup", "click"}:
                j += 1

            typed_chars = []
            last_keypress_state_idx = None
            k = j
            while k < len(timed_events) and timed_events[k][1].get("type") in {"keydown", "keypress", "keyup"}:
                sidx2, ev2 = timed_events[k]
                if ev2.get("type") == "keypress":
                    cc = ev2.get("charCode", 0)
                    if cc > 0:
                        typed_chars.append(chr(cc))
                        last_keypress_state_idx = sidx2
                k += 1

            actions.append({
                "type": "click",
                "x": x,
                "y": y,
                "source_state_idx": state_idx,
            })

            if typed_chars:
                actions.append({
                    "type": "type",
                    "x": x,
                    "y": y,
                    "text": "".join(typed_chars),
                    "source_state_idx": last_keypress_state_idx if last_keypress_state_idx is not None else state_idx,
                })
                i = k
            else:
                i = j

        elif ev_type == "keypress":
            typed_chars = []
            last_keypress_state_idx = state_idx

            while i < len(timed_events) and timed_events[i][1].get("type") in {"keydown", "keypress", "keyup"}:
                sidx2, ev2 = timed_events[i]
                if ev2.get("type") == "keypress":
                    cc = ev2.get("charCode", 0)
                    if cc > 0:
                        typed_chars.append(chr(cc))
                        last_keypress_state_idx = sidx2
                i += 1

            if typed_chars:
                actions.append({
                    "type": "type",
                    "x": 0.0,
                    "y": 0.0,
                    "text": "".join(typed_chars),
                    "source_state_idx": last_keypress_state_idx,
                })
        else:
            i += 1

    return actions


def convert_demo_to_training_traj(demo, task_name):
    utterance = demo.get("utterance", "")
    reward = float(demo.get("rawReward", demo.get("reward", 0.0)))

    if reward <= 0:
        return None

    raw_states = demo.get("states", [])
    if not raw_states:
        return None

    high_level_actions = collapse_events_with_state_indices(demo)
    if not high_level_actions:
        return None

    out_states = []
    out_actions = []
    rewards = []
    dones = []

    for step_i, action in enumerate(high_level_actions):
        sidx = action["source_state_idx"]
        if sidx < 0 or sidx >= len(raw_states):
            continue

        dom_tree = raw_states[sidx].get("dom", {})
        state_dict = build_state_from_dom(dom_tree, utterance)
        dom_elements = state_dict["dom_elements_raw"]

        if not dom_elements:
            continue

        if action["type"] == "click":
            elem_idx = find_clicked_element(dom_elements, action["x"], action["y"])
            if elem_idx < 0 or elem_idx >= len(dom_elements):
                continue

            action_type_idx = 0

        elif action["type"] == "type":
            elem_idx = -1

            if action["x"] != 0.0 or action["y"] != 0.0:
                elem_idx = find_clicked_element(dom_elements, action["x"], action["y"])

            if elem_idx < 0 or elem_idx >= len(dom_elements):
                for j, elem in enumerate(dom_elements):
                    if elem["tag"] in {"input", "textarea", "select"} and elem.get("focused", False):
                        elem_idx = j
                        break

            if elem_idx < 0 or elem_idx >= len(dom_elements):
                for j, elem in enumerate(dom_elements):
                    if elem["tag"] in {"input", "textarea", "select"}:
                        elem_idx = j
                        break

            if elem_idx < 0 or elem_idx >= len(dom_elements):
                continue

            action_type_idx = 1

        else:
            continue

        out_states.append(state_dict)
        out_actions.append({
            "action_type_idx": action_type_idx,
            "element_idx": int(elem_idx),
        })
        rewards.append(reward)
        dones.append(step_i == len(high_level_actions) - 1)

    if not out_actions:
        return None

    return {
        "states": out_states,
        "actions": out_actions,
        "rewards": [float(r) for r in rewards],
        "dones": [bool(d) for d in dones],
        "total_reward": float(sum(rewards)),
        "length": len(out_actions),
    }


def load_demo_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_demo_list(raw_data):
    """
    Accept a few common formats:
    - list of demos
    - {"demos": [...]}
    - {"trajectories": [...]}
    - single demo dict
    """
    if isinstance(raw_data, list):
        return raw_data
    if isinstance(raw_data, dict) and "demos" in raw_data and isinstance(raw_data["demos"], list):
        return raw_data["demos"]
    if isinstance(raw_data, dict) and "trajectories" in raw_data and isinstance(raw_data["trajectories"], list):
        return raw_data["trajectories"]
    if isinstance(raw_data, dict):
        return [raw_data]
    return []


def iter_task_files(task_dir):
    patterns = [
        os.path.join(task_dir, "*.json"),
        os.path.join(task_dir, "*.json.gz"),
        os.path.join(task_dir, "*.gz"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob(pattern))
    return sorted(set(paths))


def save_output(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def convert_task(task_name, input_dir, output_dir):
    task_dir = os.path.join(input_dir, task_name)
    input_files = iter_task_files(task_dir)

    if not input_files:
        print(f"[WARN] No input files found for task '{task_name}' in {task_dir}")
        return

    output_trajectories = []
    total_demos = 0
    kept_demos = 0

    for path in input_files:
        try:
            raw_data = load_demo_file(path)
            demos = extract_demo_list(raw_data)

            for demo in demos:
                total_demos += 1
                traj = convert_demo_to_training_traj(demo, task_name)
                if traj is not None:
                    output_trajectories.append(traj)
                    kept_demos += 1
        except Exception as e:
            print(f"[WARN] Failed to process {path}: {e}")

    success_rate = (kept_demos / total_demos) if total_demos > 0 else 0.0

    out = {
        "task_name": task_name,
        "num_trajectories": len(output_trajectories),
        "success_rate": success_rate,
        "trajectories": output_trajectories,
    }

    output_path = os.path.join(output_dir, f"{task_name}.json")
    save_output(out, output_path)
    print(f"Converted {task_name}: kept {kept_demos}/{total_demos} successful demos -> {output_path}")


def discover_tasks(input_dir):
    tasks = []
    if not os.path.isdir(input_dir):
        return tasks

    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isdir(path):
            tasks.append(name)
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name or 'all'")
    parser.add_argument("--input_dir", type=str, default="data/miniwob_human")
    parser.add_argument("--output_dir", type=str, default="data/miniwob_converted")
    args = parser.parse_args()

    if args.task == "all":
        tasks = discover_tasks(args.input_dir)
        if not tasks:
            print(f"[WARN] No task folders found in {args.input_dir}")
            return
        for task_name in tasks:
            convert_task(task_name, args.input_dir, args.output_dir)
    else:
        convert_task(args.task, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
