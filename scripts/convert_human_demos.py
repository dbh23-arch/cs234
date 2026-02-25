"""
Convert Stanford MiniWoB++ human demonstrations into our training format.

Usage:
    python scripts/convert_human_demos.py --task click-button
    python scripts/convert_human_demos.py --task all
"""

import os
import sys
import gzip
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state_encoder import extract_dom_features, classify_element

LEAF_TAGS = {"span", "div", "li", "td", "th", "p", "h1", "h2", "h3", "h4", "h5", "h6"}


def flatten_dom_tree(node):
    elements = []
    _flatten_recursive(node, elements)
    return elements


def _flatten_recursive(node, elements):
    tag_raw = node.get("tag", "").upper()
    if "_" in tag_raw and tag_raw.startswith("INPUT"):
        parts = tag_raw.split("_", 1)
        tag = parts[0]
        type_attr = parts[1].lower()
    else:
        tag = tag_raw
        type_attr = ""

    children = node.get("children", [])
    text = node.get("text", "")
    left = node.get("left", 0)
    top = node.get("top", 0)
    width = node.get("width", 0)
    height = node.get("height", 0)

    tag_lower = tag.lower()
    is_interactive = tag_lower in {"button", "input", "a", "select", "option", "textarea", "label"}
    is_leaf_with_text = (tag_lower in LEAF_TAGS) and text.strip() and not children
    has_size = width > 0 and height > 0

    if has_size and (is_interactive or is_leaf_with_text):
        elements.append({
            "tag": tag.lower(), "type": type_attr,
            "text": text.strip() if text else "",
            "value": node.get("value", ""),
            "left": float(left), "top": float(top),
            "width": float(width), "height": float(height),
            "visible": True, "focused": bool(node.get("focused", False)),
        })
    for child in children:
        _flatten_recursive(child, elements)


def find_clicked_element(elements, x, y, tolerance=5):
    best_idx = -1
    best_area = float("inf")
    for i, elem in enumerate(elements):
        el = elem["left"]
        et = elem["top"]
        er = el + elem["width"]
        eb = et + elem["height"]
        if (el - tolerance <= x <= er + tolerance and
                et - tolerance <= y <= eb + tolerance):
            area = elem["width"] * elem["height"]
            if area < best_area:
                best_area = area
                best_idx = i
    if best_idx == -1:
        best_dist = float("inf")
        for i, elem in enumerate(elements):
            cx = elem["left"] + elem["width"] / 2
            cy = elem["top"] + elem["height"] / 2
            dist = (cx - x) ** 2 + (cy - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx


def extract_actions(demo):
    actions = []
    states = demo.get("states", [])
    events = []
    for state in states:
        action = state.get("action")
        if action and action.get("timing") == 1:
            events.append(action)

    i = 0
    while i < len(events):
        ev = events[i]
        ev_type = ev.get("type", "")

        if ev_type == "mousedown":
            x, y = ev.get("x", 0), ev.get("y", 0)
            click_end = i + 1
            while click_end < len(events) and events[click_end].get("type") in ("mouseup", "click"):
                click_end += 1
            typed_chars = []
            type_end = click_end
            while type_end < len(events) and events[type_end].get("type") in ("keydown", "keypress", "keyup"):
                if events[type_end].get("type") == "keypress":
                    cc = events[type_end].get("charCode", 0)
                    if cc > 0:
                        typed_chars.append(chr(cc))
                type_end += 1
            if typed_chars:
                actions.append({"type": "click", "x": float(x), "y": float(y)})
                actions.append({"type": "type", "x": float(x), "y": float(y), "text": "".join(typed_chars)})
                i = type_end
            else:
                actions.append({"type": "click", "x": float(x), "y": float(y)})
                i = click_end
        elif ev_type == "keypress":
            typed_chars = []
            while i < len(events) and events[i].get("type") in ("keydown", "keypress", "keyup"):
                if events[i].get("type") == "keypress":
                    cc = events[i].get("charCode", 0)
                    if cc > 0:
                        typed_chars.append(chr(cc))
                i += 1
            if typed_chars:
                actions.append({"type": "type", "x": 0, "y": 0, "text": "".join(typed_chars)})
        else:
            i += 1
    return actions


def convert_demo(demo, task_name):
    utterance = demo.get("utterance", "")
    reward = demo.get("rawReward", 0)
    if reward <= 0:
        return None
    states = demo.get("states", [])
    if not states:
        return None
    dom_tree = states[0].get("dom", {})
    elements = flatten_dom_tree(dom_tree)
    if not elements:
        return None
    actions = extract_actions(demo)
    if not actions:
        return None
    features, valid_mask = extract_dom_features(elements, utterance)
    steps = []
    for action in actions:
        if action["type"] == "click":
            elem_idx = find_clicked_element(elements, action["x"], action["y"])
            if elem_idx < 0:
                continue
            steps.append({"action_type_idx": 0, "element_idx": int(elem_idx)})
        elif action["type"] == "type":
            elem_idx = find_clicked_element(elements, action["x"], action["y"])
            if elem_idx < 0:
                elem_idx = 0
            steps.append({"action_type_idx": 1, "element_idx": int(elem_idx), "typed_text": action["text"]})
    if not steps:
        return None
    raw_elements = []
    for elem in elements:
        raw_elements.append({
            "tag": elem["tag"], "type": elem.get("type", ""),
            "text": elem.get("text", ""), "value": elem.get("value", ""),
            "left": elem["left"], "top": elem["top"],
            "width": elem["width"], "height": elem["height"],
            "visible": elem.get("visible", True),
            "focused": elem.get("focused", False),
        })
    return {
        "task": task_name, "utterance": utterance,
        "reward": float(reward),
        "features": features.tolist(), "valid_mask": valid_mask.tolist(),
        "dom_elements": raw_elements, "steps": steps, "source": "human",
    }


def convert_task(task_name, human_raw_dir, output_dir, max_demos=None):
    all_trajectories = []
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
            if max_demos and files_converted >= max_demos:
                break
            files_found += 1
            filepath = os.path.join(task_path, demo_file)
            try:
                with gzip.open(filepath, "rt") as f:
                    demo = json.load(f)
                traj = convert_demo(demo, task_name)
                if traj is not None:
                    all_trajectories.append(traj)
                    files_converted += 1
            except Exception as e:
                print(f"  Warning: Failed to convert {filepath}: {e}")
        if max_demos and files_converted >= max_demos:
            break
    if not all_trajectories:
        print(f"  No valid trajectories found for {task_name} ({files_found} files scanned)")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}_human.json")
    with open(output_path, "w") as f:
        json.dump(all_trajectories, f)
    print(f"  Converted {files_converted}/{files_found} demos -> {output_path}")
    return files_converted


def get_available_tasks(human_raw_dir):
    tasks = set()
    for batch_folder in os.listdir(human_raw_dir):
        batch_path = os.path.join(human_raw_dir, batch_folder)
        if not os.path.isdir(batch_path):
            continue
        for task_folder in os.listdir(batch_path):
            task_path = os.path.join(batch_path, task_folder)
            if os.path.isdir(task_path):
                gz_files = [f for f in os.listdir(task_path) if f.endswith(".json.gz")]
                if gz_files:
                    tasks.add(task_folder)
    return sorted(tasks)


def main():
    parser = argparse.ArgumentParser(description="Convert Stanford human demos to training format")
    parser.add_argument("--task", type=str, default="click-button",
                        help="Task name or 'all' for all available tasks")
    parser.add_argument("--human_raw_dir", type=str, default="data/human_raw",
                        help="Path to cloned Stanford demos")
    parser.add_argument("--output_dir", type=str, default="data/demos",
                        help="Output directory for converted demos")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="Max demos per task (default: all)")
    args = parser.parse_args()

    if not os.path.isdir(args.human_raw_dir):
        print(f"Error: Human demos directory not found: {args.human_raw_dir}")
        print("Clone it with: git clone https://github.com/stanfordnlp/miniwob-plusplus-demos data/human_raw")
        sys.exit(1)

    if args.task == "all":
        tasks = get_available_tasks(args.human_raw_dir)
        print(f"Found {len(tasks)} tasks with human demos: {tasks}")
    else:
        tasks = [args.task]

    total = 0
    for task in tasks:
        print(f"\nConverting: {task}")
        count = convert_task(task, args.human_raw_dir, args.output_dir, args.max_demos)
        total += count

    print(f"\nDone! Total trajectories converted: {total}")


if __name__ == "__main__":
    main()
