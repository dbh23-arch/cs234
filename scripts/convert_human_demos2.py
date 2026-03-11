import os
import sys
import gzip
import json
import argparse
import numpy as np

from utils.state_encoder import extract_dom_features, classify_element

MAX_ELEMENTS = 64
LEAF_TAGS = {"span", "div", "li", "td", "th", "p", "h1", "h2", "h3", "h4", "h5", "h6"}


def parse_tag_and_type(tag_raw):
    tag_raw = (tag_raw or "").upper()
    if "_" in tag_raw and tag_raw.startswith("INPUT"):
        base, subtype = tag_raw.split("_", 1)
        return base.lower(), subtype.lower()
    return tag_raw.lower(), ""


def flatten_dom_tree(node):
    elements = []
    _flatten_recursive(node, elements)
    return elements


def _flatten_recursive(node, elements):
    tag, type_attr = parse_tag_and_type(node.get("tag", ""))
    children = node.get("children", [])
    text = str(node.get("text", "") or "")
    value = str(node.get("value", "") or "")
    left = float(node.get("left", 0) or 0)
    top = float(node.get("top", 0) or 0)
    width = float(node.get("width", 0) or 0)
    height = float(node.get("height", 0) or 0)
    focused = bool(node.get("focused", False))

    is_interactive = tag in {"button", "input", "a", "select", "option", "textarea", "label"}
    is_leaf_with_text = (tag in LEAF_TAGS) and text.strip() and not children
    is_leaf_with_value = (tag in LEAF_TAGS) and value.strip() and not children

    # Keep:
    # - interactive elements always
    # - leaf text nodes
    # - value-carrying leaves
    if is_interactive or is_leaf_with_text or is_leaf_with_value:
        elements.append({
            "tag": tag,
            "type": type_attr,
            "text": text.strip(),
            "value": value,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "visible": width > 0 and height > 0,
            "focused": focused,
        })

    for child in children:
        _flatten_recursive(child, elements)


def pad_features(features, mask, max_elements=MAX_ELEMENTS):
    feat_dim = features.shape[1] if len(features.shape) == 2 and features.shape[0] > 0 else 24

    out_feats = np.zeros((max_elements, feat_dim), dtype=np.float32)
    out_mask = np.zeros((max_elements,), dtype=np.float32)

    n = min(len(features), max_elements)
    if n > 0:
        out_feats[:n] = features[:n]
        out_mask[:n] = mask[:n]

    return out_feats, out_mask


def trim_dom_elements(dom_elements, max_elements=MAX_ELEMENTS):
    return dom_elements[:max_elements]


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


def collapse_events_with_state_indices(demo):
    """
    Collapse raw browser events into high-level actions, but keep the source
    state index that should provide the DOM snapshot for that action.
    """
    states = demo.get("states", [])
    events = []

    for idx, state in enumerate(states):
        action = state.get("action")
        if action and action.get("timing") == 1:
            events.append((idx, action))

    actions = []
    i = 0

    while i < len(events):
        state_idx, ev = events[i]
        ev_type = ev.get("type", "")

        if ev_type == "mousedown":
            x, y = ev.get("x", 0), ev.get("y", 0)

            click_end = i + 1
            while click_end < len(events) and events[click_end][1].get("type") in ("mouseup", "click"):
                click_end += 1

            typed_chars = []
            type_end = click_end
            type_state_idx = None

            while type_end < len(events) and events[type_end][1].get("type") in ("keydown", "keypress", "keyup"):
                sidx2, ev2 = events[type_end]
                if ev2.get("type") == "keypress":
                    cc = ev2.get("charCode", 0)
                    if cc > 0:
                        typed_chars.append(chr(cc))
                        type_state_idx = sidx2
                type_end += 1

            # Click action uses the DOM at the mousedown state
            actions.append({
                "type": "click",
                "x": float(x),
                "y": float(y),
                "source_state_idx": state_idx,
            })

            # Type action uses the DOM at the last keypress state if available
            if typed_chars:
                actions.append({
                    "type": "type",
                    "x": float(x),
                    "y": float(y),
                    "text": "".join(typed_chars),
                    "source_state_idx": type_state_idx if type_state_idx is not None else state_idx,
                })
                i = type_end
            else:
                i = click_end

        elif ev_type == "keypress":
            typed_chars = []
            last_keypress_state_idx = state_idx

            while i < len(events) and events[i][1].get("type") in ("keydown", "keypress", "keyup"):
                sidx2, ev2 = events[i]
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


def build_state_from_dom(dom_tree, utterance):
    dom_elements = flatten_dom_tree(dom_tree)
    dom_elements = trim_dom_elements(dom_elements, MAX_ELEMENTS)

    features, valid_mask = extract_dom_features(dom_elements, utterance)
    features = np.asarray(features, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.float32)

    features, valid_mask = pad_features(features, valid_mask, MAX_ELEMENTS)

    raw_elements = []
    for elem in dom_elements:
        raw_elements.append({
            "tag": elem["tag"],
            "type": elem.get("type", ""),
            "text": elem.get("text", ""),
            "value": elem.get("value", ""),
            "left": elem["left"],
            "top": elem["top"],
            "width": elem["width"],
            "height": elem["height"],
            "visible": elem.get("visible", True),
            "focused": elem.get("focused", False),
        })

    return {
        "features": features.tolist(),
        "mask": valid_mask.tolist(),
        "dom_elements_raw": raw_elements,
        "utterance": utterance,
    }


def convert_demo_to_training_traj(demo, task_name):
    utterance = demo.get("utterance", "")
    reward = float(demo.get("rawReward", 0))
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
        dom_tree = raw_states[sidx].get("dom", {})
        state_dict = build_state_from_dom(dom_tree, utterance)

        dom_elements = state_dict["dom_elements_raw"]

        if action["type"] in {"click", "type"}:
            elem_idx = find_clicked_element(dom_elements, action["x"], action["y"])

            # fallback for typing: choose focused input first
            if elem_idx < 0 and action["type"] == "type":
                for j, elem in enumerate(dom_elements):
                    if elem["tag"] in {"input", "textarea", "select"} and elem.get("focused", False):
                        elem_idx = j
                        break

            if elem_idx < 0:
                continue

            out_states.append(state_dict)
            out_actions.append({
                "action_type_idx": 0 if action["type"] == "click" else 1,
                "element_idx": int(elem_idx),
            })
            rewards.append([reward])
            dones.append([step_i == len(high_level_actions) - 1])

    if not out_actions:
        return None

    return {
        "states": out_states,
        "actions": out_actions,
        "rewards": rewards,
        "dones": dones,
        "total_reward": sum(r[0] for r in rewards),
        "length": len(out_actions),
    }