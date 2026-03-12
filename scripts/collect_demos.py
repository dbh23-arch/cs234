"""Collect heuristic demonstrations for MiniWoB++ tasks.

Each task has a hand-coded policy that solves it (or tries to).
We run these to generate training data for the offline RL methods.
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
    elements = []
    dom_info = obs.get("dom_elements", [])
    for elem in dom_info:
        
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

def make_click_element_action(env, ref):
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

class HeuristicState:
    def __init__(self):
        self.steps_done = []  
        self.data = {}  

    def has_done(self, action_desc):
        return action_desc in self.steps_done

    def mark_done(self, action_desc):
        self.steps_done.append(action_desc)

def _is_input(elem, input_types=("text", "", "search")):
    tag = elem.get("tag", "")
    typ = elem.get("type", "")
    
    if tag in ("input", "textarea") and typ in input_types:
        return True
    
    if tag.startswith("input"):
        suffix = tag[5:].lstrip("_")  
        if not suffix:
            suffix = ""
        if suffix in input_types:
            return True
    if tag == "textarea":
        return True
    return False

def _is_password(elem):
    tag = elem.get("tag", "")
    typ = elem.get("type", "")
    return (tag == "input" and typ == "password") or tag == "input_password"

def heuristic_policy(env, obs, dom_elements, utterance, task_name,
                     heuristic_state=None):
    import re
    utt_lower = utterance.lower()
    if heuristic_state is None:
        heuristic_state = HeuristicState()

    
    if "click-option" in task_name:
        
        
        
        quoted = re.findall(r'"([^"]*)"', utterance)
        if quoted:
            target = quoted[0]
        else:
            m = re.search(r'[Ss]elect\s+(\S+)', utterance)
            target = m.group(1) if m else ""
        target_lower = target.lower()

        if not heuristic_state.has_done("clicked_radio"):
            
            
            for i, elem in enumerate(dom_elements):
                if elem["tag"] == "t" and elem.get("text", "").lower().strip() == target_lower:
                    
                    for j in range(i - 1, -1, -1):
                        if dom_elements[j]["tag"] in ("input_radio", "input_checkbox"):
                            radio = dom_elements[j]
                            cx = radio["left"] + radio["width"] / 2
                            cy = radio["top"] + radio["height"] / 2
                            env_action = make_click_action(env, [cx, cy])
                            action_record = {"action_type_idx": 0,
                                             "element_idx": j, "text": ""}
                            heuristic_state.mark_done("clicked_radio")
                            return env_action, action_record
            
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if text == target_lower and elem["tag"] in ("option", "li", "div", "span", "label"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    heuristic_state.mark_done("clicked_radio")
                    return env_action, action_record

        
        if not heuristic_state.has_done("clicked_submit"):
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if elem["tag"] == "button" and text in ("submit", "ok"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    heuristic_state.mark_done("clicked_submit")
                    return env_action, action_record

    
    if "click-dialog" in task_name:
        
        quoted = re.findall(r'"([^"]*)"', utterance)
        target_label = quoted[0].lower() if quoted else "x"

        if target_label == "x":
            
            for i, elem in enumerate(dom_elements):
                if elem["tag"] == "button":
                    text = elem.get("text", "").lower().strip()
                    if text in ("", "x", "\u00d7", "close"):
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        env_action = make_click_action(env, [cx, cy])
                        action_record = {"action_type_idx": 0,
                                         "element_idx": i, "text": ""}
                        return env_action, action_record
        else:
            
            for i, elem in enumerate(dom_elements):
                if elem["tag"] == "button":
                    text = elem.get("text", "").lower().strip()
                    if text == target_label:
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        env_action = make_click_action(env, [cx, cy])
                        action_record = {"action_type_idx": 0,
                                         "element_idx": i, "text": ""}
                        return env_action, action_record

    
    if "click-checkboxes" in task_name:
        
        
        quoted = re.findall(r'"([^"]*)"', utterance)
        targets = [q.lower() for q in quoted]
        if not targets:
            
            m = re.search(r'(?:select|click|check)\s+(.+?)(?:\s+and\s+|\s*$)', utt_lower)
            if m:
                raw = m.group(1)
                targets = [t.strip().strip(',') for t in re.split(r',\s*|\s+and\s+', raw)]

        
        if not heuristic_state.has_done("checked_all"):
            checked_any = False
            for i, elem in enumerate(dom_elements):
                if elem["tag"] in ("input_checkbox", "input_radio"):
                    
                    label_text = ""
                    
                    for j in range(i + 1, min(i + 3, len(dom_elements))):
                        t = dom_elements[j].get("text", "").strip()
                        if t:
                            label_text = t.lower()
                            break
                    
                    for j in range(i + 1, min(i + 3, len(dom_elements))):
                        if dom_elements[j]["tag"] == "t":
                            label_text = dom_elements[j].get("text", "").lower().strip()
                            break
                    if label_text and label_text in targets:
                        key = f"checked_{label_text}"
                        if not heuristic_state.has_done(key):
                            cx = elem["left"] + elem["width"] / 2
                            cy = elem["top"] + elem["height"] / 2
                            env_action = make_click_action(env, [cx, cy])
                            action_record = {"action_type_idx": 0,
                                             "element_idx": i, "text": ""}
                            heuristic_state.mark_done(key)
                            return env_action, action_record

            
            heuristic_state.mark_done("checked_all")
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if elem["tag"] == "button" and text in ("submit", "ok"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    return env_action, action_record

    
    if "navigate-tree" in task_name:
        
        quoted = re.findall(r'"([^"]*)"', utterance)
        target = quoted[0].lower() if quoted else ""
        if not target:
            m = re.search(r'named\s+"?(\w+)"?', utterance)
            target = m.group(1).lower() if m else ""

        
        for i, elem in enumerate(dom_elements):
            text = elem.get("text", "").lower().strip()
            if text == target and elem["tag"] in ("span", "a", "div", "li"):
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i,
                                 "text": ""}
                return env_action, action_record

        
        for i, elem in enumerate(dom_elements):
            if elem["tag"] in ("span", "div", "li") and elem.get("text", "").strip():
                
                text = elem.get("text", "").strip()
                if text.lower() != target:  
                    key = f"expanded_{text}"
                    if not heuristic_state.has_done(key):
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        env_action = make_click_action(env, [cx, cy])
                        action_record = {"action_type_idx": 0,
                                         "element_idx": i, "text": ""}
                        heuristic_state.mark_done(key)
                        return env_action, action_record

    
    if "email-inbox" in task_name:
        
        m = re.search(r'(?:by|from)\s+(\w+)', utt_lower)
        target_sender = m.group(1).lower() if m else ""

        
        
        
        sender_idx = None
        sender_top = None
        for i, elem in enumerate(dom_elements):
            text = elem.get("text", "").lower().strip()
            if text == target_sender:
                sender_idx = i
                sender_top = elem.get("top", 0)
                break

        if sender_idx is not None:
            
            
            best_trash = None
            best_trash_idx = None
            for i, elem in enumerate(dom_elements):
                if elem["tag"] == "span" and elem.get("width", 0) < 20:
                    elem_top = elem.get("top", 0)
                    
                    if abs(elem_top - sender_top) < 15:
                        text = elem.get("text", "").strip()
                        if not text or len(text) <= 2:
                            
                            if best_trash is None or elem.get("left", 0) > best_trash.get("left", 0):
                                best_trash = elem
                                best_trash_idx = i

            if best_trash is not None:
                cx = best_trash["left"] + best_trash["width"] / 2
                cy = best_trash["top"] + best_trash["height"] / 2
                
                cx = max(1, min(cx, 159))
                cy = max(51, min(cy, 209))
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0,
                                 "element_idx": best_trash_idx, "text": ""}
                return env_action, action_record

    
    if "social-media" in task_name:
        
        user_m = re.search(r'@(\w+)', utterance)
        target_user = user_m.group(1).lower() if user_m else ""
        quoted = re.findall(r'"([^"]*)"', utterance)
        target_button = quoted[0].lower() if quoted else ""

        
        for i, elem in enumerate(dom_elements):
            text = elem.get("text", "").lower().strip()
            if text == target_button:
                
                
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i,
                                 "text": ""}
                return env_action, action_record

    
    if "click" in task_name and "option" not in task_name and "dialog" not in task_name and "checkboxes" not in task_name:
        
        for i, elem in enumerate(dom_elements):
            text = elem.get("text", "").lower().strip()
            if not text:
                continue
            if text in utt_lower and elem["tag"] in ("button", "a", "span", "div", "option", "label"):
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i,
                                 "text": ""}
                return env_action, action_record

        
        quoted = re.findall(r'"([^"]*)"', utterance)
        for q in quoted:
            q_lower = q.lower()
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if text == q_lower:
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    return env_action, action_record

        
        for i, elem in enumerate(dom_elements):
            if elem["tag"] == "t":
                text = elem.get("text", "").lower().strip()
                if text and text in utt_lower:
                    
                    for j in range(i - 1, -1, -1):
                        if dom_elements[j]["tag"] in ("button", "a", "span", "div", "label"):
                            cx = dom_elements[j]["left"] + dom_elements[j]["width"] / 2
                            cy = dom_elements[j]["top"] + dom_elements[j]["height"] / 2
                            env_action = make_click_action(env, [cx, cy])
                            action_record = {"action_type_idx": 0,
                                             "element_idx": j, "text": ""}
                            return env_action, action_record

        
        for i, elem in enumerate(dom_elements):
            if elem["tag"] == "button" and elem.get("text", "").strip():
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": i,
                                 "text": ""}
                return env_action, action_record

    
    if "login" in task_name:
        utt = utterance
        username_match = re.search(r'username\s*[:\s]\s*(\S+)', utt, re.IGNORECASE)
        password_match = re.search(r'password\s*[:\s]\s*(\S+)', utt, re.IGNORECASE)

        if username_match and password_match:
            username = username_match.group(1).strip().rstrip('.')
            password = password_match.group(1).strip().rstrip('.')

            
            if not heuristic_state.has_done("typed_username"):
                for i, elem in enumerate(dom_elements):
                    if _is_input(elem, ("text", "")):
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        focus_action = make_click_action(env, [cx, cy])
                        env.step(focus_action)
                        env_action = make_type_action(env, [cx, cy], username)
                        action_record = {"action_type_idx": 1, "element_idx": i,
                                         "text": username}
                        heuristic_state.mark_done("typed_username")
                        return env_action, action_record

            
            if not heuristic_state.has_done("typed_password"):
                for i, elem in enumerate(dom_elements):
                    if _is_password(elem):
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        focus_action = make_click_action(env, [cx, cy])
                        env.step(focus_action)
                        env_action = make_type_action(env, [cx, cy], password)
                        action_record = {"action_type_idx": 1, "element_idx": i,
                                         "text": password}
                        heuristic_state.mark_done("typed_password")
                        return env_action, action_record

            
            if not heuristic_state.has_done("clicked_submit"):
                for i, elem in enumerate(dom_elements):
                    text = elem.get("text", "").lower().strip()
                    if elem["tag"] == "button" and text in ("login", "submit", "log in", "sign in", "ok"):
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        env_action = make_click_action(env, [cx, cy])
                        action_record = {"action_type_idx": 0, "element_idx": i,
                                         "text": ""}
                        heuristic_state.mark_done("clicked_submit")
                        return env_action, action_record

    
    
    
    
    if "enter-text" in task_name:
        quoted = re.findall(r'"([^"]*)"', utterance)
        if quoted:
            target_text = quoted[0]

            
            if not heuristic_state.has_done("typed_text"):
                for i, elem in enumerate(dom_elements):
                    if _is_input(elem):
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        
                        focus_action = make_click_action(env, [cx, cy])
                        env.step(focus_action)
                        
                        env_action = make_type_action(env, [cx, cy], target_text)
                        action_record = {"action_type_idx": 1, "element_idx": i,
                                         "text": target_text}
                        heuristic_state.mark_done("typed_text")
                        return env_action, action_record

        
        if not heuristic_state.has_done("clicked_submit"):
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if elem["tag"] == "button" and text in ("submit", "ok", "go"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    heuristic_state.mark_done("clicked_submit")
                    return env_action, action_record

    
    if "search" in task_name:
        quoted = re.findall(r'"([^"]*)"', utterance)
        search_text = quoted[0] if quoted else ""
        if not search_text:
            m = re.search(r'(?:search|find|look up|query)\s+(?:for\s+)?(.+)',
                          utt_lower)
            if m:
                search_text = m.group(1).strip().rstrip('.')

        
        result_num = 1  
        m = re.search(r'(\d+)(?:st|nd|rd|th)', utt_lower)
        if m:
            result_num = int(m.group(1))

        if search_text and not heuristic_state.has_done("typed_search"):
            for i, elem in enumerate(dom_elements):
                if _is_input(elem):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    focus_action = make_click_action(env, [cx, cy])
                    env.step(focus_action)
                    env_action = make_type_action(env, [cx, cy], search_text)
                    action_record = {"action_type_idx": 1, "element_idx": i,
                                     "text": search_text}
                    heuristic_state.mark_done("typed_search")
                    return env_action, action_record

        if not heuristic_state.has_done("clicked_search"):
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if elem["tag"] == "button" and text in ("search", "submit", "go", "find"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    heuristic_state.mark_done("clicked_search")
                    return env_action, action_record

        
        if not heuristic_state.has_done("clicked_result"):
            
            
            result_links = []
            for i, elem in enumerate(dom_elements):
                if elem["tag"] in ("a", "link"):
                    text = elem.get("text", "").strip()
                    
                    if text and not text.isdigit() and text not in (">", "<", "»", "«", "..."):
                        result_links.append((i, elem))

            
            if result_num > len(result_links):
                if not heuristic_state.has_done("next_page"):
                    
                    for i, elem in enumerate(dom_elements):
                        if elem["tag"] in ("a", "link"):
                            text = elem.get("text", "").strip()
                            if text in (">", "»", "2", "3"):
                                cx = elem["left"] + elem["width"] / 2
                                cy = elem["top"] + elem["height"] / 2
                                env_action = make_click_action(env, [cx, cy])
                                action_record = {"action_type_idx": 0,
                                                 "element_idx": i, "text": ""}
                                heuristic_state.mark_done("next_page")
                                
                                heuristic_state.data["remaining_result"] = result_num - len(result_links)
                                return env_action, action_record
                else:
                    
                    remaining = heuristic_state.data.get("remaining_result", 1)
                    if remaining <= len(result_links):
                        idx, elem = result_links[remaining - 1]
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        env_action = make_click_action(env, [cx, cy])
                        action_record = {"action_type_idx": 0,
                                         "element_idx": idx, "text": ""}
                        heuristic_state.mark_done("clicked_result")
                        return env_action, action_record
            elif result_num <= len(result_links):
                idx, elem = result_links[result_num - 1]
                cx = elem["left"] + elem["width"] / 2
                cy = elem["top"] + elem["height"] / 2
                env_action = make_click_action(env, [cx, cy])
                action_record = {"action_type_idx": 0, "element_idx": idx,
                                 "text": ""}
                heuristic_state.mark_done("clicked_result")
                return env_action, action_record

    
    if "autocomplete" in task_name:
        
        starts_m = re.search(r'starts with "([^"]*)"', utterance)
        ends_m = re.search(r'ends with "([^"]*)"', utterance)
        prefix = starts_m.group(1) if starts_m else ""
        suffix = ends_m.group(1) if ends_m else ""
        
        if not prefix:
            quoted = re.findall(r'"([^"]*)"', utterance)
            prefix = quoted[0] if quoted else ""

        if prefix and not heuristic_state.has_done("typed_partial"):
            for i, elem in enumerate(dom_elements):
                if _is_input(elem):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    focus_action = make_click_action(env, [cx, cy])
                    env.step(focus_action)
                    env_action = make_type_action(env, [cx, cy], prefix)
                    action_record = {"action_type_idx": 1, "element_idx": i,
                                     "text": prefix}
                    heuristic_state.mark_done("typed_partial")
                    return env_action, action_record

        if not heuristic_state.has_done("selected_suggestion"):
            
            prefix_lower = prefix.lower()
            suffix_lower = suffix.lower()
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if not text:
                    continue
                if elem["tag"] in ("li", "div", "span", "option", "a"):
                    match = text.startswith(prefix_lower)
                    if suffix_lower:
                        match = match and text.endswith(suffix_lower)
                    if match:
                        cx = elem["left"] + elem["width"] / 2
                        cy = elem["top"] + elem["height"] / 2
                        env_action = make_click_action(env, [cx, cy])
                        action_record = {"action_type_idx": 0,
                                         "element_idx": i, "text": ""}
                        heuristic_state.mark_done("selected_suggestion")
                        return env_action, action_record

        
        if not heuristic_state.has_done("clicked_submit"):
            for i, elem in enumerate(dom_elements):
                text = elem.get("text", "").lower().strip()
                if elem["tag"] == "button" and text in ("submit", "ok"):
                    cx = elem["left"] + elem["width"] / 2
                    cy = elem["top"] + elem["height"] / 2
                    env_action = make_click_action(env, [cx, cy])
                    action_record = {"action_type_idx": 0, "element_idx": i,
                                     "text": ""}
                    heuristic_state.mark_done("clicked_submit")
                    return env_action, action_record

    return None, None

def collect_mixed_demonstrations(task_name, num_demos=500, success_rate=1.0,
                                  max_steps=20, save_dir="data/demos", seed=42):
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
        h_state = HeuristicState()

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
                env, obs, dom_elements, utterance, task_name,
                heuristic_state=h_state,
            )

            if env_action is None:
                env_action = env.action_space.sample()
                action_record = {"action_type_idx": 0, "element_idx": 0, "text": ""}

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
        h_state = HeuristicState()

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
                env, obs, dom_elements, utterance, task_name,
                heuristic_state=h_state,
            )

            if env_action is None:
                env_action = env.action_space.sample()
                action_record = {"action_type_idx": 0, "element_idx": 0, "text": ""}

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
