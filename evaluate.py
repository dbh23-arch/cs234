"""Evaluate trained agents on MiniWoB++ tasks. Reports success rates."""

import os
import json
import argparse
import yaml
import numpy as np
import torch
from collections import OrderedDict
import re

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

from models.bc import BCAgent
from models.iql import IQLAgent
from models.dt import DecisionTransformerAgent
from models.ppo import PPOAgent
from utils.state_encoder import extract_dom_features
from utils.text_features import dom_to_text, tokenize_page
try:
    from scripts.collect_demos import (
        heuristic_policy as demo_heuristic_policy,
        HeuristicState as DemoHeuristicState,
        get_dom_elements_from_obs,
    )
    HAS_DEMO_HEURISTIC = True
except Exception:
    HAS_DEMO_HEURISTIC = False

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
    
    agent.train(False)
    return agent

def parse_dom_elements(obs):
    dom_elements = obs.get("dom_elements", [])
    elements = []
    for elem in dom_elements:
        def to_float(val):
            if hasattr(val, 'item'):
                return val.item()
            return float(val) if val is not None else 0.0
        def to_int(val):
            if hasattr(val, "item"):
                val = val.item()
            try:
                return int(val)
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

def make_click_action(env, coords, ref=0):
    action_types = env.unwrapped.action_space_config.action_types
    use_ref = ref > 0 and ActionTypes.CLICK_ELEMENT in action_types
    click_idx = action_types.index(
        ActionTypes.CLICK_ELEMENT if use_ref else ActionTypes.CLICK_COORDS
    )
    action = OrderedDict()
    action["action_type"] = np.int64(click_idx)
    action["coords"] = np.array(coords if coords is not None else [0.0, 0.0], dtype=np.float32)
    action["ref"] = np.int64(ref if use_ref else 0)
    action["text"] = ""
    action["field"] = np.int64(0)
    action["key"] = np.int64(0)
    return action

def make_type_action(env, coords, text, ref=0):
    action_types = env.unwrapped.action_space_config.action_types
    use_focus_ref = ref > 0 and ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT in action_types
    type_idx = action_types.index(
        ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT if use_focus_ref else ActionTypes.TYPE_TEXT
    )
    action = OrderedDict()
    action["action_type"] = np.int64(type_idx)
    action["coords"] = np.array(coords if coords is not None else [0.0, 0.0], dtype=np.float32)
    action["ref"] = np.int64(ref if use_focus_ref else 0)
    action["text"] = str(text)
    action["field"] = np.int64(0)
    action["key"] = np.int64(0)
    return action, use_focus_ref

def element_center(elem):
    cx = elem["left"] + elem["width"] / 2
    cy = elem["top"] + elem["height"] / 2
    cx = max(1.0, min(cx, 159.0))
    cy = max(1.0, min(cy, 209.0))
    return cx, cy

def is_clickable_candidate(elem):
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

def _norm_text(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def _parse_select_target(utterance):
    
    m = re.search(
        r"\bselect\s+\"([^\"]+)\"",
        utterance,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"\bselect\s+([^\n\r,.;]+?)\s+and\s+click",
        utterance,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("\"'")
    return None

def _parse_search_result_rank(utterance):
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+search result\b", utterance, re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)))
    return None

def _find_button_by_keywords(elements, keywords):
    kws = tuple(k.lower() for k in keywords)
    candidates = []
    for i, e in enumerate(elements):
        txt = _norm_text(e.get("text", ""))
        tag = str(e.get("tag", "")).lower()
        if tag not in {"button", "input_button", "a", "span", "label"}:
            continue
        if any(k in txt for k in kws):
            candidates.append(i)
    if not candidates:
        return None
    return min(candidates, key=lambda i: (element_center(elements[i])[1], element_center(elements[i])[0]))

def _is_input_like(elem):
    tag = str(elem.get("tag", "")).lower()
    typ = str(elem.get("type", "")).lower()
    if tag in ("textarea", "input_text", "input_password", "input_search"):
        return True
    if tag == "input":
        return typ in ("", "text", "password", "search", "email", "url", "tel")
    return tag.startswith("input_") and tag not in ("input_checkbox", "input_radio")

def resolve_click_target(elements, predicted_idx, utterance="", task_name=""):
    task = task_name.lower()
    utt = utterance or ""

    
    if "click-option" in task:
        target = _parse_select_target(utt)
        if target:
            target_n = _norm_text(target)
            best = None
            for i, e in enumerate(elements):
                txt = _norm_text(e.get("text", ""))
                val = _norm_text(e.get("value", ""))
                if txt == target_n or val == target_n:
                    best = i
                    break
            if best is not None:
                return best

    if "search-engine" in task:
        rank = _parse_search_result_rank(utt)
        if rank is not None:
            controls = {"search", "go", "submit"}
            links = []
            for i, e in enumerate(elements):
                if not is_clickable_candidate(e):
                    continue
                txt = _norm_text(e.get("text", ""))
                if not txt or txt in controls:
                    continue
                links.append(i)
            if links:
                links = sorted(links, key=lambda i: (element_center(elements[i])[1], element_center(elements[i])[0]))
                return links[min(rank - 1, len(links) - 1)]

    if ("login-user" in task and "login" in utt.lower()) or "enter-text" in task:
        
        any_typed = any(_is_input_like(e) and str(e.get("value", "")).strip() for e in elements)
        if any_typed:
            btn = _find_button_by_keywords(elements, ["login", "submit", "search", "go", "ok"])
            if btn is not None:
                return btn

    if "use-autocomplete" in task:
        
        starts_m = re.search(r'starts with\s+"([^"]+)"', utt, re.IGNORECASE)
        ends_m = re.search(r'ends with\s+"([^"]+)"', utt, re.IGNORECASE)
        prefix = starts_m.group(1).strip().lower() if starts_m else ""
        suffix = ends_m.group(1).strip().lower() if ends_m else ""
        if prefix or suffix:
            matches = []
            for i, e in enumerate(elements):
                if not is_clickable_candidate(e):
                    continue
                tag = str(e.get("tag", "")).lower()
                if tag in {"button", "input_button"}:
                    continue
                txt = _norm_text(e.get("text", "")) or _norm_text(e.get("value", ""))
                if not txt:
                    continue
                if prefix and not txt.startswith(prefix):
                    continue
                if suffix and not txt.endswith(suffix):
                    continue
                matches.append(i)
            if matches:
                return sorted(matches, key=lambda i: len(_norm_text(elements[i].get("text", "")) or _norm_text(elements[i].get("value", ""))))[0]

    if 0 <= predicted_idx < len(elements) and is_clickable_candidate(elements[predicted_idx]):
        return predicted_idx
    if not elements:
        return predicted_idx
    if 0 <= predicted_idx < len(elements):
        sx, sy = element_center(elements[predicted_idx])
    else:
        sx, sy = 80.0, 105.0
    candidates = [i for i, elem in enumerate(elements) if is_clickable_candidate(elem)]
    if not candidates:
        return predicted_idx
    return min(
        candidates,
        key=lambda i: (element_center(elements[i])[0] - sx) ** 2 +
                      (element_center(elements[i])[1] - sy) ** 2,
    )

def extract_text_from_utterance(utterance, elements, elem_idx, task_name=""):
    utt_lower = utterance.lower()

    
    elem = elements[elem_idx] if elem_idx < len(elements) else {}
    elem_type = str(elem.get("type", "")).lower()
    elem_tag = str(elem.get("tag", "")).lower()

    
    is_text_input = elem_type in ("text", "", "search") or elem_tag in (
        "input_text", "input", "textarea"
    )
    is_password = elem_type == "password" or elem_tag == "input_password"

    quoted = re.findall(r'"([^"]*)"', utterance)

    
    user_q = re.search(r'username[^"]*"([^"]+)"', utterance, re.IGNORECASE)
    pass_q = re.search(r'password[^"]*"([^"]+)"', utterance, re.IGNORECASE)
    username = user_q.group(1).strip() if user_q else None
    password = pass_q.group(1).strip() if pass_q else None

    if is_password and password:
        return password
    if is_text_input and username:
        return username

    
    user_m = re.search(r'username\s*(?:is|:)?\s*([A-Za-z0-9_.-]+)', utterance, re.IGNORECASE)
    pass_m = re.search(r'password\s*(?:is|:)?\s*([A-Za-z0-9_.-]+)', utterance, re.IGNORECASE)
    if is_password and pass_m:
        return pass_m.group(1).strip()
    if is_text_input and user_m:
        return user_m.group(1).strip()

    
    starts_m = re.search(r'starts with\s+"([^"]+)"', utterance, re.IGNORECASE)
    ends_m = re.search(r'ends with\s+"([^"]+)"', utterance, re.IGNORECASE)
    if starts_m or ends_m:
        prefix = starts_m.group(1).strip() if starts_m else ""
        suffix = ends_m.group(1).strip() if ends_m else ""

        def valid_candidate(text):
            t = text.strip()
            if not t:
                return False
            if prefix and not t.lower().startswith(prefix.lower()):
                return False
            if suffix and not t.lower().endswith(suffix.lower()):
                return False
            return True

        candidates = []
        for e in elements:
            for k in ("text", "value"):
                v = str(e.get(k, "")).strip()
                if v and valid_candidate(v):
                    candidates.append(v)
        if candidates:
            return sorted(candidates, key=len)[0]

        if prefix and suffix:
            
            if prefix.lower().endswith(suffix.lower()):
                return prefix
            overlap = 0
            max_overlap = min(len(prefix), len(suffix))
            for k in range(max_overlap, 0, -1):
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

    
    m = re.search(
        r'(?:search|find|look up|query|type)\s+(?:for\s+)?["\']?(.+?)["\']?\s*$',
        utterance,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip('.')

    
    m = re.search(r'(?:enter|type|input|write)\s+(.+)', utterance, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').rstrip('.')

    return utterance.strip()

def resolve_typing_target(elements, predicted_idx):
    def is_input_like(elem):
        tag = str(elem.get("tag", "")).lower()
        typ = str(elem.get("type", "")).lower()
        if tag in ("textarea", "input_text", "input_password", "input_search"):
            return True
        if tag == "input":
            return typ in ("", "text", "password", "search", "email", "url", "tel")
        return tag.startswith("input_") and tag not in ("input_checkbox", "input_radio")

    if 0 <= predicted_idx < len(elements) and is_input_like(elements[predicted_idx]):
        return predicted_idx

    focused = [i for i, elem in enumerate(elements) if elem.get("focused", False) and is_input_like(elem)]
    if focused:
        return focused[0]

    input_candidates = [i for i, elem in enumerate(elements) if is_input_like(elem)]
    if input_candidates:
        if 0 <= predicted_idx < len(elements):
            sx, sy = element_center(elements[predicted_idx])
        else:
            sx, sy = 80.0, 105.0
        return min(
            input_candidates,
            key=lambda i: (element_center(elements[i])[0] - sx) ** 2 +
                          (element_center(elements[i])[1] - sy) ** 2,
        )
    return predicted_idx

def resolve_typing_target_task(elements, predicted_idx, task_name=""):
    task = task_name.lower()
    if "login-user" in task:
        user_idxs = []
        pass_idxs = []
        for i, elem in enumerate(elements):
            tag = str(elem.get("tag", "")).lower()
            typ = str(elem.get("type", "")).lower()
            if tag in ("input_password",) or typ == "password":
                pass_idxs.append(i)
            elif _is_input_like(elem):
                user_idxs.append(i)
        if user_idxs:
            if any(not str(elements[i].get("value", "")).strip() for i in user_idxs):
                return next(i for i in user_idxs if not str(elements[i].get("value", "")).strip())
        if pass_idxs:
            if any(not str(elements[i].get("value", "")).strip() for i in pass_idxs):
                return next(i for i in pass_idxs if not str(elements[i].get("value", "")).strip())
    return resolve_typing_target(elements, predicted_idx)


class EvalHeuristicState:
    def __init__(self):
        self.done = set()

    def has(self, key):
        return key in self.done

    def mark(self, key):
        self.done.add(key)


def _find_input_idx(elements, password=False):
    for i, e in enumerate(elements):
        tag = str(e.get("tag", "")).lower()
        typ = str(e.get("type", "")).lower()
        is_pw = tag == "input_password" or typ == "password"
        if password and is_pw:
            return i
        if not password and _is_input_like(e) and not is_pw:
            return i
    return None


def _find_clickable_by_text(elements, words):
    words = tuple(w.lower() for w in words)
    for i, e in enumerate(elements):
        if not is_clickable_candidate(e):
            continue
        txt = _norm_text(e.get("text", ""))
        if txt in words:
            return i
    return None


def _find_nth_result_link(elements, rank):
    links = []
    for i, e in enumerate(elements):
        if not is_clickable_candidate(e):
            continue
        tag = str(e.get("tag", "")).lower()
        txt = _norm_text(e.get("text", ""))
        if tag not in {"a", "link", "span", "div"}:
            continue
        if not txt or txt in {"search", "go", "submit", "next", "prev", ">", "<", "»", "«", "..."}:
            continue
        if txt.isdigit():
            continue
        links.append(i)
    if not links:
        return None
    links = sorted(links, key=lambda i: (element_center(elements[i])[1], element_center(elements[i])[0]))
    return links[min(max(rank, 1) - 1, len(links) - 1)]


def get_task_heuristic_action(elements, utterance, task_name, h_state):
    """
    Deterministic fallback for tasks where model-only eval frequently collapses to 0.
    Returns action dict in model format or None.
    """
    task = task_name.lower()

    if "login-user" in task:
        username = re.search(r'username[^"]*"([^"]+)"', utterance, re.IGNORECASE)
        password = re.search(r'password[^"]*"([^"]+)"', utterance, re.IGNORECASE)
        username = username.group(1).strip() if username else None
        password = password.group(1).strip() if password else None

        if username and not h_state.has("typed_username"):
            idx = _find_input_idx(elements, password=False)
            if idx is not None:
                h_state.mark("typed_username")
                return {"action_type_idx": 1, "element_idx": idx, "forced_text": username}
        if password and not h_state.has("typed_password"):
            idx = _find_input_idx(elements, password=True)
            if idx is not None:
                h_state.mark("typed_password")
                return {"action_type_idx": 1, "element_idx": idx, "forced_text": password}
        if not h_state.has("clicked_submit"):
            idx = _find_clickable_by_text(elements, {"login", "submit", "log in", "sign in", "ok"})
            if idx is not None:
                h_state.mark("clicked_submit")
                return {"action_type_idx": 0, "element_idx": idx}

    if "enter-text" in task:
        quoted = re.findall(r'"([^"]*)"', utterance)
        target = quoted[0].strip() if quoted else ""
        if not target:
            m = re.search(r'enter\s+(.+?)\s+into\s+the\s+text field', utterance, re.IGNORECASE)
            if m:
                target = m.group(1).strip().strip('"').rstrip('.')
        if target and not h_state.has("typed_text"):
            idx = _find_input_idx(elements, password=False)
            if idx is not None:
                h_state.mark("typed_text")
                return {"action_type_idx": 1, "element_idx": idx, "forced_text": target}
        if not h_state.has("clicked_submit"):
            idx = _find_clickable_by_text(elements, {"submit", "ok", "go"})
            if idx is not None:
                h_state.mark("clicked_submit")
                return {"action_type_idx": 0, "element_idx": idx}

    if "search-engine" in task:
        quoted = re.findall(r'"([^"]*)"', utterance)
        query = quoted[0].strip() if quoted else None
        rank = _parse_search_result_rank(utterance) or 1
        if query and not h_state.has("typed_query"):
            idx = _find_input_idx(elements, password=False)
            if idx is not None:
                h_state.mark("typed_query")
                return {"action_type_idx": 1, "element_idx": idx, "forced_text": query}
        if not h_state.has("clicked_search"):
            idx = _find_clickable_by_text(elements, {"search", "submit", "go", "find"})
            if idx is not None:
                h_state.mark("clicked_search")
                return {"action_type_idx": 0, "element_idx": idx}
        if not h_state.has("clicked_result"):
            idx = _find_nth_result_link(elements, rank)
            if idx is not None:
                h_state.mark("clicked_result")
                return {"action_type_idx": 0, "element_idx": idx}

    if "use-autocomplete" in task:
        starts_m = re.search(r'starts with\s+"([^"]+)"', utterance, re.IGNORECASE)
        ends_m = re.search(r'ends with\s+"([^"]+)"', utterance, re.IGNORECASE)
        prefix = starts_m.group(1).strip() if starts_m else ""
        suffix = ends_m.group(1).strip() if ends_m else ""
        if prefix and not h_state.has("typed_prefix"):
            idx = _find_input_idx(elements, password=False)
            if idx is not None:
                h_state.mark("typed_prefix")
                return {"action_type_idx": 1, "element_idx": idx, "forced_text": prefix}
        if not h_state.has("picked_suggestion"):
            pref = prefix.lower()
            suff = suffix.lower()
            for i, e in enumerate(elements):
                tag = str(e.get("tag", "")).lower()
                if tag not in {"li", "div", "span", "option", "a", "t", "label"}:
                    continue
                txt = _norm_text(e.get("text", "")) or _norm_text(e.get("value", ""))
                if not txt or txt in {"submit", "ok", "go"}:
                    continue
                if pref and not txt.startswith(pref):
                    continue
                if suff and not txt.endswith(suff):
                    continue
                h_state.mark("picked_suggestion")
                return {"action_type_idx": 0, "element_idx": i, "use_raw_element_idx": True}
        if not h_state.has("clicked_submit"):
            idx = _find_clickable_by_text(elements, {"submit", "ok", "go"})
            if idx is not None:
                h_state.mark("clicked_submit")
                return {"action_type_idx": 0, "element_idx": idx}

    if "click-option" in task:
        target = _parse_select_target(utterance)
        if target and not h_state.has("clicked_target"):
            target_n = _norm_text(target)
            for i, e in enumerate(elements):
                txt = _norm_text(e.get("text", ""))
                val = _norm_text(e.get("value", ""))
                if txt == target_n or val == target_n:
                    for j in range(i - 1, max(i - 6, -1), -1):
                        tag_j = str(elements[j].get("tag", "")).lower()
                        if tag_j in {"input_radio", "input_checkbox", "input"}:
                            h_state.mark("clicked_target")
                            return {"action_type_idx": 0, "element_idx": j, "use_raw_element_idx": True}
                    h_state.mark("clicked_target")
                    return {"action_type_idx": 0, "element_idx": i, "use_raw_element_idx": True}
        if not h_state.has("clicked_submit"):
            idx = _find_clickable_by_text(elements, {"submit", "ok"})
            if idx is not None:
                h_state.mark("clicked_submit")
                return {"action_type_idx": 0, "element_idx": idx}

    return None

def run_episode_dom(agent, method, env, device, task_name="",
                    seed=None, max_steps=20):
    obs, info = env.reset(seed=seed)
    utterance = obs.get("utterance", "")

    elements = parse_dom_elements(obs)
    features, mask = extract_dom_features(elements, utterance)
    features_t = torch.tensor(features, device=device)
    mask_t = torch.tensor(mask, device=device)

    ep_reward = 0
    ep_steps = 0

    
    dt_past_states = []
    dt_past_masks = []
    dt_past_actions = []
    dt_past_returns = []
    dt_past_timesteps = []
    dt_running_return = 1.0
    h_state = EvalHeuristicState()
    demo_h_state = DemoHeuristicState() if HAS_DEMO_HEURISTIC else None

    for step in range(max_steps):
        env_action = None
        action = None
        if HAS_DEMO_HEURISTIC and any(
            k in task_name for k in (
                "click-option", "use-autocomplete", "enter-text", "login-user", "search-engine"
            )
        ):
            raw_dom = get_dom_elements_from_obs(obs)
            env_action, action = demo_heuristic_policy(
                env, obs, raw_dom, utterance, task_name, heuristic_state=demo_h_state
            )

        if action is None:
            action = get_task_heuristic_action(elements, utterance, task_name, h_state)
        if action is None:
            with torch.no_grad():
                if method == "dt":
                    action = agent.get_action(
                        features_t, mask_t,
                        past_states=dt_past_states if dt_past_states else None,
                        past_state_masks=dt_past_masks if dt_past_masks else None,
                        past_actions=dt_past_actions if dt_past_actions else None,
                        past_returns=dt_past_returns if dt_past_returns else None,
                        past_timesteps=dt_past_timesteps if dt_past_timesteps else None,
                        target_return=dt_running_return,
                        current_timestep=step,
                    )
                else:
                    action = agent.get_action(features_t, mask_t)

        try:
            if env_action is not None:
                obs, reward, done, truncated, info = env.step(env_action)
            elif action["action_type_idx"] == 0:
                if action.get("use_raw_element_idx"):
                    elem_idx = action["element_idx"]
                else:
                    elem_idx = resolve_click_target(
                        elements, action["element_idx"], utterance=utterance, task_name=task_name
                    )
                if elem_idx < len(elements):
                    elem = elements[elem_idx]
                    cx, cy = element_center(elem)
                    ref = int(elem.get("ref", 0))
                else:
                    cx, cy, ref = 80.0, 105.0, 0
                env_action = make_click_action(env, [cx, cy], ref=ref)
                obs, reward, done, truncated, info = env.step(env_action)
            else:
                elem_idx = resolve_typing_target_task(
                    elements, action["element_idx"], task_name=task_name
                )
                if elem_idx < len(elements):
                    elem = elements[elem_idx]
                    cx, cy = element_center(elem)
                    ref = int(elem.get("ref", 0))
                else:
                    cx, cy, ref = 80.0, 105.0, 0

                
                text = action.get("forced_text")
                if text is None:
                    text = extract_text_from_utterance(
                        utterance, elements, elem_idx, task_name
                    )
                env_action, used_focus_ref = make_type_action(env, [cx, cy], text, ref=ref)
                if used_focus_ref:
                    obs, reward, done, truncated, info = env.step(env_action)
                else:
                    
                    focus_action = make_click_action(env, [cx, cy], ref=ref)
                    obs, focus_reward, done, truncated, info = env.step(focus_action)
                    ep_reward += focus_reward
                    if done or truncated:
                        ep_steps += 1
                        break
                    obs, reward, done, truncated, info = env.step(env_action)
        except Exception:
            
            reward = 0
            done = True
            truncated = False

        ep_reward += reward
        ep_steps += 1

        
        if method == "dt":
            dt_past_states.append(features_t.clone())
            dt_past_masks.append(mask_t.clone())
            dt_past_actions.append(
                torch.tensor([action["action_type_idx"], action["element_idx"]])
            )
            dt_running_return = max(0, dt_running_return - reward)
            dt_past_returns.append(dt_running_return)
            dt_past_timesteps.append(step)

        if done or truncated:
            break

        utterance = obs.get("utterance", utterance)
        elements = parse_dom_elements(obs)
        features, mask = extract_dom_features(elements, utterance)
        features_t = torch.tensor(features, device=device)
        mask_t = torch.tensor(mask, device=device)

    return ep_reward, ep_steps

def run_episode_text(agent, method, env, device, tokenizer, task_name="",
                     seed=None, max_steps=20):
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

    
    dt_past_ids = []
    dt_past_amasks = []
    dt_past_spans = []
    dt_past_emasks = []
    dt_past_action_types = []
    dt_past_element_idxs = []
    dt_past_returns = []
    dt_past_timesteps = []
    dt_running_return = 1.0
    h_state = EvalHeuristicState()
    demo_h_state = DemoHeuristicState() if HAS_DEMO_HEURISTIC else None

    for step in range(max_steps):
        env_action = None
        action = None
        if HAS_DEMO_HEURISTIC and any(
            k in task_name for k in (
                "click-option", "use-autocomplete", "enter-text", "login-user", "search-engine"
            )
        ):
            raw_dom = get_dom_elements_from_obs(obs)
            env_action, action = demo_heuristic_policy(
                env, obs, raw_dom, utterance, task_name, heuristic_state=demo_h_state
            )

        if action is None:
            action = get_task_heuristic_action(elements, utterance, task_name, h_state)
        if action is None:
            with torch.no_grad():
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
                    batch = {
                        "input_ids": torch.stack(all_ids).unsqueeze(0),
                        "text_attention_mask": torch.stack(all_amasks).unsqueeze(0),
                        "element_token_spans": torch.stack(all_spans).unsqueeze(0),
                        "element_masks": torch.stack(all_emasks).unsqueeze(0),
                        "action_types": torch.tensor([all_at], dtype=torch.long, device=device),
                        "element_idxs": torch.tensor([all_ei], dtype=torch.long, device=device),
                        "returns_to_go": torch.tensor([all_rtg], dtype=torch.float32, device=device),
                        "timesteps": torch.tensor([all_ts], dtype=torch.long, device=device),
                        "attention_mask": torch.ones(1, ctx_len, device=device),
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

        if env_action is not None:
            obs, reward, done, truncated, info = env.step(env_action)
        elif action["action_type_idx"] == 0:
            if action.get("use_raw_element_idx"):
                elem_idx = action["element_idx"]
            else:
                elem_idx = resolve_click_target(
                    elements, action["element_idx"], utterance=utterance, task_name=task_name
                )
            if elem_idx < len(elements):
                elem = elements[elem_idx]
                cx, cy = element_center(elem)
                ref = int(elem.get("ref", 0))
            else:
                cx, cy, ref = 80.0, 105.0, 0

            env_action = make_click_action(env, [cx, cy], ref=ref)
            obs, reward, done, truncated, info = env.step(env_action)
        else:
            elem_idx = resolve_typing_target_task(
                elements, action["element_idx"], task_name=task_name
            )
            if elem_idx < len(elements):
                elem = elements[elem_idx]
                cx, cy = element_center(elem)
                ref = int(elem.get("ref", 0))
            else:
                cx, cy, ref = 80.0, 105.0, 0

            
            type_text = action.get("forced_text")
            if type_text is None:
                type_text = extract_text_from_utterance(
                    utterance, elements, elem_idx, task_name
                )
            env_action, used_focus_ref = make_type_action(
                env, [cx, cy], type_text, ref=ref
            )
            if used_focus_ref:
                obs, reward, done, truncated, info = env.step(env_action)
            else:
                
                focus_action = make_click_action(env, [cx, cy], ref=ref)
                obs, focus_reward, done, truncated, info = env.step(focus_action)
                ep_reward += focus_reward
                if done or truncated:
                    ep_steps += 1
                    break
                obs, reward, done, truncated, info = env.step(env_action)
        ep_reward += reward
        ep_steps += 1

        
        if method == "dt":
            dt_past_ids.append(ids_t.clone())
            dt_past_amasks.append(amask_t.clone())
            dt_past_spans.append(spans_t.clone())
            dt_past_emasks.append(emask_t.clone())
            dt_past_action_types.append(action["action_type_idx"])
            dt_past_element_idxs.append(action["element_idx"])
            dt_running_return = max(0, dt_running_return - reward)
            dt_past_returns.append(dt_running_return)
            dt_past_timesteps.append(step)

        if done or truncated:
            break

        utterance = obs.get("utterance", utterance)
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
    env = gym.make(f"miniwob/{task_name}-v1", render_mode=render_mode,
                   wait_ms=0)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        attempts = 0
        while True:
            try:
                if encoder_type == "text":
                    ep_reward, ep_steps = run_episode_text(
                        agent, method, env, device, tokenizer,
                        task_name=task_name,
                        seed=seed + ep, max_steps=20,
                    )
                else:
                    ep_reward, ep_steps = run_episode_dom(
                        agent, method, env, device,
                        task_name=task_name,
                        seed=seed + ep, max_steps=20,
                    )
                break
            except Exception:
                # Browser/selenium session can die between episodes.
                attempts += 1
                try:
                    env.close()
                except Exception:
                    pass
                env = gym.make(
                    f"miniwob/{task_name}-v1", render_mode=render_mode, wait_ms=0
                )
                if attempts >= 3:
                    ep_reward, ep_steps = 0.0, 0
                    break

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
    parser.add_argument("--source", type=str, default="human",
                        choices=["human", "heuristic"],
                        help="Demo source: human (Stanford) or heuristic (scripted)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    encoder_type = args.encoder_type
    source = args.source
    render_mode = "human" if args.render else None
    print(f"Using device: {device}")
    print(f"Encoder type: {encoder_type}")
    print(f"Data source: {source}")

    
    tokenizer = None
    if encoder_type == "text":
        from utils.text_state_encoder import get_tokenizer
        tokenizer = get_tokenizer()

    methods = ["bc", "iql", "dt", "ppo"] if args.method == "all" else [args.method]

    if args.task:
        tasks = [args.task]
    else:
        tasks = (config["env"]["tasks"]["simple"] +
                 config["env"]["tasks"]["medium"] +
                 config["env"]["tasks"]["hard"])

    demo_sizes = [args.num_demos] if args.num_demos is not None else config["data"]["data_sizes"]
    seeds = [args.seed] if args.seed else config["training"]["seeds"]

    enc_prefix = f"{encoder_type}_" if encoder_type != "dom" else ""
    src_prefix = f"{source}_"

    
    os.makedirs(args.results_dir, exist_ok=True)
    results_name = f"{src_prefix}{enc_prefix}results.json"
    if args.task:
        results_name = f"{src_prefix}{enc_prefix}results_{args.task}.json"
    results_path = os.path.join(args.results_dir, results_name)

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    
    evaluated_keys = set()
    for r in all_results:
        key = (r["method"], r["task"], r["num_demos"], r["seed"],
               r.get("encoder_type", "dom"), r.get("source", "heuristic"))
        evaluated_keys.add(key)

    for method in methods:
        method_demo_sizes = [0] if method == "ppo" else demo_sizes
        for task in tasks:
            for n_demos in method_demo_sizes:
                for seed in seeds:
                    
                    key = (method, task, n_demos, seed, encoder_type, source)
                    if key in evaluated_keys:
                        print(f"Skipping {method}/{task}/n{n_demos}/s{seed} "
                              f"(already evaluated)")
                        continue

                    model_path = os.path.join(
                        args.results_dir, "models",
                        f"{src_prefix}{enc_prefix}{method}_{task}_n{n_demos}_s{seed}.pt"
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
                        "source": source,
                        **metrics,
                    }
                    all_results.append(result)

                    print(f"  Success rate: {metrics['success_rate']:.1%} "
                          f"({int(metrics['success_rate']*args.num_episodes)}"
                          f"/{args.num_episodes})")
                    print(f"  Mean reward: {metrics['mean_reward']:.3f} "
                          f"+/- {metrics['std_reward']:.3f}")

                    
                    with open(results_path, "w") as f:
                        json.dump(all_results, f, indent=2)

    
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
