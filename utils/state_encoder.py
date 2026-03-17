"""DOM feature extraction and state encoder."""

import re
import numpy as np
import torch
import torch.nn as nn

ELEMENT_TYPES = [
    "button", "input_text", "input_checkbox", "input_radio",
    "link", "select", "option", "textarea", "label", "span",
    "div", "li", "td", "other"
]
ELEMENT_TYPE_TO_IDX = {t: i for i, t in enumerate(ELEMENT_TYPES)}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _as_float(val):
    if hasattr(val, "item"):
        val = val.item()
    try:
        return float(val)
    except Exception:
        return 0.0


def _tokenize(text):
    if not text:
        return set()
    return set(TOKEN_RE.findall(str(text).lower()))


def classify_element(tag, type_attr=""):
    tag = tag.lower()
    if tag == "button":
        return "button"
    elif tag == "input" or tag.startswith("input_") or tag.startswith("input "):
        if tag.startswith("input_") or tag.startswith("input "):
            t = tag.split("_", 1)[1] if "_" in tag else tag.split(" ", 1)[1] if " " in tag else ""
        else:
            t = type_attr.lower() if type_attr else "text"
        if t in ("text", "password", "email", "search", "url", "tel", ""):
            return "input_text"
        elif t == "checkbox":
            return "input_checkbox"
        elif t == "radio":
            return "input_radio"
        else:
            return "input_text"
    elif tag == "a":
        return "link"
    elif tag == "select":
        return "select"
    elif tag == "option":
        return "option"
    elif tag == "textarea":
        return "textarea"
    elif tag == "label":
        return "label"
    elif tag == "span":
        return "span"
    elif tag == "div":
        return "div"
    elif tag == "li":
        return "li"
    elif tag in ("td", "th"):
        return "td"
    else:
        return "other"


def extract_dom_features(dom_elements, utterance, max_elements=64):
    feature_dim = len(ELEMENT_TYPES) + 10  # 14 + 10 = 24
    features = np.zeros((max_elements, feature_dim), dtype=np.float32)
    valid_mask = np.zeros(max_elements, dtype=np.float32)

    utterance_words = _tokenize(utterance)

    # text anchors for nearby label matching
    text_anchors = []
    for elem in dom_elements[:max_elements]:
        text = str(elem.get("text", "")).strip()
        value = str(elem.get("value", "")).strip()
        combined = (text + " " + value).strip()
        tokens = _tokenize(combined)
        if not tokens:
            continue
        cx = _as_float(elem.get("left", 0)) + _as_float(elem.get("width", 0)) / 2
        cy = _as_float(elem.get("top", 0)) + _as_float(elem.get("height", 0)) / 2
        text_anchors.append((cx, cy, combined, tokens))

    for i, elem in enumerate(dom_elements[:max_elements]):
        tag = elem.get("tag", "div")
        type_attr = elem.get("type", "")
        elem_type = classify_element(tag, type_attr)
        type_idx = ELEMENT_TYPE_TO_IDX.get(elem_type, len(ELEMENT_TYPES) - 1)

        features[i, type_idx] = 1.0

        left = np.clip(_as_float(elem.get("left", 0)) / 160.0, 0.0, 1.0)
        top = np.clip(_as_float(elem.get("top", 0)) / 210.0, 0.0, 1.0)
        width = np.clip(_as_float(elem.get("width", 0)) / 160.0, 0.0, 1.0)
        height = np.clip(_as_float(elem.get("height", 0)) / 210.0, 0.0, 1.0)
        offset = len(ELEMENT_TYPES)
        features[i, offset:offset + 4] = [left, top, width, height]

        features[i, offset + 4] = 1.0 if elem.get("visible", True) else 0.0
        features[i, offset + 5] = 1.0 if elem.get("focused", False) else 0.0

        text = str(elem.get("text", "")).lower()
        value = str(elem.get("value", "")).lower()
        combined_text = (text + " " + value).strip()
        text_words = _tokenize(combined_text)

        if not text_words and text_anchors:
            cx = _as_float(elem.get("left", 0)) + _as_float(elem.get("width", 0)) / 2
            cy = _as_float(elem.get("top", 0)) + _as_float(elem.get("height", 0)) / 2
            best, best_dist = None, float("inf")
            for ax, ay, anchor_text, anchor_tokens in text_anchors:
                if abs(ay - cy) > 24.0:
                    continue
                dist = (ax - cx) ** 2 + (ay - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (anchor_text, anchor_tokens)
            if best is not None:
                combined_text, text_words = best

        overlap = len(utterance_words & text_words) / max(len(utterance_words), 1)
        features[i, offset + 6] = overlap
        features[i, offset + 7] = min(len(combined_text) / 50.0, 1.0)

        clickable = elem_type in ("button", "link", "input_checkbox",
                                   "input_radio", "select", "option")
        features[i, offset + 8] = 1.0 if clickable else 0.0
        features[i, offset + 9] = 1.0 if elem_type in ("input_text", "textarea") else 0.0

        valid_mask[i] = 1.0

    return features, valid_mask


class StateEncoder(nn.Module):

    def __init__(self, element_feature_dim=24, max_elements=64,
                 embed_dim=64, state_dim=256):
        super().__init__()
        self.max_elements = max_elements
        self.embed_dim = embed_dim
        self.state_dim = state_dim

        self.element_encoder = nn.Sequential(
            nn.Linear(element_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.attention = nn.Linear(embed_dim, 1)
        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.ReLU(),
        )

    def forward(self, element_features, element_mask):
        element_embeds = self.element_encoder(element_features)
        attn_scores = self.attention(element_embeds).squeeze(-1)
        attn_scores = attn_scores.masked_fill(element_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1).nan_to_num(0.0).unsqueeze(-1)
        pooled = (element_embeds * attn_weights).sum(dim=1)
        state = self.state_proj(pooled)
        return state, element_embeds
