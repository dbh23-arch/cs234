"""Hand-crafted DOM feature extraction and learned state encoder."""

import re
import numpy as np
import torch
import torch.nn as nn

# 14 element types we track (one-hot encoded in features)
ELEMENT_TYPES = [
    "button", "input_text", "input_checkbox", "input_radio",
    "link", "select", "option", "textarea", "label", "span",
    "div", "li", "td", "other"
]
ELEMENT_TYPE_TO_IDX = {t: i for i, t in enumerate(ELEMENT_TYPES)}
TOKEN_RE = re.compile(r"[a-z0-9]+")

def _as_float(val):
    """numpy scalars have .item(), regular floats don't"""
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
        # miniwob uses both "input_text" tags and "input" with type="text"
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
    # 14 one-hot type bits + 10 numeric features = 24 per element
    feature_dim = len(ELEMENT_TYPES) + 10
    features = np.zeros((max_elements, feature_dim), dtype=np.float32)
    valid_mask = np.zeros(max_elements, dtype=np.float32)

    utterance_words = _tokenize(utterance)

    # pre-compute text anchors so we can associate labels with nearby
    # elements that don't have their own text (e.g. bare input fields)
    text_anchors = []
    for elem in dom_elements[:max_elements]:
        text = str(elem.get("text", "")).strip()
        value = str(elem.get("value", "")).strip()
        combined_text = (text + " " + value).strip()
        tokens = _tokenize(combined_text)
        if not tokens:
            continue
        cx = _as_float(elem.get("left", 0)) + _as_float(elem.get("width", 0)) / 2
        cy = _as_float(elem.get("top", 0)) + _as_float(elem.get("height", 0)) / 2
        text_anchors.append((cx, cy, combined_text, tokens))

    for i, elem in enumerate(dom_elements[:max_elements]):
        tag = elem.get("tag", "div")
        type_attr = elem.get("type", "")
        elem_type = classify_element(tag, type_attr)
        type_idx = ELEMENT_TYPE_TO_IDX.get(elem_type, len(ELEMENT_TYPES) - 1)

        # one-hot element type
        features[i, type_idx] = 1.0

        # normalized bounding box (miniwob canvas is 160x210)
        left = np.clip(_as_float(elem.get("left", 0)) / 160.0, 0.0, 1.0)
        top = np.clip(_as_float(elem.get("top", 0)) / 210.0, 0.0, 1.0)
        width = np.clip(_as_float(elem.get("width", 0)) / 160.0, 0.0, 1.0)
        height = np.clip(_as_float(elem.get("height", 0)) / 210.0, 0.0, 1.0)
        offset = len(ELEMENT_TYPES)
        features[i, offset:offset + 4] = [left, top, width, height]

        features[i, offset + 4] = 1.0 if elem.get("visible", True) else 0.0
        features[i, offset + 5] = 1.0 if elem.get("focused", False) else 0.0

        # text overlap with utterance -- key signal for which element to interact with
        text = str(elem.get("text", "")).lower()
        value = str(elem.get("value", "")).lower()
        combined_text = (text + " " + value).strip()
        text_words = _tokenize(combined_text)

        # if this element has no text, try to grab it from the nearest label
        if not text_words:
            tag = str(elem.get("tag", "")).lower()
            likely_label_target = (
                tag in {"button", "label", "option", "select", "input", "textarea"} or
                tag.startswith("input_") or tag in {"span", "div"}
            )
            if likely_label_target and text_anchors:
                cx = _as_float(elem.get("left", 0)) + _as_float(elem.get("width", 0)) / 2
                cy = _as_float(elem.get("top", 0)) + _as_float(elem.get("height", 0)) / 2
                best = None
                best_dist = float("inf")
                for ax, ay, anchor_text, anchor_tokens in text_anchors:
                    if abs(ay - cy) > 24.0:  # only look at same-row labels
                        continue
                    dist = (ax - cx) ** 2 + (ay - cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best = (anchor_text, anchor_tokens)
                if best is not None:
                    combined_text, text_words = best

        overlap = len(utterance_words & text_words) / max(len(utterance_words), 1)
        features[i, offset + 6] = overlap

        features[i, offset + 7] = min(len(combined_text) / 50.0, 1.0)  # text length

        clickable = elem_type in ("button", "link", "input_checkbox",
                                   "input_radio", "select", "option")
        features[i, offset + 8] = 1.0 if clickable else 0.0

        is_input = elem_type in ("input_text", "textarea")
        features[i, offset + 9] = 1.0 if is_input else 0.0

        valid_mask[i] = 1.0

    return features, valid_mask


class StateEncoder(nn.Module):
    """Encode a set of DOM element features into a fixed-size state vector.

    Uses learned attention to pool over variable-length element sets.
    """

    def __init__(self, element_feature_dim=24, max_elements=64,
                 embed_dim=64, state_dim=256):
        super().__init__()
        self.max_elements = max_elements

        self.element_encoder = nn.Sequential(
            nn.Linear(element_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # single-head attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.ReLU(),
        )

        self.embed_dim = embed_dim
        self.state_dim = state_dim

    def forward(self, element_features, element_mask):
        element_embeds = self.element_encoder(element_features)  # (B, N, D)

        attn_scores = self.attention(element_embeds).squeeze(-1)  # (B, N)
        attn_scores = attn_scores.masked_fill(element_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # softmax of all -inf gives nan, fix that
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = attn_weights.unsqueeze(-1)

        pooled = (element_embeds * attn_weights).sum(dim=1)  # (B, D)
        state = self.state_proj(pooled)  # (B, state_dim)

        return state, element_embeds
