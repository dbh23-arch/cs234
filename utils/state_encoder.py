"""
DOM-based state featurization for MiniWoB++ environments.
Extracts structured features from DOM elements rather than using raw pixels.
"""

import numpy as np
import torch
import torch.nn as nn


# DOM element types we care about
ELEMENT_TYPES = [
    "button", "input_text", "input_checkbox", "input_radio",
    "link", "select", "option", "textarea", "label", "span",
    "div", "li", "td", "other"
]
ELEMENT_TYPE_TO_IDX = {t: i for i, t in enumerate(ELEMENT_TYPES)}


def classify_element(tag, type_attr=""):
    """Classify a DOM element into one of our categories."""
    tag = tag.lower()
    if tag == "button":
        return "button"
    elif tag == "input":
        t = type_attr.lower() if type_attr else "text"
        if t in ("text", "password", "email", "search", "url", "tel"):
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
    """
    Extract features from a list of DOM elements.

    Each element gets a feature vector containing:
    - Element type (one-hot, 14 dims)
    - Normalized position (x, y, w, h - 4 dims)
    - Is visible (1 dim)
    - Is focused (1 dim)
    - Text overlap with utterance (1 dim)
    - Text length (1 dim)
    - Is clickable (1 dim)
    - Is input (1 dim)

    Returns: (num_elements, feature_dim) array, padded/truncated to max_elements
    """
    feature_dim = len(ELEMENT_TYPES) + 10  # 14 + 10 = 24
    features = np.zeros((max_elements, feature_dim), dtype=np.float32)
    valid_mask = np.zeros(max_elements, dtype=np.float32)

    utterance_words = set(utterance.lower().split()) if utterance else set()

    for i, elem in enumerate(dom_elements[:max_elements]):
        tag = elem.get("tag", "div")
        type_attr = elem.get("type", "")
        elem_type = classify_element(tag, type_attr)
        type_idx = ELEMENT_TYPE_TO_IDX.get(elem_type, len(ELEMENT_TYPES) - 1)

        # One-hot element type
        features[i, type_idx] = 1.0

        # Normalized position (assume 160x210 MiniWoB viewport)
        left = elem.get("left", 0) / 160.0
        top = elem.get("top", 0) / 210.0
        width = elem.get("width", 0) / 160.0
        height = elem.get("height", 0) / 210.0
        offset = len(ELEMENT_TYPES)
        features[i, offset:offset + 4] = [left, top, width, height]

        # Is visible
        features[i, offset + 4] = 1.0 if elem.get("visible", True) else 0.0

        # Is focused
        features[i, offset + 5] = 1.0 if elem.get("focused", False) else 0.0

        # Text overlap with utterance
        text = elem.get("text", "").lower()
        text_words = set(text.split()) if text else set()
        overlap = len(utterance_words & text_words) / max(len(utterance_words), 1)
        features[i, offset + 6] = overlap

        # Text length (normalized)
        features[i, offset + 7] = min(len(text) / 50.0, 1.0)

        # Is clickable
        clickable = elem_type in ("button", "link", "input_checkbox",
                                   "input_radio", "select", "option")
        features[i, offset + 8] = 1.0 if clickable else 0.0

        # Is text input
        is_input = elem_type in ("input_text", "textarea")
        features[i, offset + 9] = 1.0 if is_input else 0.0

        valid_mask[i] = 1.0

    return features, valid_mask


class StateEncoder(nn.Module):
    """
    Encodes DOM features + utterance into a fixed-size state vector.
    """

    def __init__(self, element_feature_dim=24, max_elements=64,
                 embed_dim=64, state_dim=256):
        super().__init__()
        self.max_elements = max_elements

        # Per-element encoder
        self.element_encoder = nn.Sequential(
            nn.Linear(element_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Attention pooling over elements
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )

        # Final state projection
        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.ReLU(),
        )

        self.embed_dim = embed_dim
        self.state_dim = state_dim

    def forward(self, element_features, element_mask):
        """
        Args:
            element_features: (batch, max_elements, feature_dim)
            element_mask: (batch, max_elements) - 1 for valid, 0 for padding

        Returns:
            state: (batch, state_dim)
            element_embeds: (batch, max_elements, embed_dim)
        """
        # Encode each element
        element_embeds = self.element_encoder(element_features)  # (B, N, D)

        # Attention pooling
        attn_scores = self.attention(element_embeds).squeeze(-1)  # (B, N)
        attn_scores = attn_scores.masked_fill(element_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, N)
        # Handle all-masked inputs (e.g., padded timesteps in DT sequences)
        # softmax([-inf, ...]) = NaN, replace with 0 so pooled output is zero vector
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = attn_weights.unsqueeze(-1)  # (B, N, 1)

        pooled = (element_embeds * attn_weights).sum(dim=1)  # (B, D)

        # Project to state
        state = self.state_proj(pooled)  # (B, state_dim)

        return state, element_embeds
