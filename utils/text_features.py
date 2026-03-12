"""Convert DOM trees to text for the DistilBERT encoder.

Each element gets a bracketed index like [0] button: Submit so the LM
can learn to attend to the right elements.
"""

import numpy as np


def dom_to_text(dom_elements, utterance, max_elements=64):
    parts = [f"Task: {utterance}"]
    element_char_spans = []

    for i, elem in enumerate(dom_elements[:max_elements]):
        tag = elem.get("tag", "div")
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")

        
        if text:
            desc = f"{tag}: {text}"
        elif elem_type:
            desc = f"{tag} ({elem_type})"
        else:
            desc = tag

        marker = f" | [{i}] {desc}"
        start = len(" ".join(parts)) + len(marker) - len(desc)
        end = start + len(desc)
        element_char_spans.append((start, end))
        parts.append(f"[{i}] {desc}")

    text = " | ".join(parts)

    
    while len(element_char_spans) < max_elements:
        element_char_spans.append((0, 0))

    return text, element_char_spans

def tokenize_page(text, element_char_spans, tokenizer, max_elements=64,
                  max_length=512):
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"][0]  
    attn_mask = encoded["attention_mask"][0]  
    offsets = encoded["offset_mapping"][0]  

    # map character spans to token spans so we know which tokens belong to which element
    element_token_spans = np.zeros((max_elements, 2), dtype=np.int64)
    element_mask = np.zeros(max_elements, dtype=np.float32)

    for i, (char_start, char_end) in enumerate(element_char_spans[:max_elements]):
        if char_start == 0 and char_end == 0:
            continue

        tok_start = None
        tok_end = None
        for t_idx, (t_start, t_end) in enumerate(offsets):
            if t_end == 0 and t_start == 0:
                continue
            if t_end > char_start and tok_start is None:
                tok_start = t_idx
            if t_start < char_end:
                tok_end = t_idx + 1

        if tok_start is not None and tok_end is not None:
            element_token_spans[i] = [tok_start, tok_end]
            element_mask[i] = 1.0

    return input_ids, attn_mask, element_token_spans, element_mask

def dom_elements_to_raw(dom_elements, max_elements=64):
    raw = []
    for elem in dom_elements[:max_elements]:
        def to_float(val):
            if hasattr(val, 'item'):
                return val.item()
            return float(val) if val is not None else 0.0

        raw.append({
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
        })
    return raw
