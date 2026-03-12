"""Wrapper around MiniWoB++ gym env that handles DOM feature extraction."""

import gymnasium as gym
import numpy as np
from utils.state_encoder import extract_dom_features


class MiniWoBWrapper:

    def __init__(self, task_name, max_elements=64, headless=True, wait_ms=500):
        self.task_name = task_name
        self.max_elements = max_elements
        self.headless = headless
        self.wait_ms = wait_ms

        self.env = gym.make(
            f"miniwob/{task_name}-v1",
            render_mode=None,
            wait_ms=wait_ms,
        )

        self.current_dom_elements = []
        self.utterance = ""

    def reset(self):
        obs, info = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        miniwob_action = self._convert_action(action)
        obs, reward, done, truncated, info = self.env.step(miniwob_action)
        state_features = self._process_obs(obs)
        return state_features, reward, done, truncated, info

    def _process_obs(self, obs):
        self.utterance = obs.get("utterance", "")
        dom_elements = self._get_dom_elements(obs)
        self.current_dom_elements = dom_elements

        features, mask = extract_dom_features(
            dom_elements, self.utterance, self.max_elements
        )

        return {
            "features": features,
            "mask": mask,
            "utterance": self.utterance,
            "dom_elements": dom_elements,
            "num_elements": len(dom_elements),
        }

    def _get_dom_elements(self, obs):
        elements = []

        dom_info = obs.get("dom_elements", [])
        if not dom_info:
            return elements

        for elem in dom_info:
            elements.append({
                "tag": elem.get("tag", "div"),
                "type": elem.get("type", ""),
                "text": elem.get("text", ""),
                "value": elem.get("value", ""),
                "left": elem.get("left", 0),
                "top": elem.get("top", 0),
                "width": elem.get("width", 0),
                "height": elem.get("height", 0),
                "visible": elem.get("visible", True),
                "focused": elem.get("focused", False),
                "ref": elem.get("ref", None),
            })

        return elements

    def _convert_action(self, action):
        action_type = action["action_type"]
        elem_idx = action["element_idx"]

        if elem_idx >= len(self.current_dom_elements):
            # invalid index, just do something random
            return self.env.action_space.sample()

        elem = self.current_dom_elements[elem_idx]
        cx = elem["left"] + elem["width"] / 2
        cy = elem["top"] + elem["height"] / 2

        if action_type == "click":
            return self.env.unwrapped.create_action(
                "click", left=cx, top=cy
            )
        elif action_type == "type":
            text = action.get("text", "")
            return self.env.unwrapped.create_action(
                "type", left=cx, top=cy, text=text
            )
        else:
            return self.env.unwrapped.create_action(
                "click", left=cx, top=cy
            )

    def get_num_elements(self):
        return len(self.current_dom_elements)

    def close(self):
        self.env.close()
