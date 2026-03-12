# Offline RL for Web Navigation

CS234 course project comparing offline reinforcement learning methods for web navigation on MiniWoB++ tasks.

## Overview

We compare three offline RL algorithms on the [MiniWoB++](https://miniwob.farama.org/) benchmark:

- **BC (Behavioral Cloning)** -- supervised imitation learning that directly copies expert actions
- **IQL (Implicit Q-Learning)** -- offline RL that learns Q-values without querying out-of-distribution actions
- **DT (Decision Transformer)** -- sequence modeling approach that conditions on desired returns

Each method is evaluated with two state encoders:
- **DOM encoder** -- hand-crafted 24-dim features per element (position, size, tag type, text similarity)
- **Text encoder** -- frozen DistilBERT that processes page text, producing contextual embeddings

## Project Structure

```
code/
├── train.py                    # Main training script
├── evaluate.py                 # Evaluate trained agents in MiniWoB++
├── configs/
│   └── default.yaml            # Hyperparameters and task lists
├── models/
│   ├── bc.py                   # Behavioral Cloning agent
│   ├── iql.py                  # Implicit Q-Learning agent
│   └── dt.py                   # Decision Transformer agent
├── utils/
│   ├── state_encoder.py        # DOM feature encoder (hand-crafted)
│   ├── text_state_encoder.py   # DistilBERT text encoder
│   ├── text_features.py        # DOM-to-text conversion and tokenization
│   ├── dataset.py              # PyTorch datasets for all encoder types
│   └── env_wrapper.py          # Environment utilities
├── scripts/
│   ├── collect_demos.py        # Collect demonstration data via heuristic policies
│   ├── watch_agent.py          # Watch a trained agent play in the browser
│   └── plot_results.py         # Generate result plots
├── data/demos/                 # Collected demonstration trajectories (JSON)
├── results/
│   ├── models/                 # Saved model checkpoints (.pt)
│   └── results.json            # Evaluation results
└── requirements.txt
```

## Methods

### Behavioral Cloning (BC)
Supervised learning: minimize cross-entropy between predicted and expert actions. Simple but can't distinguish good actions from bad when trained on mixed-quality data.

### Implicit Q-Learning (IQL)
Learns state-action values (Q) and state values (V) from offline data, then extracts a policy weighted by advantage = Q - V. Can handle suboptimal demonstrations because low-value actions get downweighted.

### Decision Transformer (DT)
Frames RL as sequence modeling. A GPT-style transformer takes (return-to-go, state, action) sequences and predicts the next action. At test time, conditioning on high desired return steers the agent toward successful behavior.

### Architecture Invariant
All encoders produce the same output shape:
- `state`: (B, 256) -- global page representation
- `element_embeds`: (B, 64, 64) -- per-element embeddings

This means the action heads in BC, IQL, and DT are identical regardless of which encoder is used.

## Tasks

| Difficulty | Tasks |
|---|---|
| Simple | click-button, click-link, click-option, click-dialog, click-dialog-2 |
| Medium | login-user, enter-text, search-engine, navigate-tree, click-checkboxes |
| Hard | email-inbox, social-media, use-autocomplete |

## Results

Baseline results (DOM encoder) are available for 3 simple tasks across 6 dataset sizes (10-500 demos) and 3 seeds. See `results/results.json`.
