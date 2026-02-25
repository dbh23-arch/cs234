# Quickstart

Get the project running in 5 minutes.

## 1. Setup

```bash
conda create -n cs234 python=3.10 -y
conda activate cs234
pip install -r requirements.txt
pip install transformers   # needed for text encoder
```

MiniWoB++ requires Chrome and ChromeDriver. Install ChromeDriver matching your Chrome version:
```bash
# macOS
brew install chromedriver
```

## 2. Collect Demonstrations

Collect expert demos for a task (runs headless Chrome under the hood):

```bash
python scripts/collect_demos.py --task click-button --num_demos 500
```

This saves trajectories to `data/demos/click-button.json`. Each trajectory contains DOM features, raw DOM elements, utterances, actions, and rewards.

Collect for multiple tasks:
```bash
for task in click-button click-link click-dialog-2; do
    python scripts/collect_demos.py --task $task --num_demos 500
done
```

To collect mixed-quality data (for suboptimal data experiments):
```bash
python scripts/collect_demos.py --task click-button --num_demos 500 --quality 0.5
```

## 3. Train

Train a single model:

```bash
# BC on click-button with 100 demos, DOM encoder
python train.py --method bc --task click-button --num_demos 100 --seed 42

# IQL with text (DistilBERT) encoder
python train.py --method iql --task click-button --num_demos 100 --seed 42 --encoder_type text

# Decision Transformer
python train.py --method dt --task click-button --num_demos 100 --seed 42
```

Train all methods on all tasks and dataset sizes:
```bash
python train.py --method all
```

Models are saved to `results/models/`. Naming convention: `{method}_{task}_n{demos}_s{seed}.pt` (text encoder models are prefixed with `text_`).

## 4. Evaluate

Evaluate a trained model (runs 100 episodes in headless Chrome):

```bash
python evaluate.py --method bc --task click-button --num_demos 100 --seed 42
```

Evaluate all available models:
```bash
python evaluate.py --method all
```

Results are saved to `results/results.json`.

## 5. Watch an Agent

Open a browser window and watch the agent play:

```bash
python scripts/watch_agent.py --method bc --task click-button --num_demos 100 --seed 42
```

This opens Chrome and prints step-by-step actions to the terminal.

## 6. Plot Results

Generate data-efficiency curves:

```bash
python scripts/plot_results.py
```

## Key Flags

| Flag | Description | Default |
|---|---|---|
| `--method` | `bc`, `iql`, `dt`, or `all` | `bc` |
| `--task` | MiniWoB++ task name | `click-button` |
| `--num_demos` | Number of training demos | sweeps all sizes |
| `--seed` | Random seed | sweeps all seeds |
| `--encoder_type` | `dom` (hand-crafted) or `text` (DistilBERT) | `dom` |
| `--config` | Path to YAML config | `configs/default.yaml` |

## Example: Full Experiment Pipeline

```bash
# 1. Collect data
python scripts/collect_demos.py --task click-button --num_demos 500

# 2. Train all methods with DOM encoder
python train.py --method all --task click-button

# 3. Train all methods with text encoder
python train.py --method all --task click-button --encoder_type text

# 4. Evaluate everything
python evaluate.py --method all --task click-button
python evaluate.py --method all --task click-button --encoder_type text

# 5. Plot comparison
python scripts/plot_results.py
```

## Configuration

All hyperparameters are in `configs/default.yaml`:
- Training: batch size, learning rate, epochs, seeds
- Models: hidden dims, layers, DT context length
- Encoder: LM model name, freeze/finetune
- Tasks: simple/medium/hard task lists
