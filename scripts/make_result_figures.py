import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11

TASK_TIERS = {
    "simple": [
        "click-button",
        "click-link",
        "click-option",
        "click-dialog",
        "click-dialog-2",
    ],
    "medium": [
        "click-checkboxes",
        "navigate-tree",
        "login-user",
        "enter-text",
        "search-engine",
    ],
    "hard": ["email-inbox", "social-media", "use-autocomplete"],
}

ALL_TASKS = [task for tasks in TASK_TIERS.values() for task in tasks]
METHODS = ["bc", "iql", "dt"]
METHOD_LABELS = {"bc": "BC", "iql": "IQL", "dt": "DT"}
METHOD_COLORS = {"bc": "#1f77b4", "iql": "#ff7f0e", "dt": "#2ca02c"}

def load_results(path: str) -> List[dict]:
    if os.path.isdir(path):
        records: List[dict] = []
        names = sorted(os.listdir(path))
        
        human_names = [
            n for n in names
            if n.startswith("human_results_") and n.endswith(".json")
        ]
        legacy_names = [
            n for n in names
            if n.startswith("results_") and n.endswith(".json")
        ]
        use_names = human_names if human_names else legacy_names
        for name in names:
            if name not in use_names:
                continue
            with open(os.path.join(path, name), "r") as f:
                chunk = json.load(f)
            if isinstance(chunk, list):
                records.extend(chunk)
        return records

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list) and data:
        first = data[0]
        
        if isinstance(first, dict) and first.get("task") in {"click-option"} and len(data) <= 3:
            parent_dir = os.path.dirname(path) or "."
            return load_results(parent_dir)

    return data

def aggregate(results: List[dict]) -> Dict[Tuple[str, str, int], dict]:
    grouped: Dict[Tuple[str, str, int], List[float]] = {}
    for row in results:
        task = row.get("task")
        method = row.get("method")
        n = row.get("num_demos")
        sr = row.get("success_rate")
        if task not in ALL_TASKS or method not in METHODS or n is None:
            continue
        key = (method, task, int(n))
        grouped.setdefault(key, []).append(float(sr))

    out: Dict[Tuple[str, str, int], dict] = {}
    for key, values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": int(arr.size),
            "se": float(arr.std() / np.sqrt(arr.size)),
        }
    return out

def save_both(fig: plt.Figure, outdir: str, stem: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{stem}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_overall_data_efficiency(summary: Dict[Tuple[str, str, int], dict], outdir: str) -> None:
    demo_sizes = sorted({key[2] for key in summary.keys()})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for method in METHODS:
        means = []
        errs = []
        x = []
        for n in demo_sizes:
            vals = [summary[(method, task, n)]["mean"] for task in ALL_TASKS if (method, task, n) in summary]
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            means.append(arr.mean())
            errs.append(arr.std() / np.sqrt(arr.size))
            x.append(n)

        if x:
            means_arr = np.array(means)
            errs_arr = np.array(errs)
            ax.plot(
                x,
                means_arr,
                marker="o",
                linewidth=2.2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            ax.fill_between(x, means_arr - errs_arr, means_arr + errs_arr, color=METHOD_COLORS[method], alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlabel("Number of demonstrations")
    ax.set_ylabel("Mean success rate across 13 tasks")
    ax.set_title("Offline Method Data-Efficiency")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_both(fig, outdir, "fig1_data_efficiency_overall")

def plot_task_heatmap(summary: Dict[Tuple[str, str, int], dict], outdir: str, demo_count: int) -> None:
    matrix = np.full((len(ALL_TASKS), len(METHODS)), np.nan, dtype=np.float64)
    for i, task in enumerate(ALL_TASKS):
        for j, method in enumerate(METHODS):
            key = (method, task, demo_count)
            if key in summary:
                matrix[i, j] = summary[key]["mean"]

    fig, ax = plt.subplots(figsize=(4.8, 6.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Success rate")

    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_yticks(np.arange(len(ALL_TASKS)))
    ax.set_yticklabels(ALL_TASKS)
    ax.set_title(f"Per-Task Success Heatmap (N={demo_count})")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val >= 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.tight_layout()
    save_both(fig, outdir, f"fig2_task_heatmap_n{demo_count}")

def plot_tier_bars(summary: Dict[Tuple[str, str, int], dict], outdir: str, demo_count: int) -> None:
    tiers = list(TASK_TIERS.keys())
    x = np.arange(len(tiers))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for j, method in enumerate(METHODS):
        means = []
        errs = []
        for tier in tiers:
            vals = [
                summary[(method, task, demo_count)]["mean"]
                for task in TASK_TIERS[tier]
                if (method, task, demo_count) in summary
            ]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                means.append(arr.mean())
                errs.append(arr.std() / np.sqrt(arr.size))
            else:
                means.append(np.nan)
                errs.append(0.0)

        ax.bar(
            x + (j - 1) * width,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            color=METHOD_COLORS[method],
            alpha=0.9,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tiers])
    ax.set_ylabel("Mean success rate")
    ax.set_title(f"Performance by Task Tier (N={demo_count})")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_both(fig, outdir, f"fig3_tier_comparison_n{demo_count}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--outdir", default="results/figures")
    parser.add_argument("--demo-count", type=int, default=50)
    args = parser.parse_args()

    results = load_results(args.results)
    summary = aggregate(results)

    plot_overall_data_efficiency(summary, args.outdir)
    plot_task_heatmap(summary, args.outdir, args.demo_count)
    plot_tier_bars(summary, args.outdir, args.demo_count)

    print(f"Wrote figures to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
