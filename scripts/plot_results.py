"""Generate figures for the paper/poster from evaluation results."""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

TASK_TIERS = {
    "simple": ["click-button", "click-link", "click-option",
               "click-dialog", "click-dialog-2"],
    "medium": ["login-user", "enter-text", "search-engine",
               "navigate-tree", "click-checkboxes"],
    "hard": ["email-inbox", "social-media", "use-autocomplete"],
}

METHOD_COLORS = {"bc": "#1f77b4", "iql": "#ff7f0e", "dt": "#2ca02c"}
METHOD_LABELS = {"bc": "BC", "iql": "IQL", "dt": "DT"}

def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)

def aggregate_results(results):
    aggregated = {}
    for r in results:
        key = (r["method"], r["task"], r["num_demos"])
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r["success_rate"])

    summary = {}
    for key, rates in aggregated.items():
        summary[key] = {
            "mean": np.mean(rates),
            "std": np.std(rates),
            "se": np.std(rates) / np.sqrt(len(rates)),
            "n": len(rates),
        }
    return summary

def plot_data_efficiency(results, save_dir, tier=None):
    summary = aggregate_results(results)

    if tier:
        tasks = TASK_TIERS[tier]
        title = f"Data Efficiency - {tier.capitalize()} Tasks"
        filename = f"data_efficiency_{tier}.pdf"
    else:
        tasks = [t for tier_tasks in TASK_TIERS.values() for t in tier_tasks]
        title = "Data Efficiency - All Tasks"
        filename = "data_efficiency_all.pdf"

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    demo_sizes = sorted(set(r["num_demos"] for r in results))

    for method in ["bc", "iql", "dt"]:
        means = []
        ses = []
        valid_sizes = []

        for n in demo_sizes:
            task_rates = []
            for task in tasks:
                key = (method, task, n)
                if key in summary:
                    task_rates.append(summary[key]["mean"])

            if task_rates:
                means.append(np.mean(task_rates))
                ses.append(np.std(task_rates) / np.sqrt(len(task_rates)))
                valid_sizes.append(n)

        if means:
            means = np.array(means)
            ses = np.array(ses)
            ax.plot(valid_sizes, means, 'o-',
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    linewidth=2, markersize=6)
            ax.fill_between(valid_sizes, means - ses, means + ses,
                           alpha=0.2, color=METHOD_COLORS[method])

    ax.set_xlabel("Number of Demonstrations")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_per_task_breakdown(results, save_dir, num_demos=100):
    summary = aggregate_results(results)

    all_tasks = [t for tier_tasks in TASK_TIERS.values() for t in tier_tasks]
    available_tasks = [t for t in all_tasks
                       if any((m, t, num_demos) in summary
                              for m in ["bc", "iql", "dt"])]

    if not available_tasks:
        print(f"No results for num_demos={num_demos}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    x = np.arange(len(available_tasks))
    width = 0.25

    for i, method in enumerate(["bc", "iql", "dt"]):
        means = []
        errs = []
        for task in available_tasks:
            key = (method, task, num_demos)
            if key in summary:
                means.append(summary[key]["mean"])
                errs.append(summary[key]["se"])
            else:
                means.append(0)
                errs.append(0)

        ax.bar(x + i * width, means, width, yerr=errs,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method],
               alpha=0.8, capsize=3)

    ax.set_xlabel("Task")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Per-Task Comparison (N={num_demos} demos)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(available_tasks, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"per_task_n{num_demos}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per_task_n{num_demos}.pdf")

def print_results_table(results):
    summary = aggregate_results(results)

    demo_sizes = sorted(set(r["num_demos"] for r in results))
    methods = ["bc", "iql", "dt"]

    print(f"\n{'Task':<20}", end="")
    for n in demo_sizes:
        for m in methods:
            print(f" {METHOD_LABELS[m]}@{n:<4}", end="")
    print()
    print("-" * (20 + len(demo_sizes) * len(methods) * 10))

    for tier_name, tasks in TASK_TIERS.items():
        print(f"\n--- {tier_name.upper()} ---")
        for task in tasks:
            print(f"{task:<20}", end="")
            for n in demo_sizes:
                for m in methods:
                    key = (m, task, n)
                    if key in summary:
                        print(f" {summary[key]['mean']:.1%}   ", end="")
                    else:
                        print(f"  --     ", end="")
            print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/results.json")
    parser.add_argument("--save_dir", type=str, default="results/plots")
    parser.add_argument("--num_demos", type=int, default=100,
                        help="Demo count for per-task plot")
    args = parser.parse_args()

    results = load_results(args.results)

    
    plot_data_efficiency(results, args.save_dir)
    for tier in ["simple", "medium", "hard"]:
        plot_data_efficiency(results, args.save_dir, tier=tier)

    plot_per_task_breakdown(results, args.save_dir, num_demos=args.num_demos)

    
    print_results_table(results)

if __name__ == "__main__":
    main()
