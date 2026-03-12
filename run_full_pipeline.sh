#!/bin/bash

set -e

CD_DIR="/Users/Andrew/Downloads/CS234 Reinforcement Learning/project/code"
cd "$CD_DIR"

CONDA_ENV="cs234"
TASKS=(
    click-button click-link click-option click-dialog click-dialog-2
    login-user enter-text search-engine navigate-tree click-checkboxes
    email-inbox social-media use-autocomplete
)

echo "Collecting demos for all tasks..."

for task in "${TASKS[@]}"; do
    echo "Collecting demos: $task"
    conda run -n $CONDA_ENV --no-banner python scripts/collect_demos.py \
        --task "$task" --num_demos 500 --save_dir data/demos --seed 42

    if [ -f "data/demos/${task}.json" ]; then
        SUCCESS=$(python -c "import json; d=json.load(open('data/demos/${task}.json')); print(f'{d[\"num_trajectories\"]} trajs, success={d[\"success_rate\"]:.1%}')")
        echo "  Result: $SUCCESS"
    else
        echo "  WARNING: No demo file generated for $task!"
    fi
done

echo "Training all methods across tasks..."

for method in bc iql dt ppo; do
    echo "Training: $method"
    conda run -n $CONDA_ENV --no-banner python train.py \
        --method "$method" --source human --save_dir results
done

echo "Evaluating all methods across tasks..."
rm -f results/results_*.json results/results.json \
      results/human_results_*.json results/human_results.json

for method in bc iql dt ppo; do
    echo "Evaluating: $method"
    for task in "${TASKS[@]}"; do
        echo "  $method on $task"
        conda run -n $CONDA_ENV --no-banner python evaluate.py \
            --method "$method" --task "$task" --source human \
            --num_episodes 20 --results_dir results
    done
done

echo "Pipeline complete."
echo "Results saved to results/"
