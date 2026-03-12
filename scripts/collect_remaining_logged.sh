#!/bin/bash
cd "/Users/Andrew/Downloads/CS234 Reinforcement Learning/project/code"
LOG="demo_collection.log"
PYTHON="/Users/Andrew/miniconda3/envs/cs234/bin/python"

TASKS=(click-option click-dialog click-dialog-2 login-user search-engine navigate-tree email-inbox use-autocomplete)

echo "Starting demo collection at $(date)" > "$LOG"

for task in "${TASKS[@]}"; do
    echo "" >> "$LOG"
    echo "$task start $(date)" >> "$LOG"
    $PYTHON scripts/collect_demos.py \
        --task "$task" --num_demos 500 --save_dir data/demos --seed 42 >> "$LOG" 2>&1
    echo "$task end $(date)" >> "$LOG"

    $PYTHON -c "
import json
with open('data/demos/${task}.json') as f:
    d = json.load(f)
print(f'  ${task}: {d[\"num_trajectories\"]} trajs ({d[\"success_rate\"]:.0%} success)')
" >> "$LOG" 2>&1
done

echo "" >> "$LOG"
echo "All done at $(date)" >> "$LOG"
