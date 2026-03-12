import subprocess
import sys
import json
import os

TASKS_NEEDING_COLLECTION = [
    "click-option", "click-dialog", "login-user",
    "search-engine", "navigate-tree", "email-inbox", "use-autocomplete"
]

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for task in TASKS_NEEDING_COLLECTION:
    print(f"\n{'='*50}")
    print(f"Collecting 500 demos for: {task}")
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, "scripts/collect_demos.py",
         "--task", task, "--num_demos", "500",
         "--save_dir", "data/demos", "--seed", "42"],
        cwd=CODE_DIR,
        capture_output=False
    )

    
    demo_path = os.path.join(CODE_DIR, "data", "demos", f"{task}.json")
    try:
        with open(demo_path) as f:
            d = json.load(f)
        n = d.get("num_trajectories", 0)
        sr = d.get("success_rate", 0)
        print(f"  -> {task}: {n} trajectories ({sr:.0%} success)")
    except Exception as e:
        print(f"  -> ERROR: {e}")

print("\n\nAll demo collection complete!")
print("="*50)

all_tasks = [
    "click-button", "click-link", "click-option", "click-dialog", "click-dialog-2",
    "login-user", "enter-text", "search-engine", "navigate-tree", "click-checkboxes",
    "email-inbox", "social-media", "use-autocomplete"
]
for task in all_tasks:
    demo_path = os.path.join(CODE_DIR, "data", "demos", f"{task}.json")
    try:
        with open(demo_path) as f:
            d = json.load(f)
        n = d.get("num_trajectories", 0)
        print(f"  {task}: {n} trajectories")
    except:
        print(f"  {task}: MISSING/BROKEN")
