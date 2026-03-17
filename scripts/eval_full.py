"""Print detailed eval results per seed/demo size."""
import re
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

def parse_log(path):
    results = {}
    try:
        lines = open(path).readlines()
    except FileNotFoundError:
        return results
    for i, line in enumerate(lines):
        m = re.search(r"Evaluating (\S+)\.pt", line)
        if m:
            sr = None
            for j in range(i + 1, min(i + 200, len(lines))):
                sr = re.search(r"success_rate = ([\d.]+)%", lines[j])
                if sr:
                    break
                if re.search(r"Evaluating \S+\.pt", lines[j]):
                    break
            if sr:
                name = m.group(1)
                parts = name.split("_")
                task = "_".join(parts[1:-2])
                demos = parts[-2]
                seed = parts[-1]
                key = demos + "_" + seed
                results.setdefault(task, {})[key] = float(sr.group(1))
    return results

# Gather
all_r = {}
for method in ["bc", "iql", "ppo"]:
    all_r[method] = parse_log("logs/eval_%s.log" % method)
    for w in range(4):
        for task, vals in parse_log("logs/eval_%s_w%d.log" % (method, w)).items():
            all_r[method].setdefault(task, {}).update(vals)

all_r["dt"] = {}
for w in range(4):
    for task, vals in parse_log("logs/eval_dt_w%d.log" % w).items():
        all_r["dt"].setdefault(task, {}).update(vals)

TASKS = [
    "click-button", "click-link", "click-option", "click-dialog", "click-dialog-2",
    "click-checkboxes", "login-user", "enter-text", "search-engine",
    "navigate-tree", "email-inbox", "social-media", "use-autocomplete",
]

for method_upper in ["BC", "IQL", "DT", "PPO"]:
    m = method_upper.lower()
    total = sum(len(v) for v in all_r.get(m, {}).values())
    print("=" * 70)
    print("  %s  (%d models evaluated)" % (method_upper, total))
    print("=" * 70)

    if m == "ppo":
        print("%-22s %8s %8s %8s %8s" % ("Task", "s42", "s123", "s456", "Mean"))
        print("-" * 54)
        for task in TASKS:
            data = all_r.get(m, {}).get(task, {})
            if not data:
                continue
            vals = []
            row = "%-22s" % task
            for s in ["n0_s42", "n0_s123", "n0_s456"]:
                v = data.get(s)
                if v is not None:
                    row += " %7.0f%%" % v
                    vals.append(v)
                else:
                    row += "      --"
            if vals:
                row += " %7.1f%%" % (sum(vals) / len(vals))
            print(row)
    else:
        print("%-22s | %6s %6s %6s | %6s %6s %6s | %6s %6s %6s | %6s" % (
            "Task", "10/42", "10/123", "10/456", "25/42", "25/123", "25/456",
            "50/42", "50/123", "50/456", "Mean"))
        print("-" * 100)
        for task in TASKS:
            data = all_r.get(m, {}).get(task, {})
            if not data:
                continue
            vals_all = []
            row = "%-22s |" % task
            for prefix in ["n10", "n25", "n50"]:
                for s in ["s42", "s123", "s456"]:
                    v = data.get(prefix + "_" + s)
                    if v is not None:
                        row += " %5.0f%%" % v
                        vals_all.append(v)
                    else:
                        row += "    --"
                row += " |"
            if vals_all:
                row += " %5.1f%%" % (sum(vals_all) / len(vals_all))
            print(row)
    print()

# Summary
print("=" * 70)
print("SUMMARY (avg success rate per method)")
print("=" * 70)
print("%-22s %8s %8s %8s %8s" % ("Task", "BC", "IQL", "DT", "PPO"))
print("-" * 58)
for task in TASKS:
    row = "%-22s" % task
    for m in ["bc", "iql", "dt", "ppo"]:
        data = all_r.get(m, {}).get(task, {})
        if data:
            vals = list(data.values())
            row += " %7.1f%%" % (sum(vals) / len(vals))
        else:
            row += "      --"
    print(row)
