"""Print eval results summary table."""
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
                if "Evaluating" in lines[j] and "\.pt" not in lines[j]:
                    break
                if re.search(r"Evaluating \S+\.pt", lines[j]):
                    break
            if sr:
                name = m.group(1)
                parts = name.split("_")
                task = "_".join(parts[1:-2])
                results.setdefault(task, []).append(float(sr.group(1)))
    return results

all_results = {}
for method in ["bc", "iql", "ppo"]:
    all_results[method] = parse_log("logs/eval_%s.log" % method)
    for w in range(4):
        for task, vals in parse_log("logs/eval_%s_w%d.log" % (method, w)).items():
            all_results[method].setdefault(task, []).extend(vals)

all_results["dt"] = {}
for w in range(4):
    for task, vals in parse_log("logs/eval_dt_w%d.log" % w).items():
        all_results["dt"].setdefault(task, []).extend(vals)

tasks = sorted(set(t for m in all_results.values() for t in m))
header = "%-22s %8s %8s %8s %8s" % ("Task", "BC", "IQL", "DT", "PPO")
print(header)
print("-" * 58)
for task in tasks:
    row = "%-22s" % task
    for method in ["bc", "iql", "dt", "ppo"]:
        vals = all_results[method].get(task, [])
        if vals:
            avg = sum(vals) / len(vals)
            row += " %7.1f%%" % avg
        else:
            row += "      --"
    print(row)
print("-" * 58)
counts = []
for method in ["bc", "iql", "dt", "ppo"]:
    total = sum(len(v) for v in all_results[method].values())
    counts.append("%s:%d" % (method.upper(), total))
print("  ".join(counts))
