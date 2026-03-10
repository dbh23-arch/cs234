import os
import json
from glob import glob
from tqdm import tqdm
import json
import os

import json

import json

def convert_trajs(input_file, output_file):
    with open(input_file, "r") as f:
        traj_list = json.load(f)

    output_trajectories = []

    for traj in traj_list:
        states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []

        steps = traj.get("steps", [])
        reward_value = traj.get("reward", 0.0)
        num_steps = len(steps)

        for t, step in enumerate(steps):
            actions_list.append({
                "action_type_idx": step.get("action_type_idx", 0),
                "element_idx": step.get("element_index", 0)
            })

            state = {
                "features": step.get("features", traj.get("features", [])),
                "mask": traj.get("valid_mask", []),
                "dom_elements_raw": step.get("dom_elements", traj.get("dom_elements", [])),
                "utterance": traj.get("utterance", "")
            }
            states_list.append(state)

            rewards_list.append([reward_value])
            dones_list.append([t == num_steps - 1])

        output_trajectories.append({
            "states": states_list,
            "actions": actions_list,
            "rewards": rewards_list,
            "dones": dones_list
        })

    final_output = {
        "task_name": traj_list[0].get("task", "unknown_task"),
        "num_trajectories": len(traj_list),
        "success_rate": 0.0,
        "trajectories": output_trajectories
    }

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Converted {len(traj_list)} trajectories → {output_file}")

def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = glob(os.path.join(input_folder, "*.json"))

    for file_path in tqdm(json_files, desc="Converting to MiniWoB format"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        convert_trajs(file_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder with original JSON trajectory files")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder to save MiniWoB-style JSON files")
    args = parser.parse_args()

    convert_folder(args.input_folder, args.output_folder)