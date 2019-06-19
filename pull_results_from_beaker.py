#!/usr/bin/env python
import logging
import os
import subprocess
import sys
import argparse
import json
import glob
import re
from tqdm import tqdm

# import util

def _get_experiment_result_datasets_ids(beaker_experiment_name):
    experiment_details = subprocess.check_output(["beaker", "experiment", "inspect", beaker_experiment_name]).strip()
    experiment_details = json.loads(experiment_details)
    result_ids = [node["result_id"] for node in experiment_details[0]["nodes"]]
    return result_ids

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='experiment name.')
    args = parser.parse_args()

    target_dir = f"data/subtasks_outputs/{args.experiment_name}"

    ########
    print("Pull results down...")
    if os.path.exists(target_dir):
        exit(f"Path: {target_dir} already exists!")
    else:
        os.makedirs(target_dir)

    result_datasets_ids = _get_experiment_result_datasets_ids(args.experiment_name)
    for dataset_id in result_datasets_ids:
        print(f"Pulling {dataset_id}")
        beaker_pull_command = f"beaker dataset fetch --output {target_dir} {dataset_id}"
        print(beaker_pull_command)
        subprocess.run(beaker_pull_command, shell=True)
        print(f"Pulled at: {target_dir}")


    ########
    file_paths = [file_path for file_path in glob.glob(target_dir + "/*")
                  if re.search(r"\d+", file_path)]
    file_paths = sorted(file_paths, key=lambda file_path: int(re.search(r"\d+", file_path).group()))
    print("Aggregating the outputs now in following order:")
    print("\n".join(file_paths))

    aggregated_file = target_dir + "/output-aggregated.jsonl"
    with open(aggregated_file, "w") as write_file:
        for file_path in file_paths:
            print(f"Taken {file_path}")
            with open(file_path, "r") as read_file:
                for line in tqdm(read_file.readlines()):
                    if not line.strip():
                        continue
                    write_file.write(line.strip() + "\n")

    print(f"Aggregated output at: {aggregated_file}")
