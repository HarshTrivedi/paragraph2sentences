from typing import List
import json
import os
import subprocess
import math
import re
import shutil
import argparse

def _chunks(items: List, size):
    """returns successive n-sized chunks from l."""
    return [items[index:index + size] for index in range(0, len(items), size)]

def _create_dataset(dataset_path: str):
    command = f'beaker dataset create {dataset_path}'
    output = subprocess.check_output(command, shell=True)
    beaker_dataset_id = re.search(r" ds_\w+", str(output)).group().strip()
    return beaker_dataset_id

def _get_experiment_result_datasets_ids(beaker_experiment_name):
    experiment_details = subprocess.check_output(["beaker", "experiment", "inspect", beaker_experiment_name]).strip()
    experiment_details = json.loads(experiment_details)
    return experiment_details[0]["nodes"][0]["result_id"]

def _create_docker_image(experiment_name):
    # NOTE: Please change this to what your docker and project name.
    docker_image_name = f"harshtrivedi/para2sentences:{experiment_name}"
    command = f"docker build -t {docker_image_name} ."
    returncode = subprocess.run(command, shell=True).returncode
    return docker_image_name if returncode == 0 else None

def _create_beaker_image(docker_image_name):
    command = f"beaker image create {docker_image_name}"
    output = subprocess.check_output(command, shell=True)
    beaker_image_id = re.search(r" bp_\w+ ", str(output)).group().strip()
    return beaker_image_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Splits hotpotqa paragraphs to sentences and applies coref replacement.')
    parser.add_argument('paragraphs_file', type=str, help='path to hotpotqa input paragraphs file.')
    parser.add_argument('--data-type', type=str, choices=("wiki", "hotpotqa"), default="hotpotqa")
    parser.add_argument('--experiment-name', type=str, help='experiment name, it is required.', required=True)
    parser.add_argument('--num-workers', type=int, help='number of workers.', default=10)
    parser.add_argument('--force', action='store_true', default=False, help='If specified, force do all steps.')
    parser.add_argument('--dirk', action='store_true', default=False, help='Use dirks coref resolver function.')
    # NOTE: Only data-type wiki is useful for your purpose. Use --dirk as default coref replacement flag.

    args = parser.parse_args()

    # experiment-names:
    # coref-replace-hotpotqa-para2sents-abstract-only
    # coref-replace-hotpotqa-train 
    # coref-replace-hotpotqa-dev

    experiment_description = args.experiment_name
    if args.dirk:
        experiment_description += "_dirk"

    hotpotqa_para_instances_file = args.paragraphs_file
    num_workers = args.num_workers

    data_directory = f"data/subtasks_inputs/{args.experiment_name}"

    if os.path.exists(data_directory):
        exit(f"Path: {data_directory} already exists!")
    else:
        os.makedirs(data_directory)

    image_id_store = f"{data_directory}/.beaker_image_id.txt"
    dataset_ids_store = f"{data_directory}/.dataset_ids.json"

    if not os.path.exists(image_id_store) or args.force:
        if os.path.exists(data_directory):
            shutil.rmtree(data_directory)
        os.makedirs(data_directory)
        docker_image_name = _create_docker_image(args.experiment_name)
        beaker_image_id = _create_beaker_image(docker_image_name)

        with open(image_id_store, "w") as file:
            file.write(str(beaker_image_id))
    else:
        with open(image_id_store, "r") as file:
            beaker_image_id = file.read().strip()


    if not os.path.exists(dataset_ids_store) or args.force:
        all_lines = []
        with open(hotpotqa_para_instances_file, "r") as file:
            all_lines = file.readlines()
        per_worker_num_lines = math.ceil(len(all_lines) / num_workers)
        dataset_ids = {}
        for index, lines in enumerate(_chunks(all_lines, per_worker_num_lines), start=1):
            print(f"Creating subtask dataset {index}.")
            dataset_path = f"{data_directory}/input-{index}.jsonl"
            with open(dataset_path, "w") as file:
                for line in lines:
                    file.write(line.strip() + "\n")
                # Create the beaker dataset and get the id
            # File should be closed before creating a dataset out of it.
            dataset_ids[str(index)] = _create_dataset(dataset_path)

        with open(dataset_ids_store, "w") as file:
            json.dump(dataset_ids, file)
    else:
        with open(dataset_ids_store, "r") as file:
            dataset_ids = json.load(file)
        assert len(dataset_ids) == num_workers


    tasks = []
    results_path = "/output"
    depends_on_list = []
    intermediate_ouputs = []
    for subtask_index in range(1, num_workers+1):
        input_dataset_path = f"/input/input-{subtask_index}.jsonl"
        dataset_mounts = [{
            "datasetId": dataset_ids[str(subtask_index)],
            "containerPath": input_dataset_path
        }]

        script_name = "hotpotqa_para2sentences_dirk.py" if args.dirk else "hotpotqa_para2sentences.py"
        run_command = ["python", script_name, input_dataset_path,
                       f"/output/output-{subtask_index}.jsonl", "--data-type", args.data_type]
        task_name = f"subtask-{subtask_index}"
        tasks.append({
             "spec": {
                 "description": f"subtask-{subtask_index}",
                 "image": beaker_image_id,
                 "resultPath": results_path,
                 "args": run_command,
                 "datasetMounts": dataset_mounts,
                 "requirements": {
                      "memory": "20g",
                      "cpu": 16
                 },
             },
             "name": task_name
        })

        intermediate_ouput = results_path + f"-{subtask_index}"
        intermediate_ouputs.append(intermediate_ouput + f"/output-{subtask_index}.jsonl")
        depends_on_list.append({"parentName": task_name, "containerPath": intermediate_ouput})

    aggregated_output = "/output/output.jsonl"
    tasks.append({
        "spec": {
            "description": f"merge all the outputs",
            "image": "merge_concat",
            "args": ["--input-files"] + intermediate_ouputs + ["--output-file", aggregated_output],
            "resultPath": "/output"
        },
        "name": "merge_suboutputs",
        "dependsOn": depends_on_list
    })

    config = {"description": experiment_description, "tasks": tasks}
    experiment_config_file = f".beaker-experiment.json"
    with open(experiment_config_file, "w") as output:
        output.write(json.dumps(config, indent=4))

    experiment_command = ["beaker", "experiment", "create", "--file", experiment_config_file,
                          "--name", args.experiment_name]
    print(f"\nRun the experiment with:")
    print(f"    " + " ".join(experiment_command))

    subprocess.run(experiment_command)
