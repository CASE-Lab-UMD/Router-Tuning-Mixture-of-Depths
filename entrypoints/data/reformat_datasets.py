#!/usr/bin/env python
# coding=utf-8
'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
'''

import argparse
import json
import multiprocessing
import os

from datasets import load_dataset
from tqdm import tqdm

from utils.io import create_dir, load_json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def convert_convert_ShareGPT_Vicuna_unfiltered(dataset_path, save_path):
    data_list = load_json(os.path.join(dataset_path, 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'))
    save_file = os.path.join(save_path, "data.jsonl")

    with open(save_file, "w") as fout:
        invalid_cnt = 0
        for idx, example in tqdm(enumerate(data_list)):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt" or message["from"] == "bard":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    messages.append({
                        "role": "system",
                        "content": message["value"]
                    })
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    print(idx)
                    print(example)
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "ShareGPT_Vicuna_unfiltered",
                    "id": f"ShareGPT_Vicuna_unfiltered_{idx}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in ShareGPT_Vicuna_unfiltered data: {invalid_cnt}")


def convert_WizardLM_evol_instruct_V2_143k_data(dataset_path, save_path):
    ds = load_dataset(dataset_path)
    save_file = os.path.join(save_path, "data.jsonl")

    with open(save_file, "w") as fout:
        invalid_cnt = 0
        for idx, example in tqdm(enumerate(ds['train'])):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    valid = False
                    invalid_cnt += 1
                    break
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "WizardLM_evol_instruct_V2_143k",
                    "id": f"WizardLM_evol_instruct_V2_143k_{idx}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in WizardLM_evol_instruct_V2_143k data: {invalid_cnt}")


def convert_SlimOrca_data(dataset_path, save_path):
    ds = load_dataset(dataset_path)
    save_file = os.path.join(save_path, "data.jsonl")

    with open(save_file, "w") as fout:
        invalid_cnt = 0
        for idx, example in tqdm(enumerate(ds['train'])):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    messages.append({
                        "role": "system",
                        "content": message["value"]
                    })
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "SlimOrca",
                    "id": f"SlimOrca_{idx}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in SlimOrca data: {invalid_cnt}")


def convert_MetaMathQA_data(dataset_path, save_path):
    ds = load_dataset(dataset_path)
    save_file = os.path.join(save_path, "data.jsonl")

    with open(save_file, "w") as fout:
        for idx, example in tqdm(enumerate(ds['train'])):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []

            messages.append({
                "role": "user",
                "content": example['query']
            })
            messages.append({
                "role": "assistant",
                "content": example['response']
            })
            fout.write(json.dumps({
                "dataset": "MetaMathQA",
                "id": f"MetaMathQA_{idx}",
                "messages": messages
            }) + "\n")


def convert_evol_codealpaca_v1_data(dataset_path, save_path):
    ds = load_dataset(dataset_path)
    save_file = os.path.join(save_path, "data.jsonl")

    with open(save_file, "w") as fout:
        for idx, example in tqdm(enumerate(ds['train'])):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []

            messages.append({
                "role": "user",
                "content": example['instruction']
            })
            messages.append({
                "role": "assistant",
                "content": example['output']
            })
            fout.write(json.dumps({
                "dataset": "evol_codealpaca_v1",
                "id": f"evol_codealpaca_v1_{idx}",
                "messages": messages
            }) + "\n")


def convert_alpaca_data(dataset_path, save_path):
    ds = load_dataset(dataset_path)
    save_file = os.path.join(save_path, "data.jsonl")

    with open(save_file, "w") as fout:
        for idx, example in tqdm(enumerate(ds['train'])):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []

            messages.append({
                "role": "user",
                "content": example['instruction'] + (f"\nInput: {example['input']}\n" if example['input'] != "" else "")
            })
            messages.append({
                "role": "assistant",
                "content": example['output']
            })
            fout.write(json.dumps({
                "dataset": "alpaca",
                "id": f"alpaca_{idx}",
                "messages": messages
            }) + "\n")


RAW_DATA_SUBDIR = {
    "vicuna_sharegpt": "vicuna_sharegpt",
    "evol_instruct": "evol_instruct",
    "slim_orca": "slim_orca",
    "meta_math_qa": "meta_math_qa",
    "evol_code_alpaca": "evol_code_alpaca",
    "alpaca": "alpaca",
}

DATA_PROCESS_FUNC = {
    "vicuna_sharegpt": convert_convert_ShareGPT_Vicuna_unfiltered,
    "evol_instruct": convert_WizardLM_evol_instruct_V2_143k_data,
    "slim_orca": convert_SlimOrca_data,
    "meta_math_qa": convert_MetaMathQA_data,
    "evol_code_alpaca": convert_evol_codealpaca_v1_data,

    "alpaca": convert_alpaca_data,
}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--raw_data_root", type=str, default=os.path.join(PROJECT_ROOT, "data", "raw"))
    arg_parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "reformatted"))
    args = arg_parser.parse_args()

    processes = []

    for dataset_name, dataset_subdir in RAW_DATA_SUBDIR.items():
        process_function = DATA_PROCESS_FUNC[dataset_name]
        dataset_path = os.path.join(args.raw_data_root, dataset_subdir)
        save_path = os.path.join(args.save_path, dataset_name)
        create_dir(save_path)

        process = multiprocessing.Process(target=process_function, args=(dataset_path, save_path))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All done.")
