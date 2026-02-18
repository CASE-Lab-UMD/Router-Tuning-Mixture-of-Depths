"""Mix the instruction tuning data by given portions"""

import argparse
import os
import random

from utils.io import find_files, load_jsonl, save_jsonl, create_dir
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET_NAMES = (
    "vicuna_sharegpt",
    "evol_instruct",
    "slim_orca",
    "meta_math_qa",
    "evol_code_alpaca",
)

MIX_PORTION = {
    "vicuna_sharegpt": 2.9,
    "evol_instruct": 1.0,
    "slim_orca": 1.0,
    "meta_math_qa": 1.0,
    "evol_code_alpaca": 1.0,
}


def replicate_elements(elements, portion):
    """Replicate a list by a float multiplier (e.g., 2.9x)."""
    if portion <= 0:
        return []

    integer_part = int(portion)
    fractional_part = portion - integer_part
    replicated = list(elements) * integer_part

    if fractional_part > 0 and elements:
        sampled_size = int(len(elements) * fractional_part)
        if sampled_size > 0:
            replicated.extend(random.sample(elements, sampled_size))
    return replicated

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--reformatted_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "reformatted"))
    arg_parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "mixed"))
    arg_parser.add_argument("--seed", type=int, default=233)
    args = arg_parser.parse_args()
    random.seed(args.seed)

    final_data_list = []

    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join(args.reformatted_dir, dataset_name)
        dataset_portion = MIX_PORTION[dataset_name]

        for data_file in find_files(dataset_path, "*.jsonl"):
            data_list = load_jsonl(data_file)
            print(f"{dataset_name} {data_file}: original length {len(data_list)}, portion {dataset_portion}")
            data_list = replicate_elements(data_list, dataset_portion)
            print(f"{dataset_name} {data_file}: replicated length {len(data_list)}")
            final_data_list.extend(data_list)

    print("Shuffling final data list...")
    random.shuffle(final_data_list)
    print(f"final mixed data list length: {len(final_data_list)}")

    create_dir(args.save_path)
    save_file = os.path.join(args.save_path, "data.jsonl")
    save_jsonl(final_data_list, save_file)
    print("Done.")
