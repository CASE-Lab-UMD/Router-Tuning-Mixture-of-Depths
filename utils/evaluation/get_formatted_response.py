import os
import re

import pandas as pd

try:
    from llmgate.entrypoints.evaluation.mmlu import format_example
except ImportError:
    format_example = None

from utils.io import load_json, load_jsonl


def get_alpaca_eval_responses(result_path):
    model_evaluation_file = os.path.join(result_path, "annotation_alpaca_eval_gpt4.json")
    model_evaluation_results = load_json(model_evaluation_file)
    inputs = [result["instruction"] for result in model_evaluation_results]
    responses = [result["output_2"] for result in model_evaluation_results]
    return inputs, responses, model_evaluation_results


def get_flask_responses(result_path, return_id=False):
    model_evaluation_file = os.path.join(result_path, "chatgpt_review.jsonl")
    model_evaluation_results = load_jsonl(model_evaluation_file)
    if return_id:
        inputs = [result["question_id"] for result in model_evaluation_results]
    else:
        inputs = [result["text"] for result in model_evaluation_results]
    responses = [result["answer"] for result in model_evaluation_results]
    return inputs, responses, model_evaluation_results


def get_gsm8k_responses(result_path):
    model_evaluation_file = os.path.join(result_path, "outputs.json")
    model_evaluation_results = load_json(model_evaluation_file)
    inputs = [result["question"] for result in model_evaluation_results]
    responses = [result["output"] for result in model_evaluation_results]
    return inputs, responses, model_evaluation_results


def get_human_eval_responses(result_path, return_id=False):
    model_evaluation_file = os.path.join(result_path, "eval_results.json")
    model_evaluation_results = load_json(model_evaluation_file)
    if return_id:
        inputs = [result["task_id"] for result in model_evaluation_results]
    else:
        inputs = [result["prompt"] for result in model_evaluation_results]
    responses = [result["completion"] for result in model_evaluation_results]
    return inputs, responses, model_evaluation_results


def get_mmlu_responses(result_path):
    if format_example is None:
        raise ImportError(
            "MMLU formatting requires llmgate. Install llmgate or provide a local format_example implementation."
        )

    model_evaluation_results = []

    for subject_file_name in os.listdir(result_path):
        if subject_file_name.endswith(".csv"):
            subject = subject_file_name.replace("output_", "").replace(".csv", "")
            subject_file = os.path.join(result_path, subject_file_name)

            subject_results_df = pd.read_csv(subject_file)
            subject_results = subject_results_df.to_dict(orient='records')

            # get prompted inputs
            subject_results_df = subject_results_df.drop("output", axis=1)  # drop the output for "format_example" function
            subject_prompted_inputs = [format_example(subject_results_df, i, include_answer=False) for i in range(subject_results_df.shape[0])]

            # aggregate results
            for i, result in enumerate(subject_results):
                result["prompted_input"] = subject_prompted_inputs[i]
                result["subject"] = subject
                result["output"] = str(result["output"])
                model_evaluation_results.append(result)

    prompted_inputs = [result["prompted_input"] for result in model_evaluation_results]
    responses = [result["output"] for result in model_evaluation_results]
    return prompted_inputs, responses, model_evaluation_results


def get_mt_bench_responses(result_path, return_id=False, multi_turn=False):
    """
        The results include both single-turn & multi-turn.
        Set "multi_turn=False" to use the single-turn results.
        Set "multi_turn=True" to use the multi-turn results.
    """
    model_evaluation_file = os.path.join(result_path, "answer_judged.jsonl")
    model_evaluation_results = load_jsonl(model_evaluation_file)

    if multi_turn:
        model_evaluation_results = [
            result for result in model_evaluation_results
            if result["turn"] == 2
        ]

        # get pure inputs
        pattern = re.compile(r"<\|The Start of Assistant A's Conversation with User\|>\n\n### User:\n(.*?)\n\n### Assistant", re.IGNORECASE | re.DOTALL)
        for i in range(len(model_evaluation_results)):
            prompted_result = model_evaluation_results[i]["user_prompt"]
            match = pattern.search(prompted_result)
            if match is None:
                raise ValueError("Failed to parse MT-Bench multi-turn input from `user_prompt`.")
            pure_input = match.group(1)
            model_evaluation_results[i]["pure_input"] = pure_input  # add to result

        # get pure outputs
        pattern = re.compile(r"\n\n### Assistant A:\n(.*?)\n\n### User:\n", re.IGNORECASE | re.DOTALL)
        for i in range(len(model_evaluation_results)):
            prompted_result = model_evaluation_results[i]["user_prompt"]
            match = pattern.search(prompted_result)
            if match is None:
                raise ValueError("Failed to parse MT-Bench multi-turn output from `user_prompt`.")
            pure_output = match.group(1)
            model_evaluation_results[i]["pure_output"] = pure_output  # add to result

    else:
        model_evaluation_results = [
            result for result in model_evaluation_results
            if result["turn"] == 1
        ]

        # get pure inputs
        pattern = re.compile(r"\n\n\[Question]\n(.*?)\n\n\[The Start of Assistant's Answer]\n", re.IGNORECASE | re.DOTALL)
        for i in range(len(model_evaluation_results)):
            prompted_input = model_evaluation_results[i]["user_prompt"]
            match = pattern.search(prompted_input)
            if match is None:
                raise ValueError("Failed to parse MT-Bench single-turn input from `user_prompt`.")
            pure_input = match.group(1)
            model_evaluation_results[i]["pure_input"] = pure_input  # add to result

        # get pure outputs
        pattern = re.compile(r"\n\n\[The Start of Assistant's Answer]\n(.*?)\n\[The End of Assistant's Answer]", re.IGNORECASE | re.DOTALL)
        for i in range(len(model_evaluation_results)):
            prompted_result = model_evaluation_results[i]["user_prompt"]
            match = pattern.search(prompted_result)
            if match is None:
                raise ValueError("Failed to parse MT-Bench single-turn output from `user_prompt`.")
            pure_output = match.group(1)
            model_evaluation_results[i]["pure_output"] = pure_output  # add to result

    if return_id:
        inputs = [result["question_id"] for result in model_evaluation_results]
    else:
        inputs = [result["pure_input"] for result in model_evaluation_results]
    responses = [result["pure_output"] for result in model_evaluation_results]
    return inputs, responses, model_evaluation_results


if __name__ == "__main__":
    model_name = "Llama_2_13b_chat"

    # result_path = f"/mnt/petrelfs/share_data/quxiaoye/moe-gate/evaluation_results/alpaca_eval/{model_name}/max2048-penalty1.0"
    # inputs, responses, model_evaluation_results = get_alpaca_eval_responses(result_path)
    # print("alpaca_eval\n", model_evaluation_results[0], flush=True)
    #
    # result_path = f"/mnt/petrelfs/share_data/quxiaoye/moe-gate/evaluation_results/flask/{model_name}/temp0.7-max1024"
    # inputs, responses, model_evaluation_results = get_flask_responses(result_path)
    # print("flask\n", model_evaluation_results[0], flush=True)
    #
    # result_path = f"/mnt/petrelfs/share_data/quxiaoye/moe-gate/evaluation_results/gsm8k/{model_name}/temp0.0-max512-penalty1.0"
    # inputs, responses, model_evaluation_results = get_gsm8k_responses(result_path)
    # print("gsm8k\n", model_evaluation_results[0], flush=True)
    #
    # result_path = f"/mnt/petrelfs/share_data/quxiaoye/moe-gate/evaluation_results/human_eval/{model_name}/temp0.2-top0.95-max512"
    # inputs, responses, model_evaluation_results = get_human_eval_responses(result_path)
    # print("human_eval\n", model_evaluation_results[0], flush=True)

    result_path = f"/mnt/petrelfs/share_data/quxiaoye/moe-gate/evaluation_results/mmlu/{model_name}"
    inputs, responses, model_evaluation_results = get_mmlu_responses(result_path)
    print("mmlu\n", model_evaluation_results[0], flush=True)

    # result_path = f"/mnt/petrelfs/share_data/quxiaoye/moe-gate/evaluation_results/mt_bench/{model_name}/max512"
    # inputs, responses, model_evaluation_results = get_mt_bench_responses(result_path)
    # print("mt_bench\n", model_evaluation_results[0], flush=True)
