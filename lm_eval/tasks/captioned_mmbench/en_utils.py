import yaml
import os
from pathlib import Path
import pandas as pd
import json
import openai

from loguru import logger as eval_logger
from lm_eval.tasks.captioned_mmbench.mmbench_evals import MMBench_Evaluator

with open(Path(__file__).parent / "mmbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://azure-openai-api.shenmishajing.workers.dev/v1/")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")

openai._reset_client()
setattr(openai, 'base_url', API_URL)
setattr(openai, 'api_key', API_KEY)
client = openai._load_client()


mmbench_evaluator = MMBench_Evaluator(sys_prompt=config["metadata"]["sys_prompt"], API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)



def mmbench_doc_to_text(doc, model_specific_prompt_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "L2-category": doc["L2-category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "hint": doc["hint"],
        "source": doc["source"],
        "split": doc["split"],
    }

    query_prompt = f"{data['hint']} {data['question']} {data['options']}" if pd.notna(data["hint"]) and data["hint"] != "nan" else f"{data['question']} {data['options']}"

    post = "\nAnswer with the option's letter from the given choices directly."
    pre = f'Here is the detailed description of the image: {doc["captions-sharecaptioner"]} {doc["captions-llava"]} {doc["captions-phi"]}'
    query_prompt = f"{pre}\n{query_prompt}\n{post}"

    return query_prompt


def mmbench_process_results(doc, results):
    model_response = results[0].strip()
    data = {
        "gpt_eval_score": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "hint": doc["hint"],
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmbench_aggregate_dev_results_eval(results):
    print(f"============= MMBench-EN(Dev) Detailed Results =============")
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    return overall_acc * 100