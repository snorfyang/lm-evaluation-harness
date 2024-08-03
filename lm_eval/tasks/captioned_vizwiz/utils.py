import re
import os
import json
import yaml
import pathlib

import datetime
import statistics

from lm_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

from loguru import logger as eval_logger


def vizwiz_vqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    return {
        "exact_match": accuracy,
    }


def vizwiz_vqa_doc_to_text(doc,):
    pre_prompt = f"Here is the detailed description of the image: {doc['captions-sharecaptioner']} {doc['captions-llava']} {doc['captions-phi']}\n"
    post_prompt = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
    text = f"{pre_prompt} {doc['question'].capitalize()} {post_prompt}"
    return text