import re
import os
import json

import datetime
import statistics

from lm_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


from loguru import logger as eval_logger


def vqav2_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vqav2_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        for ansDic in doc["answers"]:
            ansDic["answer"] = ansDic["answer"].replace("\n", " ")
            ansDic["answer"] = ansDic["answer"].replace("\t", " ")
            ansDic["answer"] = ansDic["answer"].strip()
        gtAcc = []
        gtAnswers = [ans["answer"] for ans in doc["answers"]]

        if len(set(gtAnswers)) > 1:
            for ansDic in doc["answers"]:
                ansDic["answer"] = eval_ai_processor.process_punctuation(ansDic["answer"])
                ansDic["answer"] = eval_ai_processor.process_digit_article(ansDic["answer"])
            resAns = eval_ai_processor.process_punctuation(resAns)
            resAns = eval_ai_processor.process_digit_article(resAns)

        for gtAnsDatum in doc["answers"]:
            otherGTAns = [item for item in doc["answers"] if item != gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item["answer"] in resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
    }


def vqav2_process_results_test(doc, result):
    res = vqav2_process_results(doc, result)
    return {
        "submission": res["submission"],
    }


def vqav2_process_results_val(doc, result):
    res = vqav2_process_results(doc, result)
    return {
        "exact_match": res["exact_match"],
    }


def vqav2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = f'Here is the detailed description of the image: {doc["captions-sharecaptioner"]} {doc["captions-llava"]} {doc["captions-phi"]}\n'
    post_prompt = "\nAnswer the question using a single word or phrase."
    return f"{pre_prompt}{doc['question']}{post_prompt}"