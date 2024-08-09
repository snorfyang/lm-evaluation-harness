from lm_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def gqa_doc_to_text(doc):
    question = doc["question"]
    pre_prompt = f'Here is the detailed description of the image: {doc["captions-sharecaptioner"]} {doc["captions-llava"]} {doc["captions-phi"]}\n'
    post_prompt = "\nAnswer the question using a single word or phrase."
    return f"{pre_prompt}{question}{post_prompt}"

def gqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    gtAns = eval_ai_processor(doc["answer"])
    if gtAns in resAns:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}