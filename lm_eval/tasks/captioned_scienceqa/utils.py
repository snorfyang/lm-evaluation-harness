def sqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if context:
        context = f"Context: {context}\n"

    pre_prompt = f'Here is the detailed description of the image: {doc["captions-sharecaptioner"]} {doc["captions-llava"]} {doc["captions-phi"]}\n'
    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    return f"{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


def sqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc).strip().lower()
    pred = results[0].strip()
    if pred.lower() == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0].lower() == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}