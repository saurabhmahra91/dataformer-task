import ast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp.decor import retry
from exp.conversion import save
from exp.types import Row
from exp.overheads import (
    device,
    get_dataset,
    get_gen_model,
    get_tokenizer,
    val_dataset,
)


def finetune_format(train_ex: Row):
    """
    Add special tokens to the text for separating system, human, and AI messages.
    """
    return (
        "<|im_start|>system\n"
        'For a given question assess whether translating the potential answer to another language might yield an inaccurate response. Avoid translation in tasks related to coding problems, alliteration, idioms, paraphrasing text, word count, spelling correction, and other linguistic constructs or contextual nuances that may affect the accuracy of the answer. When translation is deemed unsuitable, output {"translate": False}. Otherwise, output {"translate": True}.\n'
        "<|im_end|>\n"
        "<|im_start|>human\n"
        f"{train_ex["conversations"][1]["value"]}"
        "<|im_end|>\n"
        "<|im_start|>gpt"
    )


def get_result(response: str) -> bool:
    """
    Parse the LLM response JSON.
    Raises:
        json.decoder.JSONDecodeError: If it could not find a JSON in the llm response
    """

    start_idx = response.rfind("<|im_start|>")
    end_idx = response.find("<|im_end|>", start_idx)

    trimmed = response[start_idx + len("<|im_start|>gpt\n") : end_idx]

    json_resp = ast.literal_eval(trimmed)
    return json_resp["translate"]


def infer(generator: AutoModelForCausalLM, tokenizer: AutoTokenizer, row: Row) -> str:
    """
    Infer the text generation model against a given text.
    """

    tokenized = tokenizer(
        finetune_format(row),
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = tokenized["input_ids"].to(device())
    attention_mask = tokenized["attention_mask"].to(device())

    with torch.no_grad():
        output = generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.5,
            # num_return_sequences=1,
            # num_beams=4,
        )
        decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_text


# @retry(num_retries=10, avoids=(SyntaxError,))
def verdict(row: Row, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> bool:
    """
    Check if the answer returned by the LLM is correct.
    """

    true_answer = ast.literal_eval(row["conversations"][2]["value"])["translate"]
    generated_text = infer(model, tokenizer, row)
    predicted_answer = get_result(generated_text)
    save(generated_text, finetune_format(row), true_answer, predicted_answer)
    return true_answer == predicted_answer


if __name__ == "__main__":

    DATA_REPO = "satpalsr/chatml-translation-filter"
    MODEL_REPO = "satpalsr/llama2-translation-filter-full"

    dataset = get_dataset(hf_repo=DATA_REPO)
    tokenizer = get_tokenizer(hf_repo=MODEL_REPO)
    model = get_gen_model(hf_repo=MODEL_REPO)
    train_verdicts = []
    erroneous = 0
    for row in val_dataset(dataset):
        try:
            v = verdict(row, model, tokenizer)
            train_verdicts.append(v)
        except Exception as e: # Evil catch-all exception
            erroneous += 1
            print(f"Error processing row")

    print("Accuracy = ", sum(train_verdicts) / len(train_verdicts))
    print("Erroneous = ", erroneous)
