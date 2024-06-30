"""
Save the results of the fine-tuned model on the training and vaidation datasets to excel file.
"""

import pandas as pd
import json

def save_val_json_to_excel(loc: str, name: str):
    """
    Save the training or validation results jsonl file to excel
    Args:
        loc (str): The location of the jsonl file (results.jsonl)
        name (str): The name of the output excel file (training.xlsx or validation.xls)
    """

    df = pd.read_json(loc, lines=True, orient="records")
    df.to_excel(name)


def save(generated_text: str, prompt: str, actual: bool, predicted: bool):
    with open("results.json", "a+", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "generated": generated_text,
                    "prompt": prompt,
                    "actual": actual,
                    "predicted": predicted,
                }
            )
            + "\n"
        )