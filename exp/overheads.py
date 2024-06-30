import ast
import typing
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

from exp.types import Row

def get_dataset(hf_repo: str):
    """
    Get the huggingface dataset identified by it's huggingface repository url
    """
    return load_dataset(hf_repo)


def train_dataset(hf_dataset: Dataset) -> list[Row]:
    """
    Load the training dataset rows
    """
    return hf_dataset["train"]


def val_dataset(hf_dataset: Dataset) -> list[Row]:
    """
    Load the validation dataset rows
    """
    return hf_dataset["validation"]


def get_tokenizer(hf_repo: str):
    """
    Get the AutoTokenizer instance for the huggingface model at `hf_repo`
    """
    return AutoTokenizer.from_pretrained(hf_repo)

def device():
    """
    Get the torch device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_gen_model(hf_repo: str):
    """
    Get the model instance for inference.
    """
    generator = AutoModelForCausalLM.from_pretrained(hf_repo)
    generator.eval()
    generator = generator.to(device())
    return generator