from datasets import Dataset, load_dataset

from .config import DataConfig
from .constants import *


def dummy_data():
    NUM_DUMMY_SAMPLES = 100
    train_dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hi, how are you?"},
                    {"role": "assistant", "content": "I'm great thanks"},
                ]
            ]
            * NUM_DUMMY_SAMPLES
        }
    )
    eval_dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "What colour is the sky?"},
                    {"role": "assistant", "content": "The sky is blue"},
                ]
            ]
            * NUM_DUMMY_SAMPLES
        }
    )
    return train_dataset, eval_dataset


def hf_data(config: DataConfig):
    dataset: Dataset = load_dataset(config.name, split="train")
    dataset = dataset.filter(
        lambda x: x["source"] == "gpt4"
        and not x["is_bad_by_regex"]
        and x["sonnet_complexity"] != "easy"
        and x["language"] == "Russian"
    )
    dataset = dataset.select_columns("messages")

    def rename_role(sample: dict):
        msgs = sample["messages"]
        for msg in msgs:
            if msg["role"] == ROLE_BOT:
                msg["role"] = ROLE_ASSISTANT
        return sample

    dataset = dataset.map(rename_role)

    dataset = dataset.filter(lambda _, idx: idx % config.filter == 0, with_indices=True)

    train_dataset = dataset.filter(lambda _, idx: idx % 5 > 0, with_indices=True)
    eval_dataset = dataset.filter(lambda _, idx: idx % 5 == 0, with_indices=True)

    print(len(train_dataset), len(eval_dataset))

    return train_dataset, eval_dataset


def get_data(config: DataConfig):
    if config.dummy:
        return dummy_data()
    else:
        return hf_data(config)
