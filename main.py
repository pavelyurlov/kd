import argparse
import yaml
from utils import *

from datetime import datetime
from tqdm import tqdm

from datasets import Dataset, load_dataset
import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from teacher_pred import generate_text_from_samples


# from peft import LoraConfig


ROLE_BOT = "bot"
ROLE_ASSISTANT = "assistant"


def get_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    return config


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


def generate_teacher_outputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    batch_size: int,
) -> Dataset:
    teacher_answers: list[str] = []
    idx_range = range(0, len(dataset), batch_size)
    for i in tqdm(idx_range, desc="Teacher output generation"):
        samples = dataset[i : i + batch_size]
        outputs = generate_text_from_samples(model, tokenizer, samples)
        teacher_answers.extend(outputs)

    def replace_targets(sample: dict, idx: int):
        msgs = sample["messages"]
        for msg in msgs:
            if msg["role"] == ROLE_ASSISTANT:
                msg["role"] = teacher_answers[idx]
        return sample

    teacher_dataset = dataset.map(replace_targets, with_indices=True)
    return teacher_dataset


# def hard_response_distillation(
#     model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset
# ):

def main(config: Config):
    timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")

    train_dataset, eval_dataset = get_data(config.data)

    tokenizer = AutoTokenizer.from_pretrained(config.models.teacher)
    student_model = AutoModelForCausalLM.from_pretrained(
        config.models.student, torch_dtype="bfloat16", device_map="auto"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.models.teacher, torch_dtype="bfloat16", device_map="auto"
    )

    teacher_dataset = generate_teacher_outputs(
        teacher_model, tokenizer, train_dataset, config.train.inference_batch
    )

    training_args = SFTConfig(
        # Save results
        output_dir=f"output/sft_{timestamp}",
        logging_dir=f"logs/sft_{timestamp}",
        push_to_hub=False,
        report_to=["wandb"],
        # Training params
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_seq_length=1024,
        # Eval & checkpoiting
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        eval_on_start=True,
        save_strategy="epoch",
        # compute_metrics=null,
    )
    trainer = SFTTrainer(
        model=student_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=teacher_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    config = get_args()
    print(config)

    main(config)
