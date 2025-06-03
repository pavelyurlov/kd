import argparse

from datetime import datetime
from tqdm import tqdm

from datasets import Dataset, load_dataset
import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# from peft import LoraConfig


ROLE_BOT = "bot"
ROLE_ASSISTANT = "assistant"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--data", type=str, default="real", choices=["dummy", "real"])
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    model_name: str = args.model
    data_name: str = args.data
    n_epochs: int = args.epochs

    return dict(model_name=model_name, data_name=data_name, n_epochs=n_epochs)


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


def saiga_data():
    dataset_name = "IlyaGusev/saiga_scored"
    dataset: Dataset = load_dataset(dataset_name, split="train")
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

    train_dataset = dataset.filter(lambda _, idx: idx % 100 == 0, with_indices=True)
    eval_dataset = dataset.filter(lambda _, idx: idx % 300 == 1, with_indices=True)

    print(len(train_dataset), len(eval_dataset))

    return train_dataset, eval_dataset


def get_data(name: str):
    if name == "dummy":
        return dummy_data()
    elif name == "real":
        return saiga_data()
    else:
        raise NotImplementedError(name)


def model_names(name: str):
    if name == "llama":
        student_name = "unsloth/Llama-3.2-1B-Instruct"
        teacher_name = "unsloth/Llama-3.2-3B-Instruct"

    elif name == "qwen":
        student_name = "Qwen/Qwen2-0.5B-Instruct"
        teacher_name = "Qwen/Qwen2-1.5B-Instruct"

    else:
        raise NotImplementedError(name)

    return student_name, teacher_name


def generate_text_from_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample: dict,
    max_new_tokens: int = 256,
):
    msgs = sample["messages"]

    text_input = tokenizer.apply_chat_template(
        msgs[:1], tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        text=[text_input],
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="left",
    ).to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


@torch.inference_mode
def generate_teacher_outputs(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset
) -> Dataset:
    teacher_answers: list[str] = []
    for sample in tqdm(dataset, desc="Teacher output generation"):
        output = generate_text_from_sample(model, tokenizer, sample)
        teacher_answers.append(output)

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
#     pass


def main(model_name: str = "qwen", data_name: str = "dummy", n_epochs: int = 1):
    timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")

    student_name, teacher_name = model_names(model_name)
    train_dataset, eval_dataset = get_data(data_name)

    tokenizer = AutoTokenizer.from_pretrained(student_name)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype="bfloat16", device_map="auto"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype="bfloat16", device_map="auto"
    )

    teacher_dataset = generate_teacher_outputs(teacher_model, tokenizer, train_dataset)

    training_args = SFTConfig(
        # Save results
        output_dir=f"output/sft_{timestamp}",
        logging_dir=f"logs/sft_{timestamp}",
        push_to_hub=False,
        report_to=["wandb"],
        # Training params
        num_train_epochs=n_epochs,
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
    args = get_args()
    print(args)

    main(**args)
