import argparse

# from clearml import Task
from datetime import datetime

from datasets import Dataset, load_dataset
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


# timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")
# task = Task.init(
#     project_name="private/YurlovP/distil",
#     task_name=f"try_{timestamp}",
#     reuse_last_task_id=False,
# )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--data", type=str, default="dummy", choices=["dummy", "real"])
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
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(
        lambda x: x["source"] == "gpt4"
        and not x["is_bad_by_regex"]
        and x["sonnet_complexity"] != "easy"
        and x["language"] == "Russian"
    )

    def rename_role(sample):
        msgs = sample["messages"]
        for msg in msgs:
            if msg["role"] == "bot":
                msg["role"] = "assistant"
        return sample

    dataset = dataset.map(rename_role)

    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
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


def main(model_name: str = "qwen", data_name: str = "dummy", n_epochs: int = 1):
    student_name, teacher_name = model_names(model_name)
    train_dataset, eval_dataset = get_data(data_name)

    tokenizer = AutoTokenizer.from_pretrained(student_name)
    model = AutoModelForCausalLM.from_pretrained(student_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)

    timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")
    # task = Task.init(
    #     project_name="private/YurlovP/distil",
    #     task_name=f"try_{timestamp}",
    #     reuse_last_task_id=False,
    # )
    training_args = GKDConfig(
        output_dir=f"output/try_{timestamp}",
        logging_dir=f"logs/try_{timestamp}",
        num_train_epochs=n_epochs,
        eval_strategy="epoch",
        eval_on_start=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to=["wandb"],
    )
    print(training_args.report_to)
    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # eval_res = trainer.evaluate()
    # eval_res["epoch"] = 0.0
    # print(eval_res)

    trainer.train()

    # eval_res = trainer.evaluate()
    # print(eval_res)


if __name__ == "__main__":
    args = get_args()
    print(args)

    main(**args)
