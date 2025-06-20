import argparse
import yaml
from datetime import datetime

from utils.config import Config
from utils.data import get_data
from utils.inference import get_model_and_tokenizer, generate_teacher_outputs
from utils.training import train
from utils.validation import metrics_between_sets


def get_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    return config


def main(config: Config):
    timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")

    train_dataset, eval_dataset = get_data(config.data)

    teacher_model, teacher_tokenizer = get_model_and_tokenizer(config.models.teacher)
    student_model, student_tokenizer = get_model_and_tokenizer(config.models.student)

    teacher_dataset = generate_teacher_outputs(
        teacher_model,
        teacher_tokenizer,
        train_dataset,
        config.train.inference_batch,
        "teacher on train",
    )

    eval_teacher = generate_teacher_outputs(
        teacher_model,
        teacher_tokenizer,
        eval_dataset,
        config.train.inference_batch,
        "teacher on eval",
    )
    teacher_metrics = metrics_between_sets(
        eval_teacher, eval_dataset, config.data.language
    )
    teacher_metrics = {f"eval_teacher_{k}": v for k, v in teacher_metrics.items()}

    train(
        student_model,
        student_tokenizer,
        teacher_dataset,
        eval_dataset,
        config,
        timestamp,
        teacher_metrics,
    )


if __name__ == "__main__":
    config = get_args()
    print(config)

    main(config)
