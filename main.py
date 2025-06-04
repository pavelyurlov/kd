import argparse
import yaml
from datetime import datetime

from utils.config import Config
from utils.data import get_data
from utils.inference import get_model_and_tokenizer, generate_teacher_outputs
from utils.training import train


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
        teacher_model, teacher_tokenizer, train_dataset, config.train.inference_batch
    )

    train(
        student_model,
        student_tokenizer,
        teacher_dataset,
        eval_dataset,
        config,
        timestamp,
    )


if __name__ == "__main__":
    config = get_args()
    print(config)

    main(config)
