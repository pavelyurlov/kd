from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np

# Configuration
TEACHER_MODEL = "bert-large-uncased"
STUDENT_MODEL = "bert-base-uncased"
DATASET_NAME = "imdb"
OUTPUT_DIR = "./distillation_results"
BATCH_SIZE = 16


# Stage 1: Generate teacher predictions
def generate_teacher_predictions():
    print("Loading teacher model...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL
    ).to("cuda")

    print("Loading and preparing dataset...")
    dataset = load_dataset(DATASET_NAME)
    tokenized_dataset = dataset.map(
        lambda x: teacher_tokenizer(x["text"], padding=True, truncation=True),
        batched=True,
    )

    print("Generating teacher predictions...")

    def get_predictions(batch):
        inputs = {
            k: v.to("cuda")
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        with torch.no_grad():
            logits = teacher_model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        return {"teacher_labels": predictions.cpu().numpy()}

    dataset_with_teacher = tokenized_dataset.map(
        get_predictions, batched=True, batch_size=BATCH_SIZE
    )

    return dataset_with_teacher, teacher_tokenizer


# Stage 2: Train student on teacher labels
def train_student(dataset, tokenizer):
    print("Loading student model...")
    student_model = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        save_strategy="no",
    )

    print("Creating Trainer...")
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=lambda p: {
            "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
        },
    )

    print("Training student model...")
    trainer.train()

    print("Evaluation results:")
    eval_results = trainer.evaluate()
    print(eval_results)

    return student_model


# Main execution
if __name__ == "__main__":
    # Stage 1: Generate teacher labels
    dataset_with_teacher, tokenizer = generate_teacher_predictions()

    # Stage 2: Train student using teacher labels
    dataset_with_teacher.set_format(
        type="torch", columns=["input_ids", "attention_mask", "teacher_labels"]
    )
    dataset_with_teacher = dataset_with_teacher.rename_column(
        "teacher_labels", "labels"
    )

    trained_student = train_student(dataset_with_teacher, tokenizer)
    trained_student.save_pretrained(OUTPUT_DIR)
