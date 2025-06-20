from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .validation import calculate_metrics, preprocess_logits
from .inference import generate_teacher_outputs
from .validation import metrics_between_sets

# from peft import LoraConfig


def validate_student_manually(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    config: Config,
    name: str,
):
    eval_student = generate_teacher_outputs(
        model, tokenizer, eval_dataset, config.train.inference_batch, "student on eval"
    )
    student_metrics = metrics_between_sets(
        eval_student, eval_dataset, config.data.language
    )
    student_metrics = {f"eval_{name}_{k}": v for k, v in student_metrics.items()}
    return student_metrics


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: Config,
    timestamp: str,
    teacher_metrics: dict,
):

    teacher_name = config.models.teacher.split("/")[1].split("-")[0]
    student_name = config.models.student.split("/")[1].split("-")[0]
    exp_name = f"{teacher_name}_to_{student_name}"

    training_args = SFTConfig(
        # Save results
        output_dir=f"output/sft_{exp_name}_{timestamp}",
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
        eval_accumulation_steps=8,
        eval_strategy="epoch",
        eval_on_start=True,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=lambda eval_pred: calculate_metrics(
            tokenizer, eval_pred, config.data.language
        ),
    )

    trainer.log(teacher_metrics)

    student_metrics_before = validate_student_manually(
        model, tokenizer, eval_dataset, config, "student"
    )
    trainer.log(student_metrics_before)

    trainer.train()

    student_metrics_after = validate_student_manually(
        model, tokenizer, eval_dataset, config, "student"
    )
    trainer.log(student_metrics_after)
