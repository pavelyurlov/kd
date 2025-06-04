from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .validation import calculate_metrics

# from peft import LoraConfig


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: Config,
    timestamp: str,
    teacher_metrics: dict,
):

    training_args = SFTConfig(
        # Save results
        output_dir=f"output/sft_{timestamp}",
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
        compute_metrics=lambda eval_pred: calculate_metrics(
            tokenizer, eval_pred, config.data.language
        ),
    )

    trainer.log(teacher_metrics)

    trainer.train()
