from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
import torch

# ===== CONFIGURATION =====
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Larger teacher model
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller student model
DATASET_NAME = "trl-lib/capybara-preferences"  # Instruction dataset :cite[5]
OUTPUT_DIR = "./distil_qwen_output"
BATCH_SIZE = 4  # Reduced for GPU memory efficiency
MAX_SEQ_LENGTH = 1024  # Matches Qwen2.5 context limits :cite[3]


# ===== STAGE 1: GENERATE TEACHER OUTPUTS =====
def generate_teacher_responses():
    # Load teacher components
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Prepare dataset
    dataset = load_dataset(DATASET_NAME, split="train[:500]")  # Subset for demo

    def format_instruction(ex):
        return (
            f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        )

    # Generate teacher responses
    def generate(examples):
        inputs = teacher_tokenizer(
            [format_instruction(ex) for ex in examples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(teacher_model.device)

        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
            )
        decoded = teacher_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return {
            "teacher_response": [
                d.split("<|im_start|>assistant\n")[-1] for d in decoded
            ]
        }

    # Add teacher responses to dataset
    dataset = dataset.map(
        generate,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset.column_names,
    )
    return dataset, teacher_tokenizer


# ===== STAGE 2: DISTILL STUDENT MODEL =====
def distill_student(dataset, tokenizer):
    # Initialize student
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # TRL distillation setup :cite[7]
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
    )

    # Distillation trainer
    distiller = SFTTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="teacher_response",  # Distill from teacher outputs
    )

    # Start distillation
    distiller.train()
    student_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return student_model


# ===== EXECUTION PIPELINE =====
if __name__ == "__main__":
    print("=== GENERATING TEACHER RESPONSES ===")
    teacher_dataset, teacher_tok = generate_teacher_responses()

    print("\n=== DISTILLING STUDENT MODEL ===")
    distilled_student = distill_student(teacher_dataset, teacher_tok)

    print("\n=== DISTILLATION COMPLETE ===")
