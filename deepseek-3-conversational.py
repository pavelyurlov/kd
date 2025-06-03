from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
import torch

# ===== CONFIGURATION =====
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"  # Example dataset with message format
OUTPUT_DIR = "./distil_qwen_output"
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 2048
NUM_EXAMPLES = 1000  # Reduce for quick testing


# ==== For larger models, consider using LoRA adapters: ====

# from peft import LoraConfig
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     task_type="CAUSAL_LM"
# )
# # Add to SFTTrainer: peft_config=peft_config


# ===== STAGE 1: GENERATE TEACHER RESPONSES =====
def generate_teacher_responses():
    print("Loading teacher model...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=f"train_sft[:{NUM_EXAMPLES}]")

    # Format conversation for Qwen2.5
    def format_conversation(messages):
        return "\n".join(
            [
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
                for msg in messages
            ]
        )

    print("Generating teacher responses...")

    def generate_teacher_output(batch):
        new_samples = []
        for sample in batch["messages"]:
            # Format context (all messages except last assistant response)
            context = sample[:-1]
            context_str = format_conversation(context)

            # Tokenize context
            inputs = teacher_tokenizer(
                context_str,
                return_tensors="pt",
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
            ).to(teacher_model.device)

            # Generate response
            with torch.no_grad():
                outputs = teacher_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                )

            # Decode and clean response
            response = (
                teacher_tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
                )
                .split("<|im_end|>")[0]
                .strip()
            )

            # Create new sample with teacher response
            new_messages = context + [{"role": "assistant", "content": response}]
            new_samples.append({"messages": new_messages})

        return {"messages": new_samples}

    # Process dataset in batches
    teacher_dataset = dataset.map(
        generate_teacher_output,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset.column_names,
    )

    return teacher_dataset, teacher_tokenizer


# ===== STAGE 2: DISTILL STUDENT MODEL =====
def distill_student(dataset, tokenizer):
    print("Loading student model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, quantization_config=bnb_config, device_map="auto"
    )

    # Format dataset for training
    def format_for_training(examples):
        return {
            "text": "\n".join(
                [
                    f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
                    for msg in examples["messages"]
                ]
            )
        }

    train_dataset = dataset.map(
        format_for_training, remove_columns=dataset.column_names
    )

    # TRL training setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # Trainer configuration
    trainer = SFTTrainer(
        model=student_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,
    )

    # Start distillation
    print("Training student model...")
    trainer.train()

    print("Saving final model...")
    student_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return student_model


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("=== STAGE 1: GENERATING TEACHER RESPONSES ===")
    teacher_dataset, teacher_tok = generate_teacher_responses()

    print("\n=== STAGE 2: DISTILLING STUDENT MODEL ===")
    distilled_student = distill_student(teacher_dataset, teacher_tok)

    print("\n=== KNOWLEDGE DISTILLATION COMPLETE ===")
