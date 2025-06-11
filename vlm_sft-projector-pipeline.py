import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, load_from_disk
from trl import SFTTrainer, SFTConfig
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from sklearn.model_selection import train_test_split

# Загрузка датасета
data = np.load("prepare_data/cloudriver_phys.npy", allow_pickle=True)
train_dataset, val_dataset = train_test_split(data, test_size=0.1, random_state=42)

print(f"Train shape: {train_dataset.shape}, Test shape: {val_dataset.shape}")

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2_5_VLProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28
)


def generate_text_from_sample(
    model, processor, sample, max_new_tokens=1024, device="cuda"
):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[0:2],
        tokenize=False,
        add_generation_prompt=True,  # Use the sample with the system message
        # sample[1:2], tokenize=False, add_generation_prompt=True  # Use the sample withщге the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]  # Return the first decoded output text


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [
        process_vision_info(example)[0] for example in examples
    ]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = (
        -100
    )  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(
        processor, Qwen2_5_VLProcessor
    ):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [
            151652,
            151653,
            151655,
        ]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


# Configure training arguments
training_args = SFTConfig(
    output_dir="/home/jovyan/fundament/model-f/distillation/exp1_3b_phyx_shuf_merger_lm_head",  # Directory to save the model
    num_train_epochs=5,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    # gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    learning_rate=1e-5,  # Learning rate for training
    # lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    # tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="none",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    # gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=2048,  # Maximum sequence length for input
    logging_steps=100,  # Steps interval for logging
    eval_steps=100,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=100,  # Steps interval for saving
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset

z = 0
v = 0
for name, param in model.named_parameters():
    if not any(
        key in name
        for key in [
            "merger",
            "model.language_model.layers.35",
            # "vision_proj",
            # "image_proj",
            # "lm_head",
            # "norm"
        ]
    ):
        param.requires_grad = False
        v += 1
    else:
        print(name)
        param.requires_grad = True
        z += 1
z, v + z

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()
