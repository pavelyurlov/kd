from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLProcessor,
)
from tqdm import tqdm
import json
import torch
from PIL import Image

# from utils.inference import generate_text_from_samples
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
from datasets import Dataset
from PIL import Image


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_NAME = "Cloudriver/PhyX"
SYSTEM_MESSAGE = """You are a Vision Language Model specialized in sovling physics problems.
Your task is to analyze the provided image and task description and respond to question with the only letter that corresponds to the correct answer.
Focus on delivering accurate answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


def convert_to_format(image, question_description, question, options, answer):
    options_string = "\n".join(options)

    user_prompt = f"""
    You are a Vision Language Model specialized in sovling physics problems.
    Your task is to analyze the provided image and task description and respond to question with the only letter that corresponds to the correct answer.
    Focus on delivering accurate answers based on the visual information. Avoid additional explanation unless absolutely necessary.
    <DESCRIPTION>\n{question_description}</DESCRIPTION>\n\n<QUESTION>\n{question}\n</QUESTION>\n\n<OPTIONS>:\n{options_string}\n</OPTIONS>\n\nYOUR ANSWER IS ONLY ONE LETTER THAT CORRESPONDS TO THE CORRECT OPTION!!!
    """
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return conversation


def generate_text_from_sample(
    model, processor, sample, max_new_tokens=1024, device="cuda"
):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample,
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


def format_data(sample):

    options_string = "\n".join(sample["options"])

    user_prompt = f"""
    You are a Vision Language Model specialized in sovling physics problems.
    Your task is to analyze the provided image and task description and respond to question with the only letter that corresponds to the correct answer.
    Focus on delivering accurate answers based on the visual information. Avoid additional explanation unless absolutely necessary.
    <DESCRIPTION>\n{sample["question_description"]}</DESCRIPTION>\n\n<QUESTION>\n{sample["question"]}\n</QUESTION>\n\n<OPTIONS>:\n{options_string}\n</OPTIONS>\n\nYOUR ANSWER IS ONLY ONE LETTER THAT CORRESPONDS TO THE CORRECT OPTION!!!
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]


def main():

    save_name = "dataset_Cloudriver_PhyX.json"
    # load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # processor = AutoProcessor.from_pretrained(MODEL_NAME)

    processor = Qwen2_5_VLProcessor.from_pretrained(
        MODEL_NAME, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28
    )

    # load dataset
    dataset = load_dataset(DATASET_NAME, split="test_mini")
    # print('dataset ',dataset)

    chat_dataset = []
    for sample in tqdm(dataset, total=len(dataset)):

        print(type(sample["image"]))

        chat_sample = format_data(sample)

        output = generate_text_from_sample(model, processor, chat_sample)

        conversation = convert_to_format(
            sample["image"],
            sample["question_description"],
            sample["question"],
            sample["options"],
            output,
        )
        chat_dataset.append(conversation)

        # with open(save_name, mode="w", encoding="utf-8") as f:
        #     json.dump(summary, f, ensure_ascii=False, indent=4)

    np.save("cloudriver_phys.npy", chat_dataset, allow_pickle=True)

    # dataset_new.save_to_disk("cloudriver_phys") #cloudriver_phys_v1 - min_pixels(128*28*28, 256*28*28)
    # dataset_new.save_to_disk("cloudriver_phys_v1") #cloudriver_phys_v1 - min_pixels(128*28*28, 256*28*28)

    # np.save("/home/jovyan/fundament/model-f/sft/train_physx.npy", train_dataset)
    # np.save("/home/jovyan/fundament/model-f/sft/val_physx.npy", val_dataset)
    # with open(save_name, mode="w", encoding="utf-8") as f:
    #     json.dump(summary, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
