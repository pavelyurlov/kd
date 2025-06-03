from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import pandas as pd
import json


model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto",
    # low_cpu_mem_usage=True,
)


dataset_name = "IlyaGusev/saiga_scored"
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.filter(
    lambda x: x["source"] == "gpt4"
    and not x["is_bad_by_regex"]
    and x["sonnet_complexity"] != "easy"
    and x["language"] == "Russian"
)


def rename_role(sample: dict, add_system_message: bool = False):
    msgs = sample["messages"]
    for msg in msgs:
        if msg["role"] == "bot":
            msg["role"] = "assistant"
    return sample


dataset = dataset.map(rename_role)

idxs = [100 * i for i in range(len(dataset) // 100 + 1)]
miniset = dataset.select(idxs)

print(len(dataset), len(miniset))


def generate_text_from_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample: dict,
    max_new_tokens: int = 256,
):
    msgs = sample["messages"]

    text_input = tokenizer.apply_chat_template(
        msgs[:1], tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(
        text=[text_input],
        return_tensors="pt",
        # padding=True,
        # truncation=True,
        # padding_side="left",
    ).to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = tokenizer.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


print(model.device)
answers: list[str] = []
# summary = []
for sample in tqdm(miniset):
    output = generate_text_from_sample(model, tokenizer, sample)
    answers.append(output)
    # summary.append(
    #     {
    #         "question": sample["messages"][0]["content"],
    #         "target": sample["messages"][1]["content"],
    #         "answer": output,
    #     }
    # )


# with open("summary.json", mode="w", encoding="utf-8") as f:
#     json.dump(summary, f, ensure_ascii=False, indent=4)
