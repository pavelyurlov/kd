from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import pandas as pd
import json
import torch


BATCH_SIZE = 16
STEP = 100


@torch.inference_mode
def generate_text_from_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: dict,
    max_new_tokens: int = 256,
):
    msgs_batch = samples["messages"]

    text_inputs = [
        tokenizer.apply_chat_template(
            msgs[:1], tokenize=False, add_generation_prompt=True
        )
        for msgs in msgs_batch
    ]

    model_inputs = tokenizer(
        text=text_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="left",
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        # do_sample=False,
        # num_beams=1,
    )

    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_texts = tokenizer.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_texts


def main():
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

    idxs = [STEP * i for i in range(len(dataset) // STEP)]
    miniset = dataset.select(idxs)

    print(len(dataset), len(miniset))

    print(model.device)
    answers: list[str] = []
    summary = []
    for i in tqdm(
        range(0, len(miniset), BATCH_SIZE), desc=f"{model_name} {BATCH_SIZE}"
    ):
        samples = miniset[i : i + BATCH_SIZE]
        outputs = generate_text_from_samples(model, tokenizer, samples)
        answers.extend(outputs)
        for j in range(len(outputs)):
            summary.append(
                {
                    "question": samples["messages"][j][0]["content"],
                    "target": samples["messages"][j][1]["content"],
                    "answer": outputs[j],
                }
            )

    model_name = model_name.replace(":", "_")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    save_name = f"summary_{model_name}_batch{BATCH_SIZE}_step{STEP}.json"
    with open(save_name, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
