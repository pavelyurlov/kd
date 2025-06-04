from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

from utils.inference import generate_text_from_samples

BATCH_SIZE = 16
STEP = 100


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
