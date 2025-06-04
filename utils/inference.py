import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from .constants import ROLE_ASSISTANT


def get_model_and_tokenizer(
    model_name: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="bfloat16", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


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


def generate_teacher_outputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    batch_size: int,
) -> Dataset:
    teacher_answers: list[str] = []
    idx_range = range(0, len(dataset), batch_size)
    for i in tqdm(idx_range, desc="Teacher output generation"):
        samples = dataset[i : i + batch_size]
        outputs = generate_text_from_samples(model, tokenizer, samples)
        teacher_answers.extend(outputs)

    def replace_targets(sample: dict, idx: int):
        msgs = sample["messages"]
        for msg in msgs:
            if msg["role"] == ROLE_ASSISTANT:
                msg["content"] = teacher_answers[idx]
        return sample

    teacher_dataset = dataset.map(replace_targets, with_indices=True)
    return teacher_dataset
