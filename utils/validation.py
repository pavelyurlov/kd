import evaluate
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction

from .constants import ROLE_ASSISTANT


def calculate_metrics_on_text(
    predictions: list[str], references: list[str], language: str
) -> dict[str, float]:
    results = {}

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    results["rougeL"] = rouge_scores["rougeL"]

    bleu = evaluate.load("bleu")
    bleu_scores = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )
    results["bleu"] = bleu_scores["bleu"]

    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=language,
    )
    results["bert_f1"] = sum(bert_scores["f1"]) / len(bert_scores["f1"])

    return results


def metrics_between_sets(
    pred_dataset: Dataset, target_dataset: Dataset, language: str
) -> dict[str, float]:
    predictions = [
        sample["messages"][-1]["content"]
        for sample in pred_dataset
        if sample["messages"][-1]["role"] == ROLE_ASSISTANT
    ]
    references = [
        sample["messages"][-1]["content"]
        for sample in target_dataset
        if sample["messages"][-1]["role"] == ROLE_ASSISTANT
    ]

    return calculate_metrics_on_text(predictions, references, language)


def calculate_metrics(
    tokenizer: AutoTokenizer, eval_pred: EvalPrediction, language: str
) -> dict[str, float]:
    results = {}
    predictions, references = eval_pred

    pred_ids = predictions.argmax(axis=-1) if predictions.ndim == 3 else predictions
    decoded_preds = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    ref_ids = np.where(references == -100, tokenizer.pad_token_id, references)
    decoded_refs = tokenizer.batch_decode(
        ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    results = calculate_metrics_on_text(decoded_preds, decoded_refs, language)
    results = {f"student_{k}": v for k, v in results.items()}

    return results
