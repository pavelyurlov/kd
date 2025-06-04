import evaluate
import numpy as np

from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction


def calculate_metrics(
    tokenizer: AutoTokenizer, eval_pred: EvalPrediction, language: str
) -> dict:
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

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_refs)
    results["rougeL"] = rouge_scores["rougeL"]

    bleu = evaluate.load("bleu")
    bleu_scores = bleu.compute(
        predictions=decoded_preds,
        references=[[ref] for ref in decoded_refs],
    )
    results["bleu"] = bleu_scores["bleu"]

    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_refs,
        lang=language,
    )
    results["bert_f1"] = sum(bert_scores["f1"]) / len(bert_scores["f1"])

    return results
