# Load metrics
import evaluate
import numpy as np
import torch
from transformers import EvalPrediction

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """
    Convert logits -> argmax to reduce memory usage in metric computation.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def get_compute_metrics(tokenizer):
    def compute_metrics(eval_pred: EvalPrediction):
        # Unpack the logits and labels.
        # (Assuming eval_pred.predictions is a tuple: (logits, labels))
        logits, labels = eval_pred.predictions

        # Convert logits to predictions by taking argmax
        predictions = torch.from_numpy(logits).cuda().argmax(dim=-1)
        labels = torch.from_numpy(labels).cuda()

        # Compute token-level accuracy over non -100 labels.
        mask = labels != -100
        labels_masked = labels[mask]
        predictions_masked = predictions[mask]
        accuracy = (predictions_masked == labels_masked).float().mean().item()

        # Move predictions and labels back to CPU for decoding.
        # Cast predictions to np.int32 to avoid overflow errors.
        predictions_cpu = predictions.cpu().numpy().astype(np.int32)
        labels_cpu = labels.cpu().numpy()

        # Decode predictions and labels using the tokenizer.
        decoded_preds = tokenizer.batch_decode(predictions_cpu, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            np.where(labels_cpu != -100, labels_cpu, tokenizer.pad_token_id),
            skip_special_tokens=True
        )

        # Remove extra spaces.
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute text-level metrics.
        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])["bleu"]
        rouge_l = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
        meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]

        # Compute token-level precision, recall, and F1.
        precision = precision_metric.compute(
            predictions=predictions_masked.cpu().numpy(),
            references=labels_masked.cpu().numpy(),
            average='micro'
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions_masked.cpu().numpy(),
            references=labels_masked.cpu().numpy(),
            average='micro'
        )["recall"]
        f1 = f1_metric.compute(
            predictions=predictions_masked.cpu().numpy(),
            references=labels_masked.cpu().numpy(),
            average='micro'
        )["f1"]

        # Clean up GPU memory.
        del predictions, labels, mask, labels_masked, predictions_masked
        torch.cuda.empty_cache()

        return {
            "accuracy": accuracy,
            "bleu": bleu,
            "rougeL": rouge_l,
            "meteor": meteor,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics
