"""Model evaluation — metrics, confusion matrix, classification report, generation metrics.

Covers: classification, regression, NLP generation, object detection.
No code needed — call from UI after training.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def evaluate_classification(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """Full classification eval: accuracy, per-class precision/recall/F1, confusion matrix."""
    if not y_true or not y_pred:
        return {"status": "error", "message": "Empty predictions"}

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    all_labels = labels or sorted(set(y_true) | set(y_pred))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    # Confusion matrix
    cm = [[0] * len(all_labels) for _ in range(len(all_labels))]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1

    # Per-class metrics
    per_class = {}
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in y_true if t == label)
        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro/weighted averages
    macro_p = sum(v["precision"] for v in per_class.values()) / max(len(per_class), 1)
    macro_r = sum(v["recall"] for v in per_class.values()) / max(len(per_class), 1)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)

    total_support = sum(v["support"] for v in per_class.values())
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / max(total_support, 1)

    return {
        "accuracy": round(accuracy, 4),
        "total_samples": n,
        "per_class": per_class,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": {"labels": [str(l) for l in all_labels], "matrix": cm},
        "labels": [str(l) for l in all_labels],
    }


def evaluate_regression(y_true: list[float], y_pred: list[float]) -> dict:
    """Regression metrics: MSE, RMSE, MAE, R2, MAPE."""
    n = len(y_true)
    if n == 0:
        return {"status": "error"}

    errors = [t - p for t, p in zip(y_true, y_pred)]
    sq_errors = [e ** 2 for e in errors]
    abs_errors = [abs(e) for e in errors]

    mse = sum(sq_errors) / n
    rmse = math.sqrt(mse)
    mae = sum(abs_errors) / n

    mean_true = sum(y_true) / n
    ss_res = sum(sq_errors)
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    mape = sum(abs(e) / max(abs(t), 1e-8) for e, t in zip(errors, y_true)) / n * 100

    return {
        "mse": round(mse, 6), "rmse": round(rmse, 6),
        "mae": round(mae, 6), "r2": round(r2, 6),
        "mape": round(mape, 2),
        "total_samples": n,
        "residuals": {"min": round(min(errors), 4), "max": round(max(errors), 4),
                      "mean": round(sum(errors) / n, 4)},
    }


def evaluate_text_generation(references: list[str], predictions: list[str]) -> dict:
    """Text generation metrics: BLEU (1-4), ROUGE-1/2/L, exact match."""
    n = len(references)
    if n == 0:
        return {"status": "error"}

    exact_match = sum(1 for r, p in zip(references, predictions) if r.strip() == p.strip()) / n

    # BLEU (simplified n-gram precision)
    bleu_scores = {}
    for ng in [1, 2, 3, 4]:
        precisions = []
        for ref, pred in zip(references, predictions):
            ref_ngrams = _get_ngrams(ref.lower().split(), ng)
            pred_ngrams = _get_ngrams(pred.lower().split(), ng)
            if not pred_ngrams:
                precisions.append(0)
                continue
            matches = sum(min(ref_ngrams[g], pred_ngrams[g]) for g in pred_ngrams if g in ref_ngrams)
            precisions.append(matches / sum(pred_ngrams.values()))
        bleu_scores[f"bleu_{ng}"] = round(sum(precisions) / max(len(precisions), 1), 4)

    # ROUGE-1, ROUGE-2, ROUGE-L
    rouge_1, rouge_2, rouge_l = [], [], []
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()

        # ROUGE-1 (unigram overlap)
        ref_set = Counter(ref_tokens)
        pred_set = Counter(pred_tokens)
        overlap = sum(min(ref_set[t], pred_set[t]) for t in pred_set if t in ref_set)
        r1_p = overlap / max(len(pred_tokens), 1)
        r1_r = overlap / max(len(ref_tokens), 1)
        r1_f = 2 * r1_p * r1_r / max(r1_p + r1_r, 1e-8)
        rouge_1.append(r1_f)

        # ROUGE-2 (bigram)
        ref_bi = _get_ngrams(ref_tokens, 2)
        pred_bi = _get_ngrams(pred_tokens, 2)
        bi_overlap = sum(min(ref_bi[g], pred_bi[g]) for g in pred_bi if g in ref_bi)
        r2_p = bi_overlap / max(sum(pred_bi.values()), 1)
        r2_r = bi_overlap / max(sum(ref_bi.values()), 1)
        r2_f = 2 * r2_p * r2_r / max(r2_p + r2_r, 1e-8)
        rouge_2.append(r2_f)

        # ROUGE-L (longest common subsequence)
        lcs_len = _lcs_length(ref_tokens, pred_tokens)
        rl_p = lcs_len / max(len(pred_tokens), 1)
        rl_r = lcs_len / max(len(ref_tokens), 1)
        rl_f = 2 * rl_p * rl_r / max(rl_p + rl_r, 1e-8)
        rouge_l.append(rl_f)

    return {
        "exact_match": round(exact_match, 4),
        **bleu_scores,
        "rouge_1": round(sum(rouge_1) / max(len(rouge_1), 1), 4),
        "rouge_2": round(sum(rouge_2) / max(len(rouge_2), 1), 4),
        "rouge_l": round(sum(rouge_l) / max(len(rouge_l), 1), 4),
        "total_samples": n,
    }


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ── Hyperparameter Search ──

def grid_search(param_grid: dict[str, list]) -> list[dict]:
    """Generate all combinations from a parameter grid."""
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def random_search(param_ranges: dict[str, dict], n_trials: int = 20) -> list[dict]:
    """Generate random hyperparameter configs.

    Each param_range: {type: "float"|"int"|"choice", min/max or options}
    """
    import random
    configs = []
    for _ in range(n_trials):
        config = {}
        for key, spec in param_ranges.items():
            if spec["type"] == "float":
                if spec.get("log", False):
                    val = math.exp(random.uniform(math.log(spec["min"]), math.log(spec["max"])))
                else:
                    val = random.uniform(spec["min"], spec["max"])
                config[key] = round(val, 6)
            elif spec["type"] == "int":
                config[key] = random.randint(spec["min"], spec["max"])
            elif spec["type"] == "choice":
                config[key] = random.choice(spec["options"])
        configs.append(config)
    return configs
