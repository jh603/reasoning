import json
import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_precision_recall_f1(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # Exact Match
    em = float(normalized_prediction == normalized_ground_truth)

    # Tokenize both the normalized prediction and ground truth
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    # Calculate Precision and Recall
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common_tokens.values())

    # Precision = TP / (TP + FP)
    if len(prediction_tokens) > 0:
        precision = num_same / len(prediction_tokens)
    else:
        precision = 0.0

    # Recall = TP / (TP + FN)
    if len(ground_truth_tokens) > 0:
        recall = num_same / len(ground_truth_tokens)
    else:
        recall = 0.0

    # F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return em, precision, recall, f1
