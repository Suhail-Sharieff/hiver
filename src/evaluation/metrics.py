"""Evaluation Metrics module for Intent Classification, Escalation, and Text Quality."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for intents."""
    labels = sorted(list(set(y_true).union(set(y_pred))))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {}
    for idx, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(y_true.count(lbl)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": cm,
    }


def compute_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute precision, recall, F1, false negative rate, and false positive rate for escalation."""
    # Binary mapping: ESCALATE_TO_HUMAN is the positive class (1), AUTO_HANDLE is negative (0)
    pos_label = "ESCALATE_TO_HUMAN"
    neg_label = "AUTO_HANDLE"

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == neg_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == neg_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == neg_label and yp == neg_label)

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Operational risk metrics
    # False Negative Rate: escalated queries mistakenly auto-handled (critical safety breach)
    fnr = fn / max(1, tp + fn)
    # False Positive Rate: auto-handled queries unnecessarily sent to human agent (cost inefficiency)
    fpr = fp / max(1, tn + fp)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
    }


def compute_token_ngram_overlap(hyp_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Compute n-gram precision/recall overlap."""
    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0.0

    hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
    ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))

    overlap = sum((hyp_ngrams & ref_ngrams).values())
    total_hyp = sum(hyp_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    p = overlap / max(1, total_hyp)
    r = overlap / max(1, total_ref)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_lcs_length(s1: List[str], s2: List[str]) -> int:
    """Length of Longest Common Subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]


def compute_lexical_reply_metrics(
    hypotheses: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, BLEU-2, and safety PII leakage."""
    r1_scores = []
    r2_scores = []
    rl_scores = []
    bleu2_scores = []
    pii_violations = 0
    total_chars = 0

    # True PII leakage patterns: actual exposed credentials, card numbers, or SSNs
    # (Excludes advisory warnings like 'never share your password' or 'do not post passwords')
    pii_patterns = [
        r"\b(?:\d[ -]*?){13,16}\b",  # Credit card numbers
        r"\b\d{3}-\d{2}-\d{4}\b",    # SSN
        r"\b(?:my|your|the)\s+password\s+is\s+[:=]?\s*\S+\b",  # Password disclosure
        r"\bcvv\s*[:=]?\s*\d{3,4}\b",  # CVV disclosure
    ]

    for hyp, ref in zip(hypotheses, references):
        hyp_toks = re.findall(r"\w+", hyp.lower())
        ref_toks = re.findall(r"\w+", ref.lower())
        total_chars += len(hyp)

        # Check PII leakage
        for pat in pii_patterns:
            if re.search(pat, hyp.lower()):
                pii_violations += 1
                break

        # ROUGE-1
        r1_scores.append(compute_token_ngram_overlap(hyp_toks, ref_toks, 1))
        # ROUGE-2
        r2_scores.append(compute_token_ngram_overlap(hyp_toks, ref_toks, 2))

        # ROUGE-L
        lcs = compute_lcs_length(hyp_toks, ref_toks)
        p_lcs = lcs / max(1, len(hyp_toks))
        r_lcs = lcs / max(1, len(ref_toks))
        f_lcs = (2 * p_lcs * r_lcs / (p_lcs + r_lcs)) if (p_lcs + r_lcs) > 0 else 0.0
        rl_scores.append(f_lcs)

        # BLEU-2 approximation
        p1 = sum((Counter(hyp_toks) & Counter(ref_toks)).values()) / max(1, len(hyp_toks))
        hyp_2 = Counter(tuple(hyp_toks[i:i+2]) for i in range(len(hyp_toks)-1))
        ref_2 = Counter(tuple(ref_toks[i:i+2]) for i in range(len(ref_toks)-1))
        p2 = sum((hyp_2 & ref_2).values()) / max(1, len(hyp_toks)-1) if len(hyp_toks) > 1 else 0.0

        if p1 > 0 and p2 > 0:
            bp = min(1.0, math.exp(1 - len(ref_toks) / max(1, len(hyp_toks))))
            bleu = bp * math.sqrt(p1 * p2)
        else:
            bleu = 0.0
        bleu2_scores.append(bleu)

    n = max(1, len(hypotheses))
    return {
        "rouge_1": round(float(np.mean(r1_scores)), 4),
        "rouge_2": round(float(np.mean(r2_scores)), 4),
        "rouge_l": round(float(np.mean(rl_scores)), 4),
        "bleu_2": round(float(np.mean(bleu2_scores)), 4),
        "pii_leakage_rate": round(pii_violations / n, 4),
        "avg_char_length": round(total_chars / n, 1),
    }
