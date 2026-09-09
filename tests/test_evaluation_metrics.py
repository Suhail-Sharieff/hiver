"""Tests for Evaluation Metrics."""

import pytest
from src.evaluation.metrics import (
    compute_escalation_metrics,
    compute_intent_metrics,
    compute_lexical_reply_metrics,
)


def test_compute_intent_metrics():
    y_true = ["ORDER_STATUS_DELIVERY", "RETURN_REFUND", "RETURN_REFUND"]
    y_pred = ["ORDER_STATUS_DELIVERY", "RETURN_REFUND", "ORDER_STATUS_DELIVERY"]

    res = compute_intent_metrics(y_true, y_pred)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert "ORDER_STATUS_DELIVERY" in res["per_class"]
    assert len(res["confusion_matrix"]) == 2


def test_compute_escalation_metrics():
    y_true = ["ESCALATE_TO_HUMAN", "AUTO_HANDLE", "ESCALATE_TO_HUMAN", "AUTO_HANDLE"]
    y_pred = ["ESCALATE_TO_HUMAN", "AUTO_HANDLE", "AUTO_HANDLE", "AUTO_HANDLE"]

    res = compute_escalation_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert res["recall"] == 0.5
    assert res["false_negative_rate"] == 0.5
    assert res["confusion_matrix"]["TP"] == 1
    assert res["confusion_matrix"]["FN"] == 1


def test_compute_lexical_metrics():
    hypotheses = ["Please track your order at amazon.com/your-orders ^HiverBot"]
    references = ["Please check your order status at amazon.com/your-orders ^CS"]

    res = compute_lexical_reply_metrics(hypotheses, references)
    assert res["rouge_1"] > 0.0
    assert res["rouge_l"] > 0.0
    assert res["pii_leakage_rate"] == 0.0
