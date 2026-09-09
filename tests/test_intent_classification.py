"""Tests for Intent Classification module."""

import pytest
from src.agent import CalibratedIntentClassifier
from src.taxonomy import IntentType


@pytest.fixture(scope="module")
def classifier():
    return CalibratedIntentClassifier()


def test_order_status_delivery_intent(classifier):
    query = "Where is my package? The tracking has not updated for three days."
    res = classifier.classify(query)
    assert res.intent == IntentType.ORDER_STATUS_DELIVERY
    assert res.confidence >= 0.5


def test_return_refund_intent(classifier):
    query = "I want to send this item back and get a refund. Where is the return label?"
    res = classifier.classify(query)
    assert res.intent == IntentType.RETURN_REFUND
    assert res.confidence >= 0.5


def test_damaged_defective_intent(classifier):
    query = "The item arrived completely shattered and broken in the box!"
    res = classifier.classify(query)
    assert res.intent == IntentType.DAMAGED_DEFECTIVE_WRONG
    assert res.confidence >= 0.5


def test_account_payment_security_intent(classifier):
    query = "There is an unauthorized charge on my bank card from Prime that I did not make."
    res = classifier.classify(query)
    assert res.intent == IntentType.ACCOUNT_PAYMENT_SECURITY
    assert res.confidence >= 0.5


def test_escalation_complaint_intent(classifier):
    query = "This is the 3rd time your agent lied to me. I demand to speak to a manager or I will sue!"
    res = classifier.classify(query)
    assert res.intent == IntentType.ESCALATION_COMPLAINT
    assert res.confidence >= 0.9


def test_feedback_chitchat_intent(classifier):
    query = "Thank you so much, you guys are awesome and have a wonderful day!"
    res = classifier.classify(query)
    assert res.intent in [IntentType.FEEDBACK_CHITCHAT, IntentType.PRODUCT_SERVICE_INQUIRY]


def test_empty_query_fallback(classifier):
    res = classifier.classify("")
    assert isinstance(res.intent, IntentType)
    assert res.confidence > 0.0
