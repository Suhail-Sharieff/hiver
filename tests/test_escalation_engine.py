"""Tests for Policy-driven Escalation Engine."""

import pytest
from src.agent import EscalationEngine
from src.taxonomy import IntentType, EscalationDecision


@pytest.fixture(scope="module")
def engine():
    return EscalationEngine()


def test_escalate_severe_complaint(engine):
    text = "Speak to a manager immediately. This service is unacceptable."
    res = engine.evaluate(text, IntentType.ESCALATION_COMPLAINT, 0.95)
    assert res.decision == EscalationDecision.ESCALATE_TO_HUMAN
    assert res.confidence >= 0.85
    assert len(res.stated_reason) > 10


def test_escalate_security_and_fraud(engine):
    text = "Someone made an unauthorized transaction on my credit card."
    res = engine.evaluate(text, IntentType.ACCOUNT_PAYMENT_SECURITY, 0.90)
    assert res.decision == EscalationDecision.ESCALATE_TO_HUMAN
    assert "security" in res.stated_reason.lower() or "fraud" in res.stated_reason.lower()


def test_escalate_lost_package_marked_delivered(engine):
    text = "My order says delivered Saturday but was not! I was home all day."
    res = engine.evaluate(text, IntentType.ORDER_STATUS_DELIVERY, 0.88)
    assert res.decision == EscalationDecision.ESCALATE_TO_HUMAN
    assert "delivered" in res.stated_reason.lower()


def test_auto_handle_standard_tracking(engine):
    text = "Can you tell me when my order will be delivered? Tracking number is 12345."
    res = engine.evaluate(text, IntentType.ORDER_STATUS_DELIVERY, 0.85)
    assert res.decision == EscalationDecision.AUTO_HANDLE
    assert "tracking" in res.stated_reason.lower() or "orders" in res.stated_reason.lower()


def test_auto_handle_standard_returns(engine):
    text = "How do I return a shirt that is the wrong size?"
    res = engine.evaluate(text, IntentType.RETURN_REFUND, 0.85)
    assert res.decision == EscalationDecision.AUTO_HANDLE
    assert "return" in res.stated_reason.lower()
