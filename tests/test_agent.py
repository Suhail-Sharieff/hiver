"""Tests for Master AI Support Agent."""

import pytest
from src.agent import AISupportAgent, SupportAgentOutput
from src.taxonomy import IntentType, EscalationDecision


@pytest.fixture(scope="module")
def agent():
    return AISupportAgent()


def test_agent_process_pipeline(agent):
    query = "I would like to track my package, it was supposed to arrive today."
    res = agent.process(query)

    assert isinstance(res, SupportAgentOutput)
    assert res.customer_text == query
    assert res.intent == IntentType.ORDER_STATUS_DELIVERY
    assert res.intent_confidence > 0.0
    assert res.escalation_decision == EscalationDecision.AUTO_HANDLE
    assert len(res.escalation_reason) > 5
    assert len(res.drafted_reply) > 20
    assert "^HiverBot" in res.drafted_reply or "^" in res.drafted_reply
    assert len(res.retrieved_resolutions) > 0


def test_agent_escalation_flow(agent):
    query = "Unauthorized credit card charge of $200! I did not make this purchase!"
    res = agent.process(query)

    assert res.escalation_decision == EscalationDecision.ESCALATE_TO_HUMAN
    assert "security" in res.escalation_reason.lower() or "fraud" in res.escalation_reason.lower()
    assert "contact-us" in res.drafted_reply or "specialist" in res.drafted_reply or "security" in res.drafted_reply
