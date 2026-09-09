"""Tests for LLM-as-a-Judge and Rubric Evaluation."""

import pytest
from src.evaluation.judge import LLMJudge, RubricScores
from src.evaluation.human_agreement import compute_human_judge_agreement


@pytest.fixture(scope="module")
def judge():
    return LLMJudge()


def test_judge_evaluate_single(judge):
    scores = judge.evaluate_single(
        customer_text="Where is my order? It is two days late.",
        intent="ORDER_STATUS_DELIVERY",
        escalation_decision="AUTO_HANDLE",
        escalation_reason="Standard tracking inquiry.",
        drafted_reply="I apologize for the delay. You can track your shipment status here: https://www.amazon.com/your-orders. ^HiverBot",
        reference_reply="Please track your package here: https://t.co/abc ^CS",
    )

    assert isinstance(scores, RubricScores)
    assert 1.0 <= scores.groundedness <= 5.0
    assert 1.0 <= scores.empathy_and_tone <= 5.0
    assert 1.0 <= scores.actionability <= 5.0
    assert 1.0 <= scores.safety_and_pii <= 5.0
    assert 1.0 <= scores.escalation_appropriateness <= 5.0
    assert 1.0 <= scores.overall_score <= 5.0


def test_human_judge_agreement(judge):
    agreement = compute_human_judge_agreement(judge, max_samples=10)
    assert agreement["num_annotated_examples"] == 10
    assert 0.0 <= agreement["overall_exact_agreement_pct"] <= 100.0
    assert 0.0 <= agreement["overall_adjacent_agreement_pct"] <= 100.0
